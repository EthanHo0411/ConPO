"""Offline re-judge: A/B test a veto-prompt variant against the stored judge
request on logged pairs from <run>/judge_replies.jsonl (conpo/trainer.py's
_dump_judge_replies, present since commit d5fa5bb).

For each record this scores the exact rendered request the judge saw
(`judge_request`) as a baseline (also a self-consistency check on the judge:
same request, freshly sampled) and, unless --variant is "none", the same
request with the variant's sentence inserted. Both go through one vLLM burst
(conpo/judge.py's VLLMBatchedJudge._run_burst -> _vllm_burst), so a real run
loads the judge model once, not twice. Training code and the prompt it uses
are untouched; this only reads replies files and writes its own output.

Alongside --out's JSONL of per-record outcomes, a sibling `<out stem>.json`
report is written (and printed): n judged, n skipped, original/baseline/
variant keep rates, baseline-vs-original agreement, original-to-variant
transition counts, all of those split by direction, and (variant != "none"
only) the list of records where the variant's verdict differs from the
original -- see build_report's docstring for the exact field list.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from conpo import config
from conpo.judge import VLLMBatchedJudge, Verdict, _fail_closed_verdict, parse_verdict

# ---------------------------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------------------------

# build_veto_prompt's (conpo/judge.py) final paragraph, copied here as a
# literal string rather than imported: apply_variant asserts this exact text
# is present before inserting anything, so a future wording change in
# build_veto_prompt is caught loudly here instead of silently inserting the
# variant sentence in the wrong place (or not at all).
_JSON_INSTRUCTION_ANCHOR = (
    "Answer with a single JSON object and nothing else: "
    '{"degenerate": true or false, "reason": "one short sentence"}'
)

# name -> sentence inserted as its own paragraph immediately before the
# anchor above; None (only "none") leaves the request unchanged. A second
# variant is one more line here.
VARIANT_SENTENCES: dict[str, str | None] = {
    "none": None,
    "ignore_constraints": (
        "Do not answer degenerate=true because a response satisfies or fails "
        "one of the ignored constraints above; that alone is never a defect."
    ),
}


def apply_variant(judge_request: str, variant: str) -> str:
    """Apply a named prompt variant to a rendered judge request.

    "none" returns judge_request unchanged. Every other variant inserts its
    sentence as its own paragraph immediately before build_veto_prompt's
    JSON-format instruction paragraph. Raises ValueError for an unknown
    variant name, or if the anchor paragraph is not present verbatim in
    judge_request (build_veto_prompt's wording changed underneath this).
    """
    if variant not in VARIANT_SENTENCES:
        raise ValueError(f"unknown variant {variant!r} (expected one of {sorted(VARIANT_SENTENCES)})")
    sentence = VARIANT_SENTENCES[variant]
    if sentence is None:
        return judge_request
    if _JSON_INSTRUCTION_ANCHOR not in judge_request:
        raise ValueError(
            "judge_request is missing build_veto_prompt's JSON-format instruction "
            "anchor paragraph verbatim; check conpo/judge.py's build_veto_prompt for a wording change."
        )
    return judge_request.replace(_JSON_INSTRUCTION_ANCHOR, f"{sentence}\n\n{_JSON_INSTRUCTION_ANCHOR}", 1)


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------

# Injectable at judge_records(judge_fn=...): a fake in tests, run_vllm_judge
# (below) in real use. Rendered prompts in, one Verdict per prompt out, same
# order.
JudgeFn = Callable[[list[str], str], list[Verdict]]


def run_vllm_judge(prompts: list[str], model: str) -> list[Verdict]:
    """Default judge callable: one vLLM burst against `model`.

    Reuses VLLMBatchedJudge's spawned-child burst (_run_burst -> _vllm_burst,
    conpo/judge.py) instead of duplicating its sampling parameters or process
    isolation, then parses each raw reply the same way VLLMBatchedJudge.
    veto_many does: parse_verdict, falling back to _fail_closed_verdict
    (reject, parse_ok=False) on an unparseable reply rather than raising.
    """
    if not prompts:
        return []
    raw_replies = VLLMBatchedJudge(model)._run_burst(prompts)
    verdicts: list[Verdict] = []
    for raw in raw_replies:
        try:
            verdicts.append(parse_verdict(raw))
        except ValueError:
            verdicts.append(_fail_closed_verdict(raw))
    return verdicts


def load_records(
    paths: Sequence[str | Path],
    directions: Sequence[str],
    limit: int,
) -> tuple[list[dict[str, Any]], int]:
    """Load judge_replies.jsonl records from `paths` (concatenated in order),
    keeping only those whose "direction" is in `directions`. `limit` caps the
    number of direction-matching records read (0 = no cap), applied before
    the judge_request check below, so it bounds a smoke run regardless of how
    many turn out skippable.

    A record without "judge_request" (the old 8-per-burst sample format,
    before commit d5fa5bb) cannot be re-judged and is dropped from the
    returned list; the count dropped this way is the second return value.
    """
    directions_set = set(directions)
    selected: list[dict[str, Any]] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("direction") not in directions_set:
                    continue
                selected.append(record)
                if limit and len(selected) >= limit:
                    break
        if limit and len(selected) >= limit:
            break
    matched = [r for r in selected if "judge_request" in r]
    n_skipped = len(selected) - len(matched)
    return matched, n_skipped


def judge_records(
    records: list[dict[str, Any]],
    variant: str,
    judge_model: str,
    judge_fn: JudgeFn = run_vllm_judge,
) -> list[dict[str, Any]]:
    """Judge every record's stored request (baseline) and, unless variant is
    "none", the variant request -- one judge_fn call across both prompt sets.

    Each outcome carries step/direction/families/original_keep (from the
    record) plus baseline_keep/baseline_reason/baseline_parse_ok, and,
    variant != "none" only, the matching variant_* fields.
    """
    has_variant = variant != "none"
    baseline_prompts = [r["judge_request"] for r in records]
    variant_prompts = [apply_variant(r["judge_request"], variant) for r in records] if has_variant else []
    verdicts = judge_fn(baseline_prompts + variant_prompts, judge_model) if records else []
    baseline_verdicts = verdicts[: len(records)]
    variant_verdicts = verdicts[len(records) :] if has_variant else []

    outcomes: list[dict[str, Any]] = []
    for i, record in enumerate(records):
        bv = baseline_verdicts[i]
        outcome: dict[str, Any] = {
            "step": record["step"],
            "direction": record["direction"],
            "families": record["families"],
            "original_keep": record["keep"],
            "baseline_keep": bv.keep,
            "baseline_reason": bv.reason,
            "baseline_parse_ok": bv.parse_ok,
        }
        if has_variant:
            vv = variant_verdicts[i]
            outcome["variant_keep"] = vv.keep
            outcome["variant_reason"] = vv.reason
            outcome["variant_parse_ok"] = vv.parse_ok
        outcomes.append(outcome)
    return outcomes


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _stats(rows: list[dict[str, Any]], has_variant: bool) -> dict[str, Any]:
    n = len(rows)
    stats: dict[str, Any] = {
        "n": n,
        "original_keep_rate": _rate(sum(1 for r in rows if r["original_keep"]), n),
        "baseline_keep_rate": _rate(sum(1 for r in rows if r["baseline_keep"]), n),
        "baseline_agreement": _rate(sum(1 for r in rows if r["baseline_keep"] == r["original_keep"]), n),
    }
    if has_variant:
        stats["variant_keep_rate"] = _rate(sum(1 for r in rows if r["variant_keep"]), n)
        stats["original_veto_to_variant_keep"] = sum(1 for r in rows if not r["original_keep"] and r["variant_keep"])
        stats["original_keep_to_variant_veto"] = sum(1 for r in rows if r["original_keep"] and not r["variant_keep"])
    return stats


def build_report(
    outcomes: list[dict[str, Any]],
    n_skipped: int,
    variant: str,
    judge_model: str,
    directions: Sequence[str],
) -> dict[str, Any]:
    """{n_judged, n_skipped, variant, judge_model, directions,
    overall: {n, original_keep_rate, baseline_keep_rate, baseline_agreement,
              variant_keep_rate?, original_veto_to_variant_keep?, original_keep_to_variant_veto?},
    by_direction: {<direction>: <same stats as overall>, ...},
    flips?: [{step, direction, families, baseline_reason, variant_reason}, ...]}
    The variant_*/flips fields are present only when variant != "none".
    flips lists every record where original_keep != variant_keep (the
    records counted in the two transition totals above)."""
    has_variant = variant != "none"
    report: dict[str, Any] = {
        "n_judged": len(outcomes),
        "n_skipped": n_skipped,
        "variant": variant,
        "judge_model": judge_model,
        "directions": list(directions),
        "overall": _stats(outcomes, has_variant),
        "by_direction": {d: _stats([r for r in outcomes if r["direction"] == d], has_variant) for d in directions},
    }
    if has_variant:
        report["flips"] = [
            {
                "step": r["step"],
                "direction": r["direction"],
                "families": r["families"],
                "baseline_reason": r["baseline_reason"],
                "variant_reason": r["variant_reason"],
            }
            for r in outcomes
            if r["original_keep"] != r["variant_keep"]
        ]
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline A/B re-judge of logged veto verdicts: stored judge_request vs a prompt variant."
    )
    parser.add_argument("--replies", nargs="+", required=True, help="judge_replies.jsonl file(s) to re-judge")
    parser.add_argument("--variant", default="none", choices=sorted(VARIANT_SENTENCES), help="prompt variant name")
    parser.add_argument("--judge-model", default=config.JUDGE_MODEL, help="judge model for the vLLM burst")
    parser.add_argument("--out", required=True, help="output JSONL of per-record outcomes (a sibling .json report is written next to it)")
    parser.add_argument("--limit", type=int, default=0, help="judge only the first N direction-matching records, 0 = all")
    parser.add_argument("--directions", default="forward,reverse", help="comma-separated subset of forward,reverse")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    directions = [d.strip() for d in args.directions.split(",") if d.strip()]

    records, n_skipped = load_records(args.replies, directions, args.limit)
    outcomes = judge_records(records, args.variant, args.judge_model)
    report = build_report(outcomes, n_skipped, args.variant, args.judge_model, directions)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in outcomes:
            f.write(json.dumps(row) + "\n")
    report_path = out_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"wrote {out_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
