"""Offline re-judge: A/B test one or more veto-prompt variants against the
stored judge request on logged pairs from <run>/judge_replies.jsonl
(conpo/trainer.py's _dump_judge_replies, present since commit d5fa5bb).

For each record this scores the exact rendered request the judge saw
(`judge_request`) as a baseline (also a self-consistency check on the judge:
same request, freshly sampled) plus every variant named in `--variant`
(comma-separated; "none", the default, scores no variant). Every prompt --
baseline and every variant, for every record -- goes through one combined
call to `judge_fn` (conpo/judge.py's VLLMBatchedJudge._run_burst ->
_vllm_burst for a real run), so a real run loads the judge model once
regardless of how many variants are requested. Training code and the prompt
it uses are untouched; this only reads replies files and writes its own
output.

Alongside --out's JSONL of per-record outcomes, a sibling `<out stem>.json`
report is written (and printed): n judged, n skipped, baseline keep rate and
agreement with the original verdict (overall and by direction), and a
`variants` map with, per requested variant, its keep rate, agreement with
the original verdict, and transition counts (overall and by direction) plus
a flips list capped at 50 entries -- see build_report's docstring for the
exact field list.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from conpo import config
from conpo.judge import VLLMBatchedJudge, Verdict, _fail_closed_verdict, parse_verdict, strip_think

# ---------------------------------------------------------------------------
# Prompt variants: each is a pure function of the rendered judge_request
# string, doing surgery on one of build_veto_prompt's (conpo/judge.py) fixed
# anchors. Every anchor below is copied here as a literal (or a regex over
# the two possible A/B swaps) rather than imported, so a wording change in
# build_veto_prompt is caught loudly here instead of silently inserting a
# variant sentence in the wrong place, or not at all.
# ---------------------------------------------------------------------------

_MISSING_ANCHOR_SUFFIX = "; check conpo/judge.py's build_veto_prompt for a wording change."

# The final paragraph's exact text, fixed regardless of the A/B swap.
_JSON_INSTRUCTION_ANCHOR = (
    "Answer with a single JSON object and nothing else: "
    '{"degenerate": true or false, "reason": "one short sentence"}'
)

# build_veto_prompt swaps which letter carries y_minus per request
# (judge.py's veto_plus_label), so the paragraph naming "Response X is a
# deliberately modified variant of response Y..." embeds A/B either way.
# Each half of that paragraph gets its own regex so explicit_exclusions can
# replace only the second sentence and two_examples can insert before the
# first, without the two variants needing to agree on a single anchor.
_MODIFIED_VARIANT_LEAD_RE = re.compile(
    r"Response (?P<minus>[AB]) is a deliberately modified variant of "
    r"response (?P<plus>[AB])\. "
)
_JUDGE_ONLY_SENTENCE_RE = re.compile(
    r"Judge only whether response (?P<minus>[AB]) is "
    r"DEGENERATE: truncated, garbled, off-task, or changed in meaning "
    r"beyond the ignored constraints above, compared with response "
    r"(?P<plus>[AB])\.\n\n"
)

_IGNORE_CONSTRAINTS_SENTENCE = (
    "Do not answer degenerate=true because a response satisfies or fails "
    "one of the ignored constraints above; that alone is never a defect."
)

_EXPLICIT_EXCLUSIONS_TEMPLATE = (
    "Judge only whether response {minus} is DEGENERATE compared with response {plus}: "
    "truncated mid-thought, garbled or non-fluent text, off-task content, or a change of "
    "meaning that goes beyond the listed constraints. Differences that consist only of "
    "satisfying or violating the listed constraints (a missing or added keyword, a changed "
    "first or last word, punctuation, letter case, a heading, a postscript, paragraph or "
    "section breaks, bullet formatting, length, language, a quoted or wrapped answer) are "
    "expected and must be answered degenerate=false.\n\n"
)

_CHECKLIST_JSON_INSTRUCTION = (
    "Answer with a single JSON object and nothing else: "
    '{"truncated": true or false, "garbled": true or false, '
    '"off_task": true or false, "meaning_changed_beyond_constraints": true or false, '
    '"degenerate": true or false, "reason": "one short sentence"}. '
    "Set degenerate to true only if at least one of the four checks is true."
)

# Generic worked examples (not from any real pair), under 80 words each, in
# the same A/B form as the main prompt: one constraint-only difference
# (degenerate=false) and one truncation (degenerate=true).
_EXAMPLE_1 = (
    "Example 1:\nResponse A:\nPack sunscreen, water, and a map for the hike.\n\n"
    "Response B:\nPack water and a map for the hike.\n\n"
    "Response B is a deliberately modified variant of response A that omits one required "
    "keyword. Judge only whether response B is DEGENERATE: it is not, since only the listed "
    "constraint differs.\n"
    'Answer: {"degenerate": false, "reason": "only the listed constraint differs"}\n\n'
)
_EXAMPLE_2 = (
    "Example 2:\nResponse A:\nThe museum opens at nine and closes at five on weekdays.\n\n"
    "Response B:\nThe museum opens at nine and clos\n\n"
    "Response B is a deliberately modified variant of response A that stops mid-sentence. "
    "Judge only whether response B is DEGENERATE: it is, since the text is truncated.\n"
    'Answer: {"degenerate": true, "reason": "truncated"}\n\n'
)


def _apply_ignore_constraints(text: str) -> str:
    if _JSON_INSTRUCTION_ANCHOR not in text:
        raise ValueError(
            "judge_request is missing build_veto_prompt's JSON-format instruction "
            f"anchor paragraph verbatim{_MISSING_ANCHOR_SUFFIX}"
        )
    return text.replace(_JSON_INSTRUCTION_ANCHOR, f"{_IGNORE_CONSTRAINTS_SENTENCE}\n\n{_JSON_INSTRUCTION_ANCHOR}", 1)


def _apply_explicit_exclusions(text: str) -> str:
    match = _JUDGE_ONLY_SENTENCE_RE.search(text)
    if match is None:
        raise ValueError(
            'judge_request is missing build_veto_prompt\'s "Judge only whether ... is '
            f'DEGENERATE" sentence verbatim{_MISSING_ANCHOR_SUFFIX}'
        )
    replacement = _EXPLICIT_EXCLUSIONS_TEMPLATE.format(minus=match["minus"], plus=match["plus"])
    return text[: match.start()] + replacement + text[match.end() :]


def _apply_checklist_json(text: str) -> str:
    if _JSON_INSTRUCTION_ANCHOR not in text:
        raise ValueError(
            "judge_request is missing build_veto_prompt's JSON-format instruction "
            f"anchor paragraph verbatim{_MISSING_ANCHOR_SUFFIX}"
        )
    return text.replace(_JSON_INSTRUCTION_ANCHOR, _CHECKLIST_JSON_INSTRUCTION, 1)


def _apply_two_examples(text: str) -> str:
    match = _MODIFIED_VARIANT_LEAD_RE.search(text)
    if match is None:
        raise ValueError(
            'judge_request is missing build_veto_prompt\'s "Response X is a deliberately '
            f'modified variant" sentence verbatim{_MISSING_ANCHOR_SUFFIX}'
        )
    return text[: match.start()] + _EXAMPLE_1 + _EXAMPLE_2 + text[match.start() :]


# name -> pure transform of a rendered judge_request. "none" (handled by
# apply_variant directly, not listed here) is the identity.
VARIANT_FUNCS: dict[str, Callable[[str], str]] = {
    "ignore_constraints": _apply_ignore_constraints,
    "explicit_exclusions": _apply_explicit_exclusions,
    "checklist_json": _apply_checklist_json,
    "two_examples": _apply_two_examples,
    "ignore_plus_exclusions": lambda t: _apply_ignore_constraints(_apply_explicit_exclusions(t)),
}
VARIANT_NAMES: tuple[str, ...] = tuple(VARIANT_FUNCS)


def apply_variant(judge_request: str, variant: str) -> str:
    """Apply a named prompt variant to a rendered judge request.

    "none" returns judge_request unchanged. Every other variant runs its
    registered transform, which raises ValueError if its anchor is not
    present verbatim in judge_request. Raises ValueError for an unknown
    variant name.
    """
    if variant == "none":
        return judge_request
    if variant not in VARIANT_FUNCS:
        raise ValueError(f"unknown variant {variant!r} (expected one of {('none',) + VARIANT_NAMES})")
    return VARIANT_FUNCS[variant](judge_request)


def parse_variants(spec: str) -> list[str]:
    """Parse a comma-separated --variant spec into an ordered, deduplicated
    list of variant names. A bare "none" (alone or mixed with real names)
    contributes nothing: it is the "score no variant" default, not a name to
    score. Raises ValueError for an unknown name.
    """
    names: list[str] = []
    for token in (t.strip() for t in spec.split(",")):
        if not token or token == "none":
            continue
        if token not in VARIANT_FUNCS:
            raise ValueError(f"unknown variant {token!r} (expected one of {('none',) + VARIANT_NAMES})")
        if token not in names:
            names.append(token)
    return names


# ---------------------------------------------------------------------------
# Checklist boolean extraction (checklist_json variant)
# ---------------------------------------------------------------------------

_CHECKLIST_KEYS = ("truncated", "garbled", "off_task", "meaning_changed_beyond_constraints")


def extract_checklist(raw: str) -> dict[str, bool] | None:
    """Extract the checklist_json variant's four booleans from a raw judge
    reply, when all four keys are present in its first parseable JSON
    object. None for any other reply shape (a plain degenerate/reason
    reply, prose, or an unparseable one).

    Same anchor-free JSON scan as conpo.judge.parse_verdict (a reply is as
    often prose-wrapped as not), duplicated at this small size rather than
    exposed from conpo/judge.py, which has no reason to know about a
    rejudge-only prompt variant.
    """
    text = strip_think(raw)
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        start = text.find("{", idx)
        if start == -1:
            return None
        try:
            obj, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(obj, dict) and all(k in obj for k in _CHECKLIST_KEYS):
            return {k: bool(obj[k]) for k in _CHECKLIST_KEYS}
        idx = start + 1
    return None


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

    Every record read is stamped with "index", its 0-based position among
    all JSONL records read across `paths` in order -- independent of the
    direction filter and the judge_request check below, so it lines up with
    the position a reader labeling judge_replies.jsonl by hand would count.
    A record that already carries "index" keeps its own value.
    scripts/score_rejudge.py matches a labels file to rejudge's output by
    this same index.

    A record without "judge_request" (the old 8-per-burst sample format,
    before commit d5fa5bb) cannot be re-judged and is dropped from the
    returned list; the count dropped this way is the second return value.
    """
    directions_set = set(directions)
    selected: list[dict[str, Any]] = []
    index = 0
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record.setdefault("index", index)
                index += 1
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
    variants: Sequence[str],
    judge_model: str,
    judge_fn: JudgeFn = run_vllm_judge,
) -> list[dict[str, Any]]:
    """Judge every record's stored request (baseline) plus every named
    variant's rendered request, in one call to `judge_fn`: baseline prompts
    followed by each variant's prompts, concatenated, in the order of
    `variants` -- so a real run's single vLLM burst loads the judge model
    once for the whole batch regardless of how many variants are requested.

    Each outcome carries index/step/direction/families/original_keep (from
    the record), a "baseline" verdict ({keep, reason, parse_ok}), and a
    "variants" map with one entry per name in `variants` (empty when
    `variants` is empty), each entry the same three fields plus "checklist"
    (the checklist_json variant's four booleans) when the raw reply parsed
    them.
    """
    baseline_prompts = [r["judge_request"] for r in records]
    variant_prompts = {v: [apply_variant(r["judge_request"], v) for r in records] for v in variants}

    all_prompts = list(baseline_prompts)
    for v in variants:
        all_prompts.extend(variant_prompts[v])
    verdicts = judge_fn(all_prompts, judge_model) if records else []

    n = len(records)
    baseline_verdicts = verdicts[:n]
    variant_verdicts: dict[str, list[Verdict]] = {}
    offset = n
    for v in variants:
        variant_verdicts[v] = verdicts[offset : offset + n]
        offset += n

    outcomes: list[dict[str, Any]] = []
    for i, record in enumerate(records):
        bv = baseline_verdicts[i]
        outcome: dict[str, Any] = {
            "index": record["index"],
            "step": record["step"],
            "direction": record["direction"],
            "families": record["families"],
            "original_keep": record["keep"],
            "baseline": {"keep": bv.keep, "reason": bv.reason, "parse_ok": bv.parse_ok},
        }
        variants_out: dict[str, Any] = {}
        for v in variants:
            vv = variant_verdicts[v][i]
            entry: dict[str, Any] = {"keep": vv.keep, "reason": vv.reason, "parse_ok": vv.parse_ok}
            checklist = extract_checklist(vv.raw)
            if checklist is not None:
                entry["checklist"] = checklist
            variants_out[v] = entry
        outcome["variants"] = variants_out
        outcomes.append(outcome)
    return outcomes


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _baseline_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    return {
        "n": n,
        "original_keep_rate": _rate(sum(1 for r in rows if r["original_keep"]), n),
        "baseline_keep_rate": _rate(sum(1 for r in rows if r["baseline"]["keep"]), n),
        "baseline_agreement": _rate(sum(1 for r in rows if r["baseline"]["keep"] == r["original_keep"]), n),
    }


def _variant_stats(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    n = len(rows)
    keeps = [r["variants"][variant]["keep"] for r in rows]
    return {
        "n": n,
        "keep_rate": _rate(sum(keeps), n),
        "agreement_with_original": _rate(sum(1 for r, k in zip(rows, keeps) if k == r["original_keep"]), n),
        "original_veto_to_variant_keep": sum(1 for r, k in zip(rows, keeps) if not r["original_keep"] and k),
        "original_keep_to_variant_veto": sum(1 for r, k in zip(rows, keeps) if r["original_keep"] and not k),
    }


def _variant_flips(rows: list[dict[str, Any]], variant: str, cap: int = 50) -> list[dict[str, Any]]:
    flips = [
        {
            "index": r["index"],
            "step": r["step"],
            "direction": r["direction"],
            "families": r["families"],
            "baseline_reason": r["baseline"]["reason"],
            "variant_reason": r["variants"][variant]["reason"],
        }
        for r in rows
        if r["variants"][variant]["keep"] != r["original_keep"]
    ]
    return flips[:cap]


def build_report(
    outcomes: list[dict[str, Any]],
    n_skipped: int,
    variants: Sequence[str],
    judge_model: str,
    directions: Sequence[str],
) -> dict[str, Any]:
    """{n_judged, n_skipped, judge_model, directions,
    overall/by_direction: baseline-only stats
        {n, original_keep_rate, baseline_keep_rate, baseline_agreement},
    variants: {<name>: {
        overall/by_direction: {n, keep_rate, agreement_with_original,
            original_veto_to_variant_keep, original_keep_to_variant_veto},
        flips: [{index, step, direction, families, baseline_reason,
            variant_reason}, ...] (records where original_keep !=
            variant_keep, capped at 50 entries),
    }, ...} -- one entry per name in `variants` (empty map when `variants`
    is empty).
    """
    return {
        "n_judged": len(outcomes),
        "n_skipped": n_skipped,
        "judge_model": judge_model,
        "directions": list(directions),
        "overall": _baseline_stats(outcomes),
        "by_direction": {d: _baseline_stats([r for r in outcomes if r["direction"] == d]) for d in directions},
        "variants": {
            v: {
                "overall": _variant_stats(outcomes, v),
                "by_direction": {
                    d: _variant_stats([r for r in outcomes if r["direction"] == d], v) for d in directions
                },
                "flips": _variant_flips(outcomes, v),
            }
            for v in variants
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline A/B re-judge of logged veto verdicts: stored judge_request vs one or more prompt variants."
    )
    parser.add_argument("--replies", nargs="+", required=True, help="judge_replies.jsonl file(s) to re-judge")
    parser.add_argument(
        "--variant",
        default="none",
        type=parse_variants,
        help=f"comma-separated variant name(s) to score, or 'none' (default). One of {VARIANT_NAMES}",
    )
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
