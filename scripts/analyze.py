"""Build the per-benchmark and hypothesis-table report from a directory of
evaluated runs.

Layout (--runs): a directory of run subdirectories, each named
"<arm>_s<seed>" (e.g. "full_s42"), each holding the four files
scripts/eval_external.py / scripts/eval_internal.py write:
eval_external.json, eval_external.json.verdicts.jsonl, eval_internal.json,
eval_internal.json.verdicts.jsonl. A run directory missing eval_external.json
or eval_internal.json simply contributes no rows for that half of the
report.

"<arm>_s<seed>" cannot spell every arm name conpo.arms.HYPOTHESES uses --
the selection/lambda_p override rows name an arm like "full:selection=uniform",
and ":" is not a legal Windows directory character. --manifest is the
explicit alternative for those: a JSON file holding a list of
{"arm": ..., "seed": ..., "dir": ...} objects, one per run, with `dir`
naming that run's directory directly (any path, any arm string). Pass
--manifest instead of --runs when the run set includes an override arm.

Outputs: Markdown to --out, the same data as JSON to "<out>.json". Extras
(Wilson per arm, McNemar for the H1 pair, Holm across the hypothesis rows)
are computed and printed only when --extras is passed.
"""

import argparse
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from conpo import stats
from conpo.arms import HYPOTHESES, Hypothesis

_RUN_DIR_RE = re.compile(r"^(?P<arm>.+)_s(?P<seed>\d+)$")
_VERDICT_COLUMNS = ("prompt_strict", "prompt_loose")


@dataclass(frozen=True)
class RunRef:
    arm: str
    seed: int
    dir: Path


@dataclass(frozen=True)
class RunData:
    arm: str
    seed: int
    external: dict[str, Any] | None  # scripts/eval_external.py's JSON, keyed "ifeval"/"ifbench"
    internal: dict[str, Any] | None  # scripts/eval_internal.py's JSON, keyed by split name
    external_verdicts: list[dict[str, Any]]
    internal_verdicts: list[dict[str, Any]]


# --- discovery and loading ---------------------------------------------------


def discover_runs(runs_dir: Path) -> list[RunRef]:
    """Immediate subdirectories of `runs_dir` matching "<arm>_s<seed>"."""
    out = []
    for child in sorted(runs_dir.iterdir()):
        if not child.is_dir():
            continue
        match = _RUN_DIR_RE.match(child.name)
        if match is None:
            continue
        out.append(RunRef(arm=match.group("arm"), seed=int(match.group("seed")), dir=child))
    return out


def load_manifest(path: Path) -> list[RunRef]:
    """Explicit run list: a JSON array of {"arm", "seed", "dir"} objects."""
    entries = json.loads(path.read_text(encoding="utf-8"))
    return [RunRef(arm=e["arm"], seed=int(e["seed"]), dir=Path(e["dir"])) for e in entries]


def _load_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl_optional(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_run(ref: RunRef) -> RunData:
    return RunData(
        arm=ref.arm,
        seed=ref.seed,
        external=_load_json_optional(ref.dir / "eval_external.json"),
        internal=_load_json_optional(ref.dir / "eval_internal.json"),
        external_verdicts=_load_jsonl_optional(ref.dir / "eval_external.json.verdicts.jsonl"),
        internal_verdicts=_load_jsonl_optional(ref.dir / "eval_internal.json.verdicts.jsonl"),
    )


# --- (1) per-benchmark table -------------------------------------------------

_EXTERNAL_METRIC_COLUMNS = ("prompt_strict", "prompt_loose", "instruction_strict", "instruction_loose")


def external_table(runs: Sequence[RunData]) -> list[dict[str, Any]]:
    """One row per (arm, benchmark) with mean/std over seeds of the four
    ScoreReport columns, for every arm/benchmark that has data."""
    rows = []
    for arm in sorted({r.arm for r in runs}):
        arm_runs = [r for r in runs if r.arm == arm]
        for benchmark in ("ifeval", "ifbench"):
            per_column: dict[str, list[float]] = {c: [] for c in _EXTERNAL_METRIC_COLUMNS}
            for r in arm_runs:
                report = (r.external or {}).get(benchmark)
                if report is None:
                    continue
                for column in _EXTERNAL_METRIC_COLUMNS:
                    per_column[column].append(report[column])
            if not per_column["prompt_loose"]:
                continue
            row: dict[str, Any] = {"arm": arm, "benchmark": benchmark, "n_seeds": len(per_column["prompt_loose"])}
            for column, values in per_column.items():
                mean, std = stats.mean_std(values)
                row[f"{column}_mean"] = mean
                row[f"{column}_std"] = std
            rows.append(row)
    return rows


def internal_table(runs: Sequence[RunData]) -> list[dict[str, Any]]:
    """One row per (arm, split) with mean/std over seeds of PAcc."""
    rows = []
    for arm in sorted({r.arm for r in runs}):
        arm_runs = [r for r in runs if r.arm == arm]
        per_split: dict[str, list[float]] = {}
        for r in arm_runs:
            for split_name, summary in (r.internal or {}).items():
                per_split.setdefault(split_name, []).append(summary["pacc"])
        for split_name in sorted(per_split):
            mean, std = stats.mean_std(per_split[split_name])
            rows.append(
                {"arm": arm, "split": split_name, "n_seeds": len(per_split[split_name]), "pacc_mean": mean, "pacc_std": std}
            )
    return rows


# --- (2) hypothesis table -----------------------------------------------------


def _parse_metric(metric: str) -> tuple[str, str]:
    """"ifbench_prompt_loose" -> ("ifbench", "prompt_loose"): source is an
    external benchmark name ("ifeval"/"ifbench") or an internal split name;
    column is the verdicts.jsonl field to read."""
    for column in _VERDICT_COLUMNS:
        suffix = "_" + column
        if metric.endswith(suffix):
            return metric[: -len(suffix)], column
    raise ValueError(f"unrecognised metric name: {metric!r} (expected a '..._prompt_strict'/'..._prompt_loose' suffix)")


def _verdict_map(run: RunData, source: str, column: str) -> dict[Any, bool]:
    """key -> bool for `column`, from whichever of this run's two verdicts
    files has a matching benchmark (external) or split (internal) field."""
    out: dict[Any, bool] = {}
    for row in run.external_verdicts:
        if row.get("benchmark") == source:
            out[row["key"]] = row[column]
    for row in run.internal_verdicts:
        if row.get("split") == source:
            out[row["key"]] = row[column]
    return out


def _matched_seed_pairs(treatment: Sequence[RunData], control: Sequence[RunData]) -> list[tuple[RunData, RunData]]:
    """Seed i of treatment against seed i of control; when seed counts
    differ, use the seeds present in both; when no seed value is shared but
    exactly one run exists on each side, pair that single available run."""
    t_by_seed = {r.seed: r for r in treatment}
    c_by_seed = {r.seed: r for r in control}
    common = sorted(set(t_by_seed) & set(c_by_seed))
    if common:
        return [(t_by_seed[s], c_by_seed[s]) for s in common]
    if len(treatment) == 1 and len(control) == 1:
        return [(treatment[0], control[0])]
    return []


def _metric_values(runs: Sequence[RunData], source: str, column: str) -> list[float]:
    """Per-seed metric value read straight from the aggregate JSON (not the
    verdicts), for the plain mean-over-all-seeds headline number."""
    values = []
    for r in runs:
        report = (r.external or {}).get(source)
        if report is not None:
            values.append(report[column])
            continue
        summary = (r.internal or {}).get(source)
        if summary is not None:
            values.append(summary["pacc"])  # no strict/loose distinction internally; both columns read pacc
    return values


def hypothesis_table(runs: Sequence[RunData], hypotheses: Sequence[Hypothesis] = HYPOTHESES) -> list[dict[str, Any]]:
    rows = []
    for hyp in hypotheses:
        source, column = _parse_metric(hyp.metric)
        treatment_runs = [r for r in runs if r.arm == hyp.treatment]
        control_runs = [r for r in runs if r.arm == hyp.control]

        mean_t, _ = stats.mean_std(_metric_values(treatment_runs, source, column))
        mean_c, _ = stats.mean_std(_metric_values(control_runs, source, column))

        pairs = _matched_seed_pairs(treatment_runs, control_runs)
        pair_diffs, pair_los, pair_his = [], [], []
        for t_run, c_run in pairs:
            t_map = _verdict_map(t_run, source, column)
            c_map = _verdict_map(c_run, source, column)
            common_keys = sorted(set(t_map) & set(c_map), key=str)
            a = [t_map[k] for k in common_keys]
            b = [c_map[k] for k in common_keys]
            diff, lo, hi = stats.paired_bootstrap_diff(a, b)
            pair_diffs.append(diff)
            pair_los.append(lo)
            pair_his.append(hi)

        n_pairs = len(pairs)
        if n_pairs == 0:
            verdict, note = "pending", "no matched seed pairs"
            bootstrap_diff = ci_lo = ci_hi = 0.0
        else:
            bootstrap_diff = sum(pair_diffs) / n_pairs
            ci_lo = sum(pair_los) / n_pairs
            ci_hi = sum(pair_his) / n_pairs
            verdict = stats.verdict_from_ci(ci_lo, ci_hi, hyp.expected_sign)
            note = "only one seed" if n_pairs == 1 else None

        rows.append(
            {
                "id": hyp.id,
                "treatment": hyp.treatment,
                "control": hyp.control,
                "metric": hyp.metric,
                "expected_sign": hyp.expected_sign,
                "mean_diff": mean_t - mean_c,
                "bootstrap_diff": bootstrap_diff,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "n_seed_pairs": n_pairs,
                "verdict": verdict,
                "note": note,
            }
        )
    return rows


# --- extras --------------------------------------------------------------


def build_extras(runs: Sequence[RunData]) -> dict[str, Any]:
    """Wilson per arm (H1's metric), McNemar for the H1 pair, Holm across
    every hypothesis row's own McNemar p-value."""
    headline = HYPOTHESES[0]
    source, column = _parse_metric(headline.metric)

    wilson_per_arm = []
    for arm in sorted({r.arm for r in runs}):
        pooled: list[bool] = []
        for r in runs:
            if r.arm != arm:
                continue
            pooled.extend(_verdict_map(r, source, column).values())
        if not pooled:
            continue
        k = sum(1 for v in pooled if v)
        lo, hi = stats.wilson_interval(k, len(pooled))
        wilson_per_arm.append({"arm": arm, "k": k, "n": len(pooled), "lo": lo, "hi": hi})

    def _pooled_pair(treatment_arm: str, control_arm: str, src: str, col: str) -> tuple[list[bool], list[bool]]:
        t_runs = [r for r in runs if r.arm == treatment_arm]
        c_runs = [r for r in runs if r.arm == control_arm]
        a_all: list[bool] = []
        b_all: list[bool] = []
        for t_run, c_run in _matched_seed_pairs(t_runs, c_runs):
            t_map = _verdict_map(t_run, src, col)
            c_map = _verdict_map(c_run, src, col)
            common_keys = sorted(set(t_map) & set(c_map), key=str)
            a_all.extend(t_map[k] for k in common_keys)
            b_all.extend(c_map[k] for k in common_keys)
        return a_all, b_all

    a, b = _pooled_pair(headline.treatment, headline.control, source, column)
    mcnemar_headline = stats.mcnemar(a, b)._asdict() if a else None

    per_hyp_pvalues = []
    for hyp in HYPOTHESES:
        hsource, hcolumn = _parse_metric(hyp.metric)
        a_h, b_h = _pooled_pair(hyp.treatment, hyp.control, hsource, hcolumn)
        per_hyp_pvalues.append(stats.mcnemar(a_h, b_h).p_value if a_h else 1.0)
    adjusted = stats.holm(per_hyp_pvalues)
    holm_rows = [
        {"id": hyp.id, "p_value": p, "adjusted_p_value": adj}
        for hyp, p, adj in zip(HYPOTHESES, per_hyp_pvalues, adjusted)
    ]

    return {
        "wilson_per_arm": wilson_per_arm,
        "mcnemar_headline": {"treatment": headline.treatment, "control": headline.control, **(mcnemar_headline or {})}
        if mcnemar_headline
        else None,
        "holm_across_hypotheses": holm_rows,
    }


# --- rendering -----------------------------------------------------------


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def render_markdown(
    external_rows: Sequence[dict[str, Any]],
    internal_rows: Sequence[dict[str, Any]],
    hyp_rows: Sequence[dict[str, Any]],
    extras: dict[str, Any] | None,
) -> str:
    lines = ["# ConPO evaluation report", ""]

    lines += ["## External benchmarks", ""]
    lines += ["| arm | benchmark | n_seeds | prompt_strict | prompt_loose | instruction_strict | instruction_loose |"]
    lines += ["|---|---|---|---|---|---|---|"]
    for r in external_rows:
        lines.append(
            f"| {r['arm']} | {r['benchmark']} | {r['n_seeds']} "
            f"| {_fmt(r['prompt_strict_mean'])} +/- {_fmt(r['prompt_strict_std'])} "
            f"| {_fmt(r['prompt_loose_mean'])} +/- {_fmt(r['prompt_loose_std'])} "
            f"| {_fmt(r['instruction_strict_mean'])} +/- {_fmt(r['instruction_strict_std'])} "
            f"| {_fmt(r['instruction_loose_mean'])} +/- {_fmt(r['instruction_loose_std'])} |"
        )
    lines.append("")

    lines += ["## Internal PAcc by split", ""]
    lines += ["| arm | split | n_seeds | pacc |"]
    lines += ["|---|---|---|---|"]
    for r in internal_rows:
        lines.append(f"| {r['arm']} | {r['split']} | {r['n_seeds']} | {_fmt(r['pacc_mean'])} +/- {_fmt(r['pacc_std'])} |")
    lines.append("")

    lines += ["## Hypotheses", ""]
    lines += ["| id | treatment | control | metric | expected | mean_diff | bootstrap_diff | 95% CI | n_pairs | verdict | note |"]
    lines += ["|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in hyp_rows:
        ci = f"[{_fmt(r['ci_lo'])}, {_fmt(r['ci_hi'])}]"
        lines.append(
            f"| {r['id']} | {r['treatment']} | {r['control']} | {r['metric']} | {r['expected_sign']} "
            f"| {_fmt(r['mean_diff'])} | {_fmt(r['bootstrap_diff'])} | {ci} | {r['n_seed_pairs']} "
            f"| {r['verdict']} | {r['note'] or ''} |"
        )
    lines.append("")

    if extras is not None:
        lines += ["## Extras", ""]
        lines += ["### Wilson interval per arm (H1 metric)", ""]
        lines += ["| arm | k | n | lo | hi |", "|---|---|---|---|---|"]
        for w in extras["wilson_per_arm"]:
            lines.append(f"| {w['arm']} | {w['k']} | {w['n']} | {_fmt(w['lo'])} | {_fmt(w['hi'])} |")
        lines.append("")
        lines += ["### McNemar, H1 headline pair", ""]
        mc = extras["mcnemar_headline"]
        if mc is None:
            lines.append("no overlapping prompts for the H1 pair.")
        else:
            lines.append(f"n10={mc['n10']}, n01={mc['n01']}, p={_fmt(mc['p_value'])}")
        lines.append("")
        lines += ["### Holm correction across hypothesis rows", ""]
        lines += ["| id | p_value | adjusted_p_value |", "|---|---|---|"]
        for h in extras["holm_across_hypotheses"]:
            lines.append(f"| {h['id']} | {_fmt(h['p_value'])} | {_fmt(h['adjusted_p_value'])} |")
        lines.append("")

    return "\n".join(lines)


# --- CLI -------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the ConPO evaluation report from a directory of runs.")
    parser.add_argument("--runs", default=None, help='directory of run dirs named "<arm>_s<seed>"')
    parser.add_argument("--manifest", default=None, help="JSON file of explicit {arm, seed, dir} run entries")
    parser.add_argument("--out", required=True, help="Markdown output path (JSON written to <out>.json)")
    parser.add_argument("--extras", action="store_true", help="also compute Wilson/McNemar/Holm")
    args = parser.parse_args()

    if args.manifest is not None:
        refs = load_manifest(Path(args.manifest))
    elif args.runs is not None:
        refs = discover_runs(Path(args.runs))
    else:
        parser.error("one of --runs or --manifest is required")

    runs = [load_run(ref) for ref in refs]

    external_rows = external_table(runs)
    internal_rows = internal_table(runs)
    hyp_rows = hypothesis_table(runs)
    extras = build_extras(runs) if args.extras else None

    Path(args.out).write_text(render_markdown(external_rows, internal_rows, hyp_rows, extras), encoding="utf-8")
    json_out = {
        "external_table": external_rows,
        "internal_table": internal_rows,
        "hypothesis_table": hyp_rows,
        "extras": extras,
    }
    Path(f"{args.out}.json").write_text(json.dumps(json_out, indent=2), encoding="utf-8")
    print(f"wrote {args.out} and {args.out}.json")


if __name__ == "__main__":
    main()
