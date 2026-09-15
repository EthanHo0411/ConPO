"""Retention eval: zero-shot GSM8K / MMLU / BBH, greedy decoding, over the
data/retention_{gsm8k,mmlu,bbh}.jsonl rows scripts/fetch_data.py writes.

--score-only skips generation entirely and scores canned responses instead:
a jsonl of {"key", "response"} records, joined by row.key (a missing key for
a requested row raises KeyError, mirroring scripts/eval_internal.py).

Alongside `--out`'s JSON ({benchmark: accuracy, "n", "correct", "by_group",
"model", "adapter", "max_tokens", "limit", "retention_settings"} -- a
benchmark not requested is absent throughout, never written as 0),
`<out>.verdicts.jsonl` carries one line per scored prompt: {key, benchmark,
group, correct, response, prediction, gold}.

Env knobs (config.env style; a CLI flag passed explicitly still wins):
  RETENTION_SUFFIX      replaces conpo.retention.BBH_SUFFIX at the end of
                        every BBH prompt at load time (the fetched
                        retention_bbh.jsonl is left as written); unset =
                        the default suffix.
  RETENTION_MAX_TOKENS  positive integer; the --max-tokens default in place
                        of config.RETENTION_MAX_TOKENS.
  RETENTION_BBH_TASKS   comma-separated config.BBH_TASKS names; only those
                        BBH tasks are loaded and scored (filter applied
                        before --limit); an unknown name exits before any
                        model load. Unset = all tasks.
  RETENTION_BENCHMARKS  the --benchmarks default (comma-separated subset of
                        gsm8k,mmlu,bbh).
  RETENTION_LENIENT     "1" turns on conpo.retention.score_bbh's lenient
                        fallback (see its docstring); unset/"" = strict,
                        byte-identical. Any other value exits before any
                        model load.
"retention_settings" in the JSON records the five resolved values, so a
rerun under these knobs is distinguishable from a default run.
"""

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import eval_external  # noqa: E402 -- generate_greedy / _load_hf_engine / _generate_vllm live here

from conpo import config
from conpo.retention import RetentionRow, apply_bbh_suffix, bbh_suffix, extract_prediction, retention_lenient, score

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BENCHMARKS = ("gsm8k", "mmlu", "bbh")


def _row_from_json(obj: dict[str, Any]) -> RetentionRow:
    prompt = apply_bbh_suffix(obj["prompt"]) if obj["benchmark"] == "bbh" else obj["prompt"]
    return RetentionRow(key=obj["key"], benchmark=obj["benchmark"], prompt=prompt, target=obj["target"], group=obj["group"])


def load_retention_rows(
    data_dir: Path, benchmark: str, limit: int | None = None, groups: Collection[str] | None = None
) -> list[RetentionRow]:
    """Rows from `data_dir`/retention_{benchmark}.jsonl, capped to `limit`
    (a prefix slice) if given. `limit=0` yields an empty list. `groups`,
    when given, keeps only rows whose group is in it, before the cap."""
    rows: list[RetentionRow] = []
    with (data_dir / f"retention_{benchmark}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if limit is not None and len(rows) >= limit:
                break
            row = _row_from_json(json.loads(line))
            if groups is not None and row.group not in groups:
                continue
            rows.append(row)
    return rows


def _load_canned_responses(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                out[str(record["key"])] = record["response"]
    return out


def record_verdicts(rows: Sequence[RetentionRow], responses: Sequence[str]) -> list[dict[str, Any]]:
    """One verdict dict per row: key, benchmark, group, correct, response
    (full model output), prediction (the scorer's extracted answer, None if
    extraction found nothing) and gold (row.target) -- so a failed group can
    be diagnosed from the verdicts file alone."""
    out = []
    for row, response in zip(rows, responses):
        out.append(
            {
                "key": row.key,
                "benchmark": row.benchmark,
                "group": row.group,
                "correct": bool(score(row, response)),
                "response": response,
                "prediction": extract_prediction(row, response),
                "gold": row.target,
            }
        )
    return out


def write_verdicts(out_path: str, verdicts: Sequence[dict[str, Any]]) -> str:
    """Write `verdicts` to `<out_path>.verdicts.jsonl`, one JSON object per line."""
    verdicts_path = f"{out_path}.verdicts.jsonl"
    with open(verdicts_path, "w", encoding="utf-8") as f:
        for v in verdicts:
            f.write(json.dumps(v) + "\n")
    return verdicts_path


def summarise(verdicts: Sequence[dict[str, Any]], benchmarks: Sequence[str]) -> dict[str, Any]:
    """{benchmark: accuracy, "n": {...}, "correct": {...}, "by_group": {...}},
    each keyed only by benchmarks with at least one verdict."""
    result: dict[str, Any] = {}
    n: dict[str, int] = {}
    correct: dict[str, int] = {}
    by_group: dict[str, dict[str, float]] = {}
    for benchmark in benchmarks:
        rows = [v for v in verdicts if v["benchmark"] == benchmark]
        if not rows:
            continue
        n[benchmark] = len(rows)
        correct[benchmark] = sum(1 for v in rows if v["correct"])
        result[benchmark] = correct[benchmark] / n[benchmark]

        group_n: dict[str, int] = defaultdict(int)
        group_correct: dict[str, int] = defaultdict(int)
        for v in rows:
            group_n[v["group"]] += 1
            group_correct[v["group"]] += int(v["correct"])
        by_group[benchmark] = {g: group_correct[g] / group_n[g] for g in group_n}
    result["n"] = n
    result["correct"] = correct
    result["by_group"] = by_group
    return result


def retention_max_tokens() -> int:
    """RETENTION_MAX_TOKENS when set (a positive integer, else SystemExit),
    otherwise config.RETENTION_MAX_TOKENS."""
    raw = config.env("RETENTION_MAX_TOKENS", "")
    if not raw:
        return config.RETENTION_MAX_TOKENS
    if not raw.isdigit() or int(raw) == 0:
        raise SystemExit(f"RETENTION_MAX_TOKENS must be a positive integer, got {raw!r}")
    return int(raw)


def retention_bbh_tasks() -> list[str] | None:
    """RETENTION_BBH_TASKS as a list of config.BBH_TASKS names, None when
    unset (every task). An unknown name raises SystemExit naming it and
    the valid list."""
    raw = config.env("RETENTION_BBH_TASKS", "")
    if not raw:
        return None
    tasks = [task.strip() for task in raw.split(",") if task.strip()]
    unknown = [task for task in tasks if task not in config.BBH_TASKS]
    if unknown:
        raise SystemExit(
            f"RETENTION_BBH_TASKS: unknown BBH task(s) {unknown}; valid names: {', '.join(config.BBH_TASKS)}"
        )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Retention eval: GSM8K / MMLU / BBH for the ConPO policy.")
    parser.add_argument("--model", default=config.POLICY_MODEL)
    parser.add_argument("--adapter", default=None, help="LoRA adapter directory, or omit for the base model")
    parser.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument(
        "--benchmarks",
        default=config.env("RETENTION_BENCHMARKS", ",".join(BENCHMARKS)),
        help="comma-separated subset of gsm8k,mmlu,bbh (default: RETENTION_BENCHMARKS, else all three)",
    )
    parser.add_argument(
        "--groups",
        default=None,
        help="comma-separated group names (BBH task names, or MMLU subjects); applied after loading, "
        "keeps only rows whose group matches -- lets a group-filtered smoke run without a custom script",
    )
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="directory holding retention_{benchmark}.jsonl")
    parser.add_argument("--score-only", default=None, help="jsonl of {key, response}; skips generation")
    parser.add_argument("--limit", type=int, default=None, help="cap rows per benchmark, for smoke runs")
    parser.add_argument("--max-tokens", type=int, default=retention_max_tokens())
    args = parser.parse_args()

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    bbh_tasks = retention_bbh_tasks()
    lenient = retention_lenient()  # resolved once, up front: SystemExit before any model load
    data_dir = Path(args.data_dir)
    rows_by_benchmark = {
        b: load_retention_rows(data_dir, b, args.limit, bbh_tasks if b == "bbh" else None) for b in benchmarks
    }
    if args.groups is not None:
        groups = {g.strip() for g in args.groups.split(",") if g.strip()}
        rows_by_benchmark = {b: [r for r in rows if r.group in groups] for b, rows in rows_by_benchmark.items()}

    engine = None
    if args.score_only is None and args.backend == "hf":
        engine = eval_external._load_hf_engine(args.model, args.adapter)
    canned = _load_canned_responses(args.score_only) if args.score_only is not None else None

    verdicts: list[dict[str, Any]] = []
    for benchmark in benchmarks:
        rows = rows_by_benchmark[benchmark]
        prompts = [row.prompt for row in rows]
        if canned is not None:
            missing = [row.key for row in rows if row.key not in canned]
            if missing:
                raise KeyError(f"no canned response for keys {missing} in {args.score_only}")
            responses = [canned[row.key] for row in rows]
        elif engine is not None:
            responses = eval_external.generate_greedy(engine, prompts, args.max_tokens)
        else:
            responses = eval_external._generate_vllm(args.model, args.adapter, prompts, args.max_tokens)
        verdicts.extend(record_verdicts(rows, responses))

    out = summarise(verdicts, benchmarks)
    out["model"] = args.model
    out["adapter"] = args.adapter
    out["max_tokens"] = args.max_tokens
    out["limit"] = args.limit
    out["retention_settings"] = {
        "suffix": bbh_suffix(),
        "max_tokens": args.max_tokens,
        "bbh_tasks": bbh_tasks,
        "benchmarks": benchmarks,
        "lenient": lenient,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")

    verdicts_path = write_verdicts(args.out, verdicts)
    print(f"wrote {verdicts_path}")


if __name__ == "__main__":
    main()
