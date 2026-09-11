"""Retention eval: zero-shot GSM8K / MMLU / BBH, greedy decoding, over the
data/retention_{gsm8k,mmlu,bbh}.jsonl rows scripts/fetch_data.py writes.

--score-only skips generation entirely and scores canned responses instead:
a jsonl of {"key", "response"} records, joined by row.key (a missing key for
a requested row raises KeyError, mirroring scripts/eval_internal.py).

Alongside `--out`'s JSON ({benchmark: accuracy, "n", "correct", "by_group",
"model", "adapter", "max_tokens", "limit"} -- a benchmark not requested is
absent throughout, never written as 0), `<out>.verdicts.jsonl` carries one
line per scored prompt: {key, benchmark, group, correct, response,
prediction, gold}.
"""

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import eval_external  # noqa: E402 -- generate_greedy / _load_hf_engine / _generate_vllm live here

from conpo import config
from conpo.retention import RetentionRow, extract_prediction, score

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BENCHMARKS = ("gsm8k", "mmlu", "bbh")


def _row_from_json(obj: dict[str, Any]) -> RetentionRow:
    return RetentionRow(key=obj["key"], benchmark=obj["benchmark"], prompt=obj["prompt"], target=obj["target"], group=obj["group"])


def load_retention_rows(data_dir: Path, benchmark: str, limit: int | None = None) -> list[RetentionRow]:
    """Rows from `data_dir`/retention_{benchmark}.jsonl, capped to `limit`
    (a prefix slice) if given. `limit=0` yields an empty list."""
    rows: list[RetentionRow] = []
    with (data_dir / f"retention_{benchmark}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if limit is not None and len(rows) >= limit:
                break
            rows.append(_row_from_json(json.loads(line)))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Retention eval: GSM8K / MMLU / BBH for the ConPO policy.")
    parser.add_argument("--model", default=config.POLICY_MODEL)
    parser.add_argument("--adapter", default=None, help="LoRA adapter directory, or omit for the base model")
    parser.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument("--benchmarks", default=",".join(BENCHMARKS), help="comma-separated subset of gsm8k,mmlu,bbh")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="directory holding retention_{benchmark}.jsonl")
    parser.add_argument("--score-only", default=None, help="jsonl of {key, response}; skips generation")
    parser.add_argument("--limit", type=int, default=None, help="cap rows per benchmark, for smoke runs")
    parser.add_argument("--max-tokens", type=int, default=config.RETENTION_MAX_TOKENS)
    args = parser.parse_args()

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    data_dir = Path(args.data_dir)
    rows_by_benchmark = {b: load_retention_rows(data_dir, b, args.limit) for b in benchmarks}

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
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")

    verdicts_path = write_verdicts(args.out, verdicts)
    print(f"wrote {verdicts_path}")


if __name__ == "__main__":
    main()
