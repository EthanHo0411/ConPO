"""Contamination check: the training subset's base prompts against IFEval
(google/IFEval) and IFBench_test prompt texts.

Usage:
    conda run -n conpo --no-capture-output python scripts/check_overlap.py \
        [--subset subsets/<keys>] [--rows data/train.jsonl] [--limit N] \
        [--n 13] [--out overlap_report.json]

--subset points at a committed key-list file, one row key per line (blank
lines and lines starting with "#" ignored); omitted, every row in --rows is
checked. --limit caps the number of --rows lines read, for a fast sample run.
"""

import argparse
import json
from pathlib import Path

import datasets

from conpo import config, scoring
from conpo.overlap import OverlapReport, check_overlap

ROOT = Path(__file__).resolve().parent.parent


def load_subset_keys(path: Path) -> set[str]:
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            keys.add(line)
    return keys


def load_subset_base_prompts(rows_path: Path, subset_path: Path | None, limit: int | None) -> dict[str, str]:
    """key -> PoolRow.base_prompt for rows in `rows_path`, restricted to
    `subset_path`'s keys when given, capped to the first `limit` lines."""
    wanted = load_subset_keys(subset_path) if subset_path is not None else None
    texts: dict[str, str] = {}
    with rows_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["key"]
            if wanted is not None and key not in wanted:
                continue
            texts[key] = row["base_prompt"]
    return texts


def _report_dict(report: OverlapReport) -> dict:
    return {
        "exact_matches": report.exact_matches,
        "ngram_hits": report.ngram_hits,
        "hit_keys": report.hit_keys[:10],
        "total_subset": report.total_subset,
        "total_benchmark": report.total_benchmark,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default=str(ROOT / "data" / "train.jsonl"))
    parser.add_argument("--subset", default=None, help="committed key-list file; omit to use every row in --rows")
    parser.add_argument("--limit", type=int, default=None, help="cap the number of --rows lines read")
    parser.add_argument("--n", type=int, default=13)
    parser.add_argument("--out", default=str(ROOT / "overlap_report.json"))
    args = parser.parse_args()

    subset_path = Path(args.subset) if args.subset is not None else None
    subset_texts = load_subset_base_prompts(Path(args.rows), subset_path, args.limit)
    source = f"--subset {args.subset}" if args.subset else f"all of {args.rows}"
    print(f"subset: {len(subset_texts)} base prompts ({source})")

    ifeval_ds = datasets.load_dataset(config.IFEVAL_DATASET, split="train")
    ifeval_prompts = [str(row["prompt"]) for row in ifeval_ds]
    ifbench_prompts = [str(row["prompt"]) for row in scoring.load_ifbench_test()]

    ifeval_report = check_overlap(subset_texts, ifeval_prompts, n=args.n)
    ifbench_report = check_overlap(subset_texts, ifbench_prompts, n=args.n)

    print(
        f"IFEval ({len(ifeval_prompts)} prompts): "
        f"exact={ifeval_report.exact_matches} ngram={ifeval_report.ngram_hits}"
    )
    print(
        f"IFBench_test ({len(ifbench_prompts)} prompts): "
        f"exact={ifbench_report.exact_matches} ngram={ifbench_report.ngram_hits}"
    )

    result = {
        "n": args.n,
        "total_subset": len(subset_texts),
        "ifeval": _report_dict(ifeval_report),
        "ifbench_test": _report_dict(ifbench_report),
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"report -> {out_path}")


if __name__ == "__main__":
    main()
