"""Download the pool, IFEval, the retention suite (GSM8K, MMLU, BBH) and the
policy tokenizer into the local HF cache, plus nltk punkt/punkt_tab.
Idempotent: HF and nltk caching skip re-downloads.

--retention-only skips the pool/IFEval/tokenizer/nltk steps and fetches only
the retention suite, writing data/retention_{gsm8k,mmlu,bbh}.jsonl (one
RetentionRow as JSON per line, conpo.retention.RetentionRow) for
scripts/eval_retention.py's HF_DATASETS_OFFLINE=1 cluster runs.
"""

import argparse
import dataclasses
import json
from pathlib import Path

import datasets
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from transformers import AutoTokenizer

from conpo import config
from conpo.retention import RetentionRow, rows_from_bbh, rows_from_gsm8k, rows_from_mmlu
from conpo.verifiers import ensure_nltk

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _write_rows(name: str, rows: list[RetentionRow]) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"retention_{name}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dataclasses.asdict(row)) + "\n")
    return path


def fetch_retention() -> dict[str, int]:
    """Download GSM8K, MMLU and BBH test splits, write the three
    data/retention_*.jsonl files, return {benchmark: row_count}."""
    counts: dict[str, int] = {}

    print(f"GSM8K dataset: {config.GSM8K_DATASET} ({config.GSM8K_CONFIG})")
    gsm8k_hf = datasets.load_dataset(config.GSM8K_DATASET, config.GSM8K_CONFIG, split=config.GSM8K_SPLIT)
    gsm8k_rows = rows_from_gsm8k(gsm8k_hf)
    path = _write_rows("gsm8k", gsm8k_rows)
    counts["gsm8k"] = len(gsm8k_rows)
    print(f"  wrote {path}: {len(gsm8k_rows)} rows")

    print(f"MMLU dataset: {config.MMLU_DATASET} ({config.MMLU_CONFIG})")
    mmlu_hf = datasets.load_dataset(config.MMLU_DATASET, config.MMLU_CONFIG, split=config.MMLU_SPLIT)
    mmlu_rows = rows_from_mmlu(mmlu_hf)
    path = _write_rows("mmlu", mmlu_rows)
    counts["mmlu"] = len(mmlu_rows)
    print(f"  wrote {path}: {len(mmlu_rows)} rows")

    print(f"BBH dataset: {config.BBH_DATASET} ({len(config.BBH_TASKS)} tasks)")
    bbh_rows: list[RetentionRow] = []
    for task in config.BBH_TASKS:
        task_hf = datasets.load_dataset(config.BBH_DATASET, task, split="test")
        bbh_rows.extend(rows_from_bbh(task, task_hf))
    path = _write_rows("bbh", bbh_rows)
    counts["bbh"] = len(bbh_rows)
    print(f"  wrote {path}: {len(bbh_rows)} rows")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retention-only",
        action="store_true",
        help="fetch only GSM8K/MMLU/BBH, skipping the pool/IFEval/tokenizer/nltk steps",
    )
    args = parser.parse_args()

    if not args.retention_only:
        print(f"pool dataset: {config.POOL_DATASET}")
        pool = datasets.load_dataset(config.POOL_DATASET, split="train")
        print(f"  cached: {len(pool)} rows")

        print(f"IFEval dataset: {config.IFEVAL_DATASET}")
        ifeval = datasets.load_dataset(config.IFEVAL_DATASET, split="train")
        print(f"  cached: {len(ifeval)} rows")

    fetch_retention()

    if not args.retention_only:
        print(f"policy tokenizer: {config.POLICY_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(config.POLICY_MODEL)
        print(f"  cached: vocab_size={tokenizer.vocab_size}")

        print("nltk punkt / punkt_tab")
        ensure_nltk()
        print("  cached")

        # Every launcher runs with HF_HUB_OFFLINE=1, so the weights (not only
        # the tokenizer) have to be in the hub cache before the first run.
        # Offline lookup first: an entry another user staged into a shared hub
        # cache is read-only for us, and the online path would fail taking its
        # download lock even when nothing needs downloading. This stages once
        # and never syncs: a cached model is not refreshed from the Hub; delete
        # its hub entry to re-download.
        for name in (config.POLICY_MODEL, config.JUDGE_MODEL):
            print(f"model weights: {name}")
            try:
                path = snapshot_download(name, local_files_only=True)
                print(f"  already cached: {path}")
            except LocalEntryNotFoundError:
                path = snapshot_download(name)
                print(f"  cached: {path}")


if __name__ == "__main__":
    main()
