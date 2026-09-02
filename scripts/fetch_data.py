"""Download the pool, IFEval and the policy tokenizer into the local HF cache,
plus nltk punkt/punkt_tab. Idempotent: HF and nltk caching skip re-downloads."""

import datasets
from transformers import AutoTokenizer

from conpo import config
from conpo.verifiers import ensure_nltk


def main() -> None:
    print(f"pool dataset: {config.POOL_DATASET}")
    pool = datasets.load_dataset(config.POOL_DATASET, split="train")
    print(f"  cached: {len(pool)} rows")

    print(f"IFEval dataset: {config.IFEVAL_DATASET}")
    ifeval = datasets.load_dataset(config.IFEVAL_DATASET, split="train")
    print(f"  cached: {len(ifeval)} rows")

    print(f"policy tokenizer: {config.POLICY_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(config.POLICY_MODEL)
    print(f"  cached: vocab_size={tokenizer.vocab_size}")

    print("nltk punkt / punkt_tab")
    ensure_nltk()
    print("  cached")


if __name__ == "__main__":
    main()
