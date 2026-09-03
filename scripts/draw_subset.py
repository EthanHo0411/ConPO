"""Draw ONE fixed, seeded subset of the train split for every ConPO arm to
share (design choice, 2026-09-02; the user fixed the subset, the stratification is ours): stratified by num_constraints
proportionally to the train split's own histogram, largest-remainder
rounding so per-stratum sizes sum exactly to --size, then shuffled once
with --seed. Keys are deduplicated (keep the first occurrence) before
drawing: the upstream pool dataset carries a handful of duplicate keys (see
conpo/data.py's load_pool, which dedupes the same way at the source);
data/train.jsonl was built before that fix, so the duplicates are still
baked into the file this script reads, and are counted rather than drawn
twice.

Draws exactly --size keys -- no rounding. scripts/train.py's ConPOTrainer
runs on TRL's RepeatSampler with batch_size == prompts_per_update and
shuffle_dataset=False for a RUN_SUBSET run, which silently drops any
trailing chunk shorter than a full batch every epoch; when --size isn't a
multiple of prompts_per_update (config.EFFECTIVE_BATCH // NUM_GENERATIONS)
a handful of trailing prompts in the committed key order are never trained
on (e.g. 8 of 15000 prompts at 16 per update) -- negligible, and recorded in
the manifest below rather than avoided by padding --size up.

Writes subsets/train_<size>_seed<seed>.keys (one key per line, final
shuffled order) plus a sidecar subsets/train_<size>_seed<seed>.manifest.json
(size, seed, per-count sizes, duplicate_keys found in the source,
prompts_per_update and dropped_trailing_prompts, sha256 of the keys file,
source manifest.json split sizes for provenance). subsets/ is committed to
the repo; data/ is gitignored.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from conpo import config
from conpo.data import keys_digest

REPO_ROOT = Path(__file__).resolve().parent.parent

# The real (non-tiny) training run always consumes exactly this many
# prompts per optimiser update -- config.EFFECTIVE_BATCH // NUM_GENERATIONS,
# independent of num_processes (see scripts/train.py's prompts_per_update
# derivation). Recorded in the manifest so a reader can see how many
# trailing prompts, if any, TRL's RepeatSampler drops every epoch.
PROMPTS_PER_UPDATE = config.EFFECTIVE_BATCH // config.NUM_GENERATIONS


def _num_constraints_histogram(source: Path) -> tuple[dict[int, list[str]], int, int]:
    """{num_constraints: [key, ...]} in file order (deduplicated by key,
    first occurrence kept), the total row count read, and how many rows
    were skipped as duplicate keys.

    Reads only the two fields draw_subset needs (`key`, `num_constraints`),
    not a full PoolRow -- scripts/build_corpus.py writes both verbatim.
    """
    by_count: dict[int, list[str]] = {}
    seen_keys: set[str] = set()
    total = 0
    duplicate_keys = 0
    with source.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1
            key = row["key"]
            if key in seen_keys:
                duplicate_keys += 1
                continue
            seen_keys.add(key)
            by_count.setdefault(row["num_constraints"], []).append(key)
    return by_count, total, duplicate_keys


def _largest_remainder_sizes(counts: dict[int, int], target_size: int) -> dict[int, int]:
    """Per-stratum sizes proportional to `counts`, summing exactly to
    `target_size` (largest-remainder / Hamilton apportionment): floor the
    exact proportional share per stratum, then give the leftover units one
    each to the strata with the largest fractional remainder."""
    total = sum(counts.values())
    exact = {c: n * target_size / total for c, n in counts.items()}
    sizes = {c: int(v) for c, v in exact.items()}
    remainder = target_size - sum(sizes.values())
    order = sorted(counts, key=lambda c: (-(exact[c] - sizes[c]), c))
    for c in order[:remainder]:
        sizes[c] += 1
    return sizes


def draw_subset(source: Path, size: int, seed: int) -> tuple[list[str], dict[int, int], int, int]:
    """(shuffled keys, per-count sizes, source row count, duplicate key count).

    One `random.Random(seed)` instance draws every per-stratum sample (in
    ascending num_constraints order, for a fixed operation order regardless
    of the source file's own row order) and then shuffles the combined list
    once, so the whole draw is deterministic in `seed` alone.
    """
    by_count, total, duplicate_keys = _num_constraints_histogram(source)
    counts = {c: len(keys) for c, keys in by_count.items()}
    target_sizes = _largest_remainder_sizes(counts, size)

    rng = random.Random(seed)
    drawn: list[str] = []
    for c in sorted(by_count):
        drawn.extend(rng.sample(by_count[c], target_sizes[c]))
    rng.shuffle(drawn)
    return drawn, target_sizes, total, duplicate_keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", type=Path, default=REPO_ROOT / "data" / "train.jsonl")
    parser.add_argument("--out", type=Path, default=None, help="default: subsets/train_<size>_seed<seed>.keys")
    args = parser.parse_args()

    out_path = (
        args.out if args.out is not None else REPO_ROOT / "subsets" / f"train_{args.size}_seed{args.seed}.keys"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    keys, per_count_sizes, source_total, duplicate_keys = draw_subset(args.source, args.size, args.seed)
    if len(keys) != args.size:
        raise ValueError(f"drew {len(keys)} keys, expected --size {args.size}")
    assert len(set(keys)) == len(keys), f"drew {len(keys)} keys but only {len(set(keys))} are unique"

    out_path.write_text("\n".join(keys) + "\n", encoding="utf-8")
    keys_sha256 = keys_digest(out_path)

    source_manifest_path = args.source.parent / "manifest.json"
    source_manifest_split_sizes = None
    if source_manifest_path.exists():
        source_manifest_split_sizes = json.loads(source_manifest_path.read_text(encoding="utf-8")).get("split_sizes")

    manifest = {
        "size": args.size,
        "seed": args.seed,
        "source": str(args.source),
        "source_total_rows": source_total,
        "duplicate_keys": duplicate_keys,
        "per_count_sizes": {str(c): n for c, n in sorted(per_count_sizes.items())},
        "prompts_per_update": PROMPTS_PER_UPDATE,
        "dropped_trailing_prompts": args.size % PROMPTS_PER_UPDATE,
        "keys_sha256": keys_sha256,
        "source_manifest_split_sizes": source_manifest_split_sizes,
    }
    manifest_path = out_path.parent / f"{out_path.stem}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
