"""Draw ONE fixed, seeded subset of the train split for every ConPO arm to
share (locked decision, 2026-09-02): stratified by num_constraints
proportionally to the train split's own histogram, largest-remainder
rounding so per-stratum sizes sum exactly to --size, then shuffled once
with --seed.

--size is rounded to the nearest multiple of --round-to (default:
PROMPTS_PER_UPDATE, config.EFFECTIVE_BATCH // config.NUM_GENERATIONS) before
drawing: scripts/train.py's ConPOTrainer runs on TRL's RepeatSampler with
batch_size == prompts_per_update and shuffle_dataset=False for a RUN_SUBSET
run, which silently drops any trailing chunk shorter than a full batch on
every epoch -- an unrounded size would make the tail of the committed key
order permanently unreachable by every arm. Pass --round-to 1 to disable.

Writes subsets/train_<size>_seed<seed>.keys (one key per line, final
shuffled order, <size> being the rounded size actually drawn) plus a sidecar
subsets/train_<size>_seed<seed>.manifest.json (size, requested_size, round_to,
seed, per-count sizes, sha256 of the keys file, source manifest.json split
sizes for provenance). subsets/ is committed to the repo; data/ is gitignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from conpo import config

REPO_ROOT = Path(__file__).resolve().parent.parent

# The real (non-tiny) training run always consumes exactly this many
# prompts per optimiser update -- config.EFFECTIVE_BATCH // NUM_GENERATIONS,
# independent of num_processes (see scripts/train.py's prompts_per_update
# derivation and its RUN_SUBSET divisibility assert).
PROMPTS_PER_UPDATE = config.EFFECTIVE_BATCH // config.NUM_GENERATIONS


def _round_to_multiple(size: int, multiple: int) -> int:
    """Nearest multiple of `multiple` to `size` (ties round up), floored at
    `multiple` itself so a tiny --size never rounds down to zero. `multiple`
    <= 1 disables rounding."""
    if multiple <= 1:
        return size
    steps = (size + multiple // 2) // multiple
    return max(multiple, steps * multiple)


def _num_constraints_histogram(source: Path) -> tuple[dict[int, list[str]], int]:
    """{num_constraints: [key, ...]} in file order, plus the total row count.

    Reads only the two fields draw_subset needs (`key`, `num_constraints`),
    not a full PoolRow -- scripts/build_corpus.py writes both verbatim.
    """
    by_count: dict[int, list[str]] = {}
    total = 0
    with source.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_count.setdefault(row["num_constraints"], []).append(row["key"])
            total += 1
    return by_count, total


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


def draw_subset(source: Path, size: int, seed: int) -> tuple[list[str], dict[int, int], int]:
    """(shuffled keys, per-count sizes, source row count).

    One `random.Random(seed)` instance draws every per-stratum sample (in
    ascending num_constraints order, for a fixed operation order regardless
    of the source file's own row order) and then shuffles the combined list
    once, so the whole draw is deterministic in `seed` alone.
    """
    by_count, total = _num_constraints_histogram(source)
    counts = {c: len(keys) for c, keys in by_count.items()}
    target_sizes = _largest_remainder_sizes(counts, size)

    rng = random.Random(seed)
    drawn: list[str] = []
    for c in sorted(by_count):
        drawn.extend(rng.sample(by_count[c], target_sizes[c]))
    rng.shuffle(drawn)
    return drawn, target_sizes, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", type=Path, default=REPO_ROOT / "data" / "train.jsonl")
    parser.add_argument("--out", type=Path, default=None, help="default: subsets/train_<size>_seed<seed>.keys")
    parser.add_argument(
        "--round-to",
        type=int,
        default=PROMPTS_PER_UPDATE,
        help=(
            "round --size to the nearest multiple of this before drawing, so the committed "
            "subset divides evenly and TRL's RepeatSampler never drops a trailing batch "
            f"(default: prompts_per_update = {PROMPTS_PER_UPDATE}); pass 1 to disable"
        ),
    )
    args = parser.parse_args()

    target_size = _round_to_multiple(args.size, args.round_to)
    out_path = (
        args.out if args.out is not None else REPO_ROOT / "subsets" / f"train_{target_size}_seed{args.seed}.keys"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    keys, per_count_sizes, source_total = draw_subset(args.source, target_size, args.seed)
    if len(keys) != target_size:
        raise ValueError(f"drew {len(keys)} keys, expected {target_size} (requested --size {args.size})")

    out_path.write_text("\n".join(keys) + "\n", encoding="utf-8")
    keys_sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()

    source_manifest_path = args.source.parent / "manifest.json"
    source_manifest_split_sizes = None
    if source_manifest_path.exists():
        source_manifest_split_sizes = json.loads(source_manifest_path.read_text(encoding="utf-8")).get("split_sizes")

    manifest = {
        "size": target_size,
        "requested_size": args.size,
        "round_to": args.round_to,
        "seed": args.seed,
        "source": str(args.source),
        "source_total_rows": source_total,
        "per_count_sizes": {str(c): n for c, n in sorted(per_count_sizes.items())},
        "keys_sha256": keys_sha256,
        "source_manifest_split_sizes": source_manifest_split_sizes,
    }
    manifest_path = out_path.parent / f"{out_path.stem}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
