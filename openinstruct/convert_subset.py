"""Convert ConPO's IF-RLVR subset to the schema open_instruct's IFEvalVerifier expects.

Reads `data/train.jsonl` rows named by a subset `.keys` file (ConPO's own PoolRow schema:
key, base_prompt, prompt, sentences, constraints[{iid, kwargs}], english, num_constraints)
and writes one JSON object per line with exactly the three columns
open_instruct/dataset_transformation.py's rlvr_tokenize_v3 and
open_instruct/ground_truth_utils.py's IFEvalVerifier read:

    messages:     [{"role": "user", "content": <prompt>}]
    ground_truth: repr([{"instruction_id": [...], "kwargs": [...]}])  (ast.literal_eval'd, not JSON)
    dataset:      "ifeval"

`key` is carried through as a passthrough column for traceability; it is not one of the
three reserved names above and open_instruct's local-jsonl loader does not require it.

The prompt is capped with ConPO's own `conpo.prompts.cap_pool_rows` (same tokenizer, same
`config.MAX_PROMPT_TOKENS`, non-thinking) before writing, so open-instruct trains on the
same rendered prompt text ConPO's own TRL run sees for that row.

See ConPO/openinstruct/README.md for the file:line evidence this converter is built from.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

CONPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CONPO_ROOT))

from transformers import AutoTokenizer  # noqa: E402

from conpo import config  # noqa: E402
from conpo.data import PoolRow  # noqa: E402
from conpo.prompts import cap_pool_rows  # noqa: E402
from conpo.vendor.ifevalg import instructions_registry  # noqa: E402
from conpo.verifiers import Constraint  # noqa: E402

DEFAULT_TRAIN_JSONL = CONPO_ROOT / "data" / "train.jsonl"
DEFAULT_SUBSET_KEYS = CONPO_ROOT / "subsets" / "train_15000_seed42.keys"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "data"

KNOWN_FAMILIES: frozenset[str] = frozenset(instructions_registry.INSTRUCTION_DICT)


def load_subset_rows(train_jsonl: Path, keys_file: Path) -> list[PoolRow]:
    """Load the subset's `PoolRow`s from `train_jsonl`, in the `.keys` file's own order."""
    keys = [line.strip() for line in keys_file.read_text().splitlines() if line.strip()]
    key_set = set(keys)
    by_key: dict[str, PoolRow] = {}
    with train_jsonl.open() as f:
        for line in f:
            raw = json.loads(line)
            key: str = raw["key"]
            if key not in key_set:
                continue
            constraints = tuple(Constraint(iid=c["iid"], kwargs=c["kwargs"]) for c in raw["constraints"])
            by_key[key] = PoolRow(
                key=key,
                base_prompt=raw["base_prompt"],
                prompt=raw["prompt"],
                sentences=tuple(raw["sentences"]),
                constraints=constraints,
                english=raw["english"],
                num_constraints=raw["num_constraints"],
            )
    missing = sorted(key_set - by_key.keys())
    if missing:
        raise ValueError(f"{len(missing)} subset key(s) not found in {train_jsonl}, e.g. {missing[:3]}")
    return [by_key[k] for k in keys]


def to_ifeval_row(row: PoolRow) -> dict[str, Any]:
    """Build one open-instruct row from a (already prompt-capped) `PoolRow`."""
    instruction_id = [c.iid for c in row.constraints]
    kwargs_list = [dict(c.kwargs) for c in row.constraints]
    ground_truth = repr([{"instruction_id": instruction_id, "kwargs": kwargs_list}])
    return {
        "key": row.key,
        "messages": [{"role": "user", "content": row.prompt}],
        "ground_truth": ground_truth,
        "dataset": "ifeval",
    }


def count_unmapped_families(rows: list[PoolRow]) -> tuple[int, int, list[str]]:
    """(rows with >=1 unmapped-family constraint, unmapped constraint instances, unmapped family ids).

    "Unmapped" means the family id is absent from ConPO's vendored IFEvalG registry, which a
    direct diff (see README.md) confirms is byte-identical to open-instruct's own IFEvalG package.
    """
    bad_rows = 0
    bad_instances = 0
    seen_bad: set[str] = set()
    for row in rows:
        row_has_bad = False
        for c in row.constraints:
            if c.iid not in KNOWN_FAMILIES:
                bad_instances += 1
                seen_bad.add(c.iid)
                row_has_bad = True
        if row_has_bad:
            bad_rows += 1
    return bad_rows, bad_instances, sorted(seen_bad)


def convert(
    train_jsonl: Path,
    subset_keys: Path,
    out_path: Path,
    limit: int | None,
    tokenizer_name: str = config.POLICY_MODEL,
) -> dict[str, Any]:
    """Run the full convert; returns the printed summary dict for tests to check."""
    rows = load_subset_rows(train_jsonl, subset_keys)
    if limit is not None:
        rows = rows[:limit]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    rows, truncated_count = cap_pool_rows(rows, tokenizer, config.MAX_PROMPT_TOKENS)

    bad_rows, bad_instances, bad_families = count_unmapped_families(rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_constraints = 0
    with out_path.open("w") as f:
        for row in rows:
            total_constraints += len(row.constraints)
            f.write(json.dumps(to_ifeval_row(row)) + "\n")

    summary = {
        "rows_written": len(rows),
        "total_constraint_instances": total_constraints,
        "prompts_truncated": truncated_count,
        "rows_with_unmapped_family": bad_rows,
        "unmapped_constraint_instances": bad_instances,
        "unmapped_families": bad_families,
        "out_path": str(out_path),
    }
    print(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--subset-keys", type=Path, default=DEFAULT_SUBSET_KEYS)
    parser.add_argument("--out", type=Path, default=None, help="Output JSONL path.")
    parser.add_argument("--limit", type=int, default=None, help="Head-slice the subset to this many rows.")
    args = parser.parse_args(argv)

    out_path = args.out
    if out_path is None:
        name = f"ifrlvr_smoke_{args.limit}.jsonl" if args.limit is not None else "ifrlvr_15000_seed42.jsonl"
        out_path = DEFAULT_OUT_DIR / name

    convert(args.train_jsonl, args.subset_keys, out_path, args.limit)


if __name__ == "__main__":
    main()
