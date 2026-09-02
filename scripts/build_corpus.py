"""Build the ConPO training corpus: load_pool + make_splits -> data/*.jsonl + manifest.json."""

import dataclasses
import json
from pathlib import Path

from conpo.data import PoolRow, load_pool, make_splits

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED = 20260805  # fixed for reproducibility; matches stackelberg-if's probe seed


def _row_to_json(row: PoolRow) -> dict:
    return {
        "key": row.key,
        "base_prompt": row.base_prompt,
        "prompt": row.prompt,
        "sentences": list(row.sentences),
        "constraints": [{"iid": c.iid, "kwargs": dict(c.kwargs)} for c in row.constraints],
        "english": row.english,
        "num_constraints": row.num_constraints,
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows, report = load_pool(limit=None)
    print(f"loaded {report.kept} / {report.total} rows")

    splits = make_splits(rows, seed=SEED)
    sizes: dict[str, int] = {}
    for name, split_rows in splits.items():
        path = DATA_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in split_rows:
                f.write(json.dumps(_row_to_json(row)) + "\n")
        sizes[name] = len(split_rows)
        print(f"  {name}: {len(split_rows)} rows -> {path}")

    manifest = {
        "seed": SEED,
        "filter_report": dataclasses.asdict(report),
        "split_sizes": sizes,
    }
    manifest_path = DATA_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
