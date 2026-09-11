"""Score scripts/rejudge.py's output against a reader agent's labels.

Input: rejudge.py's per-record outcomes JSONL (its `--out` file, not the
sibling `.json` report) and a labels JSONL, one row per record with `index`
and `label`, `label` in {constraint_only, real_degeneracy, mixed, unclear}
for records a source vetoed, or {kept_ok, kept_degenerate} for sampled kept
records. Records are matched by `index` (scripts/rejudge.py's load_records
assigns it to every loaded record and carries it through to the outcome).

For the baseline and every variant present in the outcomes file, prints and
writes one table: the false-veto rate (constraint_only records that source
also vetoes, over all constraint_only records), the missed-degeneracy rate
(real_degeneracy records that source keeps, over all real_degeneracy
records), and the overall veto rate by direction (over every record,
regardless of label). The mixed/unclear/kept_ok/kept_degenerate labels are
read but feed neither rate: they exist for a person skimming the labels
file, not for this arithmetic. The number of unlabeled records (no entry in
the labels file for their index) is reported once, independent of source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CONSTRAINT_ONLY = "constraint_only"
REAL_DEGENERACY = "real_degeneracy"


def load_outcomes(path: str | Path) -> list[dict[str, Any]]:
    """Load rejudge.py's per-record outcomes JSONL (its --out file)."""
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_labels(path: str | Path) -> dict[int, str]:
    """Load a labels JSONL ({"index": int, "label": str} per row) into an
    index -> label mapping. Raises ValueError on a duplicate index: a
    reader-agent output naming the same record twice is a labeling bug, not
    a case to silently resolve by picking one.
    """
    labels: dict[int, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            index = row["index"]
            if index in labels:
                raise ValueError(f"duplicate label for record index {index}")
            labels[index] = row["label"]
    return labels


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _sources(outcomes: list[dict[str, Any]]) -> list[str]:
    """"baseline" plus every variant name present on the first outcome row,
    in the order rejudge.py wrote them."""
    if not outcomes:
        return ["baseline"]
    return ["baseline"] + list(outcomes[0].get("variants", {}))


def _keep(row: dict[str, Any], source: str) -> bool:
    return row["baseline"]["keep"] if source == "baseline" else row["variants"][source]["keep"]


def score(outcomes: list[dict[str, Any]], labels: dict[int, str]) -> dict[str, Any]:
    """Build the one-table score report described in this module's
    docstring: {n_records, n_unlabeled, directions, table}, `table` a list
    of one row per source ("baseline" then every variant, rejudge's order)
    with {source, n_constraint_only, false_veto_rate, n_real_degeneracy,
    missed_degeneracy_rate, veto_rate_by_direction}.
    """
    directions = sorted({r["direction"] for r in outcomes})
    n_unlabeled = sum(1 for r in outcomes if r["index"] not in labels)
    constraint_only_rows = [r for r in outcomes if labels.get(r["index"]) == CONSTRAINT_ONLY]
    real_degeneracy_rows = [r for r in outcomes if labels.get(r["index"]) == REAL_DEGENERACY]
    direction_rows = {d: [r for r in outcomes if r["direction"] == d] for d in directions}

    table: list[dict[str, Any]] = []
    for source in _sources(outcomes):
        table.append(
            {
                "source": source,
                "n_constraint_only": len(constraint_only_rows),
                "false_veto_rate": _rate(
                    sum(1 for r in constraint_only_rows if not _keep(r, source)), len(constraint_only_rows)
                ),
                "n_real_degeneracy": len(real_degeneracy_rows),
                "missed_degeneracy_rate": _rate(
                    sum(1 for r in real_degeneracy_rows if _keep(r, source)), len(real_degeneracy_rows)
                ),
                "veto_rate_by_direction": {
                    d: _rate(sum(1 for r in rows if not _keep(r, source)), len(rows))
                    for d, rows in direction_rows.items()
                },
            }
        )

    return {"n_records": len(outcomes), "n_unlabeled": n_unlabeled, "directions": directions, "table": table}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score scripts/rejudge.py's per-record outcomes against a reader agent's labels."
    )
    parser.add_argument("--rejudge-out", required=True, help="rejudge.py's --out outcomes JSONL")
    parser.add_argument("--labels", required=True, help="labels JSONL: one {index, label} row per record")
    parser.add_argument("--out", required=True, help="output path for the score report JSON")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    outcomes = load_outcomes(args.rejudge_out)
    labels = load_labels(args.labels)
    report = score(outcomes, labels)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
