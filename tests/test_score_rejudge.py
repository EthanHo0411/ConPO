"""Tests for scripts/score_rejudge.py: label loading (index matching, the
duplicate-index guard), outcomes loading, and the false-veto /
missed-degeneracy / overall-veto-rate arithmetic on eight hand-built rows.

scripts/ is not an installed package, so score_rejudge is imported by adding
scripts/ to sys.path (same pattern as tests/test_rejudge.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import score_rejudge  # noqa: E402

# ---------------------------------------------------------------------------
# Eight hand-built outcomes rows (rejudge.py's output schema: baseline +
# one variant, "ignore_constraints") and their labels, covering every label
# in the vocabulary plus one unlabeled row.
# ---------------------------------------------------------------------------


def _row(index: int, direction: str, baseline_keep: bool, variant_keep: bool) -> dict[str, Any]:
    return {
        "index": index,
        "step": 10 + index,
        "direction": direction,
        "families": ["keywords:existence"],
        "original_keep": False,
        "baseline": {"keep": baseline_keep, "reason": "baseline", "parse_ok": True},
        "variants": {"ignore_constraints": {"keep": variant_keep, "reason": "variant", "parse_ok": True}},
    }


def _eight_rows() -> list[dict[str, Any]]:
    return [
        _row(0, "forward", baseline_keep=False, variant_keep=True),  # constraint_only
        _row(1, "forward", baseline_keep=True, variant_keep=False),  # constraint_only
        _row(2, "reverse", baseline_keep=False, variant_keep=True),  # real_degeneracy
        _row(3, "reverse", baseline_keep=False, variant_keep=False),  # real_degeneracy
        _row(4, "forward", baseline_keep=False, variant_keep=True),  # mixed
        _row(5, "forward", baseline_keep=True, variant_keep=True),  # kept_ok
        _row(6, "reverse", baseline_keep=False, variant_keep=True),  # kept_degenerate
        _row(7, "reverse", baseline_keep=True, variant_keep=False),  # unlabeled
    ]


def _eight_labels() -> dict[int, str]:
    return {
        0: "constraint_only",
        1: "constraint_only",
        2: "real_degeneracy",
        3: "real_degeneracy",
        4: "mixed",
        5: "kept_ok",
        6: "kept_degenerate",
        # 7 intentionally left unlabeled
    }


def test_score_arithmetic_on_eight_hand_built_rows() -> None:
    report = score_rejudge.score(_eight_rows(), _eight_labels())

    assert report["n_records"] == 8
    assert report["n_unlabeled"] == 1
    assert report["directions"] == ["forward", "reverse"]

    table = {row["source"]: row for row in report["table"]}
    assert set(table) == {"baseline", "ignore_constraints"}

    baseline = table["baseline"]
    assert baseline["n_constraint_only"] == 2
    assert baseline["false_veto_rate"] == pytest.approx(0.5)  # row 0 vetoed, row 1 kept
    assert baseline["n_real_degeneracy"] == 2
    assert baseline["missed_degeneracy_rate"] == pytest.approx(0.0)  # rows 2,3 both correctly vetoed
    assert baseline["veto_rate_by_direction"] == {
        "forward": pytest.approx(0.5),  # rows 0,1,4,5 -> vetoed: 0,4 = 2/4
        "reverse": pytest.approx(0.75),  # rows 2,3,6,7 -> vetoed: 2,3,6 = 3/4
    }

    variant = table["ignore_constraints"]
    assert variant["false_veto_rate"] == pytest.approx(0.5)  # row 0 kept, row 1 vetoed
    assert variant["missed_degeneracy_rate"] == pytest.approx(0.5)  # row 2 kept (missed), row 3 vetoed
    assert variant["veto_rate_by_direction"] == {
        "forward": pytest.approx(0.25),  # rows 0,1,4,5 -> vetoed: 1 = 1/4
        "reverse": pytest.approx(0.5),  # rows 2,3,6,7 -> vetoed: 3,7 = 2/4
    }


def test_score_with_no_labeled_records_gives_zero_rates_not_a_crash() -> None:
    report = score_rejudge.score(_eight_rows(), labels={})
    assert report["n_unlabeled"] == 8
    for row in report["table"]:
        assert row["false_veto_rate"] == 0.0
        assert row["missed_degeneracy_rate"] == 0.0
        assert row["n_constraint_only"] == 0
        assert row["n_real_degeneracy"] == 0


def test_score_with_no_outcomes_returns_baseline_only_row() -> None:
    report = score_rejudge.score([], {})
    assert report["n_records"] == 0
    assert report["n_unlabeled"] == 0
    assert report["directions"] == []
    assert [row["source"] for row in report["table"]] == ["baseline"]


# ---------------------------------------------------------------------------
# load_labels / load_outcomes
# ---------------------------------------------------------------------------


def test_load_labels_reads_index_and_label(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text('{"index": 0, "label": "constraint_only"}\n{"index": 2, "label": "real_degeneracy"}\n', encoding="utf-8")
    assert score_rejudge.load_labels(path) == {0: "constraint_only", 2: "real_degeneracy"}


def test_load_labels_rejects_duplicate_index(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text('{"index": 0, "label": "constraint_only"}\n{"index": 0, "label": "mixed"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        score_rejudge.load_labels(path)


def test_load_outcomes_reads_jsonl_in_order(tmp_path: Path) -> None:
    path = tmp_path / "outcomes.jsonl"
    rows = _eight_rows()[:2]
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    loaded = score_rejudge.load_outcomes(path)
    assert [r["index"] for r in loaded] == [0, 1]


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def test_build_arg_parser_requires_all_three_paths() -> None:
    parser = score_rejudge.build_arg_parser()
    args = parser.parse_args(["--rejudge-out", "o.jsonl", "--labels", "l.jsonl", "--out", "r.json"])
    assert args.rejudge_out == "o.jsonl"
    assert args.labels == "l.jsonl"
    assert args.out == "r.json"
