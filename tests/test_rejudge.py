"""Tests for scripts/rejudge.py: variant insertion, judge_replies.jsonl
loading, report arithmetic, and CLI parsing.

scripts/ is not an installed package (pyproject.toml only packages `conpo*`),
so rejudge is imported by adding scripts/ to sys.path (same pattern as
tests/test_eval_scripts.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from conpo import config
from conpo.judge import Verdict

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import rejudge  # noqa: E402

# ---------------------------------------------------------------------------
# apply_variant
# ---------------------------------------------------------------------------


def test_apply_variant_none_is_identity() -> None:
    request = f"body text\n\n{rejudge._JSON_INSTRUCTION_ANCHOR}"
    assert rejudge.apply_variant(request, "none") == request


def test_apply_variant_ignore_constraints_lands_before_anchor_once() -> None:
    request = f"body text\n\n{rejudge._JSON_INSTRUCTION_ANCHOR}"
    sentence = rejudge.VARIANT_SENTENCES["ignore_constraints"]
    out = rejudge.apply_variant(request, "ignore_constraints")
    assert out.count(sentence) == 1
    assert out.index(sentence) < out.index(rejudge._JSON_INSTRUCTION_ANCHOR)
    assert "body text" in out


def test_apply_variant_unknown_variant_raises() -> None:
    with pytest.raises(ValueError):
        rejudge.apply_variant("anything", "not_a_real_variant")


def test_apply_variant_missing_anchor_raises_for_a_known_variant() -> None:
    with pytest.raises(ValueError):
        rejudge.apply_variant("no anchor text here", "ignore_constraints")


# ---------------------------------------------------------------------------
# load_records
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_load_records_skips_records_without_judge_request_and_counts_them(tmp_path: Path) -> None:
    path = tmp_path / "replies.jsonl"
    _write_jsonl(
        path,
        [
            {"direction": "forward", "step": 1},  # old 8-per-burst sample format, no judge_request
            {"direction": "forward", "step": 2, "judge_request": "req-2"},
            {"direction": "reverse", "step": 3},  # old format
        ],
    )
    records, n_skipped = rejudge.load_records([str(path)], ["forward", "reverse"], limit=0)
    assert [r["step"] for r in records] == [2]
    assert n_skipped == 2


def test_load_records_filters_by_direction(tmp_path: Path) -> None:
    path = tmp_path / "replies.jsonl"
    _write_jsonl(
        path,
        [
            {"direction": "forward", "step": 1, "judge_request": "f1"},
            {"direction": "reverse", "step": 2, "judge_request": "r1"},
        ],
    )
    records, n_skipped = rejudge.load_records([str(path)], ["forward"], limit=0)
    assert [r["step"] for r in records] == [1]
    assert n_skipped == 0


def test_load_records_limit_caps_records(tmp_path: Path) -> None:
    path = tmp_path / "replies.jsonl"
    _write_jsonl(path, [{"direction": "forward", "step": i, "judge_request": f"f{i}"} for i in range(5)])
    records, n_skipped = rejudge.load_records([str(path)], ["forward", "reverse"], limit=2)
    assert [r["step"] for r in records] == [0, 1]
    assert n_skipped == 0


def test_load_records_reads_two_files_in_order(tmp_path: Path) -> None:
    path_a = tmp_path / "a.jsonl"
    path_b = tmp_path / "b.jsonl"
    _write_jsonl(path_a, [{"direction": "forward", "step": 1, "judge_request": "a1"}])
    _write_jsonl(path_b, [{"direction": "forward", "step": 2, "judge_request": "b1"}])
    records, n_skipped = rejudge.load_records([str(path_a), str(path_b)], ["forward", "reverse"], limit=0)
    assert [r["step"] for r in records] == [1, 2]
    assert n_skipped == 0


# ---------------------------------------------------------------------------
# judge_records / build_report: a hand-built set of six records, mixed
# directions and mixed flips.
# ---------------------------------------------------------------------------


def _reply_record(step: int, direction: str, keep: bool, families: list[str], request_id: str) -> dict[str, Any]:
    """One judge_replies.jsonl record (conpo/trainer.py's _dump_judge_replies
    schema), with a distinctive judge_request so a fake judge can key off it."""
    return {
        "step": step,
        "direction": direction,
        "families": families,
        "keep": keep,
        "parse_ok": True,
        "raw": '{"degenerate": false, "reason": "training-time verdict"}',
        "prompt_chars": 10,
        "y_plus_chars": 10,
        "y_minus_chars": 5,
        "prompt": f"prompt {request_id}",
        "y_plus": "plus text",
        "y_minus": "minus text",
        "judge_request": f"record {request_id} body.\n\n{rejudge._JSON_INSTRUCTION_ANCHOR}",
        "y_plus_label": "A",
    }


def _build_fake_judge(
    records: list[dict[str, Any]], variant: str, baseline_keep: list[bool], variant_keep: list[bool]
) -> rejudge.JudgeFn:
    """A judge_fn keyed by exact rendered-prompt text: baseline_keep[i] answers
    record i's stored judge_request, variant_keep[i] answers its variant text."""
    verdict_by_prompt: dict[str, Verdict] = {}
    for i, rec in enumerate(records):
        base = rec["judge_request"]
        verdict_by_prompt[base] = Verdict(keep=baseline_keep[i], reason=f"baseline reason {i}", raw="", parse_ok=True)
        if variant != "none":
            variant_text = rejudge.apply_variant(base, variant)
            verdict_by_prompt[variant_text] = Verdict(
                keep=variant_keep[i], reason=f"variant reason {i}", raw="", parse_ok=True
            )

    def fake(prompts: list[str], model: str) -> list[Verdict]:
        return [verdict_by_prompt[p] for p in prompts]

    return fake


def test_report_arithmetic_mixed_directions_and_flips() -> None:
    records = [
        _reply_record(10, "forward", False, ["keywords:existence"], "r0"),
        _reply_record(11, "forward", True, ["length_constraints:number_sentences"], "r1"),
        _reply_record(12, "forward", True, ["keywords:existence"], "r2"),
        _reply_record(20, "reverse", False, ["keywords:forbidden_words"], "r3"),
        _reply_record(21, "reverse", False, ["keywords:existence"], "r4"),
        _reply_record(22, "reverse", True, ["length_constraints:number_words"], "r5"),
    ]
    # original_keep (from _reply_record above): F, T, T, F, F, T
    baseline_keep = [False, True, True, True, False, False]
    variant_keep = [True, False, True, False, False, True]
    fake = _build_fake_judge(records, "ignore_constraints", baseline_keep, variant_keep)

    outcomes = rejudge.judge_records(records, "ignore_constraints", "fake-model", judge_fn=fake)
    report = rejudge.build_report(
        outcomes, n_skipped=2, variant="ignore_constraints", judge_model="fake-model", directions=["forward", "reverse"]
    )

    assert report["n_judged"] == 6
    assert report["n_skipped"] == 2
    assert report["variant"] == "ignore_constraints"

    overall = report["overall"]
    assert overall["n"] == 6
    assert overall["original_keep_rate"] == pytest.approx(3 / 6)
    assert overall["baseline_keep_rate"] == pytest.approx(3 / 6)
    assert overall["baseline_agreement"] == pytest.approx(4 / 6)
    assert overall["variant_keep_rate"] == pytest.approx(3 / 6)
    assert overall["original_veto_to_variant_keep"] == 1
    assert overall["original_keep_to_variant_veto"] == 1

    fwd = report["by_direction"]["forward"]
    assert fwd["n"] == 3
    assert fwd["original_keep_rate"] == pytest.approx(2 / 3)
    assert fwd["baseline_agreement"] == pytest.approx(1.0)
    assert fwd["original_veto_to_variant_keep"] == 1
    assert fwd["original_keep_to_variant_veto"] == 1

    rev = report["by_direction"]["reverse"]
    assert rev["n"] == 3
    assert rev["original_keep_rate"] == pytest.approx(1 / 3)
    assert rev["baseline_agreement"] == pytest.approx(1 / 3)
    assert rev["original_veto_to_variant_keep"] == 0
    assert rev["original_keep_to_variant_veto"] == 0

    flips = report["flips"]
    assert {(f["step"], f["direction"]) for f in flips} == {(10, "forward"), (11, "forward")}
    for f in flips:
        assert f["baseline_reason"].startswith("baseline reason")
        assert f["variant_reason"].startswith("variant reason")
        assert "families" in f


def test_variant_none_skips_variant_judging_and_report_omits_variant_fields() -> None:
    records = [
        _reply_record(1, "forward", True, ["keywords:existence"], "a"),
        _reply_record(2, "reverse", False, ["keywords:existence"], "b"),
    ]
    seen_prompts: list[str] = []

    def fake(prompts: list[str], model: str) -> list[Verdict]:
        seen_prompts.extend(prompts)
        return [Verdict(keep=True, reason="ok", raw="", parse_ok=True) for _ in prompts]

    outcomes = rejudge.judge_records(records, "none", "fake-model", judge_fn=fake)
    assert seen_prompts == [r["judge_request"] for r in records]  # variant prompts never built or sent
    assert all("variant_keep" not in o for o in outcomes)

    report = rejudge.build_report(outcomes, n_skipped=0, variant="none", judge_model="fake-model", directions=["forward", "reverse"])
    assert "variant_keep_rate" not in report["overall"]
    assert "flips" not in report


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def test_build_arg_parser_replies_with_two_files() -> None:
    parser = rejudge.build_arg_parser()
    args = parser.parse_args(["--replies", "a.jsonl", "b.jsonl", "--out", "out.jsonl"])
    assert args.replies == ["a.jsonl", "b.jsonl"]
    assert args.variant == "none"
    assert args.judge_model == config.JUDGE_MODEL
    assert args.limit == 0
    assert args.directions == "forward,reverse"


def test_build_arg_parser_rejects_unknown_variant() -> None:
    parser = rejudge.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--replies", "a.jsonl", "--out", "out.jsonl", "--variant", "nonsense"])
