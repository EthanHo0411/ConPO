"""Tests for scripts/rejudge.py: variant anchor surgery, --variant list
parsing, checklist boolean extraction, judge_replies.jsonl loading (with
index), judging, report arithmetic, and CLI parsing.

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
from conpo.judge import VetoRequest, Verdict, build_veto_prompt, veto_plus_label

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import rejudge  # noqa: E402

# ---------------------------------------------------------------------------
# A real rendered judge_request (via conpo.judge.build_veto_prompt, not a
# hand-typed anchor) plus its minus/plus labels, since build_veto_prompt
# swaps which letter carries y_minus per request.
# ---------------------------------------------------------------------------


def _request() -> VetoRequest:
    return VetoRequest(
        prompt="Write a short poem about rivers.",
        y_plus="Rivers flow, apple trees grow, three lines in a row.",
        y_minus="Rivers flow.",
        families=("keywords:existence",),
        descriptions=("Include the word apple.",),
    )


def _rendered_and_labels() -> tuple[str, str, str]:
    """(judge_request, minus_label, plus_label) for one fixed request."""
    req = _request()
    plus_label = veto_plus_label(req)
    minus_label = "B" if plus_label == "A" else "A"
    return build_veto_prompt(req), minus_label, plus_label


# ---------------------------------------------------------------------------
# apply_variant / VARIANT_FUNCS: anchor surgery
# ---------------------------------------------------------------------------


def test_apply_variant_none_is_identity() -> None:
    request, _, _ = _rendered_and_labels()
    assert rejudge.apply_variant(request, "none") == request


def test_apply_variant_unknown_variant_raises() -> None:
    with pytest.raises(ValueError):
        rejudge.apply_variant("anything", "not_a_real_variant")


@pytest.mark.parametrize("variant", rejudge.VARIANT_NAMES)
def test_apply_variant_missing_anchor_raises_for_every_known_variant(variant: str) -> None:
    with pytest.raises(ValueError):
        rejudge.apply_variant("no anchor text here at all", variant)


def test_ignore_constraints_inserts_sentence_once_before_json_anchor() -> None:
    request, _, _ = _rendered_and_labels()
    out = rejudge.apply_variant(request, "ignore_constraints")
    assert out.count(rejudge._IGNORE_CONSTRAINTS_SENTENCE) == 1
    assert out.index(rejudge._IGNORE_CONSTRAINTS_SENTENCE) < out.index(rejudge._JSON_INSTRUCTION_ANCHOR)
    assert "Rivers flow, apple trees grow" in out  # rest of the prompt intact


def test_explicit_exclusions_replaces_judge_only_sentence_once_with_correct_labels() -> None:
    request, minus, plus = _rendered_and_labels()
    out = rejudge.apply_variant(request, "explicit_exclusions")
    expected = rejudge._EXPLICIT_EXCLUSIONS_TEMPLATE.format(minus=minus, plus=plus)
    assert out.count(expected) == 1
    assert "DEGENERATE: truncated, garbled, off-task, or changed in meaning" not in out
    assert f"Response {minus} is a deliberately modified variant of response {plus}." in out  # first sentence intact
    assert rejudge._JSON_INSTRUCTION_ANCHOR in out  # untouched


def test_checklist_json_replaces_json_instruction_once() -> None:
    request, _, _ = _rendered_and_labels()
    out = rejudge.apply_variant(request, "checklist_json")
    assert out.count(rejudge._CHECKLIST_JSON_INSTRUCTION) == 1
    assert rejudge._JSON_INSTRUCTION_ANCHOR not in out


def test_two_examples_inserts_both_examples_once_before_lead_sentence() -> None:
    request, minus, plus = _rendered_and_labels()
    out = rejudge.apply_variant(request, "two_examples")
    assert out.count(rejudge._EXAMPLE_1) == 1
    assert out.count(rejudge._EXAMPLE_2) == 1
    lead = f"Response {minus} is a deliberately modified variant of response {plus}. "
    assert out.index(rejudge._EXAMPLE_1) < out.index(rejudge._EXAMPLE_2) < out.index(lead)


def test_ignore_plus_exclusions_applies_variants_1_and_2_together() -> None:
    request, minus, plus = _rendered_and_labels()
    out = rejudge.apply_variant(request, "ignore_plus_exclusions")
    assert rejudge._IGNORE_CONSTRAINTS_SENTENCE in out
    assert rejudge._EXPLICIT_EXCLUSIONS_TEMPLATE.format(minus=minus, plus=plus) in out
    assert "DEGENERATE: truncated, garbled, off-task, or changed in meaning" not in out


def test_checklist_plus_exclusions_applies_explicit_exclusions_then_checklist_json_once() -> None:
    request, minus, plus = _rendered_and_labels()
    original = request
    out = rejudge.apply_variant(request, "checklist_plus_exclusions")
    excl = rejudge._EXPLICIT_EXCLUSIONS_TEMPLATE.format(minus=minus, plus=plus)
    assert out.count(excl) == 1
    assert out.count(rejudge._CHECKLIST_JSON_INSTRUCTION) == 1
    assert out.index(excl) < out.index(rejudge._CHECKLIST_JSON_INSTRUCTION)  # applied in order
    assert rejudge._JSON_INSTRUCTION_ANCHOR not in out
    assert "DEGENERATE: truncated, garbled, off-task, or changed in meaning" not in out
    assert request == original  # baseline untouched


def test_checklist_inline_exclusions_inserts_binding_sentence_before_json_object_once() -> None:
    request, _, _ = _rendered_and_labels()
    original = request
    out = rejudge.apply_variant(request, "checklist_inline_exclusions")
    assert out.count(rejudge._CHECKLIST_INLINE_EXCLUSIONS_INSTRUCTION) == 1
    assert out.count(rejudge._CHECKLIST_INLINE_EXCLUSIONS_SENTENCE) == 1
    assert out.index(rejudge._CHECKLIST_INLINE_EXCLUSIONS_SENTENCE) < out.index('"truncated"')  # sentence before the JSON object
    assert rejudge._JSON_INSTRUCTION_ANCHOR not in out
    assert request == original  # baseline untouched


# ---------------------------------------------------------------------------
# parse_variants
# ---------------------------------------------------------------------------


def test_parse_variants_none_is_empty_list() -> None:
    assert rejudge.parse_variants("none") == []


def test_parse_variants_splits_comma_list_and_dedups_preserving_order() -> None:
    assert rejudge.parse_variants("checklist_json,ignore_constraints,checklist_json") == [
        "checklist_json",
        "ignore_constraints",
    ]


def test_parse_variants_trims_whitespace() -> None:
    assert rejudge.parse_variants(" ignore_constraints , checklist_json ") == ["ignore_constraints", "checklist_json"]


def test_parse_variants_none_mixed_with_a_real_name_drops_the_none_token() -> None:
    assert rejudge.parse_variants("none,ignore_constraints") == ["ignore_constraints"]


def test_parse_variants_unknown_name_raises() -> None:
    with pytest.raises(ValueError):
        rejudge.parse_variants("not_a_real_variant")


# ---------------------------------------------------------------------------
# extract_checklist
# ---------------------------------------------------------------------------


def test_extract_checklist_all_four_keys_present() -> None:
    raw = (
        '{"truncated": true, "garbled": false, "off_task": false, '
        '"meaning_changed_beyond_constraints": false, "degenerate": true, "reason": "x"}'
    )
    assert rejudge.extract_checklist(raw) == {
        "truncated": True,
        "garbled": False,
        "off_task": False,
        "meaning_changed_beyond_constraints": False,
    }


def test_extract_checklist_missing_a_key_returns_none() -> None:
    raw = '{"truncated": true, "garbled": false, "off_task": false, "degenerate": true, "reason": "x"}'
    assert rejudge.extract_checklist(raw) is None


def test_extract_checklist_plain_degenerate_reply_returns_none() -> None:
    assert rejudge.extract_checklist('{"degenerate": false, "reason": "fine"}') is None


def test_extract_checklist_finds_json_wrapped_in_prose() -> None:
    raw = (
        "Sure, here you go.\n"
        '{"truncated": false, "garbled": false, "off_task": false, '
        '"meaning_changed_beyond_constraints": true, "degenerate": true, "reason": "x"}\nDone."'
    )
    checklist = rejudge.extract_checklist(raw)
    assert checklist is not None
    assert checklist["meaning_changed_beyond_constraints"] is True


def test_extract_checklist_garbage_returns_none() -> None:
    assert rejudge.extract_checklist("not json at all") is None


# ---------------------------------------------------------------------------
# load_records: direction/limit filtering plus the index it assigns
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


def test_load_records_assigns_sequential_index_across_all_lines(tmp_path: Path) -> None:
    path = tmp_path / "replies.jsonl"
    _write_jsonl(
        path,
        [
            {"direction": "reverse", "step": 1},  # index 0, dropped (no judge_request)
            {"direction": "forward", "step": 2, "judge_request": "req-2"},  # index 1, kept
            {"direction": "forward", "step": 3, "judge_request": "req-3"},  # index 2, kept
        ],
    )
    records, n_skipped = rejudge.load_records([str(path)], ["forward", "reverse"], limit=0)
    assert [r["index"] for r in records] == [1, 2]
    assert n_skipped == 1


def test_load_records_index_survives_direction_filtering(tmp_path: Path) -> None:
    path = tmp_path / "replies.jsonl"
    _write_jsonl(
        path,
        [
            {"direction": "reverse", "step": 1, "judge_request": "r1"},  # index 0, filtered out
            {"direction": "forward", "step": 2, "judge_request": "f1"},  # index 1, kept
        ],
    )
    records, n_skipped = rejudge.load_records([str(path)], ["forward"], limit=0)
    assert [r["index"] for r in records] == [1]
    assert n_skipped == 0


def test_load_records_preserves_an_existing_index(tmp_path: Path) -> None:
    path = tmp_path / "replies.jsonl"
    _write_jsonl(path, [{"direction": "forward", "step": 1, "judge_request": "f1", "index": 99}])
    records, _ = rejudge.load_records([str(path)], ["forward"], limit=0)
    assert records[0]["index"] == 99


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


def test_load_records_reads_two_files_with_a_continuing_index(tmp_path: Path) -> None:
    path_a = tmp_path / "a.jsonl"
    path_b = tmp_path / "b.jsonl"
    _write_jsonl(path_a, [{"direction": "forward", "step": 1, "judge_request": "a1"}])
    _write_jsonl(path_b, [{"direction": "forward", "step": 2, "judge_request": "b1"}])
    records, n_skipped = rejudge.load_records([str(path_a), str(path_b)], ["forward", "reverse"], limit=0)
    assert [r["step"] for r in records] == [1, 2]
    assert [r["index"] for r in records] == [0, 1]
    assert n_skipped == 0


# ---------------------------------------------------------------------------
# judge_records / build_report
# ---------------------------------------------------------------------------


def _reply_record(index: int, step: int, direction: str, keep: bool, families: list[str], request_id: str) -> dict[str, Any]:
    """One judge_replies.jsonl record as returned by load_records (carrying
    "index"), with a distinctive judge_request so a fake judge can key off
    it."""
    return {
        "index": index,
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
    records: list[dict[str, Any]],
    variants: list[str],
    baseline_keep: list[bool],
    variant_keeps: dict[str, list[bool]],
) -> tuple[rejudge.JudgeFn, list[list[str]]]:
    """A judge_fn keyed by exact rendered-prompt text, plus the list of
    prompt-list arguments it was called with (to assert one combined call)."""
    verdict_by_prompt: dict[str, Verdict] = {}
    for i, rec in enumerate(records):
        base = rec["judge_request"]
        verdict_by_prompt[base] = Verdict(keep=baseline_keep[i], reason=f"baseline reason {i}", raw="", parse_ok=True)
        for v in variants:
            text = rejudge.apply_variant(base, v)
            verdict_by_prompt[text] = Verdict(keep=variant_keeps[v][i], reason=f"{v} reason {i}", raw="", parse_ok=True)

    calls: list[list[str]] = []

    def fake(prompts: list[str], model: str) -> list[Verdict]:
        calls.append(list(prompts))
        return [verdict_by_prompt[p] for p in prompts]

    return fake, calls


def test_judge_records_runs_one_combined_burst_for_baseline_and_every_variant() -> None:
    records = [
        _reply_record(0, 10, "forward", False, ["keywords:existence"], "r0"),
        _reply_record(1, 11, "forward", True, ["length_constraints:number_sentences"], "r1"),
        _reply_record(2, 12, "forward", True, ["keywords:existence"], "r2"),
        _reply_record(3, 20, "reverse", False, ["keywords:forbidden_words"], "r3"),
        _reply_record(4, 21, "reverse", False, ["keywords:existence"], "r4"),
        _reply_record(5, 22, "reverse", True, ["length_constraints:number_words"], "r5"),
    ]
    baseline_keep = [False, True, True, True, False, False]
    variant_keeps = {
        "ignore_constraints": [True, False, True, False, False, True],
        "checklist_json": [False, False, True, True, False, True],
    }
    variants = ["ignore_constraints", "checklist_json"]
    fake, calls = _build_fake_judge(records, variants, baseline_keep, variant_keeps)

    outcomes = rejudge.judge_records(records, variants, "fake-model", judge_fn=fake)

    # one judge_fn call: baseline prompts (unmodified judge_request, even
    # with variants requested) then each variant's prompts, in order.
    assert len(calls) == 1
    expected_prompts = (
        [r["judge_request"] for r in records]
        + [rejudge.apply_variant(r["judge_request"], "ignore_constraints") for r in records]
        + [rejudge.apply_variant(r["judge_request"], "checklist_json") for r in records]
    )
    assert calls[0] == expected_prompts

    assert [o["index"] for o in outcomes] == [0, 1, 2, 3, 4, 5]
    assert outcomes[0]["baseline"] == {"keep": False, "reason": "baseline reason 0", "parse_ok": True}
    assert outcomes[0]["variants"]["ignore_constraints"]["keep"] is True
    assert outcomes[0]["variants"]["checklist_json"]["keep"] is False
    assert set(outcomes[0]["variants"]) == {"ignore_constraints", "checklist_json"}


def test_report_arithmetic_multi_variant_mixed_directions_and_flips() -> None:
    records = [
        _reply_record(0, 10, "forward", False, ["keywords:existence"], "r0"),
        _reply_record(1, 11, "forward", True, ["length_constraints:number_sentences"], "r1"),
        _reply_record(2, 12, "forward", True, ["keywords:existence"], "r2"),
        _reply_record(3, 20, "reverse", False, ["keywords:forbidden_words"], "r3"),
        _reply_record(4, 21, "reverse", False, ["keywords:existence"], "r4"),
        _reply_record(5, 22, "reverse", True, ["length_constraints:number_words"], "r5"),
    ]
    baseline_keep = [False, True, True, True, False, False]
    variant_keeps = {
        "ignore_constraints": [True, False, True, False, False, True],
        "checklist_json": [False, False, True, True, False, True],
    }
    variants = ["ignore_constraints", "checklist_json"]
    fake, _ = _build_fake_judge(records, variants, baseline_keep, variant_keeps)
    outcomes = rejudge.judge_records(records, variants, "fake-model", judge_fn=fake)

    report = rejudge.build_report(
        outcomes, n_skipped=2, variants=variants, judge_model="fake-model", directions=["forward", "reverse"]
    )

    assert report["n_judged"] == 6
    assert report["n_skipped"] == 2

    overall = report["overall"]
    assert overall["n"] == 6
    assert overall["original_keep_rate"] == pytest.approx(3 / 6)
    assert overall["baseline_keep_rate"] == pytest.approx(3 / 6)
    assert overall["baseline_agreement"] == pytest.approx(4 / 6)
    assert "keep_rate" not in overall  # baseline-only: no variant fields leak into the top level

    ic = report["variants"]["ignore_constraints"]
    assert ic["overall"]["keep_rate"] == pytest.approx(3 / 6)
    assert ic["overall"]["agreement_with_original"] == pytest.approx(4 / 6)
    assert ic["overall"]["original_veto_to_variant_keep"] == 1
    assert ic["overall"]["original_keep_to_variant_veto"] == 1
    assert ic["by_direction"]["forward"]["keep_rate"] == pytest.approx(2 / 3)
    assert ic["by_direction"]["forward"]["agreement_with_original"] == pytest.approx(1 / 3)
    assert ic["by_direction"]["reverse"]["agreement_with_original"] == pytest.approx(1.0)
    assert {f["index"] for f in ic["flips"]} == {0, 1}
    for f in ic["flips"]:
        assert f["baseline_reason"].startswith("baseline reason")
        assert f["variant_reason"].startswith("ignore_constraints reason")

    cj = report["variants"]["checklist_json"]
    assert cj["overall"]["keep_rate"] == pytest.approx(3 / 6)
    assert cj["overall"]["agreement_with_original"] == pytest.approx(4 / 6)
    assert cj["overall"]["original_veto_to_variant_keep"] == 1
    assert cj["overall"]["original_keep_to_variant_veto"] == 1
    assert cj["by_direction"]["forward"]["keep_rate"] == pytest.approx(1 / 3)
    assert cj["by_direction"]["reverse"]["keep_rate"] == pytest.approx(2 / 3)
    assert {f["index"] for f in cj["flips"]} == {1, 3}


def test_judge_records_no_variants_leaves_variants_dict_empty() -> None:
    records = [
        _reply_record(0, 1, "forward", True, ["keywords:existence"], "a"),
        _reply_record(1, 2, "reverse", False, ["keywords:existence"], "b"),
    ]
    seen_prompts: list[str] = []

    def fake(prompts: list[str], model: str) -> list[Verdict]:
        seen_prompts.extend(prompts)
        return [Verdict(keep=True, reason="ok", raw="", parse_ok=True) for _ in prompts]

    outcomes = rejudge.judge_records(records, [], "fake-model", judge_fn=fake)
    assert seen_prompts == [r["judge_request"] for r in records]  # no variant prompts built or sent
    assert all(o["variants"] == {} for o in outcomes)

    report = rejudge.build_report(
        outcomes, n_skipped=0, variants=[], judge_model="fake-model", directions=["forward", "reverse"]
    )
    assert report["variants"] == {}


def test_judge_records_attaches_checklist_only_when_present_in_the_raw_reply() -> None:
    records = [_reply_record(0, 1, "forward", True, ["keywords:existence"], "c0")]
    checklist_raw = (
        '{"truncated": true, "garbled": false, "off_task": false, '
        '"meaning_changed_beyond_constraints": false, "degenerate": true, "reason": "x"}'
    )

    def fake(prompts: list[str], model: str) -> list[Verdict]:
        verdicts = []
        for p in prompts:
            if p == records[0]["judge_request"]:
                verdicts.append(Verdict(keep=True, reason="baseline", raw='{"degenerate": false, "reason": "ok"}', parse_ok=True))
            else:
                verdicts.append(Verdict(keep=False, reason="checklist says degenerate", raw=checklist_raw, parse_ok=True))
        return verdicts

    outcomes = rejudge.judge_records(records, ["checklist_json"], "fake-model", judge_fn=fake)
    assert "checklist" not in outcomes[0]["baseline"]
    assert outcomes[0]["variants"]["checklist_json"]["checklist"] == {
        "truncated": True,
        "garbled": False,
        "off_task": False,
        "meaning_changed_beyond_constraints": False,
    }


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def test_build_arg_parser_replies_with_two_files() -> None:
    parser = rejudge.build_arg_parser()
    args = parser.parse_args(["--replies", "a.jsonl", "b.jsonl", "--out", "out.jsonl"])
    assert args.replies == ["a.jsonl", "b.jsonl"]
    assert args.variant == []
    assert args.judge_model == config.JUDGE_MODEL
    assert args.limit == 0
    assert args.directions == "forward,reverse"


def test_build_arg_parser_parses_comma_separated_variant_list() -> None:
    parser = rejudge.build_arg_parser()
    args = parser.parse_args(
        ["--replies", "a.jsonl", "--out", "out.jsonl", "--variant", "ignore_constraints,checklist_json"]
    )
    assert args.variant == ["ignore_constraints", "checklist_json"]


def test_build_arg_parser_rejects_unknown_variant() -> None:
    parser = rejudge.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--replies", "a.jsonl", "--out", "out.jsonl", "--variant", "nonsense"])
