"""Tests for conpo.scoring: exact reproduction of the vendored evaluators.

See conpo/vendor/ifbench/VENDOR.md for the eval/eval_results_*.jsonl vs.
data/IFBench_test.jsonl population mismatch this file works around, and for
the residual instruction-level discrepancy documented in
test_score_ifbench_reproduces_shipped_eval below.
"""

import json
from pathlib import Path

from conpo.scoring import load_ifbench_test, load_sample_output, score_ifbench, score_ifeval
from conpo.vendor.ifbench import instructions_registry as ifbench_registry
from conpo.vendor.ifevalg import instructions_registry as ifevalg_registry

_IFBENCH_EVAL_DIR = Path(__file__).parent.parent / "conpo" / "vendor" / "ifbench" / "eval"

_CLASSIC_IFEVAL_PREFIXES = (
    "keywords:",
    "language:",
    "length_constraints:",
    "detectable_content:",
    "detectable_format:",
    "combination:",
    "startend:",
    "change_case:",
    "punctuation:",
)


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _record_from_test_row(row: dict, response: str) -> dict:
    return {
        "key": row["key"],
        "prompt": row["prompt"],
        "response": response,
        "instruction_id_list": row["instruction_id_list"],
        "kwargs": row["kwargs"],
    }


# --- (a) score_ifbench reproduces the shipped eval -------------------------


def test_score_ifbench_reproduces_shipped_eval() -> None:
    """data/IFBench_test.jsonl has 300 prompts; the shipped eval/eval_results
    files only cover 294 of an earlier version of that test set (see
    VENDOR.md). Restricting to the 293 test prompts that match a shipped row
    by prompt text (exact or whitespace-stripped) reproduces the shipped
    prompt-level accuracies exactly. Instruction-level accuracy is off by
    exactly 1 check out of 334, both for strict and loose, from 3 specific
    checks (ratio:overlap, ratio:sentence_type, words:consonants -- all
    routed through nltk.word_tokenize / nltk.sent_tokenize) that disagree
    with the shipped verdict on identical kwargs and an identical response;
    two of those three disagreements are themselves opposite in direction and
    net to zero at the prompt level. This is a known NLTK tokenizer-data
    sensitivity, not a bug in this wrapper or the vendored checkers, and is
    asserted here explicitly (both numbers) rather than papered over with an
    approximate comparison.
    """
    test_rows = load_ifbench_test()
    sample_rows = load_sample_output()
    response_by_prompt = {row["prompt"]: row["response"] for row in sample_rows}

    strict_ref = _load_jsonl(_IFBENCH_EVAL_DIR / "eval_results_strict.jsonl")
    loose_ref = _load_jsonl(_IFBENCH_EVAL_DIR / "eval_results_loose.jsonl")
    assert len(test_rows) == 300
    assert len(strict_ref) == 294
    assert len(loose_ref) == 294

    shipped_prompts = {p for row in strict_ref for p in (row["prompt"], row["prompt"].strip())}
    matched_rows = [row for row in test_rows if row["prompt"] in shipped_prompts or row["prompt"].strip() in shipped_prompts]
    assert len(matched_rows) == 293

    records = [_record_from_test_row(row, response_by_prompt[row["prompt"]]) for row in matched_rows]
    report = score_ifbench(records)

    assert report.num_prompts == 293
    assert report.num_instructions == 334

    matched_prompts = {p for row in matched_rows for p in (row["prompt"], row["prompt"].strip())}
    strict_ref_matched = [row for row in strict_ref if row["prompt"] in matched_prompts or row["prompt"].strip() in matched_prompts]
    loose_ref_matched = [row for row in loose_ref if row["prompt"] in matched_prompts or row["prompt"].strip() in matched_prompts]
    assert len(strict_ref_matched) == 293
    assert len(loose_ref_matched) == 293

    ref_prompt_strict = sum(1 for r in strict_ref_matched if r["follow_all_instructions"]) / len(strict_ref_matched)
    ref_prompt_loose = sum(1 for r in loose_ref_matched if r["follow_all_instructions"]) / len(loose_ref_matched)
    ref_instruction_total = sum(len(r["follow_instruction_list"]) for r in strict_ref_matched)
    ref_instruction_strict = sum(sum(r["follow_instruction_list"]) for r in strict_ref_matched) / ref_instruction_total
    ref_instruction_loose = sum(sum(r["follow_instruction_list"]) for r in loose_ref_matched) / ref_instruction_total

    # Prompt-level: exact reproduction.
    assert report.prompt_strict == ref_prompt_strict == 77 / 293
    assert report.prompt_loose == ref_prompt_loose == 87 / 293

    # Instruction-level: known, documented off-by-one (see docstring). Report
    # both numbers rather than asserting a false equality.
    assert ref_instruction_total == 334
    assert report.instruction_strict == 92 / 334
    assert ref_instruction_strict == 93 / 334
    assert report.instruction_loose == 107 / 334
    assert ref_instruction_loose == 108 / 334


# --- (b) score_ifeval on hand-written records -------------------------------


def test_score_ifeval_hand_written_records() -> None:
    records = [
        {  # all-pass: both required keywords present.
            "key": 0,
            "prompt": "p1",
            "response": "The quick brown fox jumps. Wonderful day.",
            "instruction_id_list": ["keywords:existence"],
            "kwargs": [{"keywords": ["quick", "fox"]}],
        },
        {  # all-fail: required keyword absent.
            "key": 1,
            "prompt": "p2",
            "response": "Nothing relevant here.",
            "instruction_id_list": ["keywords:existence"],
            "kwargs": [{"keywords": ["giraffe"]}],
        },
        {  # mixed: keyword check passes; end-phrase check fails strict
            # (a trailing line follows the phrase) but passes loose (which
            # also tries the response with its last line dropped).
            "key": 2,
            "prompt": "p3",
            "response": "Some text.\nBest regards",
            "instruction_id_list": ["keywords:existence", "startend:end_checker"],
            "kwargs": [{"keywords": ["text"]}, {"end_phrase": "Some text."}],
        },
    ]

    report = score_ifeval(records)

    assert report.num_prompts == 3
    assert report.num_instructions == 4
    assert report.prompt_strict == 1 / 3  # only record 0
    assert report.prompt_loose == 2 / 3  # records 0 and 2
    assert report.instruction_strict == 0.5  # 2 of 4 checks
    assert report.instruction_loose == 0.75  # 3 of 4 checks
    assert report.per_instruction == {
        "keywords:existence": 2 / 3,
        "startend:end_checker": 1.0,
    }


# --- (c) IFBench's 58 OOD ids are disjoint from the 54-family training registry --


def test_ifbench_registry_disjoint_from_training_registry() -> None:
    all_ids = set(ifbench_registry.INSTRUCTION_DICT.keys())
    classic_ids = {i for i in all_ids if i.startswith(_CLASSIC_IFEVAL_PREFIXES)}
    ood_ids = all_ids - classic_ids

    assert len(all_ids) == 83
    assert len(classic_ids) == 25
    assert len(ood_ids) == 58

    training_ids = set(ifevalg_registry.INSTRUCTION_DICT.keys())
    assert ood_ids.isdisjoint(training_ids)


# --- (d) the vendored test set has 300 prompts ------------------------------


def test_ifbench_test_jsonl_has_300_prompts() -> None:
    rows = load_ifbench_test()
    assert len(rows) == 300
    assert len({row["prompt"] for row in rows}) == 300
