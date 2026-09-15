"""Tests for conpo.retention (scorers, row builders) and scripts/eval_retention.py
(--score-only path, --limit, --benchmarks, the RETENTION_* env knobs). Mirrors
tests/test_eval_scripts.py's approach for importing a scripts/ CLI as a module."""

import dataclasses
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from conpo import config
from conpo.retention import (
    RetentionRow,
    apply_bbh_suffix,
    extract_bbh_prediction,
    extract_gsm8k_prediction,
    extract_mmlu_prediction,
    extract_prediction,
    retention_lenient,
    rows_from_bbh,
    rows_from_gsm8k,
    rows_from_mmlu,
    score_bbh,
    score_gsm8k,
    score_mmlu,
)

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import eval_retention  # noqa: E402

RETENTION_ENV = (
    "RETENTION_SUFFIX",
    "RETENTION_MAX_TOKENS",
    "RETENTION_BBH_TASKS",
    "RETENTION_BENCHMARKS",
    "RETENTION_LENIENT",
)


@pytest.fixture(autouse=True)
def _clear_retention_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default-path assertions below must not depend on RETENTION_*
    values leaking in from the shell."""
    for name in RETENTION_ENV:
        monkeypatch.delenv(name, raising=False)


# --- score_gsm8k -------------------------------------------------------------


def test_score_gsm8k_correct_with_hash_marker() -> None:
    assert score_gsm8k("Step 1... Step 2...\n#### 18", "18") is True


def test_score_gsm8k_correct_without_hash_marker() -> None:
    assert score_gsm8k("Let's compute step by step. The final answer is 18", "18") is True


def test_score_gsm8k_strips_commas_in_response() -> None:
    assert score_gsm8k("Total spent.\n#### 1,234", "1234") is True


def test_score_gsm8k_wrong_last_number() -> None:
    assert score_gsm8k("Reasoning here. The answer is 42.", "18") is False


def test_score_gsm8k_marker_with_no_digits_falls_back_to_whole_response() -> None:
    # "####" present but nothing numeric after it: falls back to the last number
    # anywhere in the response instead of finding zero matches in the tail.
    assert score_gsm8k("#### see above. The final total is 18", "18") is True


# --- score_mmlu ---------------------------------------------------------------


def test_score_mmlu_bare_letter() -> None:
    assert score_mmlu("B", "B") is True


def test_score_mmlu_parenthesised_letter() -> None:
    assert score_mmlu("(B)", "B") is True


def test_score_mmlu_letter_in_sentence() -> None:
    assert score_mmlu("The answer is B.", "B") is True


def test_score_mmlu_no_letter_found() -> None:
    assert score_mmlu("I am not sure about this one.", "B") is False


def test_score_mmlu_wrong_letter() -> None:
    assert score_mmlu("A", "B") is False


# --- score_bbh -----------------------------------------------------------------


def test_score_bbh_paren_letter_target_contained_in_line() -> None:
    assert score_bbh("Reasoning...\nThe answer is (a).", "(A)") is True


def test_score_bbh_true_false() -> None:
    assert score_bbh("Working through the logic.\nTrue.", "True") is True
    assert score_bbh("Working through the logic.\nFalse.", "True") is False


def test_score_bbh_sorted_words_target_case_insensitive() -> None:
    assert score_bbh("apple, banana, cherry\nApple Banana Cherry", "apple banana cherry") is True


def test_score_bbh_trailing_full_stop_stripped() -> None:
    assert score_bbh("Reasoning.\nTrue.", "True") is True


def test_score_bbh_wrong_answer() -> None:
    assert score_bbh("Reasoning.\nFalse.", "True") is False


def test_score_bbh_paren_letter_bare_letter_inside_prose_stays_wrong() -> None:
    # Regression, Task F3 finding 2: the old bare-letter fallback matched any
    # standalone lowercase letter anywhere in the prediction, so target "(A)"
    # scored True against a sentence merely containing the word "a".
    response = "I am not sure which option is correct, it could be a tough call."
    assert score_bbh(response, "(A)") is False


def test_score_bbh_paren_letter_sentence_starting_with_letter_word_stays_wrong() -> None:
    # Same bug, target "(I)": the sentence's leading "I" is not a committed
    # marker answer, so it must not match either.
    response = "I am not sure which option is correct, it could be a tough call."
    assert score_bbh(response, "(I)") is False


# --- score_bbh: robust extractor, Task F point 2 (real Qwen3-4B responses) -----
# Every response string below is copied verbatim (full response, or a verbatim
# tail when the full response is long prose) from
# scratchpad/bbh_group_samples.jsonl -- real record_verdicts output for the
# base-model retention run, not fabricated text.


@pytest.mark.parametrize(
    "response,gold,expected",
    [
        # dyck_languages: model repeats the whole sequence, ending in the
        # gold closing brackets -- accepted (point 5 leniency).
        ("[ [ ] ]", "] ]", True),
        ("( { { } } )", ")", True),
        ("< ( ( ( [ { } ] ) ) ) >", ") ) >", True),
        # dyck_languages: repeats the sequence but never closes the outer
        # bracket -- a genuine wrong answer, still scores wrong.
        ("< [ [ ] ]", "] ] >", False),
    ],
)
def test_score_bbh_dyck_languages_whole_sequence_repeat(response: str, gold: str, expected: bool) -> None:
    assert score_bbh(response, gold) is expected


@pytest.mark.parametrize(
    "response",
    [
        # "Final Answer:" with the value on the same line, more text follows.
        "### Final Answer: **No**  \nElanor does **not** tell the truth.",
        # last marker in the text wins even when an earlier empty marker exists.
        "### ✅ Final Answer:\n\n**No, Shalonda does not tell the truth.**  \n**Shalonda lies.**\n\n**Answer: No**",
        # marker line is not the last line of the response.
        "### Final Answer: **No**  \n**Ka does not tell the truth.**",
    ],
)
def test_score_bbh_web_of_lies_final_answer_marker(response: str) -> None:
    assert score_bbh(response, "No") is True


def test_score_bbh_web_of_lies_marker_with_trailing_clause_stays_wrong() -> None:
    # Labelled residual gap: the marker's own line carries the committed answer
    # plus an explanatory clause ("Yes, Elanor tells the truth."); point 2 only
    # strips a trailing period, not a trailing clause, so this does not match
    # bare gold "Yes". Out of scope for this pass (see extract_bbh_prediction).
    response = "### Final Answer: **Yes, Elanor tells the truth.**"
    assert score_bbh(response, "Yes") is False


def test_score_bbh_web_of_lies_truncated_response_no_marker_stays_wrong() -> None:
    # 512-token cap hit mid-reasoning: no marker, no committed answer -- a
    # capacity issue, not something the extractor can or should fix.
    response = (
        "So, the only statement we can be certain of is:\n\n- **Jim lies.**\n\n"
        "Now, if **Crista says Jim lies**, and we know that **Jim lies**, "
        "then **Crista is telling the truth**.\n\nSo"
    )
    assert score_bbh(response, "Yes") is False


def test_score_bbh_word_sorting_one_word_per_line() -> None:
    assert score_bbh("syndrome  \ntherefrom", "syndrome therefrom", "word_sorting") is True


def test_extract_bbh_prediction_word_sorting_comma_separated_list() -> None:
    response = (
        "barn, delmarva, dot, embezzle, greene, it&t, panicking, prod, scorch, same, "
        "splutter, subsist, thrill, drumhead, onus, obstetric, malton, guru, damp, entirety"
    )
    gold = "barn damp delmarva dot drumhead embezzle entirety greene guru it&t malton obstetric onus panicking prod same scorch splutter subsist thrill"
    # Comma-list is accepted and space-joined (point 4) once `group` is
    # word_sorting -- the extractor never sees `gold`...
    assert extract_bbh_prediction(response, "word_sorting") == response.replace(", ", " ")
    # ...but the model's order does not match the sorted gold, so it still scores wrong.
    assert score_bbh(response, gold, "word_sorting") is False


@pytest.mark.parametrize(
    "response,gold",
    [
        ("### Final Answer:\n$$\n\\boxed{24}\n$$", "24"),
        ("**Final Answer: 58**", "58"),
        ("**Final answer:**  \n$-80$", "-80"),
    ],
)
def test_score_bbh_multistep_arithmetic_answer_marker(response: str, gold: str) -> None:
    assert score_bbh(response, gold) is True


def test_score_bbh_logical_deduction_truncated_prose_with_commas_stays_wrong() -> None:
    # 512-token cap hit mid-reasoning; the trailing line has commas but each
    # comma-separated part is multi-word prose, not a word list, so it is left
    # to the plain last-line path (point 4's "all parts single token" guard)
    # instead of being corrupted by the comma-splitting meant for word_sorting.
    response = (
        "That means **Joe and Rob are in positions 1st, 2nd, or 4th, 5th, 6th**, "
        "but **Joe must be above Rob**, and **Rob must"
    )
    assert score_bbh(response, "(D)") is False


def test_score_bbh_logical_deduction_marker_value_on_next_line_still_matches() -> None:
    # No-regression check: this sample already scored correct under the old
    # last-line extractor via paren-containment; confirm the new marker path
    # (empty "Final Answer:" line, value on the next line) still finds it.
    response = "Third-to-last is **5th place** → **Eve**.\n\n### Final Answer:\n**(B) Eve finished third-to-last**."
    assert score_bbh(response, "(B)") is True


# --- extract_*_prediction ------------------------------------------------------


def test_extract_gsm8k_prediction_uses_marker() -> None:
    assert extract_gsm8k_prediction("Step 1... Step 2...\n#### 18") == "18"


def test_extract_gsm8k_prediction_none_when_no_number() -> None:
    assert extract_gsm8k_prediction("I don't know.") is None


def test_extract_mmlu_prediction_none_when_no_letter() -> None:
    assert extract_mmlu_prediction("I am not sure about this one.") is None


def test_extract_bbh_prediction_last_line() -> None:
    assert extract_bbh_prediction("Reasoning...\nThe answer is (a).") == "the answer is (a)"


def test_extract_bbh_prediction_none_when_empty() -> None:
    assert extract_bbh_prediction("") is None


def test_extract_prediction_dispatches_by_benchmark() -> None:
    gsm8k_row = RetentionRow(key="k", benchmark="gsm8k", prompt="p", target="18", group="")
    mmlu_row = RetentionRow(key="k", benchmark="mmlu", prompt="p", target="B", group="")
    bbh_row = RetentionRow(key="k", benchmark="bbh", prompt="p", target="True", group="")
    assert extract_prediction(gsm8k_row, "#### 18") == "18"
    assert extract_prediction(mmlu_row, "B") == "B"
    assert extract_prediction(bbh_row, "True") == "true"


def test_extract_prediction_unknown_benchmark_raises() -> None:
    row = RetentionRow(key="k", benchmark="nope", prompt="p", target="x", group="")
    with pytest.raises(ValueError):
        extract_prediction(row, "anything")


# --- rows_from_* ---------------------------------------------------------------


def test_rows_from_gsm8k() -> None:
    hf_rows = [
        {"question": "Q1?", "answer": "reasoning\n#### 42"},
        {"question": "Q2?", "answer": "reasoning\n#### 1,000"},
    ]
    rows = rows_from_gsm8k(hf_rows)
    assert [r.key for r in rows] == ["gsm8k:0", "gsm8k:1"]
    assert all(r.benchmark == "gsm8k" for r in rows)
    assert all(r.group == "" for r in rows)
    assert rows[0].target == "42"
    assert rows[1].target == "1000"
    assert "Q1?" in rows[0].prompt
    assert "#### <number>" in rows[0].prompt


def test_rows_from_mmlu() -> None:
    hf_rows = [{"question": "Q1?", "choices": ["c0", "c1", "c2", "c3"], "answer": 2, "subject": "anatomy"}]
    rows = rows_from_mmlu(hf_rows)
    assert rows[0].key == "mmlu:0"
    assert rows[0].benchmark == "mmlu"
    assert rows[0].target == "C"
    assert rows[0].group == "anatomy"
    assert "A. c0" in rows[0].prompt
    assert "C. c2" in rows[0].prompt
    assert "D. c3" in rows[0].prompt


def test_rows_from_bbh() -> None:
    hf_rows = [{"input": "not ( True ) and ( True ) is", "target": "False"}]
    rows = rows_from_bbh("boolean_expressions", hf_rows)
    assert rows[0].key == "bbh:boolean_expressions:0"
    assert rows[0].benchmark == "bbh"
    assert rows[0].target == "False"
    assert rows[0].group == "boolean_expressions"
    assert "not ( True ) and ( True ) is" in rows[0].prompt


# --- scripts/eval_retention.py --score-only ------------------------------------


def _write_rows_jsonl(path: Path, rows: list[RetentionRow]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dataclasses.asdict(row)) + "\n")


def _fixture_rows(tmp_path: Path) -> Path:
    """3 rows per benchmark, written to tmp_path/retention_{benchmark}.jsonl."""
    gsm8k_rows = [
        RetentionRow(key="gsm8k:0", benchmark="gsm8k", prompt="p", target="1", group=""),
        RetentionRow(key="gsm8k:1", benchmark="gsm8k", prompt="p", target="2", group=""),
        RetentionRow(key="gsm8k:2", benchmark="gsm8k", prompt="p", target="3", group=""),
    ]
    mmlu_rows = [
        RetentionRow(key="mmlu:0", benchmark="mmlu", prompt="p", target="A", group="sub1"),
        RetentionRow(key="mmlu:1", benchmark="mmlu", prompt="p", target="B", group="sub1"),
        RetentionRow(key="mmlu:2", benchmark="mmlu", prompt="p", target="C", group="sub2"),
    ]
    bbh_rows = [
        RetentionRow(key="bbh:task_a:0", benchmark="bbh", prompt="p", target="True", group="task_a"),
        RetentionRow(key="bbh:task_a:1", benchmark="bbh", prompt="p", target="False", group="task_a"),
        RetentionRow(key="bbh:task_b:0", benchmark="bbh", prompt="p", target="(A)", group="task_b"),
    ]
    _write_rows_jsonl(tmp_path / "retention_gsm8k.jsonl", gsm8k_rows)
    _write_rows_jsonl(tmp_path / "retention_mmlu.jsonl", mmlu_rows)
    _write_rows_jsonl(tmp_path / "retention_bbh.jsonl", bbh_rows)
    return tmp_path


def _fixture_canned(tmp_path: Path) -> Path:
    """2/3 gsm8k correct, 2/3 mmlu correct, 2/3 bbh correct, spread across groups."""
    canned = [
        {"key": "gsm8k:0", "response": "#### 1"},  # correct
        {"key": "gsm8k:1", "response": "#### 2"},  # correct
        {"key": "gsm8k:2", "response": "#### 99"},  # wrong
        {"key": "mmlu:0", "response": "A"},  # correct, sub1
        {"key": "mmlu:1", "response": "B"},  # correct, sub1
        {"key": "mmlu:2", "response": "D"},  # wrong, sub2
        {"key": "bbh:task_a:0", "response": "True"},  # correct
        {"key": "bbh:task_a:1", "response": "banana"},  # wrong
        {"key": "bbh:task_b:0", "response": "the answer is (a)"},  # correct
    ]
    path = tmp_path / "canned.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for record in canned:
            f.write(json.dumps(record) + "\n")
    return path


def test_eval_retention_score_only_all_benchmarks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = _fixture_rows(tmp_path)
    canned_path = _fixture_canned(tmp_path)
    out_path = tmp_path / "out.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_retention.py",
            "--data-dir",
            str(data_dir),
            "--score-only",
            str(canned_path),
            "--out",
            str(out_path),
        ],
    )
    eval_retention.main()
    result = json.loads(out_path.read_text(encoding="utf-8"))

    assert result["gsm8k"] == pytest.approx(2 / 3)
    assert result["mmlu"] == pytest.approx(2 / 3)
    assert result["bbh"] == pytest.approx(2 / 3)
    assert result["n"] == {"gsm8k": 3, "mmlu": 3, "bbh": 3}
    assert result["correct"] == {"gsm8k": 2, "mmlu": 2, "bbh": 2}
    assert result["by_group"]["gsm8k"] == {"": pytest.approx(2 / 3)}
    assert result["by_group"]["mmlu"] == {"sub1": 1.0, "sub2": 0.0}
    assert result["by_group"]["bbh"] == {"task_a": 0.5, "task_b": 1.0}
    assert result["limit"] is None
    assert result["adapter"] is None

    verdicts_path = tmp_path / "out.json.verdicts.jsonl"
    assert verdicts_path.exists()
    lines = [json.loads(line) for line in verdicts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 9
    by_key = {v["key"]: v for v in lines}
    assert by_key["gsm8k:2"]["correct"] is False
    assert by_key["mmlu:2"]["group"] == "sub2"
    assert by_key["bbh:task_b:0"]["correct"] is True

    # response/prediction/gold: inspectable without re-running generation.
    assert by_key["gsm8k:2"]["response"] == "#### 99"
    assert by_key["gsm8k:2"]["prediction"] == "99"
    assert by_key["gsm8k:2"]["gold"] == "3"
    assert by_key["mmlu:2"]["response"] == "D"
    assert by_key["mmlu:2"]["prediction"] == "D"
    assert by_key["mmlu:2"]["gold"] == "C"
    assert by_key["bbh:task_a:1"]["response"] == "banana"
    assert by_key["bbh:task_a:1"]["prediction"] == "banana"
    assert by_key["bbh:task_a:1"]["gold"] == "False"


def test_eval_retention_score_only_missing_key_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = _fixture_rows(tmp_path)
    canned_path = tmp_path / "short.jsonl"
    canned_path.write_text(json.dumps({"key": "gsm8k:0", "response": "#### 1"}) + "\n", encoding="utf-8")
    out_path = tmp_path / "out.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_retention.py", "--data-dir", str(data_dir), "--score-only", str(canned_path), "--out", str(out_path)],
    )
    with pytest.raises(KeyError):
        eval_retention.main()


def test_eval_retention_limit_caps_rows_per_benchmark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = _fixture_rows(tmp_path)
    canned = [
        {"key": "gsm8k:0", "response": "#### 1"},
        {"key": "gsm8k:1", "response": "#### 2"},
        {"key": "mmlu:0", "response": "A"},
        {"key": "mmlu:1", "response": "B"},
        {"key": "bbh:task_a:0", "response": "True"},
        {"key": "bbh:task_a:1", "response": "banana"},
    ]
    canned_path = tmp_path / "canned_limit.jsonl"
    with canned_path.open("w", encoding="utf-8") as f:
        for record in canned:
            f.write(json.dumps(record) + "\n")

    out_path = tmp_path / "out.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_retention.py",
            "--data-dir",
            str(data_dir),
            "--limit",
            "2",
            "--score-only",
            str(canned_path),
            "--out",
            str(out_path),
        ],
    )
    eval_retention.main()
    result = json.loads(out_path.read_text(encoding="utf-8"))
    assert result["n"] == {"gsm8k": 2, "mmlu": 2, "bbh": 2}
    assert result["limit"] == 2


def test_load_retention_rows_limit_zero_returns_empty(tmp_path: Path) -> None:
    data_dir = _fixture_rows(tmp_path)
    assert eval_retention.load_retention_rows(data_dir, "gsm8k", limit=0) == []


def test_eval_retention_groups_filters_rows_by_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = _fixture_rows(tmp_path)
    canned_path = _fixture_canned(tmp_path)
    out_path = tmp_path / "out.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_retention.py",
            "--data-dir",
            str(data_dir),
            "--groups",
            "task_b",
            "--score-only",
            str(canned_path),
            "--out",
            str(out_path),
        ],
    )
    eval_retention.main()
    result = json.loads(out_path.read_text(encoding="utf-8"))
    # gsm8k (group "") and mmlu (sub1/sub2) rows are all filtered out; only
    # bbh:task_b:0 matches, so gsm8k/mmlu are absent (0 rows), same as
    # test_eval_retention_benchmarks_subset_omits_others's "absent, not 0" rule.
    assert set(result["n"]) == {"bbh"}
    assert result["n"]["bbh"] == 1
    assert result["by_group"]["bbh"] == {"task_b": 1.0}


def test_eval_retention_benchmarks_subset_omits_others(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = _fixture_rows(tmp_path)
    canned_path = _fixture_canned(tmp_path)
    out_path = tmp_path / "out.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_retention.py",
            "--data-dir",
            str(data_dir),
            "--benchmarks",
            "gsm8k,bbh",
            "--score-only",
            str(canned_path),
            "--out",
            str(out_path),
        ],
    )
    eval_retention.main()
    result = json.loads(out_path.read_text(encoding="utf-8"))
    assert "mmlu" not in result
    assert "mmlu" not in result["n"]
    assert "mmlu" not in result["correct"]
    assert "mmlu" not in result["by_group"]
    assert set(result["n"]) == {"gsm8k", "bbh"}


# --- RETENTION_* env knobs -----------------------------------------------------
# RETENTION_SUFFIX / RETENTION_MAX_TOKENS / RETENTION_BBH_TASKS / RETENTION_BENCHMARKS;
# every default must stay byte-identical to the pre-knob behaviour.

DEFAULT_BBH_SUFFIX = "Give only the final answer on the last line."
CUSTOM_SUFFIX = "Think step by step, then write the final answer on the last line."
BBH_INPUT = "not ( True ) and ( True ) is"


def test_rows_from_bbh_default_prompt_exact_string() -> None:
    rows = rows_from_bbh("boolean_expressions", [{"input": BBH_INPUT, "target": "False"}])
    assert rows[0].prompt == "not ( True ) and ( True ) is\n\nGive only the final answer on the last line."


def test_rows_from_bbh_env_suffix_replaces_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETENTION_SUFFIX", CUSTOM_SUFFIX)
    rows = rows_from_bbh("boolean_expressions", [{"input": BBH_INPUT, "target": "False"}])
    assert rows[0].prompt == f"{BBH_INPUT}\n\n{CUSTOM_SUFFIX}"


def test_apply_bbh_suffix_identity_by_default_and_env_swaps_trailing_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt = f"{BBH_INPUT}\n\n{DEFAULT_BBH_SUFFIX}"
    assert apply_bbh_suffix(prompt) == prompt
    monkeypatch.setenv("RETENTION_SUFFIX", CUSTOM_SUFFIX)
    assert apply_bbh_suffix(prompt) == f"{BBH_INPUT}\n\n{CUSTOM_SUFFIX}"
    # A prompt written under some other suffix is left alone.
    assert apply_bbh_suffix(f"{BBH_INPUT}\n\nother") == f"{BBH_INPUT}\n\nother"


def test_load_retention_rows_env_suffix_rewrites_on_disk_bbh_prompts_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bbh_prompt = f"{BBH_INPUT}\n\n{DEFAULT_BBH_SUFFIX}"
    bbh_rows = [RetentionRow(key="bbh:web_of_lies:0", benchmark="bbh", prompt=bbh_prompt, target="Yes", group="web_of_lies")]
    # A non-BBH row whose prompt happens to end with the BBH suffix must not be rewritten.
    gsm8k_rows = [RetentionRow(key="gsm8k:0", benchmark="gsm8k", prompt=bbh_prompt, target="1", group="")]
    _write_rows_jsonl(tmp_path / "retention_bbh.jsonl", bbh_rows)
    _write_rows_jsonl(tmp_path / "retention_gsm8k.jsonl", gsm8k_rows)
    assert eval_retention.load_retention_rows(tmp_path, "bbh")[0].prompt == bbh_prompt
    monkeypatch.setenv("RETENTION_SUFFIX", CUSTOM_SUFFIX)
    assert eval_retention.load_retention_rows(tmp_path, "bbh")[0].prompt == f"{BBH_INPUT}\n\n{CUSTOM_SUFFIX}"
    assert eval_retention.load_retention_rows(tmp_path, "gsm8k")[0].prompt == bbh_prompt


def test_retention_max_tokens_env_overrides_config_default(monkeypatch: pytest.MonkeyPatch) -> None:
    assert eval_retention.retention_max_tokens() == config.RETENTION_MAX_TOKENS == 512
    monkeypatch.setenv("RETENTION_MAX_TOKENS", "2048")
    assert eval_retention.retention_max_tokens() == 2048


@pytest.mark.parametrize("raw", ["0", "-5", "abc", "1.5"])
def test_retention_max_tokens_env_rejects_non_positive_integers(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("RETENTION_MAX_TOKENS", raw)
    with pytest.raises(SystemExit, match="RETENTION_MAX_TOKENS"):
        eval_retention.retention_max_tokens()


def test_eval_retention_max_tokens_env_reaches_generation_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = _fixture_rows(tmp_path)
    out_path = tmp_path / "out.json"
    seen: list[int] = []

    def fake_generate(engine: object, prompts: list[str], max_tokens: int) -> list[str]:
        seen.append(max_tokens)
        return ["x"] * len(prompts)

    monkeypatch.setattr(eval_retention.eval_external, "_load_hf_engine", lambda model, adapter: object())
    monkeypatch.setattr(eval_retention.eval_external, "generate_greedy", fake_generate)
    monkeypatch.setenv("RETENTION_MAX_TOKENS", "2048")
    monkeypatch.setattr(
        sys, "argv", ["eval_retention.py", "--data-dir", str(data_dir), "--benchmarks", "bbh", "--out", str(out_path)]
    )
    eval_retention.main()
    assert seen == [2048]
    result = json.loads(out_path.read_text(encoding="utf-8"))
    assert result["max_tokens"] == 2048
    assert result["retention_settings"]["max_tokens"] == 2048


def _fixture_bbh_tasks(tmp_path: Path) -> tuple[Path, Path]:
    """BBH rows under two real config.BBH_TASKS names plus canned responses
    for all of them; returns (data_dir, canned_path)."""
    rows = [
        RetentionRow(key="bbh:boolean_expressions:0", benchmark="bbh", prompt="p", target="True", group="boolean_expressions"),
        RetentionRow(key="bbh:boolean_expressions:1", benchmark="bbh", prompt="p", target="False", group="boolean_expressions"),
        RetentionRow(key="bbh:web_of_lies:0", benchmark="bbh", prompt="p", target="Yes", group="web_of_lies"),
    ]
    _write_rows_jsonl(tmp_path / "retention_bbh.jsonl", rows)
    canned_path = tmp_path / "canned_bbh.jsonl"
    with canned_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({"key": row.key, "response": row.target}) + "\n")
    return tmp_path, canned_path


def test_retention_bbh_tasks_env_filters_rows_on_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir, canned_path = _fixture_bbh_tasks(tmp_path)
    out_path = tmp_path / "out.json"
    monkeypatch.setenv("RETENTION_BBH_TASKS", "web_of_lies")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_retention.py",
            "--data-dir",
            str(data_dir),
            "--benchmarks",
            "bbh",
            "--score-only",
            str(canned_path),
            "--out",
            str(out_path),
        ],
    )
    eval_retention.main()
    result = json.loads(out_path.read_text(encoding="utf-8"))
    assert result["n"] == {"bbh": 1}
    assert result["by_group"]["bbh"] == {"web_of_lies": 1.0}
    assert result["retention_settings"]["bbh_tasks"] == ["web_of_lies"]


def test_retention_bbh_tasks_filter_applies_before_limit(tmp_path: Path) -> None:
    data_dir, _ = _fixture_bbh_tasks(tmp_path)
    rows = eval_retention.load_retention_rows(data_dir, "bbh", limit=1, groups=["web_of_lies"])
    assert [r.key for r in rows] == ["bbh:web_of_lies:0"]


def test_retention_bbh_tasks_env_unknown_name_exits_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir, _ = _fixture_bbh_tasks(tmp_path)

    def _no_load(model: str, adapter: str | None) -> object:
        raise AssertionError("model load reached")

    monkeypatch.setattr(eval_retention.eval_external, "_load_hf_engine", _no_load)
    monkeypatch.setenv("RETENTION_BBH_TASKS", "web_of_lies,nope")
    monkeypatch.setattr(
        sys, "argv", ["eval_retention.py", "--data-dir", str(data_dir), "--benchmarks", "bbh", "--out", str(tmp_path / "o.json")]
    )
    with pytest.raises(SystemExit) as excinfo:
        eval_retention.main()
    message = str(excinfo.value)
    assert "nope" in message
    assert all(task in message for task in config.BBH_TASKS)


def test_retention_benchmarks_env_selects_subset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = _fixture_rows(tmp_path)
    canned_path = _fixture_canned(tmp_path)
    out_path = tmp_path / "out.json"
    monkeypatch.setenv("RETENTION_BENCHMARKS", "bbh")
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_retention.py", "--data-dir", str(data_dir), "--score-only", str(canned_path), "--out", str(out_path)],
    )
    eval_retention.main()
    result = json.loads(out_path.read_text(encoding="utf-8"))
    assert set(result["n"]) == {"bbh"}
    assert "gsm8k" not in result and "mmlu" not in result
    assert result["retention_settings"]["benchmarks"] == ["bbh"]


def test_eval_retention_default_summary_is_unchanged_plus_retention_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = _fixture_rows(tmp_path)
    canned_path = _fixture_canned(tmp_path)
    out_path = tmp_path / "out.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_retention.py", "--data-dir", str(data_dir), "--score-only", str(canned_path), "--out", str(out_path)],
    )
    eval_retention.main()
    result = json.loads(out_path.read_text(encoding="utf-8"))
    assert set(result) == {
        "gsm8k", "mmlu", "bbh", "n", "correct", "by_group", "model", "adapter", "max_tokens", "limit", "retention_settings",
    }
    assert result["max_tokens"] == 512
    assert result["retention_settings"] == {
        "suffix": DEFAULT_BBH_SUFFIX,
        "max_tokens": 512,
        "bbh_tasks": None,
        "benchmarks": ["gsm8k", "mmlu", "bbh"],
        "lenient": False,
    }


# --- RETENTION_LENIENT ---------------------------------------------------------


def test_retention_lenient_unset_is_false() -> None:
    assert retention_lenient() is False


def test_retention_lenient_env_one_is_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETENTION_LENIENT", "1")
    assert retention_lenient() is True


def test_retention_lenient_env_invalid_raises_systemexit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETENTION_LENIENT", "yes")
    with pytest.raises(SystemExit, match="RETENTION_LENIENT"):
        retention_lenient()


def test_eval_retention_lenient_env_invalid_exits_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = _fixture_rows(tmp_path)

    def _no_load(model: str, adapter: str | None) -> object:
        raise AssertionError("model load reached")

    monkeypatch.setattr(eval_retention.eval_external, "_load_hf_engine", _no_load)
    monkeypatch.setenv("RETENTION_LENIENT", "yes")
    monkeypatch.setattr(
        sys, "argv", ["eval_retention.py", "--data-dir", str(data_dir), "--out", str(tmp_path / "o.json")]
    )
    with pytest.raises(SystemExit, match="RETENTION_LENIENT"):
        eval_retention.main()


# score_bbh: strict unchanged with the knob unset (already covered by every score_bbh
# assertion above, run under the autouse _clear_retention_env fixture) -- these three
# cases are the lenient fallback's own behaviour, knob on.


def test_score_bbh_lenient_paren_letter_anywhere_in_response(monkeypatch: pytest.MonkeyPatch) -> None:
    # The committed "(C)" sits mid-response, not on the last line, so strict's
    # last-line containment check misses it.
    response = "Tracking the throws: the ball is with (C).\n\nWait, let me double check my count once more."
    assert score_bbh(response, "(C)") is False  # strict: last line has no "(c)"
    monkeypatch.setenv("RETENTION_LENIENT", "1")
    assert score_bbh(response, "(C)") is True


def test_score_bbh_lenient_yes_no_negative_phrase(monkeypatch: pytest.MonkeyPatch) -> None:
    response = "Given the chain of claims, Elanor does not tell the truth in this scenario."
    assert score_bbh(response, "No") is False  # strict: no marker, last line is not "no"
    monkeypatch.setenv("RETENTION_LENIENT", "1")
    assert score_bbh(response, "No") is True


def test_score_bbh_lenient_yes_no_positive_phrase(monkeypatch: pytest.MonkeyPatch) -> None:
    response = "Given the chain of claims, Elanor tells the truth in this scenario."
    assert score_bbh(response, "Yes") is False  # strict: no marker, last line is not "yes"
    monkeypatch.setenv("RETENTION_LENIENT", "1")
    assert score_bbh(response, "Yes") is True


def test_score_bbh_lenient_yes_no_picks_later_phrase(monkeypatch: pytest.MonkeyPatch) -> None:
    # "lies" (negative) precedes "is telling the truth" (positive) -- the later one wins.
    response = "Crista says Jim lies, and Crista is telling the truth."
    monkeypatch.setenv("RETENTION_LENIENT", "1")
    assert score_bbh(response, "Yes") is True
    assert score_bbh(response, "No") is False


def test_score_bbh_lenient_marker_still_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    # "(A)" is mentioned early, but a "Final Answer:" marker commits to
    # something else entirely -- a marker's presence must block the lenient
    # paren-anywhere fallback, not let it fall through to the "(A)" mention,
    # even though target "(A)" would otherwise match under RETENTION_LENIENT.
    response = "(A) is one possibility.\n\nAfter more thought:\n### Final Answer: unclear, leaning elsewhere"
    monkeypatch.setenv("RETENTION_LENIENT", "1")
    assert score_bbh(response, "(A)") is False


def test_eval_retention_lenient_env_recorded_in_retention_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = _fixture_rows(tmp_path)
    canned_path = _fixture_canned(tmp_path)
    out_path = tmp_path / "out.json"
    monkeypatch.setenv("RETENTION_LENIENT", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_retention.py", "--data-dir", str(data_dir), "--score-only", str(canned_path), "--out", str(out_path)],
    )
    eval_retention.main()
    result = json.loads(out_path.read_text(encoding="utf-8"))
    assert result["retention_settings"]["lenient"] is True


# --- EVAL_VLLM_GPU_MEMORY_UTILIZATION (retention's vllm backend generates via
# eval_external._generate_vllm -> _build_vllm_engine, local/retention.sh's
# header; same builder, tested directly here through eval_retention's own
# import of eval_external) --------------------------------------------------


def test_retention_vllm_engine_gpu_memory_utilization_unset_omits_kwarg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVAL_VLLM_GPU_MEMORY_UTILIZATION", raising=False)
    eval_retention.eval_external._build_vllm_engine.cache_clear()
    captured: list[dict[str, Any]] = []

    class FakeLLM:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = FakeLLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    eval_retention.eval_external._build_vllm_engine("model-a", None)

    assert captured == [{"model": "model-a", "enable_lora": False, "max_lora_rank": 64}]


def test_retention_vllm_engine_gpu_memory_utilization_set_reaches_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_VLLM_GPU_MEMORY_UTILIZATION", "0.35")
    eval_retention.eval_external._build_vllm_engine.cache_clear()
    captured: list[dict[str, Any]] = []

    class FakeLLM:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = FakeLLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    eval_retention.eval_external._build_vllm_engine("model-a", None)

    assert captured == [
        {"model": "model-a", "enable_lora": False, "max_lora_rank": 64, "gpu_memory_utilization": 0.35}
    ]


@pytest.mark.parametrize("raw", ["0", "1.5", "abc"])
def test_retention_vllm_engine_gpu_memory_utilization_invalid_raises_before_builder(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv("EVAL_VLLM_GPU_MEMORY_UTILIZATION", raw)
    eval_retention.eval_external._build_vllm_engine.cache_clear()

    def _unreachable(**kwargs: Any) -> Any:
        raise AssertionError("vllm.LLM must not be constructed for an invalid EVAL_VLLM_GPU_MEMORY_UTILIZATION")

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = _unreachable  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    with pytest.raises(SystemExit, match="EVAL_VLLM_GPU_MEMORY_UTILIZATION"):
        eval_retention.eval_external._build_vllm_engine("model-a", None)
