"""Tests for conpo.retention (scorers, row builders) and scripts/eval_retention.py
(--score-only path, --limit, --benchmarks). Mirrors tests/test_eval_scripts.py's
approach for importing a scripts/ CLI as a module."""

import dataclasses
import json
import sys
from pathlib import Path

import pytest

from conpo.retention import (
    RetentionRow,
    extract_bbh_prediction,
    extract_gsm8k_prediction,
    extract_mmlu_prediction,
    extract_prediction,
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
        "### Final Answer:\n\n**No, Shalonda does not tell the truth.**  \n**Shalonda lies.**\n\n**Answer: No**",
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
    assert score_bbh("syndrome  \ntherefrom", "syndrome therefrom") is True


def test_extract_bbh_prediction_word_sorting_comma_separated_list() -> None:
    response = (
        "barn, delmarva, dot, embezzle, greene, it&t, panicking, prod, scorch, same, "
        "splutter, subsist, thrill, drumhead, onus, obstetric, malton, guru, damp, entirety"
    )
    gold = "barn damp delmarva dot drumhead embezzle entirety greene guru it&t malton obstetric onus panicking prod same scorch splutter subsist thrill"
    # Comma-list is accepted and space-joined (point 4) once `target` shows a
    # multi-word gold is expected...
    assert extract_bbh_prediction(response, gold) == response.replace(", ", " ")
    # ...but the model's order does not match the sorted gold, so it still scores wrong.
    assert score_bbh(response, gold) is False


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
    response = "Third-to-last is **5th place**.\n\n### Final Answer:\n**(B) Eve finished third-to-last**."
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
