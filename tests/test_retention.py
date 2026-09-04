"""Tests for conpo.retention (scorers, row builders) and scripts/eval_retention.py
(--score-only path, --limit, --benchmarks). Mirrors tests/test_eval_scripts.py's
approach for importing a scripts/ CLI as a module."""

import dataclasses
import json
import sys
from pathlib import Path

import pytest

from conpo.retention import RetentionRow, rows_from_bbh, rows_from_gsm8k, rows_from_mmlu, score_bbh, score_gsm8k, score_mmlu

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
