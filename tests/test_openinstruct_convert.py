"""openinstruct/convert_subset.py: schema, family-id identity mapping, --limit.

CPU only: a real Qwen3-4B tokenizer (cached, no GPU), same convention as test_prompt_cap.py.
Fixture rows are built in-memory (a tiny train.jsonl + .keys file under tmp_path), not the
real 81,897-row corpus, so this only re-verifies converter mechanics; the data review's own
scan of the real corpus (README.md) is the evidence that 0 real rows hit an unmapped family.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from openinstruct.convert_subset import KNOWN_FAMILIES, convert

FIXTURE_ROWS = [
    {
        "key": "row_one",
        "base_prompt": "Write a short poem about the sea.",
        "prompt": "Write a short poem about the sea. Do not use any commas in your response.",
        "sentences": ["Do not use any commas in your response."],
        "constraints": [{"iid": "punctuation:no_comma", "kwargs": {}}],
        "english": True,
        "num_constraints": 1,
    },
    {
        "key": "row_two",
        "base_prompt": "List three fruits.",
        "prompt": "List three fruits. Your entire response should be in English and in all lowercase letters.",
        "sentences": ["Your entire response should be in English and in all lowercase letters."],
        "constraints": [{"iid": "change_case:english_lowercase", "kwargs": {}}],
        "english": True,
        "num_constraints": 1,
    },
    {
        "key": "row_three",
        "base_prompt": "Describe your favourite season.",
        "prompt": "Describe your favourite season. At least 3 letters must appear at least twice each.",
        "sentences": ["At least 3 letters must appear at least twice each."],
        "constraints": [{"iid": "letters:letter_counting", "kwargs": {"N": 3, "relation": "at least"}}],
        "english": True,
        "num_constraints": 1,
    },
]


def _write_fixture(tmp_path: Path, rows: list[dict]) -> tuple[Path, Path]:
    train_jsonl = tmp_path / "train.jsonl"
    keys_file = tmp_path / "subset.keys"
    with train_jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    keys_file.write_text("\n".join(row["key"] for row in rows) + "\n")
    return train_jsonl, keys_file


def test_schema_columns(tmp_path: Path) -> None:
    train_jsonl, keys_file = _write_fixture(tmp_path, FIXTURE_ROWS)
    out_path = tmp_path / "out.jsonl"

    summary = convert(train_jsonl, keys_file, out_path, limit=None)

    assert summary["rows_written"] == 3
    assert summary["rows_with_unmapped_family"] == 0
    lines = out_path.read_text().splitlines()
    assert len(lines) == 3
    for line in lines:
        row = json.loads(line)
        assert set(row) == {"key", "messages", "ground_truth", "dataset"}
        assert row["dataset"] == "ifeval"
        assert row["messages"] == [{"role": "user", "content": row["messages"][0]["content"]}]
        assert row["messages"][0]["role"] == "user"
        # IFEvalVerifier parses `ground_truth` with ast.literal_eval, never json.loads.
        parsed = ast.literal_eval(row["ground_truth"])
        assert isinstance(parsed, list) and len(parsed) == 1
        assert set(parsed[0]) == {"instruction_id", "kwargs"}
        assert len(parsed[0]["instruction_id"]) == len(parsed[0]["kwargs"])


def test_family_id_identity_mapping(tmp_path: Path) -> None:
    train_jsonl, keys_file = _write_fixture(tmp_path, FIXTURE_ROWS)
    out_path = tmp_path / "out.jsonl"

    convert(train_jsonl, keys_file, out_path, limit=None)

    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    by_key = {row["key"]: row for row in rows}
    for fixture_row in FIXTURE_ROWS:
        parsed = ast.literal_eval(by_key[fixture_row["key"]]["ground_truth"])[0]
        expected_ids = [c["iid"] for c in fixture_row["constraints"]]
        expected_kwargs = [c["kwargs"] for c in fixture_row["constraints"]]
        # No rename table: constraints[i]["iid"] passes straight through as instruction_id[i].
        assert parsed["instruction_id"] == expected_ids
        assert parsed["kwargs"] == expected_kwargs
        for iid in expected_ids:
            assert iid in KNOWN_FAMILIES


def test_unmapped_family_is_reported(tmp_path: Path) -> None:
    rows = [*FIXTURE_ROWS, {
        "key": "row_bad",
        "base_prompt": "Do something unusual.",
        "prompt": "Do something unusual. Obey the made-up rule.",
        "sentences": ["Obey the made-up rule."],
        "constraints": [{"iid": "nonexistent:not_a_real_family", "kwargs": {}}],
        "english": True,
        "num_constraints": 1,
    }]
    train_jsonl, keys_file = _write_fixture(tmp_path, rows)
    out_path = tmp_path / "out.jsonl"

    summary = convert(train_jsonl, keys_file, out_path, limit=None)

    assert summary["rows_written"] == 4
    assert summary["rows_with_unmapped_family"] == 1
    assert summary["unmapped_constraint_instances"] == 1
    assert summary["unmapped_families"] == ["nonexistent:not_a_real_family"]


def test_limit_head_slices_before_writing(tmp_path: Path) -> None:
    train_jsonl, keys_file = _write_fixture(tmp_path, FIXTURE_ROWS)
    out_path = tmp_path / "out.jsonl"

    summary = convert(train_jsonl, keys_file, out_path, limit=2)

    assert summary["rows_written"] == 2
    lines = out_path.read_text().splitlines()
    assert len(lines) == 2
    written_keys = [json.loads(line)["key"] for line in lines]
    assert written_keys == [row["key"] for row in FIXTURE_ROWS[:2]]


def test_missing_key_raises(tmp_path: Path) -> None:
    train_jsonl, keys_file = _write_fixture(tmp_path, FIXTURE_ROWS)
    keys_file.write_text(keys_file.read_text() + "no_such_key\n")
    out_path = tmp_path / "out.jsonl"

    try:
        convert(train_jsonl, keys_file, out_path, limit=None)
    except ValueError as exc:
        assert "no_such_key" in str(exc)
    else:
        raise AssertionError("expected ValueError for a subset key missing from train_jsonl")
