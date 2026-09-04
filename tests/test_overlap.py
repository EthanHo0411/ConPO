"""Tests for conpo.overlap and scripts/check_overlap.py.

scripts/ is not an installed package (see tests/test_eval_scripts.py's
docstring), so check_overlap is imported the same way: adding scripts/ to
sys.path.
"""

import json
import sys
from pathlib import Path

import pytest

from conpo.overlap import check_overlap, ngrams, normalise

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import check_overlap as check_overlap_script  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_SHARED_13 = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike"
_SHARED_12 = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima"


# --- normalise ---------------------------------------------------------------


def test_normalise_ignores_case_and_punctuation() -> None:
    assert normalise("Hello,  World!!  Foo.") == "hello world foo"
    assert normalise("HELLO WORLD FOO") == "hello world foo"


# --- ngrams -------------------------------------------------------------------


def test_ngrams_thirteen_tokens_yields_one_gram() -> None:
    grams = ngrams(_SHARED_13, n=13)
    assert grams == {tuple(_SHARED_13.split(" "))}


def test_ngrams_below_n_tokens_yields_empty_set() -> None:
    assert ngrams(_SHARED_12, n=13) == set()


def test_ngrams_normalises_before_tokenising() -> None:
    assert ngrams("Alpha, Bravo!", n=2) == {("alpha", "bravo")}


# --- check_overlap: planted exact duplicate -----------------------------------


def test_check_overlap_detects_planted_exact_duplicate() -> None:
    text = "The quick brown fox jumps over the lazy dog."
    subset = {"k1": text}
    benchmark = ["some other prompt", text]

    report = check_overlap(subset, benchmark, n=13)

    assert report.exact_matches == 1
    assert report.ngram_hits == 0  # 9 tokens < 13, so ngrams() yields no spans at all
    assert report.hit_keys == ["k1"]  # hit via the exact-match check
    assert report.total_subset == 1
    assert report.total_benchmark == 2


def test_check_overlap_exact_match_ignores_case_and_punctuation() -> None:
    subset = {"k1": "THE QUICK, BROWN fox!"}
    benchmark = ["the quick brown fox"]

    report = check_overlap(subset, benchmark, n=13)

    assert report.exact_matches == 1
    assert report.ngram_hits == 0  # 4 tokens < 13


# --- check_overlap: planted 13-gram overlap, near-miss not detected ----------


def test_check_overlap_detects_planted_13gram_not_full_duplicate() -> None:
    subset_text = f"prefix words here {_SHARED_13} suffix words too"
    benchmark_text = f"totally different opening {_SHARED_13} and closing bits"
    subset = {"k1": subset_text}

    report = check_overlap(subset, [benchmark_text], n=13)

    assert report.ngram_hits == 1
    assert report.exact_matches == 0  # full texts differ
    assert report.hit_keys == ["k1"]


def test_check_overlap_twelve_gram_near_miss_not_detected() -> None:
    subset_text = f"prefix words here {_SHARED_12} suffix words too"
    benchmark_text = f"totally different opening {_SHARED_12} and closing bits"
    subset = {"k1": subset_text}

    report = check_overlap(subset, [benchmark_text], n=13)

    assert report.ngram_hits == 0
    assert report.exact_matches == 0
    assert report.hit_keys == []


def test_check_overlap_no_overlap_at_all() -> None:
    report = check_overlap({"k1": "completely unrelated text here"}, ["something else entirely"], n=13)
    assert report.exact_matches == 0
    assert report.ngram_hits == 0
    assert report.hit_keys == []


# --- check_overlap: hit_keys order follows subset_texts iteration order ------


def test_check_overlap_hit_keys_order_follows_subset_iteration() -> None:
    text = "shared duplicate prompt text"
    subset = {"z_key": text, "a_key": text, "m_key": "no overlap at all here"}

    report = check_overlap(subset, [text], n=13)

    assert report.hit_keys == ["z_key", "a_key"]


# --- scripts/check_overlap.py: subset-key loading -----------------------------


def test_load_subset_keys_skips_blank_and_comment_lines(tmp_path: Path) -> None:
    path = tmp_path / "keys.txt"
    path.write_text("key1\n\n# a comment\nkey2\n  key3  \n", encoding="utf-8")

    assert check_overlap_script.load_subset_keys(path) == {"key1", "key2", "key3"}


def test_load_subset_base_prompts_filters_by_subset_and_limit(tmp_path: Path) -> None:
    rows_path = tmp_path / "rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as f:
        for i in range(5):
            f.write(json.dumps({"key": f"k{i}", "base_prompt": f"prompt {i}"}) + "\n")

    subset_path = tmp_path / "keys.txt"
    subset_path.write_text("k1\nk3\n", encoding="utf-8")

    all_texts = check_overlap_script.load_subset_base_prompts(rows_path, None, None)
    assert all_texts == {f"k{i}": f"prompt {i}" for i in range(5)}

    filtered = check_overlap_script.load_subset_base_prompts(rows_path, subset_path, None)
    assert filtered == {"k1": "prompt 1", "k3": "prompt 3"}

    limited = check_overlap_script.load_subset_base_prompts(rows_path, None, limit=2)
    assert limited == {"k0": "prompt 0", "k1": "prompt 1"}


# --- scripts/check_overlap.py: --internal-splits -----------------------------


def test_load_eval_split_keys_reads_key_field(tmp_path: Path) -> None:
    path = tmp_path / "eval_i.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"key": "k0", "base_prompt": "p0"}) + "\n")
        f.write(json.dumps({"key": "k1", "base_prompt": "p1"}) + "\n")

    assert check_overlap_script.load_eval_split_keys(path) == {"k0", "k1"}


def test_check_internal_splits_reports_zero_overlap_when_disjoint() -> None:
    report = check_overlap_script.check_internal_splits(
        subset_keys={"s0", "s1", "s2"},
        eval_i_keys={"i0", "i1"},
        eval_ii_keys={"ii0"},
        eval_iii_keys={"iii0", "iii1"},
    )
    assert report == {
        "subset": 3,
        "eval_i": 2,
        "eval_ii": 1,
        "eval_iii": 2,
        "overlap_eval_i": 0,
        "overlap_eval_ii": 0,
        "overlap_eval_iii": 0,
    }


def test_check_internal_splits_detects_injected_overlap() -> None:
    report = check_overlap_script.check_internal_splits(
        subset_keys={"s0", "s1", "shared"},
        eval_i_keys={"i0", "shared"},  # injected overlap
        eval_ii_keys={"ii0"},
        eval_iii_keys={"iii0"},
    )
    assert report["overlap_eval_i"] == 1
    assert report["overlap_eval_ii"] == 0
    assert report["overlap_eval_iii"] == 0


def _write_eval_split(path: Path, keys: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for key in keys:
            f.write(json.dumps({"key": key, "base_prompt": f"prompt for {key}"}) + "\n")


def test_check_overlap_script_internal_splits_exits_zero_when_disjoint(tmp_path: Path, monkeypatch) -> None:
    subset_path = tmp_path / "keys.txt"
    subset_path.write_text("s0\ns1\n", encoding="utf-8")
    eval_i = tmp_path / "eval_i.jsonl"
    eval_ii = tmp_path / "eval_ii.jsonl"
    eval_iii = tmp_path / "eval_iii.jsonl"
    _write_eval_split(eval_i, ["i0"])
    _write_eval_split(eval_ii, ["ii0"])
    _write_eval_split(eval_iii, ["iii0"])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_overlap.py", "--internal-splits",
            "--subset-keys", str(subset_path),
            "--eval-i", str(eval_i), "--eval-ii", str(eval_ii), "--eval-iii", str(eval_iii),
        ],
    )
    check_overlap_script.main()  # no SystemExit raised


def test_check_overlap_script_internal_splits_exits_one_when_overlapping(tmp_path: Path, monkeypatch) -> None:
    subset_path = tmp_path / "keys.txt"
    subset_path.write_text("s0\nshared\n", encoding="utf-8")
    eval_i = tmp_path / "eval_i.jsonl"
    eval_ii = tmp_path / "eval_ii.jsonl"
    eval_iii = tmp_path / "eval_iii.jsonl"
    _write_eval_split(eval_i, ["shared"])  # injected overlap with the subset
    _write_eval_split(eval_ii, ["ii0"])
    _write_eval_split(eval_iii, ["iii0"])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_overlap.py", "--internal-splits",
            "--subset-keys", str(subset_path),
            "--eval-i", str(eval_i), "--eval-ii", str(eval_ii), "--eval-iii", str(eval_iii),
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        check_overlap_script.main()
    assert excinfo.value.code == 1


# --- scripts/check_overlap.py: full run on the real sample --------------------


def test_check_overlap_script_runs_on_data_train_sample(tmp_path: Path, monkeypatch, capsys) -> None:
    """--limit 2000 sample against the real google/IFEval (HF cache) and the
    vendored IFBench_test; asserts it runs and produces a well-shaped report."""
    out_path = tmp_path / "overlap_report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_overlap.py",
            "--rows",
            str(DATA_DIR / "train.jsonl"),
            "--limit",
            "2000",
            "--out",
            str(out_path),
        ],
    )
    check_overlap_script.main()
    captured = capsys.readouterr()
    print(captured.out)  # surfaced in the receipt via -s / -q output capture

    result = json.loads(out_path.read_text(encoding="utf-8"))
    assert result["total_subset"] == 2000
    for benchmark in ("ifeval", "ifbench_test"):
        report = result[benchmark]
        assert 0 <= report["exact_matches"] <= 2000
        assert 0 <= report["ngram_hits"] <= 2000
        assert len(report["hit_keys"]) <= 10
        assert report["total_subset"] == 2000
        assert report["total_benchmark"] > 0
