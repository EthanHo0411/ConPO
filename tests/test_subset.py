"""Tests for scripts/draw_subset.py, conpo.data.load_subset, and
scripts/train.py's Pass1CheckpointCallback (fixed shared training subset +
first-pass checkpoint, locked decision 2026-09-02).

scripts/ is not an installed package, so draw_subset and train are imported
by adding scripts/ to sys.path -- the same pattern tests/test_eval_scripts.py
already uses for eval_external/eval_internal.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from trl import GRPOConfig

from conpo import config
from conpo.arms import resolve_arm
from conpo.data import PoolRow, TokenMeter, keys_digest, load_subset
from conpo.edges import EdgeBuffer, FeasibleBuffer
from conpo.engine import HFEngine
from conpo.judge import StaticJudge
from conpo.selection import FamilyStats
from conpo.trainer import ConPOTrainer
from conpo.verifiers import Constraint

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import draw_subset  # noqa: E402
import train as train_script  # noqa: E402

# ---------------------------------------------------------------------------
# scripts/draw_subset.py
# ---------------------------------------------------------------------------


def _write_synthetic_train(path: Path, per_count: dict[int, int]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for count, size in per_count.items():
            for i in range(size):
                f.write(json.dumps({"key": f"row-{count}-{i}", "num_constraints": count}) + "\n")


def test_draw_subset_exact_stratified_sizes(tmp_path: Path) -> None:
    per_count = {1: 100, 2: 150, 3: 120, 4: 80, 5: 50}  # 500 rows total
    source = tmp_path / "train.jsonl"
    _write_synthetic_train(source, per_count)

    keys, sizes, total, duplicate_keys = draw_subset.draw_subset(source, size=200, seed=42)

    assert total == 500
    assert duplicate_keys == 0
    assert len(keys) == 200
    assert len(set(keys)) == 200  # no duplicates
    assert sum(sizes.values()) == 200
    # 200 / 500 = 0.4 exactly, so every stratum's share is exact -- no
    # largest-remainder rounding kicks in for this histogram.
    assert sizes == {1: 40, 2: 60, 3: 48, 4: 32, 5: 20}


def test_draw_subset_largest_remainder_rounding_sums_exactly(tmp_path: Path) -> None:
    per_count = {1: 7, 2: 3}  # 10 rows total, target 6 -> non-exact shares
    source = tmp_path / "train.jsonl"
    _write_synthetic_train(source, per_count)

    _, sizes, total, duplicate_keys = draw_subset.draw_subset(source, size=6, seed=0)

    assert total == 10
    assert duplicate_keys == 0
    assert sum(sizes.values()) == 6
    # exact shares: 1 -> 4.2, 2 -> 1.8; floors 4+1=5, leftover 1 goes to the
    # larger fractional remainder (2's 0.8 > 1's 0.2).
    assert sizes == {1: 4, 2: 2}


def test_draw_subset_deterministic_for_seed_different_for_another(tmp_path: Path) -> None:
    per_count = {1: 100, 2: 150, 3: 120, 4: 80, 5: 50}
    source = tmp_path / "train.jsonl"
    _write_synthetic_train(source, per_count)

    keys_a, _, _, _ = draw_subset.draw_subset(source, size=200, seed=42)
    keys_a_again, _, _, _ = draw_subset.draw_subset(source, size=200, seed=42)
    keys_b, _, _, _ = draw_subset.draw_subset(source, size=200, seed=7)

    assert keys_a == keys_a_again
    assert keys_a != keys_b


# ---------------------------------------------------------------------------
# scripts/draw_subset.py: duplicate-key dedup (the review fix -- pool keys
# duplicated upstream, e.g. hard_coded_5 x3, were leaking into the drawn key
# list; see conpo/data.py's load_pool for the source-side dedup)
# ---------------------------------------------------------------------------


def test_num_constraints_histogram_dedupes_by_key_keep_first(tmp_path: Path) -> None:
    source = tmp_path / "train.jsonl"
    # "dup" appears 3x; the first occurrence's num_constraints (2) must be
    # the one kept, and the later repeats (num_constraints=4) must not
    # double-count into the histogram.
    rows = [
        {"key": "dup", "num_constraints": 2},
        {"key": "a", "num_constraints": 2},
        {"key": "dup", "num_constraints": 4},
        {"key": "dup", "num_constraints": 4},
        {"key": "b", "num_constraints": 3},
    ]
    source.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    by_count, total, duplicate_keys = draw_subset._num_constraints_histogram(source)

    assert total == 5
    assert duplicate_keys == 2
    assert by_count == {2: ["dup", "a"], 3: ["b"]}


def test_draw_subset_drops_duplicate_keys_before_sampling(tmp_path: Path) -> None:
    source = tmp_path / "train.jsonl"
    rows = [{"key": "dup", "num_constraints": 1}] * 3 + [
        {"key": f"row-{i}", "num_constraints": 1} for i in range(9)
    ]
    source.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    keys, sizes, total, duplicate_keys = draw_subset.draw_subset(source, size=5, seed=0)

    assert total == 12
    assert duplicate_keys == 2  # "dup" kept once, its 2 repeats skipped
    assert sum(sizes.values()) == 5
    assert len(keys) == 5
    assert len(set(keys)) == 5  # no duplicates leaked into the drawn keys
    assert keys.count("dup") <= 1


def test_draw_subset_cli_writes_keys_and_manifest(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "train.jsonl"
    _write_synthetic_train(source, {1: 20, 2: 30})
    out_path = tmp_path / "subsets" / "train_25_seed3.keys"

    monkeypatch.setattr(
        sys,
        "argv",
        ["draw_subset.py", "--size", "25", "--seed", "3", "--source", str(source), "--out", str(out_path)],
    )
    draw_subset.main()

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 25
    assert len(set(lines)) == 25

    manifest = json.loads((tmp_path / "subsets" / "train_25_seed3.manifest.json").read_text(encoding="utf-8"))
    assert manifest["size"] == 25
    assert manifest["seed"] == 3
    assert manifest["duplicate_keys"] == 0
    assert sum(manifest["per_count_sizes"].values()) == 25
    assert manifest["prompts_per_update"] == draw_subset.PROMPTS_PER_UPDATE
    assert manifest["dropped_trailing_prompts"] == 25 % draw_subset.PROMPTS_PER_UPDATE
    assert manifest["keys_sha256"] == keys_digest(out_path)
    assert manifest["source_manifest_split_sizes"] is None  # no data/manifest.json next to this synthetic source


# ---------------------------------------------------------------------------
# scripts/draw_subset.py: exact --size, no rounding (the review fix -- the
# module used to round --size up to a multiple of PROMPTS_PER_UPDATE and
# called it a "locked decision" that was never made; 15000 was silently
# rounded to 15008)
# ---------------------------------------------------------------------------


def test_draw_subset_cli_draws_exact_size_no_rounding(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "train.jsonl"
    _write_synthetic_train(source, {1: 40, 2: 60})  # 100 rows, plenty for size=25 stratified
    out_dir = tmp_path / "subsets"

    monkeypatch.setattr(sys, "argv", ["draw_subset.py", "--size", "25", "--seed", "3", "--source", str(source)])
    monkeypatch.setattr(draw_subset, "REPO_ROOT", tmp_path)
    draw_subset.main()

    # 25 is not a multiple of PROMPTS_PER_UPDATE (16): the old default
    # rounded this up to 32 and named the file accordingly. The fix draws
    # and names the file by the literal --size.
    out_path = out_dir / "train_25_seed3.keys"
    assert out_path.exists()
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 25
    assert len(set(lines)) == 25

    manifest = json.loads((out_dir / "train_25_seed3.manifest.json").read_text(encoding="utf-8"))
    assert manifest["size"] == 25
    assert "requested_size" not in manifest  # retired along with the rounding it explained
    assert "round_to" not in manifest
    assert manifest["dropped_trailing_prompts"] == 25 % draw_subset.PROMPTS_PER_UPDATE


# ---------------------------------------------------------------------------
# conpo/data.py: load_subset
# ---------------------------------------------------------------------------

_ROW_A = PoolRow(
    key="a", base_prompt="A", prompt="A", sentences=(), constraints=(), english=True, num_constraints=0
)
_ROW_B = PoolRow(
    key="b", base_prompt="B", prompt="B", sentences=(), constraints=(), english=True, num_constraints=0
)
_ROW_C = PoolRow(
    key="c", base_prompt="C", prompt="C", sentences=(), constraints=(), english=True, num_constraints=0
)


def test_load_subset_returns_rows_in_keys_file_order(tmp_path: Path) -> None:
    keys_path = tmp_path / "subset.keys"
    keys_path.write_text("c\na\n", encoding="utf-8")

    result = load_subset([_ROW_A, _ROW_B, _ROW_C], keys_path)

    assert [r.key for r in result] == ["c", "a"]


def test_load_subset_raises_on_missing_key(tmp_path: Path) -> None:
    keys_path = tmp_path / "subset.keys"
    keys_path.write_text("a\nzzz\n", encoding="utf-8")

    try:
        load_subset([_ROW_A, _ROW_B], keys_path)
        raise AssertionError("expected KeyError")
    except KeyError as exc:
        assert "zzz" in str(exc)


# ---------------------------------------------------------------------------
# conpo/data.py: keys_digest (line-ending independence -- Windows write_text
# emits CRLF, git checks the same file out as LF on the Linux cluster)
# ---------------------------------------------------------------------------


def test_keys_digest_matches_across_line_endings(tmp_path: Path) -> None:
    keys = ["a", "b", "c"]
    lf_path = tmp_path / "lf.keys"
    crlf_path = tmp_path / "crlf.keys"
    lf_path.write_bytes(("\n".join(keys) + "\n").encode("utf-8"))
    crlf_path.write_bytes(("\r\n".join(keys) + "\r\n").encode("utf-8"))

    assert keys_digest(lf_path) == keys_digest(crlf_path)


# ---------------------------------------------------------------------------
# scripts/train.py: _report_subset_prompts_per_update
#
# draw_subset.py no longer rounds --size up to a multiple of
# prompts_per_update (see the exact-size tests above), so a RUN_SUBSET whose
# size doesn't divide evenly -- e.g. the committed 15000-key subset at
# prompts_per_update=16 -- is now the expected, negligible case (TRL's
# RepeatSampler drops the trailing remainder every epoch) rather than a sign
# of a hand-edited or corrupted subset; this must not raise.
# ---------------------------------------------------------------------------


def test_report_subset_prompts_per_update_silent_when_exact(capsys: pytest.CaptureFixture[str]) -> None:
    train_script._report_subset_prompts_per_update(32, 16)  # no raise
    assert capsys.readouterr().out == ""


def test_report_subset_prompts_per_update_prints_note_when_not_exact(capsys: pytest.CaptureFixture[str]) -> None:
    train_script._report_subset_prompts_per_update(15000, 16)  # no raise
    assert "15000" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# scripts/train.py: Pass1CheckpointCallback
# ---------------------------------------------------------------------------

_PASS1_NO_COMMA = Constraint("punctuation:no_comma", {})


def _pass1_row(key: str, base: str) -> PoolRow:
    sentence = "Do not use any commas."
    return PoolRow(
        key=key,
        base_prompt=base,
        prompt=f"{base} {sentence}",
        sentences=(sentence,),
        constraints=(_PASS1_NO_COMMA,),
        english=True,
        num_constraints=1,
    )


_PASS1_ROWS = [_pass1_row(f"pass1-row-{i}", f"Explain topic {i}.") for i in range(4)]


def _pass1_model_and_tokenizer() -> tuple[Qwen3ForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(config.POLICY_MODEL)
    cfg = Qwen3Config(
        vocab_size=151936,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=512,
    )
    return Qwen3ForCausalLM(cfg), tokenizer


def test_pass1_callback_fires_once_at_pass_boundary(tmp_path: Path) -> None:
    model, tokenizer = _pass1_model_and_tokenizer()
    rows_by_key = {r.key: r for r in _PASS1_ROWS}
    dataset = Dataset.from_list(
        [{"prompt": [{"role": "user", "content": r.prompt}], "key": r.key} for r in _PASS1_ROWS]
    )

    grpo_config = GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=16,
        # 4 rows, 1 prompt/optimizer-step (see prompts_per_update below) ->
        # pass1_step == 4; max_steps == 5 runs slightly past that boundary.
        max_steps=5,
        use_vllm=False,
        beta=config.KL_BETA,
        loss_type="grpo",
        learning_rate=1e-3,
        report_to=[],
        logging_steps=1,
        gradient_accumulation_steps=1,
        chat_template_kwargs=config.POLICY_CHAT_TEMPLATE_KWARGS,
        seed=0,
        use_cpu=True,
        bf16=False,
        fp16=False,
        shuffle_dataset=False,
    )
    arm, overrides = resolve_arm("ifrlvr")
    callback = train_script.Pass1CheckpointCallback(dataset_size=len(_PASS1_ROWS), prompts_per_update=1)
    assert callback.pass1_step == 4

    trainer = ConPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        arm=arm,
        rows_by_key=rows_by_key,
        edge_buffer=EdgeBuffer(),
        feasible_buffer=FeasibleBuffer(),
        stats=FamilyStats(),
        judge=StaticJudge(keep=True),
        engine_factory=lambda t: HFEngine(t.accelerator.unwrap_model(t.model), t.processing_class, device="cpu"),
        meter=TokenMeter(budget=0),
        mechanism_log_path=tmp_path / "mechanism.jsonl",
        seed=0,
        overrides=overrides,
        peft_config=LoraConfig(
            r=4, lora_alpha=8, lora_dropout=0.0, target_modules=list(config.LORA_TARGET_MODULES), task_type="CAUSAL_LM"
        ),
        callbacks=[callback],
    )
    callback.trainer = trainer

    trainer.train()

    assert trainer.state.global_step == 5
    assert callback.fired is True

    checkpoint_dir = tmp_path / "checkpoint_pass1"
    assert checkpoint_dir.exists()
    pass1_meta = json.loads((tmp_path / "pass1.json").read_text(encoding="utf-8"))
    assert pass1_meta["global_step"] == 4  # fired at the pass boundary, not at step 5
    assert pass1_meta["tokens_so_far"] >= 0
    assert pass1_meta["wall_time_seconds"] >= 0


# ---------------------------------------------------------------------------
# scripts/train.py: _load_training_rows (the review fix -- --tiny used to
# truncate to the pool file's first 4 rows BEFORE a RUN_SUBSET was applied,
# so any real subset whose keys weren't among those first 4 rows crashed
# load_subset with KeyError)
# ---------------------------------------------------------------------------


def _write_pool_jsonl(path: Path, rows: list[PoolRow]) -> None:
    """Inverse of scripts/train.py's `_row_from_json` -- same shape as
    scripts/build_corpus.py's `_row_to_json`."""
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            payload = {
                "key": row.key,
                "base_prompt": row.base_prompt,
                "prompt": row.prompt,
                "sentences": list(row.sentences),
                "constraints": [{"iid": c.iid, "kwargs": dict(c.kwargs)} for c in row.constraints],
                "english": row.english,
                "num_constraints": row.num_constraints,
            }
            f.write(json.dumps(payload) + "\n")


def test_load_training_rows_tiny_with_subset_truncates_after_subset_order(tmp_path: Path) -> None:
    pool_rows = [_pass1_row(f"row-{i}", f"Explain topic {i}.") for i in range(30)]
    pool_path = tmp_path / "train.jsonl"
    _write_pool_jsonl(pool_path, pool_rows)

    # 20 keys, deliberately excluding the pool's first 4 rows and in a
    # shuffled (descending) order, so truncating the pool file to 4 rows
    # before applying the subset would miss every subset key.
    subset_keys = [f"row-{i}" for i in range(29, 9, -1)]
    assert len(subset_keys) == 20
    subset_path = tmp_path / "subset.keys"
    subset_path.write_text("\n".join(subset_keys) + "\n", encoding="utf-8")

    rows = train_script._load_training_rows(pool_path, str(subset_path), tiny=True)

    assert [r.key for r in rows] == subset_keys[:4]


def test_load_training_rows_tiny_with_real_15000_subset_does_not_raise() -> None:
    """RUN_SUBSET pointing at the committed 15000-key subset with --tiny must
    not KeyError (regression for the truncate-before-subset bug)."""
    subset_path = REPO_ROOT / "subsets" / "train_15000_seed42.keys"

    rows = train_script._load_training_rows(train_script.DATA_DIR / "train.jsonl", str(subset_path), tiny=True)

    assert len(rows) == 4
    subset_order = subset_path.read_text(encoding="utf-8").splitlines()
    assert [r.key for r in rows] == subset_order[:4]


def test_load_training_rows_tiny_without_subset_unchanged(tmp_path: Path) -> None:
    pool_rows = [_pass1_row(f"row-{i}", f"Explain topic {i}.") for i in range(10)]
    pool_path = tmp_path / "train.jsonl"
    _write_pool_jsonl(pool_path, pool_rows)

    rows = train_script._load_training_rows(pool_path, "", tiny=True)

    assert [r.key for r in rows] == [f"row-{i}" for i in range(4)]


# ---------------------------------------------------------------------------
# scripts/train.py: _vllm_sleep_mode (review fix -- vllm_enable_sleep_mode
# must not turn on for RUN_JUDGE_MODE=http, where the judge never contends
# with the colocated engine for local GPU memory)
# ---------------------------------------------------------------------------


def test_vllm_sleep_mode_off_without_vllm() -> None:
    assert train_script._vllm_sleep_mode(use_vllm=False, judge_mode="batched") is False
    assert train_script._vllm_sleep_mode(use_vllm=False, judge_mode="http") is False


def test_vllm_sleep_mode_on_only_for_batched_judge() -> None:
    assert train_script._vllm_sleep_mode(use_vllm=True, judge_mode="batched") is True
    assert train_script._vllm_sleep_mode(use_vllm=True, judge_mode="http") is False


# ---------------------------------------------------------------------------
# scripts/train.py: main() --tiny writes RUN_JUDGE_MODE/RUN_JUDGE_MODEL/
# RUN_VETO_PERIOD/RUN_LIMIT into run_metadata.json (review4/round2 finding 3).
# CPU, tiny random model, RUN_NO_VETO=1 so no judge model is ever loaded.
# ---------------------------------------------------------------------------


def _write_synthetic_pool_rows(path: Path, n: int) -> None:
    sentence = "Do not use any commas."
    constraint = {"iid": "punctuation:no_comma", "kwargs": {}}
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            base = f"Explain topic number {i}."
            row = {
                "key": f"row-{i}",
                "base_prompt": base,
                "prompt": f"{base} {sentence}",
                "sentences": [sentence],
                "constraints": [constraint],
                "english": True,
                "num_constraints": 1,
            }
            f.write(json.dumps(row) + "\n")


def test_main_tiny_writes_judge_and_veto_run_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_synthetic_pool_rows(data_dir / "train.jsonl", n=4)
    monkeypatch.setattr(train_script, "DATA_DIR", data_dir)

    out_dir = tmp_path / "out"
    monkeypatch.setenv("RUN_ARM", "solve_only")
    monkeypatch.setenv("RUN_STEPS", "1")
    monkeypatch.setenv("RUN_OUT_DIR", str(out_dir))
    # RUN_NO_VETO=1 wins over RUN_JUDGE_MODE/RUN_JUDGE_MODEL in build_judge
    # (StaticJudge, no model load), but scripts/train.py still records what
    # was asked for -- exactly the gap this test closes.
    monkeypatch.setenv("RUN_NO_VETO", "1")
    monkeypatch.setenv("RUN_JUDGE_MODE", "http")
    monkeypatch.setenv("RUN_JUDGE_MODEL", "some/test-judge-model")
    monkeypatch.setenv("RUN_VETO_PERIOD", "7")
    monkeypatch.setenv("RUN_LIMIT", "2")
    monkeypatch.setattr(sys, "argv", ["train.py", "--tiny"])

    train_script.main()

    metadata = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["judge_mode"] == "http"
    assert metadata["judge_model"] == "some/test-judge-model"
    assert metadata["veto_period_k"] == 7
    assert metadata["row_limit"] == 2
    assert metadata["dataset_size"] == 2  # RUN_LIMIT applied after --tiny's 4-row cap


def test_main_tiny_row_limit_zero_writes_null_row_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_synthetic_pool_rows(data_dir / "train.jsonl", n=4)
    monkeypatch.setattr(train_script, "DATA_DIR", data_dir)

    out_dir = tmp_path / "out"
    monkeypatch.setenv("RUN_ARM", "solve_only")
    monkeypatch.setenv("RUN_STEPS", "1")
    monkeypatch.setenv("RUN_OUT_DIR", str(out_dir))
    monkeypatch.setenv("RUN_NO_VETO", "1")
    monkeypatch.setattr(sys, "argv", ["train.py", "--tiny"])

    train_script.main()

    metadata = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["row_limit"] is None  # RUN_LIMIT unset (0 = no cap), not written as 0
    assert metadata["judge_mode"] == "batched"  # config.JUDGE_MODE_DEFAULT
    assert metadata["veto_period_k"] == config.VETO_PERIOD_K
