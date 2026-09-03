"""Tests for scripts/draw_subset.py, conpo.data.load_subset, and
scripts/train.py's Pass1CheckpointCallback (fixed shared training subset +
first-pass checkpoint, locked decision 2026-09-02).

scripts/ is not an installed package, so draw_subset and train are imported
by adding scripts/ to sys.path -- the same pattern tests/test_eval_scripts.py
already uses for eval_external/eval_internal.
"""

from __future__ import annotations

import hashlib
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
from conpo.data import PoolRow, TokenMeter, load_subset
from conpo.edges import EdgeBuffer, FeasibleBuffer
from conpo.engine import HFEngine
from conpo.judge import StaticJudge
from conpo.selection import FamilyStats
from conpo.trainer import ConPOTrainer
from conpo.verifiers import Constraint

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
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

    keys, sizes, total = draw_subset.draw_subset(source, size=200, seed=42)

    assert total == 500
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

    _, sizes, total = draw_subset.draw_subset(source, size=6, seed=0)

    assert total == 10
    assert sum(sizes.values()) == 6
    # exact shares: 1 -> 4.2, 2 -> 1.8; floors 4+1=5, leftover 1 goes to the
    # larger fractional remainder (2's 0.8 > 1's 0.2).
    assert sizes == {1: 4, 2: 2}


def test_draw_subset_deterministic_for_seed_different_for_another(tmp_path: Path) -> None:
    per_count = {1: 100, 2: 150, 3: 120, 4: 80, 5: 50}
    source = tmp_path / "train.jsonl"
    _write_synthetic_train(source, per_count)

    keys_a, _, _ = draw_subset.draw_subset(source, size=200, seed=42)
    keys_a_again, _, _ = draw_subset.draw_subset(source, size=200, seed=42)
    keys_b, _, _ = draw_subset.draw_subset(source, size=200, seed=7)

    assert keys_a == keys_a_again
    assert keys_a != keys_b


def test_draw_subset_cli_writes_keys_and_manifest(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "train.jsonl"
    _write_synthetic_train(source, {1: 20, 2: 30})
    out_path = tmp_path / "subsets" / "train_25_seed3.keys"

    # --round-to 1 opts out of the divisibility rounding (tested separately
    # below) so this test can assert the plumbing against the literal --size.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "draw_subset.py",
            "--size",
            "25",
            "--seed",
            "3",
            "--source",
            str(source),
            "--out",
            str(out_path),
            "--round-to",
            "1",
        ],
    )
    draw_subset.main()

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 25
    assert len(set(lines)) == 25

    manifest = json.loads((tmp_path / "subsets" / "train_25_seed3.manifest.json").read_text(encoding="utf-8"))
    assert manifest["size"] == 25
    assert manifest["requested_size"] == 25
    assert manifest["round_to"] == 1
    assert manifest["seed"] == 3
    assert sum(manifest["per_count_sizes"].values()) == 25
    assert manifest["keys_sha256"] == hashlib.sha256(out_path.read_bytes()).hexdigest()
    assert manifest["source_manifest_split_sizes"] is None  # no data/manifest.json next to this synthetic source


# ---------------------------------------------------------------------------
# scripts/draw_subset.py: --round-to (RepeatSampler divisibility, the review
# fix -- 15000 % PROMPTS_PER_UPDATE != 0 was silently unreachable, see
# scripts/train.py's _assert_subset_divides_prompts_per_update)
# ---------------------------------------------------------------------------


def test_round_to_multiple_rounds_ties_up_and_floors_at_multiple() -> None:
    assert draw_subset._round_to_multiple(15000, 16) == 15008  # 937.5 -> 938
    assert draw_subset._round_to_multiple(20000, 16) == 20000  # already exact
    assert draw_subset._round_to_multiple(100, 16) == 96  # 6.25 -> 6, no tie
    assert draw_subset._round_to_multiple(3, 16) == 16  # floored, never 0
    assert draw_subset._round_to_multiple(25, 1) == 25  # disabled


def test_draw_subset_cli_default_round_to_names_file_by_rounded_size(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "train.jsonl"
    _write_synthetic_train(source, {1: 40, 2: 60})  # 100 rows, plenty for size=25 stratified
    out_dir = tmp_path / "subsets"

    monkeypatch.setattr(sys, "argv", ["draw_subset.py", "--size", "25", "--seed", "3", "--source", str(source)])
    monkeypatch.setattr(draw_subset, "REPO_ROOT", tmp_path)
    draw_subset.main()

    rounded = draw_subset._round_to_multiple(25, draw_subset.PROMPTS_PER_UPDATE)
    out_path = out_dir / f"train_{rounded}_seed3.keys"
    assert out_path.exists()
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == rounded
    assert rounded % draw_subset.PROMPTS_PER_UPDATE == 0

    manifest = json.loads((out_dir / f"train_{rounded}_seed3.manifest.json").read_text(encoding="utf-8"))
    assert manifest["size"] == rounded
    assert manifest["requested_size"] == 25
    assert manifest["round_to"] == draw_subset.PROMPTS_PER_UPDATE


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
# scripts/train.py: _assert_subset_divides_prompts_per_update
# ---------------------------------------------------------------------------


def test_assert_subset_divides_prompts_per_update_passes_when_exact() -> None:
    train_script._assert_subset_divides_prompts_per_update(32, 16)  # no raise


def test_assert_subset_divides_prompts_per_update_raises_when_not_exact() -> None:
    with pytest.raises(AssertionError, match="15000"):
        train_script._assert_subset_divides_prompts_per_update(15000, 16)


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
