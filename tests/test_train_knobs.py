"""Tests for scripts/train.py's smoke-shape env knobs (RUN_NUM_GENERATIONS,
RUN_MAX_COMPLETION_TOKENS, RUN_EFFECTIVE_BATCH) and the dataset-size /
prompts_per_update fail-loud guard (design 2026-09-04, from the RTX 5060
smoke on 2026-09-03: the trainer's fixed shape OOMs an 8 GB card, and
RUN_LIMIT=8 < prompts_per_update=16 used to silently train nothing while
main() still printed "done: 2 step(s)").

scripts/ is not an installed package, so train is imported by adding
scripts/ to sys.path -- the same pattern tests/test_subset.py already uses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from datasets import Dataset
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from trl import GRPOConfig

from conpo import config
from conpo.arms import resolve_arm
from conpo.data import TokenMeter
from conpo.edges import EdgeBuffer, FeasibleBuffer
from conpo.judge import StaticJudge
from conpo.selection import FamilyStats
from conpo.trainer import ConPOTrainer

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train as train_script  # noqa: E402

# ---------------------------------------------------------------------------
# scripts/train.py: _check_dataset_size (the fail-loud guard)
# ---------------------------------------------------------------------------


def test_check_dataset_size_raises_with_both_numbers_in_message() -> None:
    with pytest.raises(SystemExit) as exc_info:
        train_script._check_dataset_size(dataset_size=8, prompts_per_update=16)

    message = str(exc_info.value)
    assert "8" in message
    assert "16" in message
    assert "RUN_LIMIT" in message
    assert "RUN_EFFECTIVE_BATCH" in message


def test_check_dataset_size_silent_when_exact_or_larger() -> None:
    train_script._check_dataset_size(dataset_size=16, prompts_per_update=16)  # no raise
    train_script._check_dataset_size(dataset_size=32, prompts_per_update=16)  # no raise


# ---------------------------------------------------------------------------
# scripts/train.py: _run_shape (RUN_NUM_GENERATIONS / RUN_MAX_COMPLETION_TOKENS
# / RUN_EFFECTIVE_BATCH -- smoke and shape tests only, cluster runs never set
# them so config.env's default reproduces config.py's own values)
# ---------------------------------------------------------------------------


def test_run_shape_tiny_ignores_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUN_NUM_GENERATIONS", "99")
    monkeypatch.setenv("RUN_MAX_COMPLETION_TOKENS", "99")
    monkeypatch.setenv("RUN_EFFECTIVE_BATCH", "99")

    assert train_script._run_shape(tiny=True) == (2, 16, config.EFFECTIVE_BATCH)


def test_run_shape_defaults_to_config_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUN_NUM_GENERATIONS", raising=False)
    monkeypatch.delenv("RUN_MAX_COMPLETION_TOKENS", raising=False)
    monkeypatch.delenv("RUN_EFFECTIVE_BATCH", raising=False)

    assert train_script._run_shape(tiny=False) == (
        config.NUM_GENERATIONS,
        config.MAX_COMPLETION_TOKENS,
        config.EFFECTIVE_BATCH,
    )


def test_run_shape_reads_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUN_NUM_GENERATIONS", "2")
    monkeypatch.setenv("RUN_MAX_COMPLETION_TOKENS", "64")
    monkeypatch.setenv("RUN_EFFECTIVE_BATCH", "4")

    assert train_script._run_shape(tiny=False) == (2, 64, 4)


def test_run_shape_knobs_reach_grpo_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The knobs feed GRPOConfig the same way main() wires them: num_generations,
    max_completion_length, and per_device_train_batch_size (== num_generations,
    the shape main() builds GRPOConfig with)."""
    monkeypatch.setenv("RUN_NUM_GENERATIONS", "2")
    monkeypatch.setenv("RUN_MAX_COMPLETION_TOKENS", "64")
    monkeypatch.setenv("RUN_EFFECTIVE_BATCH", "4")

    num_generations, max_completion_length, effective_batch = train_script._run_shape(tiny=False)
    per_device_batch = num_generations
    gradient_accumulation_steps = effective_batch // (per_device_batch * 1)

    grpo_config = GRPOConfig(
        output_dir=str(tmp_path),
        num_generations=num_generations,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_completion_length=max_completion_length,
        max_steps=1,
        use_vllm=False,
        beta=config.KL_BETA,
        loss_type="grpo",
        learning_rate=1e-3,
        report_to=[],
        use_cpu=True,
        bf16=False,
        fp16=False,
    )

    assert grpo_config.num_generations == 2
    assert grpo_config.per_device_train_batch_size == 2
    assert grpo_config.max_completion_length == 64
    assert grpo_config.gradient_accumulation_steps == 2  # 4 // 2


# ---------------------------------------------------------------------------
# scripts/train.py: main() --tiny writes num_generations/max_completion_tokens/
# effective_batch to run_metadata.json (--tiny ignores the env overrides, per
# _run_shape above, so these are the tiny-shape constants regardless of env)
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


def test_main_tiny_writes_shape_knobs_to_run_metadata(
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
    # --tiny ignores these; run_metadata.json must still show the tiny shape,
    # not the overrides, proving --tiny short-circuits _run_shape.
    monkeypatch.setenv("RUN_NUM_GENERATIONS", "99")
    monkeypatch.setenv("RUN_MAX_COMPLETION_TOKENS", "99")
    monkeypatch.setenv("RUN_EFFECTIVE_BATCH", "99")
    monkeypatch.setattr(sys, "argv", ["train.py", "--tiny"])

    train_script.main()

    metadata = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["num_generations"] == 2
    assert metadata["max_completion_tokens"] == 16
    assert metadata["effective_batch"] == config.EFFECTIVE_BATCH


def test_main_tiny_raises_when_run_limit_below_prompts_per_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RUN_LIMIT below prompts_per_update must raise instead of training with
    a silently empty dataset (the RTX 5060 smoke's RUN_LIMIT=8 / 2026-09-03
    failure, reproduced here at --tiny's own scale: prompts_per_update=1,
    RUN_LIMIT=0 rows selected by an out-of-range limit is not representable,
    so this drives dataset_size to 0 via an empty pool instead)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_synthetic_pool_rows(data_dir / "train.jsonl", n=0)
    monkeypatch.setattr(train_script, "DATA_DIR", data_dir)

    out_dir = tmp_path / "out"
    monkeypatch.setenv("RUN_ARM", "solve_only")
    monkeypatch.setenv("RUN_STEPS", "1")
    monkeypatch.setenv("RUN_OUT_DIR", str(out_dir))
    monkeypatch.setenv("RUN_NO_VETO", "1")
    monkeypatch.setattr(sys, "argv", ["train.py", "--tiny"])

    with pytest.raises(SystemExit) as exc_info:
        train_script.main()

    message = str(exc_info.value)
    assert "0" in message
    assert "1" in message  # prompts_per_update on a single CPU process
    assert not (out_dir / "run_metadata.json").exists()


# ---------------------------------------------------------------------------
# scripts/train.py: main() with RUN_FULL_FT=1 (design 2026-09-04, real run
# since B7 2026-09-04: conpo/trainer.py's ConPOTrainer._reference_model now
# scores pi_ref under TRL's own self.ref_model when there is no adapter to
# disable, so a real ConPOTrainer.train() completes instead of raising
# AttributeError -- see conpo/trainer.py for the reference-scoring path)
# ---------------------------------------------------------------------------


def test_main_tiny_run_full_ft_writes_flag_and_no_adapter_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RUN_FULL_FT=1 with RUN_ARM=full (exercises every reference-scoring
    site: forward/reverse edge admission and the prompt-margin term) runs a
    real training step, records full_ft: true, writes no adapter_config.json
    under out_dir/adapter (peft_config=None -> plain model.save_pretrained),
    and logs exactly one mechanism.jsonl record for the one optimiser step."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_synthetic_pool_rows(data_dir / "train.jsonl", n=4)
    monkeypatch.setattr(train_script, "DATA_DIR", data_dir)

    out_dir = tmp_path / "out"
    monkeypatch.setenv("RUN_ARM", "full")
    monkeypatch.setenv("RUN_STEPS", "1")
    monkeypatch.setenv("RUN_OUT_DIR", str(out_dir))
    monkeypatch.setenv("RUN_NO_VETO", "1")
    monkeypatch.setenv("RUN_FULL_FT", "1")
    monkeypatch.setattr(sys, "argv", ["train.py", "--tiny"])

    train_script.main()

    metadata = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["full_ft"] is True
    assert not (out_dir / "adapter" / "adapter_config.json").exists()
    mechanism_lines = (out_dir / "mechanism.jsonl").read_text(encoding="utf-8").strip().splitlines()
    mechanism_lines = [line for line in mechanism_lines if "event" not in json.loads(line)]  # step_timing records live beside the per-step record
    assert len(mechanism_lines) == 1


# ---------------------------------------------------------------------------
# conpo/trainer.py: ConPOTrainer.__init__'s construction-time guard (review
# finding, B7 2026-09-04) -- a non-PEFT model with beta == 0.0 leaves TRL's
# self.ref_model None, which _reference_model would otherwise hand to a
# reference-scoring site with nothing to score under; the guard must reject
# that combination before super().__init__() returns, not on the first step.
# ---------------------------------------------------------------------------


def _no_ref_model_trainer_args(tmp_path: Path) -> tuple[Qwen3ForCausalLM, AutoTokenizer, GRPOConfig]:
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
    model = Qwen3ForCausalLM(cfg)
    grpo_config = GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=8,
        max_steps=1,
        use_vllm=False,
        beta=0.0,  # trl only builds self.ref_model when beta != 0.0
        loss_type="grpo",
        learning_rate=1e-3,
        report_to=[],
        use_cpu=True,
        bf16=False,
        fp16=False,
    )
    return model, tokenizer, grpo_config


def test_full_ft_with_zero_beta_raises_before_engine_factory_runs(tmp_path: Path) -> None:
    model, tokenizer, grpo_config = _no_ref_model_trainer_args(tmp_path)
    row = {"prompt": [{"role": "user", "content": "Explain gravity."}], "key": "row-0"}
    dataset = Dataset.from_list([row])
    arm, overrides = resolve_arm("solve_only")

    def _engine_factory(_trainer: ConPOTrainer) -> object:
        raise AssertionError("engine_factory must not run: the guard raises during super().__init__()")

    with pytest.raises(ValueError, match="RUN_FULL_FT"):
        ConPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=dataset,
            processing_class=tokenizer,
            arm=arm,
            rows_by_key={},
            edge_buffer=EdgeBuffer(),
            feasible_buffer=FeasibleBuffer(),
            stats=FamilyStats(),
            judge=StaticJudge(keep=True),
            engine_factory=_engine_factory,
            meter=TokenMeter(budget=0),
            mechanism_log_path=tmp_path / "mechanism.jsonl",
            seed=0,
            overrides=overrides,
            peft_config=None,  # non-PEFT model
        )


# ---------------------------------------------------------------------------
# scripts/train.py: main() must raise the guard before loading any model or
# tokenizer (design 2026-09-04, from the RTX 5060 smoke on 2026-09-03 where a
# wrong RUN_LIMIT was reported only after a 56-minute Qwen3-4B download/load)
# ---------------------------------------------------------------------------


def test_main_raises_before_loading_model_or_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_synthetic_pool_rows(data_dir / "train.jsonl", n=1)
    monkeypatch.setattr(train_script, "DATA_DIR", data_dir)

    def _fail_loader(*args: object, **kwargs: object) -> tuple[object, object]:
        raise AssertionError("model loaded before the guard")

    monkeypatch.setattr(train_script, "_real_model_and_tokenizer", _fail_loader)
    monkeypatch.setattr(train_script, "_tiny_model_and_tokenizer", _fail_loader)

    out_dir = tmp_path / "out"
    monkeypatch.setenv("RUN_ARM", "solve_only")
    monkeypatch.setenv("RUN_STEPS", "1")
    monkeypatch.setenv("RUN_OUT_DIR", str(out_dir))
    monkeypatch.setenv("RUN_NO_VETO", "1")
    monkeypatch.setattr(sys, "argv", ["train.py"])  # non-tiny: prompts_per_update=16 (8 * 128 // 8)

    with pytest.raises(SystemExit) as exc_info:
        train_script.main()

    message = str(exc_info.value)
    assert "dataset_size (1)" in message
    assert "prompts_per_update (16)" in message
    assert "RUN_LIMIT" in message
    assert not (out_dir / "run_metadata.json").exists()
