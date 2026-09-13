"""Tests for `ConPOTrainer._install_vllm_reload_guard` (conpo/trainer.py).

TRL's `VLLMGeneration.generate` calls `self.llm.collective_rpc("reload_weights")`
on every call when sleep mode is on (trl/generation/vllm_generation.py lines
540-548, "work around for vllm issue 29341"), and vLLM's `reload_weights` with
no arguments reloads the base model from disk
(vllm/v1/worker/gpu_model_runner.py line 5480), clobbering the policy weights
`sync_weights()` had already pushed for this step. The guard wraps
`collective_rpc` so a `"reload_weights"` call is always followed by a resync.

No GPU is needed here: the trainer is built with `use_vllm=False`
(tests/test_trainer_tiny.py's and tests/test_train_knobs.py's pattern for a
fast, real ConPOTrainer on CPU), then a fake `vllm_generation` is attached
after construction to exercise the guard's wrapping behaviour in isolation.
GPU-side verification (the reload restoring the right log-probs) lives
outside pytest, in scripts/diag_logp.py.

Perf-pass instrumentation (Ethan devbox, 2026-09-11): `_FakeLLM`/
`_FakeVLLMGeneration` also stand in for `wake_up`/`sleep`/`generate`, so
`test_engine_seconds_accumulates_across_generate_calls` below can exercise
the same guard's `self._engine_seconds` timing without a GPU either.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from trl import GRPOConfig

from conpo import config
from conpo.arms import resolve_arm
from conpo.data import TokenMeter
from conpo.edges import EdgeBuffer, FeasibleBuffer
from conpo.engine import HFEngine
from conpo.judge import StaticJudge
from conpo.selection import FamilyStats
from conpo.trainer import ConPOTrainer


def _tiny_model_and_tokenizer() -> tuple[Qwen3ForCausalLM, AutoTokenizer]:
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


def _lora_config() -> LoraConfig:
    return LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, target_modules=list(config.LORA_TARGET_MODULES), task_type="CAUSAL_LM"
    )


def _build_trainer(tmp_path: Path) -> ConPOTrainer:
    """A minimal, fast, real ConPOTrainer (no `.train()` call needed): only
    `__init__` (and therefore `_install_vllm_reload_guard`'s call site) has
    to run. `use_vllm=False` means `self.vllm_generation` does not exist
    yet after construction, matching every non-vLLM run this trainer
    supports (RUN_USE_VLLM=0, HFEngine)."""
    model, tokenizer = _tiny_model_and_tokenizer()
    row = {"prompt": [{"role": "user", "content": "Explain gravity."}], "key": "row-0"}
    dataset = Dataset.from_list([row])
    grpo_config = GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=8,
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
    arm, overrides = resolve_arm("solve_only")

    def _engine_factory(trainer: ConPOTrainer) -> HFEngine:
        return HFEngine(trainer.accelerator.unwrap_model(trainer.model), trainer.processing_class, device="cpu")

    return ConPOTrainer(
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
        peft_config=_lora_config(),
    )


class _FakeLLM:
    """Stand-in for `vllm.LLM`: records every `collective_rpc` method name,
    plus (perf-pass instrumentation, 2026-09-11) `wake_up`/`sleep` calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def collective_rpc(self, method: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append(method)

    def wake_up(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(f"wake_up:{kwargs.get('tags')}")

    def sleep(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append("sleep")


class _FakeVLLMGeneration:
    """Stand-in for `trl.generation.vllm_generation.VLLMGeneration`."""

    def __init__(self, llm: _FakeLLM) -> None:
        self.llm = llm

    def sync_weights(self) -> None:
        self.llm.calls.append("sync")

    def generate(self, *args: Any, **kwargs: Any) -> str:
        """Mimics `VLLMGeneration.generate`'s colocate-mode body (trl
        vllm_generation.py lines 536-548, 650-702): wake weights, reload,
        wake kv_cache, sample (here: a few ms sleep standing in for vLLM's
        own `llm.generate` call), then sleep(level=2)."""
        self.llm.wake_up(tags=["weights"])
        self.llm.collective_rpc("reload_weights")
        self.llm.wake_up(tags=["kv_cache"])
        time.sleep(0.005)
        self.llm.sleep(level=2)
        return "generated"


def test_reload_weights_triggers_resync(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path)
    llm = _FakeLLM()
    trainer.vllm_generation = _FakeVLLMGeneration(llm)

    trainer._install_vllm_reload_guard()
    trainer.vllm_generation.llm.collective_rpc("reload_weights")

    assert llm.calls == ["reload_weights", "sync"]


def test_other_methods_pass_through_unchanged(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path)
    llm = _FakeLLM()
    trainer.vllm_generation = _FakeVLLMGeneration(llm)

    trainer._install_vllm_reload_guard()
    trainer.vllm_generation.llm.collective_rpc("load_weights", "unused_arg")

    assert llm.calls == ["load_weights"]


def test_installing_twice_does_not_double_wrap(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path)
    llm = _FakeLLM()
    trainer.vllm_generation = _FakeVLLMGeneration(llm)

    trainer._install_vllm_reload_guard()
    trainer._install_vllm_reload_guard()
    trainer.vllm_generation.llm.collective_rpc("reload_weights")

    assert llm.calls == ["reload_weights", "sync"]  # not ["reload_weights", "sync", "sync"]


def test_trainer_without_vllm_generation_installs_nothing(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path)
    assert not hasattr(trainer, "vllm_generation")  # RUN_USE_VLLM=0: trl never sets it

    trainer._install_vllm_reload_guard()  # must not raise
    # HFEngine run: no colocated vLLM engine ever existed, so the per-step
    # record must never carry an "engine_seconds" key (2026-09-11 perf pass).
    assert trainer._engine_seconds is None


def test_engine_seconds_accumulates_across_generate_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_install_vllm_reload_guard`'s wake_up/sleep/generate wrapping
    (Ethan devbox perf pass, 2026-09-11): two `generate()` calls, each
    going through the real trl colocate sequence via `_FakeVLLMGeneration`
    above (wake weights, reload, wake kv_cache, ~5ms of "sampling", sleep),
    accumulate wake_reload_resync/generate_only/sleep seconds and bump
    generate_calls/reload_calls -- the same wrapped callables
    conpo/engine.py's `VLLMColocateEngine.generate` and TRL's own rollout
    both call through in a real run.

    RUN_VLLM_PUSH_ONCE_PER_STEP=0 (L1 devbox perf pass, 2026-09-13): this
    test's two direct `generate()` calls, with no `_generate_and_score_completions`
    step boundary in between, are exactly the "same step" shape push-once
    would collapse -- pinned off here so this test keeps guarding today's
    unconditional per-call reload+resync+sleep path unchanged. See
    test_push_once_per_step_skips_repeat_sync_and_defers_sleep below for the
    default (flag on) behaviour.
    """
    monkeypatch.setenv("RUN_VLLM_PUSH_ONCE_PER_STEP", "0")
    trainer = _build_trainer(tmp_path)
    llm = _FakeLLM()
    trainer.vllm_generation = _FakeVLLMGeneration(llm)
    trainer._install_vllm_reload_guard()

    trainer.vllm_generation.generate()
    trainer.vllm_generation.generate()

    seconds = trainer._engine_seconds
    assert seconds is not None
    assert seconds["generate_calls"] == 2
    assert seconds["reload_calls"] == 2
    assert seconds["reload_calls_skipped"] == 0
    # Two ~5ms "sampling" sleeps, each between the kv_cache wake and
    # llm.sleep -- time.sleep(x) never returns before x seconds elapse.
    assert seconds["generate_only"] >= 0.008
    assert seconds["wake_reload_resync"] >= 0.0
    assert seconds["sleep"] >= 0.0
    # The reload guard's own effect (test_reload_weights_triggers_resync's
    # assertion) still holds with the extra wrapping installed.
    assert llm.calls.count("sync") == 2
    assert llm.calls.count("sleep") == 2


def test_push_once_per_step_skips_repeat_sync_and_defers_sleep(tmp_path: Path) -> None:
    """RUN_VLLM_PUSH_ONCE_PER_STEP defaults to "1" (L1 devbox perf pass,
    2026-09-13): a second `generate()` call within the same step (no
    `_generate_and_score_completions` boundary in between, same shape as
    the two direct calls above) must not re-run TRL's reload_weights RPC or
    ConPOTrainer's own resync -- weights are still resident from the first
    call -- and must not really sleep the engine either, so the next call's
    wake_up is a genuine no-op instead of restoring garbage weights. The
    deferred sleep only runs for real once `_flush_deferred_vllm_sleep`
    (called from `_generate_and_score_completions`, right before every
    flush_pending_veto trigger) says so. A step boundary
    (`_vllm_synced_this_step` reset, what `_generate_and_score_completions`
    does at its own top) forces a real resync and sleep again."""
    trainer = _build_trainer(tmp_path)
    llm = _FakeLLM()
    trainer.vllm_generation = _FakeVLLMGeneration(llm)
    trainer._install_vllm_reload_guard()

    trainer.vllm_generation.generate()
    trainer.vllm_generation.generate()

    seconds = trainer._engine_seconds
    assert seconds is not None
    assert seconds["generate_calls"] == 2
    assert seconds["reload_calls"] == 1
    assert seconds["reload_calls_skipped"] == 1
    assert llm.calls.count("sync") == 1
    assert llm.calls.count("sleep") == 0  # deferred, not dropped
    assert trainer._vllm_sleep_pending is True

    trainer._flush_deferred_vllm_sleep()
    assert llm.calls.count("sleep") == 1
    assert trainer._vllm_sleep_pending is False
    assert seconds["sleep"] >= 0.0

    # New step: reset the same two trackers _generate_and_score_completions
    # resets at its own top, then confirm a real resync+sleep happens again.
    trainer._vllm_synced_this_step = False
    trainer._vllm_sleep_pending = False

    trainer.vllm_generation.generate()

    assert seconds["reload_calls"] == 2
    assert seconds["reload_calls_skipped"] == 1  # unchanged: this call resynced for real
    assert llm.calls.count("sync") == 2
    trainer._flush_deferred_vllm_sleep()
    assert llm.calls.count("sleep") == 2
