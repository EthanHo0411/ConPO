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
    # B4 (audit 2026-09-14): status flips to "finished" once main() returns
    # normally, and started_at is stamped by the pre-train write.
    assert metadata["status"] == "finished"
    assert "started_at" in metadata


# ---------------------------------------------------------------------------
# scripts/train.py: run_metadata.json is written before trainer.train() too
# (B4, audit 2026-09-14), so a run killed mid-train() still leaves a
# "running" artefact behind instead of none.
# ---------------------------------------------------------------------------


def test_main_tiny_leaves_running_run_metadata_when_train_raises(
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

    def _raising_train(self: ConPOTrainer, resume_from_checkpoint: str | None = None) -> None:
        raise RuntimeError("simulated kill mid-train")

    monkeypatch.setattr(ConPOTrainer, "train", _raising_train)

    with pytest.raises(RuntimeError, match="simulated kill mid-train"):
        train_script.main()

    metadata = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "running"
    assert "started_at" in metadata
    # Outcome-only fields never reach the file when train() never returns.
    assert "tokens_generated" not in metadata
    assert "pass1" not in metadata
    assert "actual_h100h" not in metadata
    # A configured knob, to show the pre-train write is not an empty stub.
    assert metadata["arm"] == "solve_only"


def test_main_tiny_run_tau_and_tau_p_reach_config_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RUN_TAU / RUN_TAU_P override config.TAU / config.TAU_P before the
    trainer is built (scripts/train.py); conpo/trainer.py's compute_loss
    reads config.TAU / config.TAU_P fresh at call time, so the override
    reaching config is what reaches training, and run_metadata.json
    must record the effective values.

    monkeypatch.setattr registers each pre-test value for restoration at
    teardown regardless of what main()'s own plain `config.TAU = tau`
    assignment does in between, so config.TAU/TAU_P are back to their
    pre-test values for every later test.
    """
    monkeypatch.setattr(config, "TAU", config.TAU)
    monkeypatch.setattr(config, "TAU_P", config.TAU_P)
    monkeypatch.setattr(config, "KAPPA", config.KAPPA)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_synthetic_pool_rows(data_dir / "train.jsonl", n=4)
    monkeypatch.setattr(train_script, "DATA_DIR", data_dir)

    out_dir = tmp_path / "out"
    monkeypatch.setenv("RUN_ARM", "solve_only")
    monkeypatch.setenv("RUN_STEPS", "1")
    monkeypatch.setenv("RUN_OUT_DIR", str(out_dir))
    monkeypatch.setenv("RUN_NO_VETO", "1")
    monkeypatch.setenv("RUN_TAU", "0.25")
    monkeypatch.setenv("RUN_TAU_P", "0.4")
    monkeypatch.setenv("RUN_KAPPA", "2.5")
    monkeypatch.setattr(sys, "argv", ["train.py", "--tiny"])

    train_script.main()

    assert config.TAU == pytest.approx(0.25)
    assert config.TAU_P == pytest.approx(0.4)
    assert config.KAPPA == pytest.approx(2.5)

    metadata = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["tau"] == pytest.approx(0.25)
    assert metadata["tau_p"] == pytest.approx(0.4)
    assert metadata["kappa"] == pytest.approx(2.5)


def test_main_tiny_run_loss_type_reaches_config_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RUN_LOSS_TYPE overrides config.LOSS_TYPE before the trainer is built
    (scripts/train.py); the GRPOConfig(...) call reads config.LOSS_TYPE at
    that point (loss_type=config.LOSS_TYPE), so the override reaching config
    is what reaches TRL's GRPOTrainer, and run_metadata.json must record the
    effective value.

    monkeypatch.setattr registers the pre-test value for restoration at
    teardown regardless of what main()'s own plain `config.LOSS_TYPE = loss_type`
    assignment does in between, so config.LOSS_TYPE is back to its pre-test
    value for every later test.
    """
    monkeypatch.setattr(config, "LOSS_TYPE", config.LOSS_TYPE)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_synthetic_pool_rows(data_dir / "train.jsonl", n=4)
    monkeypatch.setattr(train_script, "DATA_DIR", data_dir)

    out_dir = tmp_path / "out"
    monkeypatch.setenv("RUN_ARM", "solve_only")
    monkeypatch.setenv("RUN_STEPS", "1")
    monkeypatch.setenv("RUN_OUT_DIR", str(out_dir))
    monkeypatch.setenv("RUN_NO_VETO", "1")
    monkeypatch.setenv("RUN_LOSS_TYPE", "dapo")
    monkeypatch.setattr(sys, "argv", ["train.py", "--tiny"])

    train_script.main()

    assert config.LOSS_TYPE == "dapo"

    metadata = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["loss_type"] == "dapo"


def test_main_tiny_run_pair_loss_weight_reaches_config_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RUN_PAIR_LOSS_WEIGHT overrides config.PAIR_LOSS_WEIGHT before the
    trainer is built (scripts/train.py); conpo/trainer.py's compute_loss
    reads config.PAIR_LOSS_WEIGHT fresh at call time, so the override
    reaching config is what reaches training, and run_metadata.json must
    record the effective value.

    monkeypatch.setattr registers the pre-test value for restoration at
    teardown regardless of what main()'s own plain
    `config.PAIR_LOSS_WEIGHT = pair_loss_weight` assignment does in between,
    so config.PAIR_LOSS_WEIGHT is back to its pre-test value for every later
    test.
    """
    monkeypatch.setattr(config, "PAIR_LOSS_WEIGHT", config.PAIR_LOSS_WEIGHT)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_synthetic_pool_rows(data_dir / "train.jsonl", n=4)
    monkeypatch.setattr(train_script, "DATA_DIR", data_dir)

    out_dir = tmp_path / "out"
    monkeypatch.setenv("RUN_ARM", "solve_only")
    monkeypatch.setenv("RUN_STEPS", "1")
    monkeypatch.setenv("RUN_OUT_DIR", str(out_dir))
    monkeypatch.setenv("RUN_NO_VETO", "1")
    monkeypatch.setenv("RUN_PAIR_LOSS_WEIGHT", "2.5")
    monkeypatch.setattr(sys, "argv", ["train.py", "--tiny"])

    train_script.main()

    assert config.PAIR_LOSS_WEIGHT == pytest.approx(2.5)

    metadata = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["pair_loss_weight"] == pytest.approx(2.5)


def test_main_tiny_run_vllm_is_mode_default_and_override_reach_grpo_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RUN_VLLM_IS_MODE is a pure passthrough to TRL's own GRPOConfig field
    `vllm_importance_sampling_mode` (no ConPO-internal config.py counterpart,
    unlike RUN_LOSS_TYPE) -- scripts/train.py sets it on grpo_config only
    when the env var is set. --tiny never runs with use_vllm=True, so this
    pins the wiring, not vLLM behaviour, by capturing the GRPOConfig that
    reaches ConPOTrainer.__init__ via a thin subclass.
    """

    class _CapturingTrainer(train_script.ConPOTrainer):
        captured_mode: str | None = None

        def __init__(self, *a: object, **kw: object) -> None:
            _CapturingTrainer.captured_mode = kw["args"].vllm_importance_sampling_mode  # type: ignore[attr-defined]
            super().__init__(*a, **kw)

    monkeypatch.setattr(train_script, "ConPOTrainer", _CapturingTrainer)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_synthetic_pool_rows(data_dir / "train.jsonl", n=4)
    monkeypatch.setattr(train_script, "DATA_DIR", data_dir)

    monkeypatch.setenv("RUN_ARM", "solve_only")
    monkeypatch.setenv("RUN_STEPS", "1")
    monkeypatch.setenv("RUN_NO_VETO", "1")
    monkeypatch.setattr(sys, "argv", ["train.py", "--tiny"])

    out_dir_default = tmp_path / "out_default"
    monkeypatch.setenv("RUN_OUT_DIR", str(out_dir_default))
    monkeypatch.delenv("RUN_VLLM_IS_MODE", raising=False)
    train_script.main()
    assert _CapturingTrainer.captured_mode == GRPOConfig().vllm_importance_sampling_mode

    out_dir_override = tmp_path / "out_override"
    monkeypatch.setenv("RUN_OUT_DIR", str(out_dir_override))
    monkeypatch.setenv("RUN_VLLM_IS_MODE", "token_mask")
    train_script.main()
    assert _CapturingTrainer.captured_mode == "token_mask"


def test_main_tiny_run_vllm_is_mode_kl_clamp_resample_and_token_logratio_clamp_reach_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """K2/K4/K5/K6 (fixer review 2026-09-13): RUN_VLLM_IS_MODE,
    RUN_KL_CLAMP, RUN_RESAMPLE_ZERO_STD/RUN_RESAMPLE_MAX_ROUNDS and
    RUN_TOKEN_LOGRATIO_CLAMP all reach the trainer (this file's own
    vllm_is_mode test above, and tests/test_trainer_tiny.py's
    _kl_clamp/_token_logratio_clamp/_resample_zero_std tests) but, before
    this fix, none of the four reached run_metadata.json -- unlike
    RUN_LOSS_TYPE/RUN_PAIR_LOSS_WEIGHT above, so a finished run's own
    artefacts could not say which of these knobs were active for it."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_synthetic_pool_rows(data_dir / "train.jsonl", n=4)
    monkeypatch.setattr(train_script, "DATA_DIR", data_dir)

    monkeypatch.setenv("RUN_ARM", "solve_only")
    monkeypatch.setenv("RUN_STEPS", "1")
    monkeypatch.setenv("RUN_NO_VETO", "1")
    monkeypatch.setattr(sys, "argv", ["train.py", "--tiny"])

    out_dir_default = tmp_path / "out_default"
    monkeypatch.setenv("RUN_OUT_DIR", str(out_dir_default))
    for env_var in (
        "RUN_VLLM_IS_MODE",
        "RUN_KL_CLAMP",
        "RUN_RESAMPLE_ZERO_STD",
        "RUN_RESAMPLE_MAX_ROUNDS",
        "RUN_TOKEN_LOGRATIO_CLAMP",
    ):
        monkeypatch.delenv(env_var, raising=False)
    train_script.main()
    metadata_default = json.loads((out_dir_default / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata_default["vllm_is_mode"] is None
    # Loss-aggregation decision (Ethan, 2026-09-14): RUN_KL_CLAMP unset now
    # defaults to config.KL_CLAMP (40.0), not None -- see
    # test_main_tiny_defaults_are_dapo_and_kl_clamp_40_disable_gives_none
    # below for the "none" disable spelling.
    assert metadata_default["kl_clamp"] == pytest.approx(config.KL_CLAMP)
    assert metadata_default["resample_zero_std"] is False
    assert metadata_default["resample_max_rounds"] == 3
    assert metadata_default["token_logratio_clamp"] is None

    out_dir_override = tmp_path / "out_override"
    monkeypatch.setenv("RUN_OUT_DIR", str(out_dir_override))
    monkeypatch.setenv("RUN_VLLM_IS_MODE", "token_mask")
    monkeypatch.setenv("RUN_KL_CLAMP", "40")
    monkeypatch.setenv("RUN_RESAMPLE_ZERO_STD", "1")
    monkeypatch.setenv("RUN_RESAMPLE_MAX_ROUNDS", "2")
    monkeypatch.setenv("RUN_TOKEN_LOGRATIO_CLAMP", "0.5")
    train_script.main()
    metadata_override = json.loads((out_dir_override / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata_override["vllm_is_mode"] == "token_mask"
    assert metadata_override["kl_clamp"] == pytest.approx(40.0)
    assert metadata_override["resample_zero_std"] is True
    assert metadata_override["resample_max_rounds"] == 2
    assert metadata_override["token_logratio_clamp"] == pytest.approx(0.5)


def test_main_tiny_defaults_are_dapo_and_kl_clamp_40_disable_gives_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loss-aggregation decision (Ethan, 2026-09-14): config.LOSS_TYPE and
    config.KL_CLAMP default to "dapo" and 40.0, and both RUN_LOSS_TYPE/
    RUN_KL_CLAMP unset reach run_metadata.json as such. RUN_KL_CLAMP="none"
    is the explicit disable spelling (a numeric "0" is a real clamp-to-zero,
    see tests/test_trainer_tiny.py::test_compute_loss_kl_clamp_zero_collapses_kl_to_zero,
    so it is not reused as the disable value): paired with
    RUN_LOSS_TYPE=grpo it reproduces the pre-2026-09-14 (no clamp)
    behaviour, and run_metadata.json must record kl_clamp as None for it.
    """
    monkeypatch.setattr(config, "LOSS_TYPE", config.LOSS_TYPE)
    assert config.LOSS_TYPE == "dapo"
    assert config.KL_CLAMP == 40.0

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_synthetic_pool_rows(data_dir / "train.jsonl", n=4)
    monkeypatch.setattr(train_script, "DATA_DIR", data_dir)

    monkeypatch.setenv("RUN_ARM", "solve_only")
    monkeypatch.setenv("RUN_STEPS", "1")
    monkeypatch.setenv("RUN_NO_VETO", "1")
    monkeypatch.delenv("RUN_LOSS_TYPE", raising=False)
    monkeypatch.delenv("RUN_KL_CLAMP", raising=False)
    monkeypatch.setattr(sys, "argv", ["train.py", "--tiny"])

    out_dir_default = tmp_path / "out_default"
    monkeypatch.setenv("RUN_OUT_DIR", str(out_dir_default))
    train_script.main()
    metadata_default = json.loads((out_dir_default / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata_default["loss_type"] == "dapo"
    assert metadata_default["kl_clamp"] == pytest.approx(40.0)

    out_dir_old = tmp_path / "out_old"
    monkeypatch.setenv("RUN_OUT_DIR", str(out_dir_old))
    monkeypatch.setenv("RUN_LOSS_TYPE", "grpo")
    monkeypatch.setenv("RUN_KL_CLAMP", "none")
    train_script.main()
    metadata_old = json.loads((out_dir_old / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata_old["loss_type"] == "grpo"
    assert metadata_old["kl_clamp"] is None


def test_main_tiny_run_pair_scale_auto_default_and_override_reach_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RUN_PAIR_SCALE_AUTO (fix devbox 2026-09-14): same "read back off the
    constructed trainer instance" pattern as pair_objective/pos_anchor_weight
    below. --tiny fixes gradient_accumulation_steps=1, so pair_scale_factor
    is 1.0 either way here; tests/test_trainer_tiny.py covers the
    gradient_accumulation_steps>1 arithmetic and its loss_type-independence
    directly."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_synthetic_pool_rows(data_dir / "train.jsonl", n=4)
    monkeypatch.setattr(train_script, "DATA_DIR", data_dir)

    monkeypatch.setenv("RUN_ARM", "solve_only")
    monkeypatch.setenv("RUN_STEPS", "1")
    monkeypatch.setenv("RUN_NO_VETO", "1")
    monkeypatch.setattr(sys, "argv", ["train.py", "--tiny"])

    out_dir_default = tmp_path / "out_default"
    monkeypatch.setenv("RUN_OUT_DIR", str(out_dir_default))
    monkeypatch.delenv("RUN_PAIR_SCALE_AUTO", raising=False)
    train_script.main()
    metadata_default = json.loads((out_dir_default / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata_default["pair_scale_auto"] is True
    assert metadata_default["pair_scale_factor"] == pytest.approx(1.0)

    out_dir_override = tmp_path / "out_override"
    monkeypatch.setenv("RUN_OUT_DIR", str(out_dir_override))
    monkeypatch.setenv("RUN_PAIR_SCALE_AUTO", "0")
    train_script.main()
    metadata_override = json.loads((out_dir_override / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata_override["pair_scale_auto"] is False
    assert metadata_override["pair_scale_factor"] == pytest.approx(1.0)


def test_main_tiny_run_pair_objective_and_pos_anchor_weight_reach_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RUN_PAIR_OBJECTIVE / RUN_POS_ANCHOR_WEIGHT (K8, devbox 2026-09-14):
    same "read back off the constructed trainer instance" pattern as
    kl_clamp/token_logratio_clamp above, since main() never binds their raw
    env values to a local variable."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_synthetic_pool_rows(data_dir / "train.jsonl", n=4)
    monkeypatch.setattr(train_script, "DATA_DIR", data_dir)

    monkeypatch.setenv("RUN_ARM", "solve_only")
    monkeypatch.setenv("RUN_STEPS", "1")
    monkeypatch.setenv("RUN_NO_VETO", "1")
    monkeypatch.setattr(sys, "argv", ["train.py", "--tiny"])

    out_dir_default = tmp_path / "out_default"
    monkeypatch.setenv("RUN_OUT_DIR", str(out_dir_default))
    monkeypatch.delenv("RUN_PAIR_OBJECTIVE", raising=False)
    monkeypatch.delenv("RUN_POS_ANCHOR_WEIGHT", raising=False)
    train_script.main()
    metadata_default = json.loads((out_dir_default / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata_default["pair_objective"] == "sequence"
    assert metadata_default["pos_anchor_weight"] == pytest.approx(0.0)

    out_dir_override = tmp_path / "out_override"
    monkeypatch.setenv("RUN_OUT_DIR", str(out_dir_override))
    monkeypatch.setenv("RUN_PAIR_OBJECTIVE", "span_local")
    monkeypatch.setenv("RUN_POS_ANCHOR_WEIGHT", "0.3")
    train_script.main()
    metadata_override = json.loads((out_dir_override / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata_override["pair_objective"] == "span_local"
    assert metadata_override["pos_anchor_weight"] == pytest.approx(0.3)


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


def test_main_tiny_run_vllm_push_once_per_step_default_and_override_reach_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RUN_VLLM_PUSH_ONCE_PER_STEP (L1, devbox perf pass 2026-09-13):
    scripts/train.py records the effective value in run_metadata.json, same
    RUN_TAU/RUN_KAPPA override-and-record pattern -- --tiny runs with
    use_vllm=False, so this only pins the recorded flag, not vLLM behaviour
    itself (tests/test_vllm_reload_guard.py covers that)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_synthetic_pool_rows(data_dir / "train.jsonl", n=4)
    monkeypatch.setattr(train_script, "DATA_DIR", data_dir)

    monkeypatch.setenv("RUN_ARM", "solve_only")
    monkeypatch.setenv("RUN_STEPS", "1")
    monkeypatch.setenv("RUN_NO_VETO", "1")
    monkeypatch.setattr(sys, "argv", ["train.py", "--tiny"])

    out_dir_default = tmp_path / "out_default"
    monkeypatch.setenv("RUN_OUT_DIR", str(out_dir_default))
    train_script.main()
    metadata_default = json.loads((out_dir_default / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata_default["vllm_push_once_per_step"] is True

    out_dir_off = tmp_path / "out_off"
    monkeypatch.setenv("RUN_OUT_DIR", str(out_dir_off))
    monkeypatch.setenv("RUN_VLLM_PUSH_ONCE_PER_STEP", "0")
    train_script.main()
    metadata_off = json.loads((out_dir_off / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata_off["vllm_push_once_per_step"] is False


def test_main_tiny_run_repair_batch_generation_default_and_override_reach_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RUN_REPAIR_BATCH_GENERATION (L2, devbox perf pass 2026-09-13): same
    override-and-record pattern as RUN_VLLM_PUSH_ONCE_PER_STEP above --
    conpo/forward.py's/conpo/trainer.py's own behaviour under each setting
    is covered in tests/test_forward.py and tests/test_trainer_tiny.py,
    this only pins the recorded flag."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_synthetic_pool_rows(data_dir / "train.jsonl", n=4)
    monkeypatch.setattr(train_script, "DATA_DIR", data_dir)

    monkeypatch.setenv("RUN_ARM", "solve_only")
    monkeypatch.setenv("RUN_STEPS", "1")
    monkeypatch.setenv("RUN_NO_VETO", "1")
    monkeypatch.setattr(sys, "argv", ["train.py", "--tiny"])

    out_dir_default = tmp_path / "out_default"
    monkeypatch.setenv("RUN_OUT_DIR", str(out_dir_default))
    train_script.main()
    metadata_default = json.loads((out_dir_default / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata_default["repair_batch_generation"] is True

    out_dir_off = tmp_path / "out_off"
    monkeypatch.setenv("RUN_OUT_DIR", str(out_dir_off))
    monkeypatch.setenv("RUN_REPAIR_BATCH_GENERATION", "0")
    train_script.main()
    metadata_off = json.loads((out_dir_off / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata_off["repair_batch_generation"] is False


def test_main_tiny_run_cache_ref_logprobs_default_and_override_reach_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RUN_CACHE_REF_LOGPROBS (L3, devbox perf pass 2026-09-13): same
    override-and-record pattern as RUN_VLLM_PUSH_ONCE_PER_STEP above --
    conpo/trainer.py's own behaviour under each setting is covered in
    tests/test_h6_multi.py and tests/test_edges.py, this only pins the
    recorded flag."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_synthetic_pool_rows(data_dir / "train.jsonl", n=4)
    monkeypatch.setattr(train_script, "DATA_DIR", data_dir)

    monkeypatch.setenv("RUN_ARM", "solve_only")
    monkeypatch.setenv("RUN_STEPS", "1")
    monkeypatch.setenv("RUN_NO_VETO", "1")
    monkeypatch.setattr(sys, "argv", ["train.py", "--tiny"])

    out_dir_default = tmp_path / "out_default"
    monkeypatch.setenv("RUN_OUT_DIR", str(out_dir_default))
    train_script.main()
    metadata_default = json.loads((out_dir_default / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata_default["cache_ref_logprobs"] is True

    out_dir_off = tmp_path / "out_off"
    monkeypatch.setenv("RUN_OUT_DIR", str(out_dir_off))
    monkeypatch.setenv("RUN_CACHE_REF_LOGPROBS", "0")
    train_script.main()
    metadata_off = json.loads((out_dir_off / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata_off["cache_ref_logprobs"] is False


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


# ---------------------------------------------------------------------------
# scripts/train.py: RUN_RESUME (devbox 2026-09-11, node-local disk wiped
# about every 48 hours, so a run must be resumable from its newest harvested
# checkpoint instead of restarted cold on top of it).
#
# _find_checkpoints/_resolve_resume unit tests below never construct a model;
# the two through main() at the bottom do (tiny, CPU), exercising the whole
# RUN_RESUME contract: ConPOTrainer._save_conpo_state/_load_conpo_state,
# trainer.train(resume_from_checkpoint=...), and run_metadata.json.
# ---------------------------------------------------------------------------


def _make_checkpoint(out_dir: Path, step: int, name: str | None = None) -> Path:
    checkpoint_dir = out_dir / (name or f"checkpoint-{step}")
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "trainer_state.json").write_text(json.dumps({"global_step": step}), encoding="utf-8")
    return checkpoint_dir


def test_find_checkpoints_reads_global_step_and_handles_missing_state(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    checkpoint_1 = _make_checkpoint(out_dir, step=1)
    broken = out_dir / "checkpoint-broken"
    broken.mkdir()  # no trainer_state.json: still "a checkpoint exists", step reads as 0

    found = dict(train_script._find_checkpoints(out_dir))
    assert found[checkpoint_1] == 1
    assert found[broken] == 0


def test_resolve_resume_fresh_run_raises_when_checkpoint_exists(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _make_checkpoint(out_dir, step=3)

    with pytest.raises(SystemExit) as exc_info:
        train_script._resolve_resume(out_dir, run_resume=False)
    message = str(exc_info.value)
    assert "RUN_RESUME" in message
    assert "checkpoint" in message


def test_resolve_resume_fresh_run_silent_when_out_dir_has_no_checkpoint(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"  # never created
    assert train_script._resolve_resume(out_dir, run_resume=False) == (None, None)


def test_resolve_resume_picks_newest_checkpoint_by_step_not_lexicographic_order(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _make_checkpoint(out_dir, step=9)
    newest = _make_checkpoint(out_dir, step=10)  # "checkpoint-10" sorts before "checkpoint-9" as a string

    checkpoint_dir, step = train_script._resolve_resume(out_dir, run_resume=True)
    assert checkpoint_dir == newest
    assert step == 10


def test_resolve_resume_raises_when_nothing_to_resume(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"  # never created
    with pytest.raises(SystemExit) as exc_info:
        train_script._resolve_resume(out_dir, run_resume=True)
    assert "RUN_RESUME=1" in str(exc_info.value)


def _run_main_tiny(
    data_dir: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch, steps: int, resume: bool
) -> None:
    monkeypatch.setattr(train_script, "DATA_DIR", data_dir)
    monkeypatch.setenv("RUN_ARM", "full")
    monkeypatch.setenv("RUN_STEPS", str(steps))
    # RUN_SAVE_STEPS (design choice, builder, 2026-09-11): checkpoint every
    # step instead of every config.SAVE_STEPS (25), so this CPU test reaches
    # a checkpoint in 1 step instead of 25.
    monkeypatch.setenv("RUN_SAVE_STEPS", "1")
    monkeypatch.setenv("RUN_OUT_DIR", str(out_dir))
    monkeypatch.setenv("RUN_NO_VETO", "1")
    monkeypatch.setenv("RUN_RESUME", "1" if resume else "0")
    monkeypatch.setattr(sys, "argv", ["train.py", "--tiny"])
    train_script.main()


def test_main_resume_continues_from_saved_step_and_conpo_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RUN_RESUME end to end: a fresh 1-step run checkpoints at
    checkpoint-1, then a resumed invocation with RUN_STEPS=2 continues from
    it instead of restarting -- exactly 2 per-step mechanism records total
    (steps 0 and 1, never two step-0 records), the resume event's
    edge_buffer_size/tokens_cumulative match what checkpoint-1 itself saved,
    and run_metadata.json's resumed_from_step is 1."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_synthetic_pool_rows(data_dir / "train.jsonl", n=4)
    out_dir = tmp_path / "out"

    _run_main_tiny(data_dir, out_dir, monkeypatch, steps=1, resume=False)

    checkpoint_1 = out_dir / "checkpoint-1"
    assert checkpoint_1.is_dir()
    saved_state = json.loads((checkpoint_1 / "conpo_state.json").read_text(encoding="utf-8"))
    saved_edges = sum(1 for _ in (checkpoint_1 / "conpo_edges.jsonl").open(encoding="utf-8"))

    _run_main_tiny(data_dir, out_dir, monkeypatch, steps=2, resume=True)

    lines = [
        json.loads(line)
        for line in (out_dir / "mechanism.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    step_records = [r for r in lines if "event" not in r]
    assert sorted(r["step"] for r in step_records) == [0, 1]  # continued, not restarted at 0 twice

    resume_records = [r for r in lines if r.get("event") == "resume"]
    assert len(resume_records) == 1
    assert resume_records[0]["resumed_from_step"] == 1
    assert resume_records[0]["edge_buffer_size"] == saved_edges
    assert resume_records[0]["tokens_cumulative"] == saved_state["meter_total"]

    metadata = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["resumed_from_step"] == 1


def test_main_fresh_run_refuses_out_dir_with_existing_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh run (RUN_RESUME unset) must fail loud, before loading any
    model, if RUN_OUT_DIR already holds a checkpoint -- a node-local-disk
    wipe followed by a plain restart must never silently start training
    over on top of harvested data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_synthetic_pool_rows(data_dir / "train.jsonl", n=4)
    monkeypatch.setattr(train_script, "DATA_DIR", data_dir)

    def _fail_loader(*args: object, **kwargs: object) -> tuple[object, object]:
        raise AssertionError("model loaded before the RUN_RESUME guard")

    monkeypatch.setattr(train_script, "_real_model_and_tokenizer", _fail_loader)
    monkeypatch.setattr(train_script, "_tiny_model_and_tokenizer", _fail_loader)

    out_dir = tmp_path / "out"
    _make_checkpoint(out_dir, step=1)
    monkeypatch.setenv("RUN_ARM", "solve_only")
    monkeypatch.setenv("RUN_STEPS", "1")
    monkeypatch.setenv("RUN_OUT_DIR", str(out_dir))
    monkeypatch.setenv("RUN_NO_VETO", "1")
    monkeypatch.delenv("RUN_RESUME", raising=False)
    monkeypatch.setattr(sys, "argv", ["train.py", "--tiny"])

    with pytest.raises(SystemExit) as exc_info:
        train_script.main()
    message = str(exc_info.value)
    assert "RUN_RESUME" in message
    assert "checkpoint" in message


# ---------------------------------------------------------------------------
# scripts/train.py: SaveCadenceCallback (resume keeps RUN_SAVE_STEPS)
# ---------------------------------------------------------------------------


def test_save_cadence_callback_reasserts_args_on_train_begin() -> None:
    from types import SimpleNamespace

    from transformers import TrainerState

    # transformers reloads this from the checkpoint after copying the args in.
    state = TrainerState(save_steps=25, logging_steps=25)
    args = SimpleNamespace(save_steps=15, logging_steps=1)
    train_script.SaveCadenceCallback().on_train_begin(args, state, None)
    assert state.save_steps == 15
    assert state.logging_steps == 1


# ---------------------------------------------------------------------------
# scripts/train.py: StopAtStepCallback (RUN_STOP_AT_STEP)
# ---------------------------------------------------------------------------


def test_stop_at_step_callback_saves_and_stops_once_target_step_is_reached() -> None:
    from types import SimpleNamespace

    from transformers import TrainerControl, TrainerState

    callback = train_script.StopAtStepCallback(stop_at_step=2)

    below_target = TrainerState(global_step=1)
    control = TrainerControl()
    callback.on_step_end(SimpleNamespace(), below_target, control)
    assert control.should_save is False
    assert control.should_training_stop is False

    at_target = TrainerState(global_step=2)
    control = TrainerControl()
    callback.on_step_end(SimpleNamespace(), at_target, control)
    assert control.should_save is True
    assert control.should_training_stop is True


def test_main_tiny_run_stop_at_step_halts_training_and_records_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RUN_STOP_AT_STEP must halt the run (not just save an extra checkpoint)
    at the given step while leaving max_steps -- and so the cosine
    schedule -- at the full RUN_STEPS value; unset, it must not touch the
    run at all."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_synthetic_pool_rows(data_dir / "train.jsonl", n=4)
    monkeypatch.setattr(train_script, "DATA_DIR", data_dir)

    monkeypatch.setenv("RUN_ARM", "solve_only")
    monkeypatch.setenv("RUN_STEPS", "4")
    monkeypatch.setenv("RUN_NO_VETO", "1")
    monkeypatch.setattr(sys, "argv", ["train.py", "--tiny"])

    # Default: RUN_STOP_AT_STEP unset must not halt the run early. RUN_SAVE_STEPS=1
    # so a checkpoint at every step would reveal an early halt if the callback
    # fired anyway.
    out_dir_default = tmp_path / "out_default"
    monkeypatch.setenv("RUN_OUT_DIR", str(out_dir_default))
    monkeypatch.setenv("RUN_SAVE_STEPS", "1")
    monkeypatch.delenv("RUN_STOP_AT_STEP", raising=False)
    train_script.main()
    assert (out_dir_default / "checkpoint-4").is_dir()
    metadata_default = json.loads((out_dir_default / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata_default["stop_at_step"] is None

    # Override: RUN_STOP_AT_STEP=2 must save at step 2 and stop there, never
    # reaching step 4 -- config.SAVE_STEPS' own cadence (25) would not save
    # at step 2 on its own, so checkpoint-2 existing proves the callback's
    # should_save, and checkpoint-4 absent proves should_training_stop.
    out_dir_stop = tmp_path / "out_stop"
    monkeypatch.setenv("RUN_OUT_DIR", str(out_dir_stop))
    monkeypatch.delenv("RUN_SAVE_STEPS", raising=False)
    monkeypatch.setenv("RUN_STOP_AT_STEP", "2")
    train_script.main()
    checkpoint_2 = out_dir_stop / "checkpoint-2"
    assert checkpoint_2.is_dir()
    trainer_state = json.loads((checkpoint_2 / "trainer_state.json").read_text(encoding="utf-8"))
    assert trainer_state["global_step"] == 2
    assert not (out_dir_stop / "checkpoint-4").exists()
    metadata_stop = json.loads((out_dir_stop / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata_stop["stop_at_step"] == 2


def test_main_tiny_done_message_reports_completed_steps_not_max_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """B5 (vii, audit 2026-09-14): the final "done: N step(s)" message must
    report trainer.state.global_step (steps run), not the RUN_STEPS
    budget -- RUN_STOP_AT_STEP (same fixture as the halt test above) stops
    the run at step 2 while RUN_STEPS asks for 4, so the old `steps`
    (max_steps) value in the message overstated how far training got."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_synthetic_pool_rows(data_dir / "train.jsonl", n=4)
    monkeypatch.setattr(train_script, "DATA_DIR", data_dir)

    out_dir = tmp_path / "out"
    monkeypatch.setenv("RUN_ARM", "solve_only")
    monkeypatch.setenv("RUN_STEPS", "4")
    monkeypatch.setenv("RUN_STOP_AT_STEP", "2")
    monkeypatch.setenv("RUN_OUT_DIR", str(out_dir))
    monkeypatch.setenv("RUN_NO_VETO", "1")
    monkeypatch.setattr(sys, "argv", ["train.py", "--tiny"])

    train_script.main()

    out = capsys.readouterr().out
    assert "done: 2 step(s)" in out
    assert "done: 4 step(s)" not in out
