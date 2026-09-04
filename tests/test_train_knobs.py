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
from trl import GRPOConfig

from conpo import config

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
