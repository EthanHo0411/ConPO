"""scripts/train_dpo.py --tiny: one DPO step on a tiny random Qwen3 model
(CPU, must finish in well under 3 minutes).

scripts/ is not an installed package (pyproject.toml only packages `conpo*`),
so the script is imported by adding scripts/ to sys.path, same as
tests/test_eval_scripts.py does for scripts/eval_external.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from safetensors.torch import load_file

from conpo.pairs import Pair, save_pairs

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train_dpo  # noqa: E402

_PAIRS = [
    Pair("p1", "Explain how gravity works.", "Gravity pulls masses together.", "I refuse to answer.", "vf", {}),
    Pair("p2", "Describe photosynthesis.", "Plants convert light into energy.", "banana banana banana.", "vf", {}),
    Pair("p3", "Summarize the water cycle.", "Water evaporates, condenses, and falls as rain.", "42.", "vf", {}),
    Pair("p4", "Outline the rock cycle.", "Rocks form, erode, and reform over time.", "no.", "vf", {}),
]


def test_train_dpo_tiny_one_step_adapter_changes_and_metadata_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs_path = tmp_path / "pairs.jsonl"
    save_pairs(_PAIRS, pairs_path)
    out_dir = tmp_path / "out"

    monkeypatch.setenv("RUN_ARM", "vf_dpo")
    monkeypatch.setenv("RUN_PAIRS", str(pairs_path))
    monkeypatch.setenv("RUN_SEED", "0")
    monkeypatch.setenv("RUN_STEPS", "1")
    monkeypatch.setenv("RUN_OUT_DIR", str(out_dir))
    monkeypatch.setattr(sys, "argv", ["train_dpo.py", "--tiny"])

    train_dpo.main()

    # Adapter saved and changed: peft zero-initialises every lora_B matrix,
    # so a nonzero lora_B after one optimizer step proves training moved it.
    adapter_dir = out_dir / "adapter"
    assert adapter_dir.is_dir()
    state_dict = load_file(str(adapter_dir / "adapter_model.safetensors"))
    lora_b_tensors = [t for name, t in state_dict.items() if "lora_B" in name]
    assert lora_b_tensors
    assert any(t.abs().sum().item() > 0 for t in lora_b_tensors)

    # Metadata written with the fields the token-budget/arm bookkeeping needs.
    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["arm"] == "vf_dpo"
    assert metadata["n_pairs"] == 4
    assert metadata["pairs_path"] == str(pairs_path)
    assert metadata["seed"] == 0
    assert metadata["steps"] == 1
    assert metadata["pair_build_tokens"] == 0  # no manifest file alongside pairs_path in this test


def test_train_dpo_rejects_a_grpo_arm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pairs_path = tmp_path / "pairs.jsonl"
    save_pairs(_PAIRS, pairs_path)

    monkeypatch.setenv("RUN_ARM", "full")  # a "grpo" arm, not trainable by this script
    monkeypatch.setenv("RUN_PAIRS", str(pairs_path))
    monkeypatch.setenv("RUN_OUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(sys, "argv", ["train_dpo.py", "--tiny"])

    with pytest.raises(ValueError, match="grpo"):
        train_dpo.main()


def test_train_dpo_reads_pair_build_tokens_from_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pairs_path = tmp_path / "pairs.jsonl"
    save_pairs(_PAIRS, pairs_path)
    manifest_path = pairs_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps({"tokens_spent": 12345}), encoding="utf-8")
    out_dir = tmp_path / "out"

    monkeypatch.setenv("RUN_ARM", "vf_dpo")
    monkeypatch.setenv("RUN_PAIRS", str(pairs_path))
    monkeypatch.setenv("RUN_STEPS", "1")
    monkeypatch.setenv("RUN_OUT_DIR", str(out_dir))
    monkeypatch.setattr(sys, "argv", ["train_dpo.py", "--tiny"])

    train_dpo.main()

    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["pair_build_tokens"] == 12345
