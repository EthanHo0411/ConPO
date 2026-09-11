"""Export open-instruct's saved LoRA adapter into a clean PEFT-format directory for ConPO's eval.

grpo_fast.py's `save_model` (patched by patches/lora_peft.patch) already writes standard PEFT
format directly into `--output_dir` when `--use_peft` was set: `adapter_config.json` +
`adapter_model.safetensors`. `--deepspeed_stage 2` keeps parameters unpartitioned (ZeRO-2 only
shards optimizer state and gradients), so no ZeRO-3 gather step is needed and no
`adapter_model.bin` gets written either (that branch is `if self.stage == 3`). See README.md's
"LoRA" section for the file:line evidence.

This script copies just those two files into a clean destination (dropping the run's full
tokenizer files and `.checkpoint_complete` marker that also live in `--output_dir`) and, unless
`--skip-verify`, loads the result on CPU with PEFT exactly as ConPO/scripts/eval_external.py's
`_load_hf_engine` does, to prove the exported directory is a working adapter, not just present
files.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

CONPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CONPO_ROOT))

from conpo import config  # noqa: E402

ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHT_NAMES = ("adapter_model.safetensors", "adapter_model.bin")


def export(run_dir: Path, dest_dir: Path) -> list[str]:
    """Copy the adapter's two files from `run_dir` (a training --output_dir) into `dest_dir`.

    Returns the copied file names. Raises FileNotFoundError with a specific reason if the run
    directory does not look like a PEFT run's save_model output.
    """
    config_path = run_dir / ADAPTER_CONFIG_NAME
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} not found; grpo_fast.py's save_model only writes it when "
            "--use_peft was set on the training run (see README.md's LoRA section)."
        )
    weight_path = next((run_dir / name for name in ADAPTER_WEIGHT_NAMES if (run_dir / name).exists()), None)
    if weight_path is None:
        raise FileNotFoundError(f"no {' or '.join(ADAPTER_WEIGHT_NAMES)} found next to {config_path}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in (config_path, weight_path):
        dst = dest_dir / src.name
        shutil.copyfile(src, dst)
        copied.append(dst.name)
    return copied


def verify_on_cpu(dest_dir: Path, base_model: str) -> dict[str, Any]:
    """Load `dest_dir` with PEFT on CPU, the same call `scripts/eval_external.py`'s
    `_load_hf_engine` makes, to prove the exported directory is a working adapter."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    peft_model = PeftModel.from_pretrained(model, str(dest_dir))
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    return {"loaded": True, "trainable_params": trainable, "total_params": total}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Training --output_dir to export from.")
    parser.add_argument("--dest", type=Path, required=True, help="Clean adapter output directory to write.")
    parser.add_argument("--base-model", default=config.POLICY_MODEL, help="Base model for the CPU verify load.")
    parser.add_argument("--skip-verify", action="store_true", help="Skip the PEFT-on-CPU load check.")
    args = parser.parse_args(argv)

    copied = export(args.run_dir, args.dest)
    result: dict[str, Any] = {"copied": copied, "dest": str(args.dest)}
    if not args.skip_verify:
        result["verify"] = verify_on_cpu(args.dest, args.base_model)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
