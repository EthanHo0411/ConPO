"""DPO baseline training entrypoint (vf_dpo | relax_dpo | edge_dpo, locked
decision 2026-09-02(c)). See conpo/pairs.py for pair construction and
scripts/build_pairs.py for the pair-building CLI that produces RUN_PAIRS.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from trl import DPOConfig, DPOTrainer

from conpo import config
from conpo.arms import resolve_arm
from conpo.pairs import Pair, load_pairs

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Default RUN_PAIRS per arm, matching scripts/build_pairs.py's --out
# convention (data/pairs_<mode>.jsonl, mode in {vf, relax, edges}).
_DEFAULT_PAIRS = {
    "vf_dpo": DATA_DIR / "pairs_vf.jsonl",
    "relax_dpo": DATA_DIR / "pairs_relax.jsonl",
    "edge_dpo": DATA_DIR / "pairs_edges.jsonl",
}


def _tiny_model_and_tokenizer() -> tuple[torch.nn.Module, AutoTokenizer]:
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


def _real_model_and_tokenizer() -> tuple[torch.nn.Module, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(config.POLICY_MODEL)
    # UNTESTED: no GPU in this environment, see scripts/train.py's own note.
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(config.POLICY_MODEL, dtype=dtype)
    return model, tokenizer


def _dataset_from_pairs(pairs: list[Pair]) -> Dataset:
    """Conversational prompt/chosen/rejected columns. `chat_template_kwargs`
    is a per-example column trl.DPOTrainer._prepare_dataset's tokenize_fn
    reads (`example.get("chat_template_kwargs", {})`) and merges into every
    apply_chat_template call, the hook that applies
    config.POLICY_CHAT_TEMPLATE_KWARGS consistently for prompt, chosen and
    rejected tokenization alike -- DPOConfig itself has no top-level
    chat_template_kwargs field (unlike GRPOConfig in scripts/train.py).
    """
    return Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": p.prompt}],
                "chosen": [{"role": "assistant", "content": p.chosen}],
                "rejected": [{"role": "assistant", "content": p.rejected}],
                "chat_template_kwargs": dict(config.POLICY_CHAT_TEMPLATE_KWARGS),
            }
            for p in pairs
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny", action="store_true", help="tiny random model + 4 pairs, for smoke tests")
    args = parser.parse_args()

    run_start = time.monotonic()
    arm_name = config.env("RUN_ARM", "vf_dpo")
    arm, _overrides = resolve_arm(arm_name)
    if arm.trainer != "dpo":
        raise ValueError(f"RUN_ARM={arm_name!r} is a {arm.trainer!r} arm; scripts/train_dpo.py trains 'dpo' arms only")

    pairs_path = Path(config.env("RUN_PAIRS", str(_DEFAULT_PAIRS[arm_name])))
    seed = int(config.env("RUN_SEED", "0"))
    steps = int(config.env("RUN_STEPS", "1"))
    out_dir = Path(config.env("RUN_OUT_DIR", "runs/tmp"))
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)

    pairs = load_pairs(pairs_path)
    if args.tiny:
        pairs = pairs[:4]
    if not pairs:
        raise ValueError(f"no pairs loaded from {pairs_path}")
    dataset = _dataset_from_pairs(pairs)

    model, tokenizer = _tiny_model_and_tokenizer() if args.tiny else _real_model_and_tokenizer()
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=list(config.LORA_TARGET_MODULES),
        task_type="CAUSAL_LM",
    )

    # DPO batch sizing is not paper-specified (config.EFFECTIVE_BATCH is a
    # GRPO per-prompt-group concept, not applicable to a flat pairs
    # dataset); a small fixed batch is the DEV_SET default pending a real-GPU
    # tuning pass, --tiny shrinks it further for smoke-test speed.
    per_device_batch = 2 if args.tiny else 4
    # --tiny's RUN_STEPS default (1) makes num_warmup_steps == max_steps under
    # config.WARMUP_RATIO (ceil(0.03 * 1) == 1): the whole run falls inside the
    # warmup ramp and the single optimizer step applies at LR == 0. Warmup only
    # matters for a real multi-hundred-step run, so --tiny skips it.
    warmup_ratio = 0.0 if args.tiny else config.WARMUP_RATIO
    dpo_config = DPOConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=per_device_batch,
        gradient_checkpointing=False,
        max_steps=steps,
        learning_rate=config.LR,
        beta=config.DPO_BETA,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=config.LR_SCHEDULER,
        max_grad_norm=config.MAX_GRAD_NORM,
        seed=seed,
        report_to=[],
        logging_steps=1,
        use_cpu=not torch.cuda.is_available(),
        # UNTESTED: no GPU in this environment, see _real_model_and_tokenizer's dtype note above.
        bf16=torch.cuda.is_available(),
        fp16=False,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(str(out_dir / "adapter"))

    manifest_path = pairs_path.with_suffix(".manifest.json")
    pair_build_tokens = 0
    if manifest_path.is_file():
        pair_build_tokens = json.loads(manifest_path.read_text(encoding="utf-8")).get("tokens_spent", 0)

    wall_hours = (time.monotonic() - run_start) / 3600
    metadata = {
        "arm": arm_name,
        "pairs_path": str(pairs_path),
        "n_pairs": len(pairs),
        "seed": seed,
        "steps": steps,
        "beta": config.DPO_BETA,
        "learning_rate": config.LR,
        "pair_build_tokens": pair_build_tokens,
        "actual_h100h": wall_hours,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(
        f"done: {steps} step(s), arm={arm_name}, pairs={len(pairs)}, "
        f"pair_build_tokens={pair_build_tokens}, out_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
