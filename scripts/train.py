"""ConPO training entrypoint. See conpo/trainer.py (ConPOTrainer) and conpo/arms.py (ArmSpec)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from accelerate import PartialState
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from trl import GRPOConfig

import conpo.operators  # noqa: F401  registers all corruption operators
from conpo import config
from conpo.arms import resolve_arm
from conpo.data import PoolRow, TokenMeter
from conpo.edges import EdgeBuffer, FeasibleBuffer
from conpo.engine import HFEngine, VLLMColocateEngine
from conpo.judge import HttpJudge, StaticJudge, resolve_endpoint
from conpo.selection import FamilyStats
from conpo.trainer import ConPOTrainer
from conpo.verifiers import Constraint

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _row_from_json(data: dict) -> PoolRow:
    """Inverse of scripts/build_corpus.py's `_row_to_json`."""
    return PoolRow(
        key=data["key"],
        base_prompt=data["base_prompt"],
        prompt=data["prompt"],
        sentences=tuple(data["sentences"]),
        constraints=tuple(Constraint(c["iid"], c["kwargs"]) for c in data["constraints"]),
        english=data["english"],
        num_constraints=data["num_constraints"],
    )


def _load_rows(path: Path, limit: int | None) -> list[PoolRow]:
    rows: list[PoolRow] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(rows) >= limit:
                break
            rows.append(_row_from_json(json.loads(line)))
    return rows


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
    # UNTESTED: no GPU in this environment, see engine.py's VLLMColocateEngine docstring
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(config.POLICY_MODEL, dtype=dtype)
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny", action="store_true", help="tiny random model + 4 rows, for smoke tests")
    args = parser.parse_args()

    arm_name = config.env("RUN_ARM", "full")
    seed = int(config.env("RUN_SEED", "0"))
    steps = int(config.env("RUN_STEPS", "1"))
    use_vllm = config.env("RUN_USE_VLLM", "0") == "1"
    out_dir = Path(config.env("RUN_OUT_DIR", "runs/tmp"))
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    torch.manual_seed(seed)

    arm, overrides = resolve_arm(arm_name)

    rows = _load_rows(DATA_DIR / "train.jsonl", limit=4 if args.tiny else None)
    rows_by_key = {r.key: r for r in rows}
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": r.prompt}], "key": r.key} for r in rows])

    model, tokenizer = _tiny_model_and_tokenizer() if args.tiny else _real_model_and_tokenizer()
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=list(config.LORA_TARGET_MODULES),
        task_type="CAUSAL_LM",
    )

    # --tiny shrinks generation shape too, so a smoke test stays fast; the
    # real run uses the paper's own sampling shape from config.py.
    num_generations = 2 if args.tiny else config.NUM_GENERATIONS
    max_completion_length = 16 if args.tiny else config.MAX_COMPLETION_TOKENS
    per_device_batch = 2 if args.tiny else num_generations

    # TRL derives generation_batch_size = per_device_train_batch_size *
    # num_processes * gradient_accumulation_steps (trl.GRPOConfig). With
    # per_device_train_batch_size == num_generations (one prompt's completions
    # per micro-batch), accumulation is the only lever left to reach
    # config.EFFECTIVE_BATCH (128, paper section 4.1) prompts per optimiser
    # update; left at the trainer default (1) this silently trained at 1
    # prompt/update instead. --tiny only ever runs a smoke test on 1 device,
    # so it stays at accumulation 1.
    if args.tiny:
        gradient_accumulation_steps = 1
    else:
        num_processes = PartialState().num_processes
        gradient_accumulation_steps = config.EFFECTIVE_BATCH // (per_device_batch * num_processes)
        assert per_device_batch * num_processes * gradient_accumulation_steps == config.EFFECTIVE_BATCH, (
            f"config.EFFECTIVE_BATCH ({config.EFFECTIVE_BATCH}) must be an exact multiple of "
            f"per_device_train_batch_size ({per_device_batch}) * num_processes ({num_processes})"
        )
        prompts_per_update = num_processes * gradient_accumulation_steps
        print(
            f"gradient_accumulation_steps={gradient_accumulation_steps} "
            f"(num_processes={num_processes}, per_device_train_batch_size={per_device_batch}) "
            f"-> {prompts_per_update} prompt(s) per optimizer update"
        )

    grpo_config = GRPOConfig(
        output_dir=str(out_dir),
        num_generations=num_generations,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        temperature=config.TEMPERATURE,
        top_p=config.TOP_P,
        max_completion_length=max_completion_length,
        chat_template_kwargs=config.POLICY_CHAT_TEMPLATE_KWARGS,
        learning_rate=config.LR,
        warmup_ratio=config.WARMUP_RATIO,
        lr_scheduler_type=config.LR_SCHEDULER,
        max_grad_norm=config.MAX_GRAD_NORM,
        beta=config.KL_BETA,
        epsilon=config.EPSILON,
        loss_type=config.LOSS_TYPE,
        use_vllm=use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.25,
        vllm_max_model_length=config.MAX_MODEL_LEN,
        max_steps=steps,
        seed=seed,
        report_to=[],
        logging_steps=1,
        use_cpu=not torch.cuda.is_available(),
        # UNTESTED: no GPU in this environment, see _real_model_and_tokenizer's dtype note above.
        bf16=torch.cuda.is_available(),
        fp16=False,
    )

    judge = StaticJudge(keep=True) if config.env("RUN_NO_VETO", "0") == "1" else HttpJudge(resolve_endpoint(""))

    def engine_factory(trainer: ConPOTrainer):
        if use_vllm:
            # UNTESTED: no GPU in this environment, see engine.py's VLLMColocateEngine docstring
            return VLLMColocateEngine(trainer)
        return HFEngine(
            trainer.accelerator.unwrap_model(trainer.model),
            trainer.processing_class,
            device=str(trainer.accelerator.device),
        )

    edge_buffer = EdgeBuffer()
    feasible_buffer = FeasibleBuffer()
    stats = FamilyStats()
    meter = TokenMeter()
    mechanism_log_path = out_dir / "mechanism.jsonl"

    trainer = ConPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        arm=arm,
        rows_by_key=rows_by_key,
        edge_buffer=edge_buffer,
        feasible_buffer=feasible_buffer,
        stats=stats,
        judge=judge,
        engine_factory=engine_factory,
        meter=meter,
        mechanism_log_path=mechanism_log_path,
        seed=seed,
        overrides=overrides,
        peft_config=lora_config,
    )

    trainer.train()

    trainer.save_model(str(out_dir / "adapter"))
    edge_buffer.save(out_dir / "edges.jsonl")
    (out_dir / "family_stats.json").write_text(json.dumps(stats.snapshot(), indent=2), encoding="utf-8")

    print(f"done: {steps} step(s), arm={arm_name}, tokens={meter.total}, out_dir={out_dir}")


if __name__ == "__main__":
    main()
