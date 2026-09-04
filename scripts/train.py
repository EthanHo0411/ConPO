"""ConPO training entrypoint. See conpo/trainer.py (ConPOTrainer) and conpo/arms.py (ArmSpec).

RUN_FULL_FT=1 (design choice, builder, 2026-09-04, read via config.env
inside main()): trains every weight instead of a LoRA adapter (peft_config=
None), for the full-FT point-check (slurm/train_fullft.sbatch). Default "0"
(LoRA, every other run). Recorded as run_metadata.json's "full_ft" field.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import random
import shutil
import tempfile
import time
from pathlib import Path

import torch
from accelerate import PartialState
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM, TrainerCallback
from trl import GRPOConfig

import conpo.operators  # noqa: F401  registers all corruption operators
from conpo import config
from conpo.arms import resolve_arm
from conpo.data import PoolRow, TokenMeter, keys_digest, load_subset
from conpo.edges import EdgeBuffer, FeasibleBuffer
from conpo.engine import Engine, HFEngine, VLLMColocateEngine
from conpo.judge import build_judge
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


def _load_training_rows(data_path: Path, subset_path: str, tiny: bool) -> list[PoolRow]:
    """Rows for one training run.

    RUN_SUBSET (`subset_path` non-empty): the full pool is read first and
    reordered to the subset's key order, THEN `tiny` truncates to the first
    4 keys of that order -- truncating the raw pool file first (the old
    behaviour) would almost always miss every subset key and load_subset
    would raise KeyError. No RUN_SUBSET: a plain read of `data_path`, capped
    at 4 rows for `tiny`.
    """
    if subset_path:
        rows = load_subset(_load_rows(data_path, limit=None), subset_path)
        return rows[:4] if tiny else rows
    return _load_rows(data_path, limit=4 if tiny else None)


def _report_subset_prompts_per_update(dataset_size: int, prompts_per_update: int) -> None:
    """Prints a note when a RUN_SUBSET's dataset_size is not an exact
    multiple of prompts_per_update: a RUN_SUBSET run pins TRL's
    RepeatSampler to shuffle_dataset=False with batch_size ==
    prompts_per_update, and RepeatSampler drops any trailing chunk shorter
    than a full batch on every epoch (trl/trainer/utils.py). scripts/
    draw_subset.py draws exactly --size keys with no rounding, so a
    remainder here is expected whenever --size isn't a multiple of
    prompts_per_update, not a sign of a hand-edited or corrupted subset --
    see its module docstring for why the dropped tail is negligible.
    """
    remainder = dataset_size % prompts_per_update
    if remainder:
        print(
            f"note: RUN_SUBSET dataset_size ({dataset_size}) is not a multiple of "
            f"prompts_per_update ({prompts_per_update}); TRL's RepeatSampler will drop the "
            f"trailing {remainder} row(s) every epoch (negligible)."
        )


def _run_shape(tiny: bool) -> tuple[int, int, int]:
    """(num_generations, max_completion_length, effective_batch) for one run.

    --tiny shrinks generation shape for a fast CPU smoke test and ignores
    every override below. Otherwise RUN_NUM_GENERATIONS/
    RUN_MAX_COMPLETION_TOKENS/RUN_EFFECTIVE_BATCH (smoke and shape tests
    only, design 2026-09-04: cluster runs never set them) override
    config.py's NUM_GENERATIONS/MAX_COMPLETION_TOKENS/EFFECTIVE_BATCH so a
    real-model dry run fits a small GPU.
    """
    if tiny:
        return 2, 16, config.EFFECTIVE_BATCH
    num_generations = int(config.env("RUN_NUM_GENERATIONS", str(config.NUM_GENERATIONS)))
    max_completion_length = int(config.env("RUN_MAX_COMPLETION_TOKENS", str(config.MAX_COMPLETION_TOKENS)))
    effective_batch = int(config.env("RUN_EFFECTIVE_BATCH", str(config.EFFECTIVE_BATCH)))
    return num_generations, max_completion_length, effective_batch


def _check_dataset_size(dataset_size: int, prompts_per_update: int) -> None:
    """Fail loud (design 2026-09-04) instead of silently training nothing.

    TRL's RepeatSampler (trl/trainer/utils.py) never emits a batch shorter
    than prompts_per_update, so a dataset smaller than that drops every row
    on every epoch -- the RUN_LIMIT=8 / prompts_per_update=16 case where
    scripts/train.py still printed "done: N step(s)" with zero rows trained.
    """
    if dataset_size < prompts_per_update:
        raise SystemExit(
            f"dataset_size ({dataset_size}) is smaller than prompts_per_update ({prompts_per_update}); "
            "TRL's RepeatSampler would drop every row every epoch and train nothing. Raise RUN_LIMIT to "
            f"at least {prompts_per_update}, or lower RUN_EFFECTIVE_BATCH so prompts_per_update <= "
            f"dataset_size ({dataset_size})."
        )


def _tiny_model_and_tokenizer(full_ft: bool) -> tuple[torch.nn.Module, AutoTokenizer]:
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
    if not full_ft:
        return model, tokenizer
    # RUN_FULL_FT=1 only: its reference-model path (trl.GRPOTrainer, beta !=
    # 0, no PEFT adapter to disable) reloads the policy model from
    # model.config._name_or_path (trl.trainer.utils.create_model_from_path);
    # a bare Qwen3ForCausalLM(cfg) leaves that field "". Saving to and
    # reloading from a throwaway local directory gives it a resolvable path
    # (design choice, builder, 2026-09-04) without changing the weights. The
    # LoRA path never touches self.ref_model, so it skips this save/reload.
    # ponytail: atexit.register(shutil.rmtree, ...) accumulates one handler
    # and one temp dir per full-FT --tiny run for the life of the process
    # (harmless, ignore_errors=True); explicit cleanup right after
    # GRPOTrainer.__init__ if this ever runs enough full-FT tests to matter.
    tmp_dir = tempfile.mkdtemp(prefix="conpo_tiny_model_")
    atexit.register(shutil.rmtree, tmp_dir, ignore_errors=True)
    model.save_pretrained(tmp_dir)
    return Qwen3ForCausalLM.from_pretrained(tmp_dir), tokenizer


def _vllm_sleep_mode(use_vllm: bool, judge_mode: str) -> bool:
    """Whether GRPOConfig's vllm_enable_sleep_mode should be on.

    Only worth the sleep/wake round trip TRL pays on every generate() call
    (trl/generation/vllm_generation.py) when the colocated vLLM engine and
    the batched-veto judge contend for the same local GPU memory:
    RUN_USE_VLLM=1 (a colocated engine exists at all) AND RUN_JUDGE_MODE ==
    "batched" (build_judge loads its model locally between bursts). Under
    RUN_JUDGE_MODE == "http" the judge runs on a separate GPU/process over
    HTTP and never competes with the training engine, so there is nothing
    for the sleep/wake dance to protect against.
    """
    return use_vllm and judge_mode == "batched"


def _real_model_and_tokenizer(model_name: str) -> tuple[torch.nn.Module, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # UNTESTED: no GPU in this environment, see engine.py's VLLMColocateEngine docstring
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    return model, tokenizer


class Pass1CheckpointCallback(TrainerCallback):
    """Saves RUN_OUT_DIR/checkpoint_pass1 and pass1.json the first optimiser
    step at which the trainer has consumed every dataset row at least once
    (the prompt-matched comparison point the IF-RLVR arm's checkpoint must
    also land on -- see conpo/experiments.md's token-budget stop rule).

    main()'s per_device_train_batch_size == num_generations makes one
    micro-batch exactly one prompt's completions, so each optimiser update
    consumes exactly `prompts_per_update` distinct prompts and the pass
    boundary is ceil(dataset_size / prompts_per_update) optimiser steps.
    Applies to every arm (only reads state.global_step and the token meter,
    nothing arm-specific); fires at most once.
    """

    def __init__(self, dataset_size: int, prompts_per_update: int) -> None:
        self.trainer: object | None = None
        self.pass1_step = max(1, math.ceil(dataset_size / prompts_per_update))
        self.fired = False
        self._start = time.monotonic()

    def on_step_end(self, args: object, state: object, control: object, **kwargs: object) -> None:
        if self.fired or state.global_step < self.pass1_step:  # type: ignore[attr-defined]
            return
        self.fired = True
        trainer = self.trainer
        assert trainer is not None, "Pass1CheckpointCallback.trainer must be set before training starts"
        out_dir = Path(trainer.args.output_dir)  # type: ignore[attr-defined]
        trainer.save_model(str(out_dir / "checkpoint_pass1"))  # type: ignore[attr-defined]
        payload = {
            "global_step": state.global_step,  # type: ignore[attr-defined]
            "tokens_so_far": trainer.meter.total,  # type: ignore[attr-defined]
            "wall_time_seconds": time.monotonic() - self._start,
        }
        (out_dir / "pass1.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny", action="store_true", help="tiny random model + 4 rows, for smoke tests")
    args = parser.parse_args()

    run_start = time.monotonic()
    arm_name = config.env("RUN_ARM", "full")
    seed = int(config.env("RUN_SEED", "0"))
    steps = int(config.env("RUN_STEPS", "1"))
    use_vllm = config.env("RUN_USE_VLLM", "0") == "1"
    model_name = config.env("RUN_MODEL", config.POLICY_MODEL)
    veto_period_k = int(config.env("RUN_VETO_PERIOD", str(config.VETO_PERIOD_K)))
    out_dir = Path(config.env("RUN_OUT_DIR", "runs/tmp"))
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    torch.manual_seed(seed)

    arm, overrides = resolve_arm(arm_name)

    subset_path = config.env("RUN_SUBSET", "")
    rows = _load_training_rows(DATA_DIR / "train.jsonl", subset_path, args.tiny)
    # RUN_LIMIT: applied after RUN_SUBSET/--tiny, for a smoke run on a small
    # GPU that wants real data shape without the full row count.
    row_limit = int(config.env("RUN_LIMIT", "0"))
    if row_limit:
        rows = rows[:row_limit]
    subset_sha256 = keys_digest(subset_path) if subset_path else None
    rows_by_key = {r.key: r for r in rows}
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": r.prompt}], "key": r.key} for r in rows])

    num_generations, max_completion_length, effective_batch = _run_shape(args.tiny)
    per_device_batch = 2 if args.tiny else num_generations

    # TRL derives generation_batch_size = per_device_train_batch_size *
    # num_processes * gradient_accumulation_steps (trl.GRPOConfig). With
    # per_device_train_batch_size == num_generations (one prompt's completions
    # per micro-batch), accumulation is the only lever left to reach
    # effective_batch (128 by default, paper section 4.1) prompts per
    # optimiser update; left at the trainer default (1) this silently trained
    # at 1 prompt/update instead. --tiny only ever runs a smoke test on 1
    # device, so it stays at accumulation 1.
    # num_processes fetched unconditionally (not only in the non-tiny branch
    # below) so prompts_per_update -- how many dataset rows one optimiser
    # update consumes -- is defined for --tiny too, for Pass1CheckpointCallback.
    # PartialState() needs no trainer or model instance, so it (and everything
    # below through _check_dataset_size) reads before the model/tokenizer load.
    num_processes = PartialState().num_processes
    if args.tiny:
        gradient_accumulation_steps = 1
    else:
        gradient_accumulation_steps = effective_batch // (per_device_batch * num_processes)
        assert per_device_batch * num_processes * gradient_accumulation_steps == effective_batch, (
            f"effective_batch ({effective_batch}) must be an exact multiple of "
            f"per_device_train_batch_size ({per_device_batch}) * num_processes ({num_processes})"
        )
    prompts_per_update = num_processes * gradient_accumulation_steps
    # Moved ahead of the model/tokenizer load (design 2026-09-04): a dataset
    # too small for one optimizer update now fails within seconds instead of
    # after the policy model download/load.
    _check_dataset_size(len(rows), prompts_per_update)
    if subset_path:
        _report_subset_prompts_per_update(len(rows), prompts_per_update)
    if not args.tiny:
        print(
            f"gradient_accumulation_steps={gradient_accumulation_steps} "
            f"(num_processes={num_processes}, per_device_train_batch_size={per_device_batch}) "
            f"-> {prompts_per_update} prompt(s) per optimizer update"
        )

    # RUN_FULL_FT=1 (design choice, builder, 2026-09-04): trains every weight
    # (peft_config=None below, no adapter_config.json under out_dir/adapter).
    # Reference scoring then runs under TRL's own self.ref_model instead of a
    # disabled adapter (see conpo/trainer.py's _reference_model), populated
    # automatically since config.KL_BETA != 0 and the model isn't PEFT-wrapped.
    full_ft = config.env("RUN_FULL_FT", "0") == "1"
    model, tokenizer = _tiny_model_and_tokenizer(full_ft) if args.tiny else _real_model_and_tokenizer(model_name)
    lora_config = None if full_ft else LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=list(config.LORA_TARGET_MODULES),
        task_type="CAUSAL_LM",
    )

    judge_mode = config.env("RUN_JUDGE_MODE", getattr(config, "JUDGE_MODE_DEFAULT", "batched"))
    # Same resolution formula as conpo/judge.py:build_judge itself, only for
    # run_metadata.json's own record of the run -- build_judge re-reads
    # RUN_JUDGE_MODEL from the environment independently, so this never
    # drives which judge gets built, only what gets written down.
    judge_model = config.env("RUN_JUDGE_MODEL", "") or config.JUDGE_MODEL

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
        # Batched veto (USER DECISIONS 2026-09-02): sleep the colocated vLLM
        # engine around each veto burst (ConPOTrainer.flush_pending_veto)
        # rather than keep its KV cache resident while the judge scores.
        # Only on when the judge loads locally (RUN_JUDGE_MODE=
        # "batched"): under "http" the judge runs on a separate GPU/process
        # over HTTP, so the sleep/wake round trip would be pure overhead
        # (see _vllm_sleep_mode). UNTESTED: no GPU in this environment.
        vllm_enable_sleep_mode=_vllm_sleep_mode(use_vllm, judge_mode),
        max_steps=steps,
        seed=seed,
        report_to=[],
        logging_steps=1,
        # A RUN_SUBSET was already shuffled once (its own seed) and every arm
        # must see it in that same committed order; TRL's own per-epoch
        # shuffle (keyed on RUN_SEED, which differs per arm) would break
        # that. No subset -> no order is committed, so leave TRL's default.
        shuffle_dataset=(subset_path == ""),
        use_cpu=not torch.cuda.is_available(),
        # UNTESTED: no GPU in this environment, see _real_model_and_tokenizer's dtype note above.
        bf16=torch.cuda.is_available(),
        fp16=False,
    )

    # before_load/after_unload are left unset: build_judge runs before the
    # trainer exists, so there is no engine handle to hand it yet.
    # ConPOTrainer.flush_pending_veto sleeps/wakes the colocated vLLM engine
    # around the burst itself instead (_sleep_engine_if_enabled/_wake_engine).
    judge = build_judge(
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )

    def engine_factory(trainer: ConPOTrainer) -> Engine:
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
    pass1_callback = Pass1CheckpointCallback(dataset_size=len(rows), prompts_per_update=prompts_per_update)

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
        veto_period_k=veto_period_k,
        out_dir=out_dir,
        peft_config=lora_config,
        callbacks=[pass1_callback],
    )
    pass1_callback.trainer = trainer

    trainer.train()
    # The fourth flush condition (USER DECISIONS 2026-09-02): end of
    # training, so nothing sits unscored in trainer.pending_queue when the
    # run finishes normally (the other three -- period, queue size, and
    # before a checkpoint save -- run inside ConPOTrainer itself).
    trainer.flush_pending_veto()
    # H6 part B: persists RUN_OUT_DIR/pair_stats.json at train end too (a
    # no-op when RUN_MULTI is off, save_pair_stats's own contract).
    trainer.save_pair_stats()

    trainer.save_model(str(out_dir / "adapter"))
    edge_buffer.save(out_dir / "edges.jsonl")
    (out_dir / "family_stats.json").write_text(json.dumps(stats.snapshot(), indent=2), encoding="utf-8")

    pass1_info: dict[str, object] = {"pass1_step": pass1_callback.pass1_step, "fired": pass1_callback.fired}
    if not pass1_callback.fired:
        # Token budget stopped the run before the first full pass: record
        # how much of it completed instead of the (never-written) pass1.json.
        pass1_info["fraction_completed"] = min(1.0, trainer.state.global_step / pass1_callback.pass1_step)
    wall_hours = (time.monotonic() - run_start) / 3600
    run_metadata = {
        "arm": arm_name,
        "seed": seed,
        "model": model_name,
        "dataset_size": len(rows),
        "prompts_per_update": prompts_per_update,
        "subset_path": subset_path or None,
        "subset_sha256": subset_sha256,
        "pass1": pass1_info,
        "token_budget": meter.budget,
        "tokens_generated": meter.total,
        "actual_h100h": wall_hours * num_processes,
        # Judge placement/model, veto period, and row cap: without these a
        # finished run's artefacts alone cannot say which judge placement it
        # used, what veto period it ran with, whether RUN_JUDGE_MODEL
        # overrode the default judge, or whether RUN_LIMIT truncated its
        # training rows (review4/round2 finding 3).
        "judge_mode": judge_mode,
        "judge_model": judge_model,
        "veto_period_k": veto_period_k,
        "row_limit": row_limit or None,
        # Smoke-shape knobs (design 2026-09-04): what shape this run trained
        # with, whether or not RUN_NUM_GENERATIONS/
        # RUN_MAX_COMPLETION_TOKENS/RUN_EFFECTIVE_BATCH overrode the default.
        "num_generations": num_generations,
        "max_completion_tokens": max_completion_length,
        "effective_batch": effective_batch,
        # H6 part B: whether this run attempted multi-family reverse edges.
        "multi": overrides.multi,
        # RUN_FULL_FT (design choice, builder, 2026-09-04): True when this
        # run trained every weight (peft_config=None above) instead of a
        # LoRA adapter.
        "full_ft": full_ft,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    print(f"done: {steps} step(s), arm={arm_name}, tokens={meter.total}, out_dir={out_dir}")


if __name__ == "__main__":
    main()
