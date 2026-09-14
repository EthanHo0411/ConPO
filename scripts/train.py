"""ConPO training entrypoint. See conpo/trainer.py (ConPOTrainer) and conpo/arms.py (ArmSpec).

RUN_FULL_FT=1 (design choice, builder, 2026-09-04, read via config.env
inside main()): trains every weight instead of a LoRA adapter (peft_config=
None), for the full-FT point-check (local/train_fullft.sh). Default "0"
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
from conpo.prompts import cap_pool_rows
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


def _find_checkpoints(out_dir: Path) -> list[tuple[Path, int]]:
    """Every `checkpoint-*` directory directly under `out_dir`, paired with
    its optimiser step (from `trainer_state.json`'s `global_step`, the same
    file TRL's own `Trainer.train(resume_from_checkpoint=...)` reads; 0 for a
    directory missing that file -- still counted as "a checkpoint exists"
    for the fresh-run guard below). Empty when `out_dir` holds none.
    """
    found: list[tuple[Path, int]] = []
    for candidate in sorted(out_dir.glob("checkpoint-*")):
        if not candidate.is_dir():
            continue
        state_file = candidate / "trainer_state.json"
        step = json.loads(state_file.read_text(encoding="utf-8"))["global_step"] if state_file.exists() else 0
        found.append((candidate, step))
    return found


def _resolve_resume(out_dir: Path, run_resume: bool) -> tuple[Path | None, int | None]:
    """RUN_RESUME knob (devbox 2026-09-11): the node-local disk is wiped
    about every 48 hours, so a run must be resumable from its newest
    harvested checkpoint rather than restarted cold on top of it.

    RUN_RESUME unset/0 (fresh run): fails loud if `out_dir` already holds a
    checkpoint -- a wiped-and-restarted job must never silently start over
    on data a prior process already checkpointed there. RUN_RESUME=1:
    fails loud if `out_dir` holds no checkpoint at all, otherwise returns
    the newest one (by step) and its step number for `trainer.train(
    resume_from_checkpoint=...)` and `run_metadata.json`'s `resumed_from_step`.
    """
    checkpoints = _find_checkpoints(out_dir) if out_dir.exists() else []
    if run_resume:
        if not checkpoints:
            raise SystemExit(f"RUN_RESUME=1 but no checkpoint-* directory found under {out_dir}")
        checkpoint_dir, step = max(checkpoints, key=lambda pair: pair[1])
        return checkpoint_dir, step
    if checkpoints:
        raise SystemExit(
            f"RUN_OUT_DIR {out_dir} already holds {len(checkpoints)} checkpoint-* director"
            f"{'y' if len(checkpoints) == 1 else 'ies'}; refusing to start a fresh run on top of it "
            "(a wiped-and-restarted job must not silently start over on harvested data). Set "
            "RUN_RESUME=1 to resume from it, or point RUN_OUT_DIR at an empty directory."
        )
    return None, None


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


class SaveCadenceCallback(TrainerCallback):
    """Re-asserts the launch-time checkpoint cadence on resume. transformers
    copies args.save_steps into TrainerState, then replaces the state with the
    checkpoint's trainer_state.json, and DefaultFlowCallback saves on
    state.save_steps: a run resumed with a different RUN_SAVE_STEPS kept the
    old cadence (observed 2026-09-11: full_s42 resumed from checkpoint-25 with
    RUN_SAVE_STEPS=15 saved nothing at step 30). on_train_begin runs after
    that reload.
    """

    def on_train_begin(self, args: object, state: object, control: object, **kwargs: object) -> None:
        state.save_steps = args.save_steps  # type: ignore[attr-defined]
        state.logging_steps = args.logging_steps  # type: ignore[attr-defined]


class StopAtStepCallback(TrainerCallback):
    """Halts training once state.global_step reaches stop_at_step, saving a
    checkpoint at that step first (RUN_STOP_AT_STEP). transformers' inner
    loop runs on_step_end, then _maybe_log_save_evaluate (which honours
    control.should_save and writes checkpoint-<global_step> via the standard
    path), then checks control.should_training_stop -- so setting both here
    saves-then-stops at the same step. max_steps is left untouched by the
    caller, so a checkpoint saved this way keeps the full run's cosine
    schedule for a later resume.
    """

    def __init__(self, stop_at_step: int) -> None:
        self.stop_at_step = stop_at_step

    def on_step_end(self, args: object, state: object, control: object, **kwargs: object) -> None:
        if state.global_step >= self.stop_at_step:  # type: ignore[attr-defined]
            control.should_save = True  # type: ignore[attr-defined]
            control.should_training_stop = True  # type: ignore[attr-defined]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny", action="store_true", help="tiny random model + 4 rows, for smoke tests")
    args = parser.parse_args()

    arm_name = config.env("RUN_ARM", "full")
    seed = int(config.env("RUN_SEED", "0"))
    steps = int(config.env("RUN_STEPS", "1"))
    use_vllm = config.env("RUN_USE_VLLM", "0") == "1"
    # RUN_VLLM_PUSH_ONCE_PER_STEP (L1, devbox perf pass): read here only to
    # record the effective value in run_metadata below -- conpo/trainer.py's
    # _install_vllm_reload_guard reads the same env var itself, directly, at
    # guard-install time.
    vllm_push_once_per_step = config.env("RUN_VLLM_PUSH_ONCE_PER_STEP", "1") == "1"
    # RUN_REPAIR_BATCH_GENERATION (L2, devbox perf pass): same pattern --
    # read here only to record the effective value in run_metadata below;
    # conpo/trainer.py's ConPOTrainer.__init__ reads the same env var
    # itself, directly.
    repair_batch_generation = config.env("RUN_REPAIR_BATCH_GENERATION", "1") == "1"
    # RUN_CACHE_REF_LOGPROBS (L3, devbox perf pass): same pattern -- read
    # here only to record the effective value in run_metadata below;
    # conpo/trainer.py's ConPOTrainer.__init__ reads the same env var
    # itself, directly.
    cache_ref_logprobs = config.env("RUN_CACHE_REF_LOGPROBS", "1") == "1"
    model_name = config.env("RUN_MODEL", config.POLICY_MODEL)
    veto_period_k = int(config.env("RUN_VETO_PERIOD", str(config.VETO_PERIOD_K)))
    # RUN_SAVE_STEPS (design choice, builder, 2026-09-11): overrides
    # config.SAVE_STEPS the same way RUN_VETO_PERIOD overrides
    # config.VETO_PERIOD_K -- config.py's locked value is untouched, this
    # only lets a CPU test force a checkpoint every step instead of every 25.
    save_steps = int(config.env("RUN_SAVE_STEPS", str(config.SAVE_STEPS)))
    # RUN_STOP_AT_STEP (devbox 2026-09-13): when set, StopAtStepCallback
    # (below) halts training and saves a checkpoint once state.global_step
    # reaches this value, without changing max_steps (`steps` below) -- so a
    # checkpoint resumed later still follows the full run's cosine schedule.
    stop_at_step_raw = config.env("RUN_STOP_AT_STEP", "")
    stop_at_step = int(stop_at_step_raw) if stop_at_step_raw else None
    # RUN_TAU / RUN_TAU_P: same override pattern as RUN_VETO_PERIOD /
    # RUN_SAVE_STEPS above, but for config.TAU / config.TAU_P directly --
    # conpo/trainer.py's compute_loss and conpo/losses.py's resp_loss /
    # prompt_loss defaults both read these two module attributes fresh at
    # call time (no importer binds them earlier), so assigning here before
    # ConPOTrainer is constructed is enough to take effect for the whole run.
    tau = float(config.env("RUN_TAU", str(config.TAU)))
    tau_p = float(config.env("RUN_TAU_P", str(config.TAU_P)))
    config.TAU = tau
    config.TAU_P = tau_p
    # RUN_KAPPA (devbox 2026-09-13): same pattern for config.KAPPA, the
    # resp_loss retirement bar and the saturated-edge eviction threshold
    # (kappa * tau), read fresh at call time by conpo/trainer.py.
    kappa = float(config.env("RUN_KAPPA", str(config.KAPPA)))
    config.KAPPA = kappa
    # RUN_LOSS_TYPE (devbox 2026-09-13): same pattern for config.LOSS_TYPE,
    # read fresh by the GRPOConfig(...) call below (loss_type=config.LOSS_TYPE).
    loss_type = config.env("RUN_LOSS_TYPE", config.LOSS_TYPE)
    config.LOSS_TYPE = loss_type
    # RUN_PAIR_LOSS_WEIGHT (devbox 2026-09-13): same pattern for
    # config.PAIR_LOSS_WEIGHT, the multiplier on the summed pair losses read
    # fresh at call time by conpo/trainer.py's compute_loss.
    pair_loss_weight = float(config.env("RUN_PAIR_LOSS_WEIGHT", str(config.PAIR_LOSS_WEIGHT)))
    config.PAIR_LOSS_WEIGHT = pair_loss_weight
    # RUN_VLLM_IS_MODE (devbox 2026-09-13): pure passthrough to TRL's own
    # GRPOConfig.vllm_importance_sampling_mode (default "sequence_mask") --
    # unlike RUN_LOSS_TYPE this has no ConPO-internal config.py counterpart,
    # so it is applied directly to grpo_config below only when set, after
    # the GRPOConfig(...) call closes.
    vllm_is_mode = config.env("RUN_VLLM_IS_MODE", "")
    out_dir = Path(config.env("RUN_OUT_DIR", "runs/tmp"))
    run_resume = config.env("RUN_RESUME", "0") == "1"
    resume_checkpoint_dir, resumed_from_step = _resolve_resume(out_dir, run_resume)
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

    # TASK G (design choice, builder, 2026-09-11): TRL's GRPOConfig has no
    # max_prompt_length in this trl version, so nothing upstream ever caps a
    # prompt before it reaches an engine -- cap every row's prompt here, once,
    # before rows_by_key/dataset are built, so every downstream consumer
    # (TRL generation via `dataset` below, conpo/forward.py's repair prompts,
    # conpo/reverse.py's corruption and conpo/prompts.counterfactual_prompts_set,
    # conpo/judge.py's veto prompt) reads the same already-capped
    # `row.prompt` (conpo/prompts.py's cap_prompt_messages/cap_pool_rows).
    # A throwaway tokenizer load here (the real one loads again below with
    # the model) keeps this independent of _real_model_and_tokenizer/
    # _tiny_model_and_tokenizer's own tokenizer, at the cost of loading it
    # twice -- negligible next to the model load either path takes.
    cap_tokenizer = AutoTokenizer.from_pretrained(config.POLICY_MODEL if args.tiny else model_name)
    rows, prompt_truncated_count = cap_pool_rows(rows, cap_tokenizer, config.MAX_PROMPT_TOKENS)
    print(
        f"prompt cap: {prompt_truncated_count}/{len(rows)} row(s) truncated to "
        f"<= {config.MAX_PROMPT_TOKENS} tokens (config.MAX_PROMPT_TOKENS)"
    )

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
        # Checkpoint every SAVE_STEPS optimiser steps (equal to the veto
        # period, so each save follows a veto flush) and keep the last two:
        # a production run outlives the node-local disk's 48-hour wipe only
        # through checkpoints harvested to NFS (local/harvest.sh).
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
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
    if vllm_is_mode:
        grpo_config.vllm_importance_sampling_mode = vllm_is_mode

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
    callbacks: list[TrainerCallback] = [pass1_callback, SaveCadenceCallback()]
    if stop_at_step is not None:
        callbacks.append(StopAtStepCallback(stop_at_step))

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
        callbacks=callbacks,
        resume_from=resume_checkpoint_dir,
    )
    pass1_callback.trainer = trainer

    trainer.train(resume_from_checkpoint=str(resume_checkpoint_dir) if resume_checkpoint_dir is not None else None)
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
    # RUN_RESUME: cumulative_wall_seconds() adds this process's own elapsed
    # time (since trainer construction) on top of whatever a prior process's
    # conpo_state.json recorded, so actual_h100h reflects the whole run
    # across every resumed process, not only this one.
    wall_hours = trainer.cumulative_wall_seconds() / 3600
    run_metadata = {
        "arm": arm_name,
        "seed": seed,
        "model": model_name,
        "dataset_size": len(rows),
        "prompts_per_update": prompts_per_update,
        "subset_path": subset_path or None,
        "subset_sha256": subset_sha256,
        # TASK G: how many of this run's rows had their prompt left-truncated
        # to config.MAX_PROMPT_TOKENS at load time (cap_pool_rows above).
        "prompt_truncated_rows": prompt_truncated_count,
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
        # RUN_TAU / RUN_TAU_P (devbox 2026-09-12): effective config.TAU /
        # config.TAU_P this run trained with, config.py's defaults unless
        # overridden.
        "tau": tau,
        "tau_p": tau_p,
        "kappa": kappa,
        "loss_type": loss_type,
        "pair_loss_weight": pair_loss_weight,
        # RUN_VLLM_IS_MODE / RUN_KL_CLAMP / RUN_RESAMPLE_ZERO_STD /
        # RUN_RESAMPLE_MAX_ROUNDS / RUN_TOKEN_LOGRATIO_CLAMP (K2/K4/K5/K6,
        # fixer review 2026-09-13): these four knobs reached the trainer but
        # not run_metadata.json, so a finished run's artefacts alone could
        # not say which of them were active for it (same reasoning as the
        # judge_mode/judge_model/veto_period_k comment above). kl_clamp,
        # resample_zero_std, resample_max_rounds and token_logratio_clamp
        # are read back off the constructed trainer instance rather than
        # re-parsed here, since main() never binds their raw env values to a
        # local variable the way it does for loss_type/pair_loss_weight.
        "vllm_is_mode": vllm_is_mode or None,
        "kl_clamp": trainer._kl_clamp,
        "resample_zero_std": trainer._resample_zero_std,
        "resample_max_rounds": trainer._resample_max_rounds,
        "token_logratio_clamp": trainer._token_logratio_clamp,
        # RUN_PAIR_OBJECTIVE / RUN_POS_ANCHOR_WEIGHT (K8, devbox 2026-09-14):
        # same "read back off the trainer instance" pattern as the four
        # knobs above.
        "pair_objective": trainer._pair_objective,
        "pos_anchor_weight": trainer._pos_anchor_weight,
        # RUN_PAIR_SCALE_AUTO (fix devbox 2026-09-14): pair_scale_auto is
        # read back off the trainer instance (same pattern as above);
        # pair_scale_factor is the steady-state divisor conpo/trainer.py's
        # compute_loss applies (1/gradient_accumulation_steps when on, 1.0
        # when off) -- the per-micro-batch exact value is also on record via
        # last_loss_stats/the trainer's own metrics logging (stats's
        # "resp_pair_scale").
        "pair_scale_auto": trainer._pair_scale_auto,
        "pair_scale_factor": (1.0 / gradient_accumulation_steps) if trainer._pair_scale_auto else 1.0,
        "row_limit": row_limit or None,
        # RUN_RESUME (devbox 2026-09-11): the checkpoint step this run
        # resumed from, or None for a fresh run.
        "resumed_from_step": resumed_from_step,
        # RUN_STOP_AT_STEP (devbox 2026-09-13): step this run was told to
        # halt at, or None when unset (max_steps stays the full run's).
        "stop_at_step": stop_at_step,
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
        # RUN_VLLM_PUSH_ONCE_PER_STEP (L1, devbox perf pass): whether this
        # run pushed the policy into vLLM once per step (default) instead of
        # before every generate() call.
        "vllm_push_once_per_step": vllm_push_once_per_step,
        # RUN_REPAIR_BATCH_GENERATION (L2, devbox perf pass): whether this
        # run merged every failing group's forward-repair engine.generate
        # calls into one call per repair round for the whole step (default)
        # instead of one repair_failed_group call per group.
        "repair_batch_generation": repair_batch_generation,
        # RUN_CACHE_REF_LOGPROBS (L3, devbox perf pass): whether this run
        # cached an edge's x^-A reference log-probs on first draw (default)
        # instead of recomputing them from the reference model on every
        # replay (conpo/trainer.py's _prompt_margins).
        "cache_ref_logprobs": cache_ref_logprobs,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    print(f"done: {steps} step(s), arm={arm_name}, tokens={meter.total}, out_dir={out_dir}")


if __name__ == "__main__":
    main()
