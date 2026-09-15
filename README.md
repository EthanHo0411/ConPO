# ConPO

Post-training for verifiable instruction following: repaired/corrupted response pairs trained alongside a clean-generation GRPO objective.

Locked: Qwen3-4B policy every run (later ablations: Qwen3-8B, Llama-3.1-Tulu-3-8B-SFT, Olmo-3-7B-Instruct-SFT, unused today); LoRA r64 alpha128 dropout0.05; 54 constraint families (IFEval 25 + IFBench-Train 29); pool `allenai/IF_multi_constraints_upto5`, eval on `google/IFEval` + held-out `allenai/IFBench_test` (58 disjoint families); official vendored IFEvalG scorers only; judge Qwen3-14B.

Tests: `uv run --extra cpu --extra dev python -m pytest tests/test_verifiers.py -q`

Layout: `conpo/config.py` constants; `conpo/verifiers.py` Constraint type + build/check/describe_failure + family registry; `conpo/vendor/ifevalg/` vendored checkers (see its VENDOR.md); `tests/` suite + kwargs fixture.

## Arms and hypotheses

The full run plan (arms, seeds, H1-H6, run count) lives in
`conpo/experiments.md`. Summary:

| Arm | RUN_ARM / override | Seeds |
|---|---|---|
| full | `RUN_ARM=full` | 42, 43, 44 |
| ifrlvr | `RUN_ARM=ifrlvr` | 42, 43, 44 |
| solve_only | `RUN_ARM=solve_only` | 42 |
| forward_only | `RUN_ARM=forward_only` | 42 |
| reverse_only | `RUN_ARM=reverse_only` | 42 |
| selection ablation | `RUN_ARM=full RUN_SELECTION=uniform\|half` | 42 |
| no-veto ablation | `RUN_ARM=full RUN_NO_VETO=1` | 42 |
| no-exactness ablation | `RUN_ARM=full RUN_NO_EXACTNESS=1` | 42 |
| no-distance ablation | `RUN_ARM=full RUN_NO_DISTANCE=1` | 42 |
| no-prompt-loss ablation | `RUN_ARM=full RUN_LAMBDA_P=0` | 42 |

16 runs answer the hypothesis graph (H1-H6), plus 2 baselines outside it
(inference-time verify-and-repair; the IF-RLVR paper's own literal
hyperparameters as a labelled replication), plus 2 model-ablation runs
(`Qwen/Qwen3-8B`, `allenai/Olmo-3-7B-Instruct-SFT`, `RUN_ARM=full` only),
plus 1 full fine-tune point-check -- 21 runs total.

## The RUN_* contract

Every training and eval knob is an environment variable read by
`config.env(name, default)`, so `local/train.sh` and `local/eval.sh`
export env only and never retype a value `scripts/train.py` can read from
`conpo/config.py` itself.

| Variable | Values | Read by |
|---|---|---|
| `RUN_ARM` | `full` \| `ifrlvr` \| `solve_only` \| `forward_only` \| `reverse_only` | `scripts/train.py` -> `conpo/arms.py:resolve_arm` (`ARMS` dict) |
| `RUN_SEED` | integer | `scripts/train.py`, default `0` there (`local/train.sh` defaults it to `42`) |
| `RUN_STEPS` | integer, optimiser-step budget | `scripts/train.py`, default `1` there (`local/train.sh` defaults it to `500`) |
| `RUN_TOKEN_BUDGET` | integer, generated-token budget, `0` = no cap | `conpo/data.py:TokenMeter` |
| `RUN_USE_VLLM` | `1` sample through vLLM, `0` HF generate | `scripts/train.py`, default `0` there (`local/train.sh` defaults it to `1`, overridable). B1 (audit 2026-09-14): forward-repair candidates sampled through the colocated vLLM engine (`conpo/engine.py:VLLMColocateEngine`) no longer carry a per-call seed, so they are reproducible only from vLLM's own run-level generator, not from `RUN_SEED` alone; recorded as `run_metadata.json`'s `vllm_repair_sampling` (`"unseeded"` when set, `null` under HF generate). |
| `RUN_JUDGE_ENDPOINT` | base URL, used verbatim | `conpo/judge.py:resolve_endpoint` (`RUN_JUDGE_MODE=http` only) |
| `RUN_JUDGE_MODE` | `batched` (default, `config.JUDGE_MODE_DEFAULT`) \| `http` | `scripts/train.py:build_judge`, also written to `run_metadata.json` |
| `RUN_JUDGE_MODEL` | model name, default `config.JUDGE_MODEL` | `scripts/train.py:build_judge`, also written to `run_metadata.json` |
| `RUN_VETO_PERIOD` | integer, default `config.VETO_PERIOD_K` (25) | `scripts/train.py` -> `ConPOTrainer(veto_period_k=...)`, also written to `run_metadata.json` |
| `RUN_MODEL` | model name, default `config.POLICY_MODEL` | `scripts/train.py`, also written to `run_metadata.json` |
| `RUN_LIMIT` | integer, `0` (default) = no cap | `scripts/train.py`, caps training rows after `RUN_SUBSET`/`--tiny`, also written to `run_metadata.json` |
| `RUN_OUT_DIR` | output directory | `scripts/train.py`, default `runs/tmp` there |
| `RUN_SELECTION` | `uniform` \| `half` (default `transfer`, the paper's active selection) | `conpo/arms.py:resolve_arm` -> `Overrides.selection` |
| `RUN_NO_VETO` | `1` | `conpo/arms.py:resolve_arm` -> `Overrides.no_veto`; `scripts/train.py:build_judge` returns `StaticJudge(keep=True)` |
| `RUN_NO_EXACTNESS` | `1` | `conpo/arms.py:resolve_arm` -> `Overrides.no_exactness` |
| `RUN_NO_DISTANCE` | `1` | `conpo/arms.py:resolve_arm` -> `Overrides.no_distance` |
| `RUN_LAMBDA_P` | float, default `config.LAMBDA_P` (1.0) | `conpo/arms.py:resolve_arm` -> `Overrides.lambda_p` |
| `RUN_MULTI` | `1` | `conpo/arms.py:resolve_arm` -> `Overrides.multi`; `conpo/trainer.py` then owns a `conpo.compositor.PairStats` and passes it to `conpo/reverse.py:reverse_batch`, which attempts a multi-family edge on a share of its draws (`config.MULTI_EDGE_SHARE`) instead of always drawing one family (H6) |
| `RUN_NUM_GENERATIONS` | integer, default `config.NUM_GENERATIONS` (8) | `scripts/train.py`, also written to `run_metadata.json` as `num_generations`. Smoke and shape tests only -- cluster runs never set this. |
| `RUN_MAX_COMPLETION_TOKENS` | integer, default `config.MAX_COMPLETION_TOKENS` (1024) | `scripts/train.py`, also written to `run_metadata.json` as `max_completion_tokens`. Smoke and shape tests only -- cluster runs never set this. |
| `RUN_EFFECTIVE_BATCH` | integer, default `config.EFFECTIVE_BATCH` (128) | `scripts/train.py`, also written to `run_metadata.json` as `effective_batch`. Smoke and shape tests only -- cluster runs never set this. |
| `RUN_FULL_FT` | `1` | `scripts/train.py`, default `0` there: `1` trains every weight (`peft_config=None`) instead of a LoRA adapter, for the full-FT point-check (`local/train_fullft.sh`); recorded as `run_metadata.json`'s `full_ft`. Reference scoring runs under TRL's own `self.ref_model` instead of a disabled adapter (`conpo/trainer.py`'s `_reference_model`); see "What is verified on the devbox, and what is not". |
| `RUN_RESUME` | `1` | `scripts/train.py`, default `0` there (fresh run): `1` resumes from the newest `checkpoint-N` under `RUN_OUT_DIR` (by `global_step` in its `trainer_state.json`), passed to `trainer.train(resume_from_checkpoint=...)` and to `ConPOTrainer(resume_from=...)`, which restores ConPO's own state (edge buffer, feasible buffer, family/selection stats, `PairStats`, token budget, cumulative counters, rng, wall time) from that checkpoint's `conpo_state.json`/`conpo_edges.jsonl`/`conpo_pair_stats.json`. Fails loud either way: a fresh run refuses a `RUN_OUT_DIR` that already holds a checkpoint (a wiped-and-restarted job must not silently start over), and `RUN_RESUME=1` refuses one that holds none. Recorded as `run_metadata.json`'s `resumed_from_step` (`null` for a fresh run); logs a `"resume"` event to `mechanism.jsonl`. Devbox 2026-09-11: the node-local disk is wiped about every 48 hours (see `local/harvest.sh`). |
| `RUN_SAVE_STEPS` | integer, default `config.SAVE_STEPS` (25) | `scripts/train.py` (design choice, builder, 2026-09-11, same pattern as `RUN_VETO_PERIOD` overriding `config.VETO_PERIOD_K`): checkpoint cadence for `GRPOConfig(save_steps=...)`. Smoke and test knob only -- cluster runs never set it. |
| `RUN_LOSS_TYPE` | `dapo` (default, `config.LOSS_TYPE`) \| `grpo` \| any TRL `loss_type` | `scripts/train.py` -> `GRPOConfig(loss_type=...)`, also written to `run_metadata.json`. Loss-aggregation decision (Ethan, 2026-09-14); `grpo` reproduces the pre-2026-09-14 behaviour (pair with `RUN_KL_CLAMP=none` below). |
| `RUN_KL_CLAMP` | float, default `config.KL_CLAMP` (40.0) \| `none` disables the clamp | `conpo/trainer.py`'s `ConPOTrainer.__init__` -> `self._kl_clamp`, also written to `run_metadata.json`. Clamps the KL estimator's diff to +/- this many nats (decision 2026-09-14). `none` (case-insensitive) or an explicitly empty value disables the clamp (`None`); a numeric `0` is a real clamp, not the disable spelling -- it collapses the KL estimate to exactly zero. `none` reproduces the pre-2026-09-14 (no clamp) behaviour. |
| `RUN_KL_BETA` | float, default `config.KL_BETA` (0.001) | `scripts/train.py` -> `GRPOConfig(beta=...)`, also written to `run_metadata.json`. K1 ablation (devbox 2026-09-15): `RUN_KL_BETA=0` switches the explicit KL term off the way TRL 1.9.2 does for `beta=0.0` (no reference model, no KL term in the loss); the pair path (`score_response` under `_reference_model`) keeps working unchanged for a LoRA policy, since it scores against the frozen initial policy by disabling the adapter rather than through TRL's `ref_model`. |
| `RUN_SKIP_STEP_ON_GAP` | float bar in nats, default unset (disabled) | `conpo/trainer.py`'s `ConPOTrainer.__init__` -> `self._skip_step_on_gap_bar`, also written to `run_metadata.json` as `skip_step_on_gap` (`null` when unset). K2 ablation (devbox 2026-09-15): when set, `_update_skip_step_on_gap` reads TRL's own `sampling/sampling_logp_difference/max` metric (the trainer's per-token log-prob vs vLLM's, `self._metrics["train"]`, populated by TRL's `_generate_and_score_completions` only when `use_vllm` and `vllm_importance_sampling_correction`) for the current optimiser step; if it exceeds the bar, `compute_loss` scales the returned loss to `0.0` for every micro-batch of that step (the graph stays attached, so `backward()`/grad clipping still run) instead of skipping the micro-batch outright. Counted as `skipped_steps_on_gap` (cumulative) in the trainer's own logged metrics and as `skipped_on_gap`/`sampling_gap_max` in `mechanism.jsonl`, both present only when the flag is set. |

`conpo/arms.py:resolve_arm` and the `ConPOTrainer` it configures are
exercised on CPU by `tests/test_trainer_tiny.py`, which constructs the
trainer directly rather than through the script; `scripts/train.py`'s own
`main()` (CLI entrypoint) has one direct `--tiny` test in
`tests/test_subset.py`, covering `run_metadata.json`'s RUN_* fields.

## Local test commands

Run from the repo root. uv is the only env manager. Bootstrap once: `uv sync
--extra cpu --extra dev` creates `.venv` (torch 2.11.0+cpu, transformers
5.14.1, trl 1.9.2, peft 0.20.0).

```bash
# whole suite
uv run --extra cpu --extra dev python -m pytest tests -q

# one file
uv run --extra cpu --extra dev python -m pytest tests/test_verifiers.py -q

# corpus rebuild (CPU only, no network once the HF cache is staged)
uv run --extra cpu --extra dev python scripts/build_corpus.py
```

As of 2026-09-06 in the uv venv: 882 passed, 14 skipped, 0 failed.

On this devbox, after the cu130 sync (step 0 below):
`source local/env.sh && uv run --frozen --extra cu130 --extra cluster --extra dev python -m pytest tests -q`
(Python 3.12.11, torch 2.11.0+cu130, vllm 0.26.0, transformers 5.14.1,
trl 1.9.2, peft 0.20.0): 888 passed, 14 skipped on 2026-09-08 in 7m34s. The suite
needs the HF cache staged (step 1) or network for the policy tokenizer.

## Devbox launch sequence

Runs on this devbox (2x H100 80 GB, no job scheduler) via the uv venv
(`torch 2.11.0+cu130`, `transformers 5.14.1`, `trl 1.9.2`, `peft 0.20.0`,
`vllm 0.26.0`). `GPU=<idx>` picks the card via `CUDA_VISIBLE_DEVICES`
(default 0 for training/eval/retention/dpo, 1 for the judge server, `0,1`
for the full-FT point-check), so two single-GPU runs can share the box with
`GPU=0` and `GPU=1`.

Every launcher's output directory and log now default under `$CONPO_RUNS_DIR`
(`local/env.sh`, default `/node-local-cache/cache/conpo-runs`) instead of
`${PROJECT}/runs` and `${PROJECT}/logs`: the node-local disk is wiped about
every 48 hours, so a fresh run lives there until it is worth keeping.
`bash local/harvest.sh <run>` copies one named run (every run under
`$CONPO_RUNS_DIR`, with no argument) into `runs/` with `rsync -a`, skipping
`checkpoint-*/` unless `KEEP_CHECKPOINTS=1`; `DRY_RUN=1` previews the
`rsync` commands without copying. An explicit `RUN_OUT_DIR` / `EVAL_OUT_DIR`
still wins over the default, as always.

0. **Sync the uv environment** (once, from the repo root):
   ```bash
   source local/env.sh && uv sync --frozen --extra cu130 --extra cluster --extra dev
   ```
   Measured on this box: 217 packages in 1m34s the first time (uv cache on
   `/node-local-cache/cache/uv`), 3 s on a rebuild. `local/env.sh` puts the
   venv on the node-local disk as well (`CONPO_VENV=/node-local-cache/cache/conpo-venv`,
   Python 3.12.11 per `.python-version`: the apt 3.12.3 has no `Python.h`
   and Triton's JIT, hence vLLM, fails on it). Importing torch and
   transformers takes 10 s from that venv against 30 s to 150 s from one on
   NFS, which is why the repo `.venv` is only the fallback. The node-local
   venv is gone after a node move: re-run this step. `cluster` (vllm 0.26.0,
   a CUDA 13 build) pairs with `cu130` only -- pyproject.toml's conflict
   markers refuse `cluster` with `cpu` or `cu128`. The launchers activate
   `$CONPO_VENV`.

1. **Stage the HF cache and model weights** (once, `HF_HUB_OFFLINE` unset):
   ```bash
   source local/env.sh && "$CONPO_VENV/bin/python" scripts/fetch_data.py
   ```
   Caches `config.POOL_DATASET`, `config.IFEVAL_DATASET`, nltk
   punkt/punkt_tab, and now also `snapshot_download`s the
   `config.POLICY_MODEL`, `config.JUDGE_MODEL`, and every model in
   `config.LATER_ABLATION_MODELS` (the model-axis replications, staged now
   rather than at replication time) into `local/env.sh`'s shared hub
   (`/node-local-cache/huggingface-home/hub`). Every launcher below runs
   with `HF_HUB_OFFLINE=1` against this cache.

2. **Build the training corpus** (CPU only):
   ```bash
   source local/env.sh && HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 "$CONPO_VENV/bin/python" scripts/build_corpus.py
   ```
   Measured: 3m26s; train 81897 / dev 1000 / eval_i 1500 / eval_ii 2952 /
   eval_iii 6947 rows from the 95373-row pool.
   Writes `data/{train,dev,eval_i,eval_ii,eval_iii}.jsonl` and
   `data/manifest.json`.

3. **Launch a training run** (one H100; `RUN_SUBSET` defaults to
   `subsets/train_15000_seed42.keys` in the launcher). The default judge
   placement is the batched veto (USER DECISIONS 2026-09-02): no judge
   server needed, so this is the whole sequence for a normal run:
   ```bash
   RUN_ARM=full RUN_SEED=42 GPU=0 bash local/train.sh
   ```
   `scripts/train.py`'s `build_judge` loads/unloads a `VLLMBatchedJudge`/
   `HFBatchedJudge` on the same card, in bursts, every `RUN_VETO_PERIOD`
   optimiser steps (`ConPOTrainer.flush_pending_veto`); writes to
   `$CONPO_RUNS_DIR/full_s42/`. Add ablation overrides the same way, e.g.
   `RUN_ARM=full RUN_NO_VETO=1 GPU=0 bash local/train.sh`. Once the run is
   worth keeping past the node-local wipe: `bash local/harvest.sh full_s42`.
   If the node-local disk gets wiped mid-run, resume from that harvest
   instead of restarting cold: `KEEP_CHECKPOINTS=1 bash local/harvest.sh
   full_s42` (checkpoints are skipped by default), copy `runs/full_s42/`
   back under `CONPO_RUNS_DIR` (or point `RUN_OUT_DIR` at it directly), then
   `RUN_ARM=full RUN_SEED=42 RUN_RESUME=1 GPU=0 bash local/train.sh` -- see
   the `RUN_RESUME` row above and `local/harvest.sh`'s header.

   **`RUN_JUDGE_MODE=http` fallback** (a resident judge server on the second
   H100, serialising every training run against it -- for comparison or
   debugging the batched path):
   ```bash
   GPU=1 bash local/judge_server.sh
   ```
   Wait for `runs/serve/judge.endpoint` to appear (the launcher also polls
   its own `/health` before writing it), then
   `RUN_ARM=full RUN_SEED=42 RUN_JUDGE_MODE=http GPU=0 bash local/train.sh`
   (`local/train.sh` waits for the judge endpoint only in this mode).

4. **DPO baseline launch** (one H100, `vf_dpo` | `edge_dpo`, USER DECISIONS
   2026-09-02(c)): `local/train_dpo.sh` builds `RUN_PAIRS` with
   `scripts/build_pairs.py` first (skipped when a pairs file already
   exists), then trains with `scripts/train_dpo.py`.
   ```bash
   RUN_ARM=vf_dpo RUN_SEED=42 GPU=0 bash local/train_dpo.sh
   RUN_ARM=edge_dpo RUN_SEED=42 RUN_EDGES=runs/full_s42/edges.jsonl \
     GPU=0 bash local/train_dpo.sh
   ```
   `edge_dpo` needs a finished `full` run's `edges.jsonl` (default
   `$CONPO_RUNS_DIR/full_s${RUN_SEED}/edges.jsonl`) and fails loud when it
   is missing.

5. **Full fine-tune point-check** (two H100s, `local/train_fullft.sh`):
   `RUN_FULL_FT=1` (`scripts/train.py`, `peft_config=None`) under an
   `accelerate launch --config_file local/accelerate_fullft.yaml
   --num_processes 2` FSDP launch, sharding the full optimizer state across
   both cards (torch-native, not DeepSpeed ZeRO-2: `deepspeed` is not in
   `uv.lock`). `RUN_OUT_DIR` is explicit (a plain
   `$CONPO_RUNS_DIR/${RUN_ARM}_s${RUN_SEED}` default would collide with the
   LoRA `full_s42` run):
   ```bash
   RUN_ARM=full RUN_SEED=42 RUN_FULL_FT=1 RUN_OUT_DIR=runs/full_fullft_s42 \
     bash local/train_fullft.sh
   ```
   See "What is verified on the devbox, and what is not" for the open questions this script has never run
   against.

6. **Model-axis replications** (one H100, `RUN_ARM=full` only, `RUN_MODEL`
   overriding `config.POLICY_MODEL`):
   ```bash
   RUN_ARM=full RUN_SEED=42 RUN_MODEL=Qwen/Qwen3-8B RUN_OUT_DIR=runs/full_qwen3_8b_s42 \
     GPU=0 bash local/train.sh
   RUN_ARM=full RUN_SEED=42 RUN_MODEL=allenai/Olmo-3-7B-Instruct-SFT RUN_OUT_DIR=runs/full_olmo3_7b_sft_s42 \
     GPU=0 bash local/train.sh
   ```

7. **Evaluate a finished run**:
   ```bash
   GPU=0 bash local/eval.sh runs/full_s42/adapter
   ```
   Omit the adapter argument to evaluate the base model. Decoding follows the
   locked protocol (temperature 0, one response per prompt, no system
   prompt, thinking off, up to `config.EVAL_MAX_NEW_TOKENS` = 2048 new
   tokens) and left-truncates prompts to `config.MAX_PROMPT_TOKENS` like
   training, with truncated-prompt counts recorded in each output JSON.

   **Evaluate every run in one sweep** (`local/eval_sweep.sh`, plain bash,
   sequential -- each launcher owns whichever GPU it is given until it
   exits, since this devbox has no queue to serialise them):
   ```bash
   bash local/eval_sweep.sh                      # eval.sh: base + every $CONPO_RUNS_DIR/*/adapter
   RETENTION=1 bash local/eval_sweep.sh           # + retention.sh (GSM8K/MMLU/BBH, USER DECISIONS)
   DRY_RUN=1 bash local/eval_sweep.sh             # print the launch lines instead of running
   ```
   `RETENTION=1` runs `local/retention.sh` for the base model and for the
   run ids in `RETENTION_RUNS` (default `full_s42 full_s43 full_s44
   ifrlvr_s42 ifrlvr_s43 ifrlvr_s44`), e.g. `bash local/retention.sh
   runs/full_s42/adapter`. `local/retention.sh` mirrors `local/eval.sh`'s
   own "positional adapter dir, omit for base model".

## What is verified on the devbox, and what is not

Verified end to end on this box on 2026-09-08 (2x H100 80 GB, node-local
venv, the real `Qwen/Qwen3-4B` checkpoint, smoke knobs `RUN_STEPS=2
RUN_LIMIT=64 RUN_NUM_GENERATIONS=4 RUN_MAX_COMPLETION_TOKENS=64
RUN_EFFECTIVE_BATCH=8`):

- Steps 0 to 2 of the launch sequence (sync, staging of the policy and judge
  weights plus the IF datasets into the shared hub, corpus build) and the
  full CPU suite (888 passed, 14 skipped in the cu130 venv; the 882 under
  "Local test commands" is the 2026-09-06 cpu-venv baseline before the six
  tests this adaptation added).
- `local/train.sh` in four configurations, each writing `adapter/`,
  `checkpoint-2/`, `run_metadata.json`, `family_stats.json`,
  `mechanism.jsonl`, `edges.jsonl` (and `judge_replies.jsonl` when a judge
  ran): `RUN_USE_VLLM=0 RUN_NO_VETO=1` (HF generate, 495 s per step, smoke
  only); `RUN_USE_VLLM=1 RUN_NO_VETO=1` (TRL colocate vLLM through
  `VLLMColocateEngine`, 38 s per step); the default batched veto
  (`VLLMBatchedJudge` burst in a spawned child process with the trainer's
  engine asleep, `RUN_VETO_PERIOD=1`, 120 s per step, 8 verdicts parsed);
  and `RUN_JUDGE_MODE=http` against `GPU=1 bash local/judge_server.sh`
  (`HttpJudge`, 8 calls answered 200).
- `local/eval.sh` on the base model with `--backend vllm`
  (`_generate_vllm`, `EVAL_LIMIT=8`): `external.json` and `internal.json`
  written, process exits cleanly (in-process vLLM engine, see
  `local/env.sh`).
- `local/eval_sweep.sh`: `bash -n` and `DRY_RUN=1` only.

Still untested:

- `local/eval.sh` with an adapter (the `LoRARequest` path) and
  `--backend hf` generation.
- `local/retention.sh` (never launched).
- `local/train_dpo.sh`: the `build_pairs.py --engine vllm` step and
  `train_dpo.py` on the real model.
- `local/train_fullft.sh` / `local/accelerate_fullft.yaml`: the two-process
  accelerate FSDP launch (torch-native, not DeepSpeed ZeRO-2, since
  `deepspeed` is not in `uv.lock`) and TRL colocate generation under it.
  `RUN_FULL_FT=1`'s reference scoring itself is a tested path
  (`tests/test_train_knobs.py`).
- `RUN_ARM=ifrlvr` and every ablation override on the real model
  (unit-tested at the mechanism level only); production knobs
  (`RUN_STEPS=500`, `RUN_EFFECTIVE_BATCH=128`, 8 generations, 1024
  completion tokens) and their memory; two concurrent single-GPU runs.
- Model-axis replications: `Qwen/Qwen3-8B` is in the shared hub;
  `allenai/Olmo-3-7B-Instruct-SFT` and `allenai/Llama-3.1-Tulu-3-8B-SFT`
  are not staged.
- `HFBatchedJudge` on a GPU (CPU tests only).
- `RUN_RESUME` on the real devbox (unit- and CPU-tiny-tested only, `tests/
  test_train_knobs.py`): resuming a real GPU checkpoint's LoRA adapter,
  optimiser and scheduler state through TRL's own `resume_from_checkpoint`.

Version note: trl 1.9.2 warns that it supports vLLM 0.17.0 to 0.25.1 while
`uv.lock` pins vllm 0.26.0; colocate generation, the batched judge, and the
judge server all ran on that pairing.
