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
`config.env(name, default)`, so `slurm/train.sbatch` and `slurm/eval.sbatch`
export env only and never retype a value `scripts/train.py` can read from
`conpo/config.py` itself.

| Variable | Values | Read by |
|---|---|---|
| `RUN_ARM` | `full` \| `ifrlvr` \| `solve_only` \| `forward_only` \| `reverse_only` | `scripts/train.py` -> `conpo/arms.py:resolve_arm` (`ARMS` dict) |
| `RUN_SEED` | integer | `scripts/train.py`, default `0` there (`slurm/train.sbatch` defaults it to `42`) |
| `RUN_STEPS` | integer, optimiser-step budget | `scripts/train.py`, default `1` there (`slurm/train.sbatch` defaults it to `500`) |
| `RUN_TOKEN_BUDGET` | integer, generated-token budget, `0` = no cap | `conpo/data.py:TokenMeter` |
| `RUN_USE_VLLM` | `1` sample through vLLM, `0` HF generate | `scripts/train.py`, default `0` there (`slurm/train.sbatch` always sets `1`) |
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
| `RUN_FULL_FT` | `1` | `scripts/train.py`, default `0` there: `1` trains every weight (`peft_config=None`) instead of a LoRA adapter, for the full-FT point-check (`slurm/train_fullft.sbatch`); recorded as `run_metadata.json`'s `full_ft`. Reference scoring runs under TRL's own `self.ref_model` instead of a disabled adapter (`conpo/trainer.py`'s `_reference_model`); see "What is untested" for what still isn't. |

`conpo/arms.py:resolve_arm` and the `ConPOTrainer` it configures are
exercised on CPU by `tests/test_trainer_tiny.py`, which constructs the
trainer directly rather than through the script; `scripts/train.py`'s own
`main()` (SLURM entrypoint) has one direct `--tiny` test in
`tests/test_subset.py`, covering `run_metadata.json`'s RUN_* fields.

## Local test commands

Run from the repo root. Default env manager is uv (the conda envs `conpo`
and `conpo-gpu` are kept as a fallback -- USER DECISIONS 2026-09-06).
Bootstrap once: `uv sync --extra cpu --extra dev` creates `.venv` with the
same pins as the conda env `conpo` (torch 2.11.0+cpu, transformers 5.14.1,
trl 1.9.2, peft 0.20.0).

```bash
# whole suite
uv run --extra cpu --extra dev python -m pytest tests -q

# one file
uv run --extra cpu --extra dev python -m pytest tests/test_verifiers.py -q

# corpus rebuild (CPU only, no network once the HF cache is staged)
uv run --extra cpu --extra dev python scripts/build_corpus.py
```

As of 2026-09-06 in the uv venv: 882 passed, 14 skipped, 0 failed (same
counts as the conda env `conpo`).

## Cluster launch sequence

Runs on the mlp SLURM cluster via the uv venv
(`torch 2.11.0+cu130`, `transformers 5.14.1`, `trl 1.9.2`, `peft 0.20.0`,
`vllm 0.26.0`). Two H100 80 GB GPUs: `--gres=gpu:h100:1` per LoRA run (one
run per card, two runs concurrently) and for the judge server;
`--gres=gpu:h100:2` for the full-FT point-check. The gres strings are
unverified on the cluster.

0. **Bootstrap uv on the cluster** (once, from the repo root; install uv if
   missing):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv sync --frozen --extra cu130 --extra cluster
   ```
   `cluster` (vllm 0.26.0, a CUDA 13 build) pairs with `cu130` only --
   pyproject.toml's conflict markers refuse `cluster` with `cpu` or `cu128`.
   This needs an NVIDIA driver of 580 or newer; nobody has checked the
   cluster driver, and this uv bootstrap itself has never run on a GPU box.
   The launchers activate `${PROJECT}/.venv` (override with `CONPO_VENV`).

1. **Stage the HF cache** (once, from a node with network access,
   `HF_HUB_OFFLINE` unset):
   ```bash
   source "${CONPO_VENV:-${PROJECT}/.venv}/bin/activate"
   python scripts/fetch_data.py
   ```
   Caches `config.POOL_DATASET`, `config.IFEVAL_DATASET`, the
   `config.POLICY_MODEL` tokenizer, and nltk punkt/punkt_tab. Every later
   step runs with `HF_HUB_OFFLINE=1` against this cache.

2. **Build the training corpus** (CPU only):
   ```bash
   python scripts/build_corpus.py
   ```
   Writes `data/{train,dev,eval_i,eval_ii,eval_iii}.jsonl` and
   `data/manifest.json`.

3. **Launch a training run** (one H100; `RUN_SUBSET` defaults to
   `subsets/train_15000_seed42.keys` in the launcher). The default judge
   placement is the batched veto (USER DECISIONS 2026-09-02): no judge
   server needed, so this is the whole sequence for a normal run:
   ```bash
   RUN_ARM=full RUN_SEED=42 \
     sbatch --export=ALL,RUN_ARM,RUN_SEED slurm/train.sbatch
   ```
   `scripts/train.py`'s `build_judge` loads/unloads a `VLLMBatchedJudge`/
   `HFBatchedJudge` on the same card, in bursts, every `RUN_VETO_PERIOD`
   optimiser steps (`ConPOTrainer.flush_pending_veto`); writes to
   `runs/full_s42/`. Add ablation overrides the same way, e.g.
   `RUN_ARM=full RUN_NO_VETO=1 sbatch --export=ALL,RUN_ARM,RUN_NO_VETO slurm/train.sbatch`.

   **`RUN_JUDGE_MODE=http` fallback** (a resident judge server on the second
   H100, serialising every training run against it -- for comparison or
   debugging the batched path):
   ```bash
   sbatch slurm/judge_server.sbatch
   ```
   Wait for `runs/serve/judge.endpoint` to appear (the job also polls its
   own `/health` before writing it), then
   `RUN_ARM=full RUN_SEED=42 RUN_JUDGE_MODE=http sbatch --export=ALL,RUN_ARM,RUN_SEED,RUN_JUDGE_MODE slurm/train.sbatch`
   (`slurm/train.sbatch` waits for the judge endpoint only in this mode).

4. **DPO baseline launch** (one H100, `vf_dpo` | `edge_dpo`, USER DECISIONS
   2026-09-02(c)): `slurm/train_dpo.sbatch` builds `RUN_PAIRS` with
   `scripts/build_pairs.py` first (skipped when a pairs file already
   exists), then trains with `scripts/train_dpo.py`.
   ```bash
   RUN_ARM=vf_dpo RUN_SEED=42 \
     sbatch --export=ALL,RUN_ARM,RUN_SEED slurm/train_dpo.sbatch
   RUN_ARM=edge_dpo RUN_SEED=42 RUN_EDGES=runs/full_s42/edges.jsonl \
     sbatch --export=ALL,RUN_ARM,RUN_SEED,RUN_EDGES slurm/train_dpo.sbatch
   ```
   `edge_dpo` needs a finished `full` run's `edges.jsonl` (default
   `runs/full_s${RUN_SEED}/edges.jsonl`) and fails loud when it is missing.

5. **Full fine-tune point-check** (two H100s, `slurm/train_fullft.sbatch`):
   `RUN_FULL_FT=1` (`scripts/train.py`, `peft_config=None`) under an
   `accelerate launch --config_file slurm/accelerate_fullft.yaml
   --num_processes 2` ZeRO-2 launch, sharding the full optimizer state
   across both cards. `RUN_OUT_DIR` is explicit (a plain
   `runs/${RUN_ARM}_s${RUN_SEED}` default would collide with the LoRA
   `full_s42` run):
   ```bash
   sbatch --export=ALL,RUN_ARM,RUN_SEED,RUN_FULL_FT,RUN_OUT_DIR \
     slurm/train_fullft.sbatch
   ```
   (`RUN_ARM=full RUN_SEED=42 RUN_FULL_FT=1 RUN_OUT_DIR=runs/full_fullft_s42`
   -- the `full_fullft_s42` card's own launch line.) See "What is untested"
   for the two open questions this script has never run against.

6. **Model-axis replications** (one H100, `RUN_ARM=full` only, `RUN_MODEL`
   overriding `config.POLICY_MODEL`):
   ```bash
   RUN_ARM=full RUN_SEED=42 RUN_MODEL=Qwen/Qwen3-8B RUN_OUT_DIR=runs/full_qwen3_8b_s42 \
     sbatch --export=ALL,RUN_ARM,RUN_SEED,RUN_MODEL,RUN_OUT_DIR slurm/train.sbatch
   RUN_ARM=full RUN_SEED=42 RUN_MODEL=allenai/Olmo-3-7B-Instruct-SFT RUN_OUT_DIR=runs/full_olmo3_7b_sft_s42 \
     sbatch --export=ALL,RUN_ARM,RUN_SEED,RUN_MODEL,RUN_OUT_DIR slurm/train.sbatch
   ```

7. **Evaluate a finished run**:
   ```bash
   sbatch slurm/eval.sbatch runs/full_s42/adapter
   ```
   Omit the adapter argument to evaluate the base model.

   **Evaluate every run in one sweep** (`slurm/eval_sweep.sh`, plain bash,
   not an sbatch script -- run it from a login/head node):
   ```bash
   bash slurm/eval_sweep.sh                     # eval.sbatch: base + every runs/*/adapter
   RETENTION=1 bash slurm/eval_sweep.sh          # + retention.sbatch (GSM8K/MMLU/BBH, USER DECISIONS)
   DRY_RUN=1 bash slurm/eval_sweep.sh            # print the sbatch lines instead of submitting
   ```
   `RETENTION=1` submits `slurm/retention.sbatch` for the base model and for
   the run ids in `RETENTION_RUNS` (default `full_s42 full_s43 full_s44
   ifrlvr_s42 ifrlvr_s43 ifrlvr_s44`), e.g. `sbatch slurm/retention.sbatch
   runs/full_s42/adapter`. `slurm/retention.sbatch` itself is being written
   by another builder in parallel -- this is its assumed CLI shape, mirroring
   `slurm/eval.sbatch`'s own "positional adapter dir, omit for base model".

## What is untested

No GPU exists on this development machine and `vllm` is not installed here.
Of the uv extras, only `cpu` has been synced and verified (`uv sync --extra
cpu --extra dev`, matching the conda env `conpo`); `cu128` (RTX 5060 smoke),
`cu130`, and `cluster` have never been synced on a GPU box, and the uv
cluster bootstrap itself (`uv sync --frozen --extra cu130 --extra cluster`)
has never run. So every GPU-dependent path below has never run, even though
the surrounding Python modules now exist and are CPU-tested:

- `slurm/judge_server.sbatch` -- never submitted; the vLLM OpenAI-server
  invocation, the `--gres=gpu:h100:1` line, and the health-poll
  timing are unverified.
- `slurm/train.sbatch` -- never submitted; the `RUN_JUDGE_MODE=http`
  judge-endpoint wait loop, the `RUN_OUT_DIR` layout, and the two-GPU
  full-FT comment variant are unverified.
- `slurm/eval.sbatch` -- never submitted; the `--out` file paths it builds
  are checked against `scripts/eval_external.py`/`scripts/eval_internal.py`'s
  real `argparse` contract, but only their `--score-only` path (no
  generation) is exercised by `tests/test_eval_scripts.py` -- real
  generation, both `--backend hf` and `--backend vllm`, is untested.
- `scripts/train.py` -- exists and reads the RUN_* contract above, but as
  an entrypoint it has no direct test; `use_vllm=False` (`RUN_USE_VLLM=0`)
  and `--tiny` are its only combination exercised anywhere, and only
  indirectly, by `tests/test_trainer_tiny.py` constructing `ConPOTrainer`
  directly rather than running the script.
- `scripts/eval_external.py`'s and `scripts/eval_internal.py`'s
  `_generate_vllm` -- lazily imports `vllm.LLM`, `vllm.SamplingParams`,
  `vllm.lora.request.LoRARequest`; never called by anything on this
  machine.
- `conpo/judge.py:HttpJudge` -- the real HTTP path against a live vLLM
  server. Only `StaticJudge`, `parse_verdict`, and `build_veto_prompt` are
  exercised by `tests/test_judge.py`.
- `conpo/judge.py:VLLMBatchedJudge` -- the batched veto's default judge
  (loads/unloads a colocated `vllm.LLM` per burst). No vLLM install on this
  machine; only `HFBatchedJudge` and `PendingQueue` are CPU-tested
  (`tests/test_batched_veto.py`), and `ConPOTrainer.flush_pending_veto`'s
  wiring is tested against a recording fake judge, not a real batched one
  (`tests/test_batched_veto_trainer.py`).
- `ConPOTrainer._sleep_engine_if_enabled`/`_wake_engine` -- the
  `vllm_enable_sleep_mode` sleep-before/wake-after path around a veto
  burst, verified only by reading `trl/generation/vllm_generation.py`'s own
  `sleep(level=2)`/`wake_up(tags=...)` calls; never run against a live
  colocated vLLM engine.
- `conpo/engine.py:VLLMColocateEngine` -- wraps a live TRL `GRPOTrainer`'s
  colocated vLLM generation. Only `HFEngine` and `FakeEngine` are
  exercised by tests.
- Any `RUN_USE_VLLM=1` code path end to end, and TRL's own
  `vllm_mode="colocate"` integration on a real GPU.
- `RUN_ARM=full` and `RUN_ARM=ifrlvr` are exercised end to end at
  `RUN_STEPS<=2` on a tiny random model (`tests/test_trainer_tiny.py`);
  neither has run on the real `Qwen/Qwen3-4B` checkpoint or a real GPU.
  The ablation overrides are unit-tested at the mechanism level (e.g. the
  `selection` modes in `tests/test_reverse.py`), not end to end through
  `ConPOTrainer`.
- `slurm/train_dpo.sbatch` -- never submitted; the `build_pairs.py --engine
  vllm` step and `train_dpo.py`'s real-model path are unverified, same
  caveats as `slurm/train.sbatch`'s own header.
- `slurm/train_fullft.sbatch` / `slurm/accelerate_fullft.yaml` -- never
  submitted; the ZeRO-2 ("no offload") config is a guess at fitting full
  fp32 AdamW state across two H100s, and TRL's `vllm_mode="colocate"`
  generation under a two-process accelerate/deepspeed launch has never been
  exercised. `RUN_FULL_FT=1`'s reference scoring is a real, tested path
  (`conpo/trainer.py`'s `_reference_model` scores under TRL's own
  `self.ref_model` when there is no adapter to disable, see
  `tests/test_train_knobs.py`'s full-FT test) -- only the two-process launch
  above remains unverified.
- `slurm/eval_sweep.sh` -- `bash -n` and one `DRY_RUN=1` run are checked on
  this machine (no `runs/` directories to iterate); submitting it to Slurm
  itself is untested. Its assumption about `slurm/retention.sbatch`'s CLI
  (positional adapter dir, omit for base model) was confirmed by reading the
  landed `slurm/retention.sbatch`, written by another builder.
