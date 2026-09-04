# ConPO

Post-training for verifiable instruction following: repaired/corrupted response pairs trained alongside a clean-generation GRPO objective.

Locked: Qwen3-4B policy every run (later ablations: Qwen3-8B, Llama-3.1-Tulu-3-8B-SFT, Olmo-3-7B-Instruct-SFT, unused today); LoRA r64 alpha128 dropout0.05; 54 constraint families (IFEval 25 + IFBench-Train 29); pool `allenai/IF_multi_constraints_upto5`, eval on `google/IFEval` + held-out `allenai/IFBench_test` (58 disjoint families); official vendored IFEvalG scorers only; judge Qwen3-14B.

Tests: `conda run -n conpo --no-capture-output python -m pytest tests/test_verifiers.py -q`

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

`conpo/arms.py:resolve_arm` and the `ConPOTrainer` it configures are
exercised on CPU by `tests/test_trainer_tiny.py`, which constructs the
trainer directly rather than through the script; `scripts/train.py`'s own
`main()` (SLURM entrypoint) has one direct `--tiny` test in
`tests/test_subset.py`, covering `run_metadata.json`'s RUN_* fields.

## Local test commands

Run from the repo root, conda env `conpo`:

```bash
# whole suite
conda run -n conpo --no-capture-output python -m pytest tests -q

# one file
conda run -n conpo --no-capture-output python -m pytest tests/test_verifiers.py -q

# corpus rebuild (CPU only, no network once the HF cache is staged)
conda run -n conpo --no-capture-output python scripts/build_corpus.py
```

As of 2026-09-03: 673 passed, 14 skipped, 0 failed.

## Cluster launch sequence

Runs on the mlp SLURM cluster, conda env `vllm`
(`torch 2.11.0+cu130`, `transformers 5.14.1`, `trl 1.9.2`, `peft 0.20.0`,
`vllm 0.26.0`). Two H100 80 GB GPUs: `--gres=gpu:h100:1` per LoRA run (one
run per card, two runs concurrently) and for the judge server;
`--gres=gpu:h100:2` for the full-FT point-check. The gres strings are
unverified on the cluster.

1. **Stage the HF cache** (once, from a node with network access,
   `HF_HUB_OFFLINE` unset):
   ```bash
   conda activate vllm
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

4. **Evaluate a finished run**:
   ```bash
   sbatch slurm/eval.sbatch runs/full_s42/adapter
   ```
   Omit the adapter argument to evaluate the base model.

## What is untested

No GPU exists on this development machine and `vllm` is not installed here
(the `conpo` conda env is CPU-only, `pyproject.toml`'s `cluster` optional
group is not installed), so every GPU-dependent path below has never run,
even though the surrounding Python modules now exist and are CPU-tested:

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
