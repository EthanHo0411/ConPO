# ConPO on the shared devbox: local-GPU + uv adaptation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every ConPO training/eval path launchable on this box (2x H100, no SLURM) from the uv venv, with model/data caches on the node-local disks.

**Architecture:** The `RUN_*` env contract (README) stays untouched. `slurm/` is replaced by `local/`: plain bash launchers that source `local/env.sh` (uv cache, HF hub, venv, log tee, `JOB_ID`) and pick GPUs with `GPU=<idx>`. `scripts/fetch_data.py` also stages model weights so the offline launchers never hit the network.

**Tech Stack:** uv 0.8.22, Python 3.12, torch 2.11.0+cu130, vllm 0.26.0, transformers 5.14.1, trl 1.9.2, peft 0.20.0. Driver 580.126.20 (CUDA 13 capable). Shell bash.

**Spec:** Ethan's request 2026-09-08 (this session): explore ConPO, make it trainable on the local GPUs with uv instead of conda, review `/node-local-cache/cache` and `/node-local-cache/huggingface-home` and adapt uv / model caching to them. All changes inside ConPO pre-approved; reversible writes in the two node-local dirs approved.

**Sizing (judgment.md 1b):** heavy. 5+ files, a new launcher contract, design ambiguity (cache placement). Process: plan here, Sonnet 5 xhigh subagent execution, fresh-context verifier, review tail.

## Global Constraints

- Writes only inside `/home/dev/project/ethan` plus reversible writes under `/node-local-cache/cache` and `/node-local-cache/huggingface-home` (Ethan, 2026-09-08).
- `RUN_*` variable names and defaults unchanged (README "The RUN_* contract").
- No new Python dependencies; `uv.lock` stays as committed (no `uv lock`).
- Commit after every task; never push (external effect, Ethan's call).
- No em dashes anywhere; type hints on all new Python.

## Decisions taken without Ethan (autonomous run, gates collapsed)

- D1 git: ConPO is already a repo (24 commits, `origin` = `git@github.com:EthanHo0411/ConPO.git`, Ethan's identity). No init. Commits local only.
- D2 uv: cache moves to `/node-local-cache/cache/uv` (a real uv cache our earlier runs left there, 277 MB, dev-owned). `.venv` stays in the repo on NFS so it survives pod restarts and node moves. `UV_LINK_MODE=copy` because the two live on different filesystems. Scoped through `local/env.sh`, not through `.ethan-env` (SERVER.md §9 redirection stays global).
- D3 HF: launchers use the shared hub `/node-local-cache/huggingface-home/hub` as `HF_HUB_CACHE`. Already there: `Qwen/Qwen3-14B` (judge, 28 GB), `Qwen/Qwen3-8B` (ablation, 16 GB), `Qwen/Qwen3-0.6B`, `openai/gsm8k`, `cais/mmlu`. Missing and staged there by Task 2: `Qwen/Qwen3-4B` (policy, 8.06 GB, 13 files), `allenai/IF_multi_constraints_upto5`, `google/IFEval`, `lukaemon/bbh`. Not there and not staged (later ablations only): `allenai/Olmo-3-7B-Instruct-SFT` (only the non-SFT Instruct is cached), `allenai/Llama-3.1-Tulu-3-8B-SFT`. Processed-dataset cache (`HF_DATASETS_CACHE`) stays inside the boundary.
- D4 launchers: `slurm/` is deleted; `local/` replaces it one-for-one. `GPU` env picks `CUDA_VISIBLE_DEVICES` (train/eval/retention/dpo default `0`, judge server default `1`, full-FT `0,1`). `JOB_ID` replaces `SLURM_JOB_ID` in output names.
- D5 full-FT: `deepspeed` is not in `uv.lock` and cannot be added without re-locking, so `local/accelerate_fullft.yaml` becomes an accelerate FSDP config (torch-native, no new dependency). Still UNTESTED, same as before.
- D6 `fetch_data.py` gains `snapshot_download` of `config.POLICY_MODEL` and `config.JUDGE_MODEL` (weights, not just tokenizer), since every launcher runs `HF_HUB_OFFLINE=1`.
- D7 conda references removed (README, `scripts/check_overlap.py` docstring). uv is the only env manager.
- D8 No EnterPlanMode / brainstorming gate: Ethan is away; decisions D1-D7 are the gate output.

---

### Task 1: GPU uv environment (main session, commands only)

**Files:** none changed; creates `.venv/` (gitignored) and populates `/node-local-cache/cache/uv`.

- [ ] Step 1: `source local/env.sh` (sets `UV_CACHE_DIR`, `UV_LINK_MODE`).
- [ ] Step 2: `uv sync --frozen --extra cu130 --extra cluster --extra dev` (background; multi-GB download).
- [ ] Step 3 proving check: `uv run --frozen --extra cu130 --extra cluster --extra dev python -c "import torch, vllm; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count(), vllm.__version__)"` prints `2.11.0+cu130 True 2 0.26.0`.
- [ ] Step 4 regression: `.venv/bin/python -m pytest tests -q` (README baseline in the cpu venv: 882 passed, 14 skipped). Record the counts here.
- [ ] Step 5: measure `time .venv/bin/python -c "import torch, transformers, trl, vllm"` (NFS venv import cost) and record it below.

Result: (fill in)

### Task 2: Stage models and data into the shared hub (main session)

**Files:**
- Modify: `scripts/fetch_data.py` (main(): add model snapshot downloads)

- [ ] Step 1: in `scripts/fetch_data.py` add, after the tokenizer block inside `if not args.retention_only:`

```python
        from huggingface_hub import snapshot_download

        for name in (config.POLICY_MODEL, config.JUDGE_MODEL):
            print(f"model weights: {name}")
            path = snapshot_download(name)
            print(f"  cached: {path}")
```

and `import huggingface_hub` is NOT needed at top level (keep the lazy import so `--retention-only` stays light).
- [ ] Step 2: `source local/env.sh && .venv/bin/python scripts/fetch_data.py` (background; writes to `/node-local-cache/huggingface-home/hub` and `HF_DATASETS_CACHE`; also writes `data/retention/*.jsonl` and `nltk_data/`).
- [ ] Step 3 proving check: `ls /node-local-cache/huggingface-home/hub/models--Qwen--Qwen3-4B/snapshots/*/model*.safetensors` lists 3 shards; `ls .../datasets--allenai--IF_multi_constraints_upto5 .../datasets--google--IFEval .../datasets--lukaemon--bbh` exist.
- [ ] Step 4: `source local/env.sh && HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python scripts/build_corpus.py` writes `data/{train,dev,eval_i,eval_ii,eval_iii}.jsonl` + `data/manifest.json` (background).
- [ ] Step 5: commit `scripts/fetch_data.py`.

### Task 3a: Local launchers replace slurm/ (Sonnet 5 xhigh subagent)

**Files:**
- Create: `local/train.sh`, `local/eval.sh`, `local/judge_server.sh`, `local/retention.sh`, `local/train_dpo.sh`, `local/train_fullft.sh`, `local/eval_sweep.sh`, `local/accelerate_fullft.yaml`
- Delete: `slurm/` (all 8 files) via `git rm -r slurm`
- Consumes: `local/env.sh` (already written): `PROJECT`, `JOB_ID`, `conpo_launch_setup` (needs `NAME` set first; activates venv, tees to `logs/${NAME}_${JOB_ID}.out`).
- Produces: launcher CLI shapes identical to the sbatch ones (positional adapter dir for eval/retention; env-driven for train/dpo/fullft; `eval_sweep.sh` runs launchers with `bash` sequentially instead of `sbatch`).

Port rules (apply to every script):
1. Drop every `#SBATCH` line and the "sbatch --export=ALL,..." usage examples; usage becomes `RUN_ARM=full RUN_SEED=42 bash local/train.sh`.
2. Replace the prologue (`PROJECT=...; cd; mkdir logs; echo node=...; nvidia-smi; source venv; echo env=`) with:
```bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
NAME=conpo_train            # per script: conpo_eval, judge_server, conpo_retention, conpo_train_dpo, conpo_train_fullft
export CUDA_VISIBLE_DEVICES="${GPU:-0}"   # judge_server: ${GPU:-1}; train_fullft: ${GPU:-0,1}
conpo_launch_setup
```
3. Replace every `${SLURM_JOB_ID}` with `${JOB_ID}`; `judge_server` port default becomes `${JS_PORT:-8000}`.
4. Keep `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1` exports; drop the `no_proxy` block (single node; nothing leaves localhost) and the `TOKENIZERS_PARALLELISM`/`VLLM_WORKER_MULTIPROC_METHOD`/`PYTORCH_CUDA_ALLOC_CONF` exports (now in env.sh).
5. Keep the judge-endpoint wait loop in train/fullft/dpo exactly (file format `hostname port jobid` still written by `local/judge_server.sh`).
6. Rewrite the header comments: "UNTESTED on the mlp cluster" wording becomes "local devbox, 2x H100; GPU=<idx> picks the card". Keep the description of what each script does. No em dashes.
7. `local/accelerate_fullft.yaml`: replace the DeepSpeed block with
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_version: 2
  fsdp_reshard_after_forward: false   # ZeRO-2 style: shard grads + optimizer, keep params gathered
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_offload_params: false
mixed_precision: bf16
num_machines: 1
num_processes: 2
main_training_function: main
rdzv_backend: static
same_network: true
use_cpu: false
```
with a header comment saying deepspeed is not in uv.lock (D5) and that this path is UNTESTED.
8. `local/eval_sweep.sh`: `submit()` runs `bash "$@"` (foreground, sequential) instead of `sbatch "$@"`; `DRY_RUN=1` prints `bash <args>`.

- [ ] Step 1: `bash -n` every new script; `DRY_RUN=1 bash local/eval_sweep.sh` prints lines and exits 0.
- [ ] Step 2: `grep -rn 'SLURM\|sbatch\|#SBATCH\|gres' local/` returns nothing.
- [ ] Step 3: `.venv/bin/python -m pytest tests -q` count unchanged from Task 1.
- [ ] Step 4: commit: "local/: bash launchers for the 2x H100 devbox replace slurm/".

### Task 3b: Repo references to slurm/conda (Sonnet 5 xhigh subagent)

**Files:**
- Modify: `README.md` (sections "Local test commands", "Cluster launch sequence", "What is untested"), `scripts/run_card.py:124-142,404-428` (launch lines), `scripts/dashboard.py:704` (comment), `conpo/config.py:307` (comment), `conpo/judge.py:100` (docstring), `conpo/experiments.md` (2 mentions), `scripts/check_overlap.py:5` (docstring: `conda run -n conpo ...` becomes `uv run --extra cu130 --extra cluster python scripts/check_overlap.py ...`).

- [ ] Step 1: `grep -rn 'sbatch\|slurm\|conda\|SLURM' --include='*.py' --include='*.md' --include='*.sh' --include='*.yaml' .` and rewrite every hit (except `conpo/vendor/ifbench/VENDOR.md`, historical) to the `local/` equivalents: `sbatch --export=ALL,A,B slurm/train.sbatch` becomes `A=.. B=.. bash local/train.sh`; `sbatch slurm/eval.sbatch X` becomes `bash local/eval.sh X`.
- [ ] Step 2: README "Cluster launch sequence" becomes "Devbox launch sequence": step 0 = `source local/env.sh && uv sync --frozen --extra cu130 --extra cluster --extra dev`; step 1 = `.venv/bin/python scripts/fetch_data.py` (now stages weights too); steps 3-7 use `bash local/*.sh` with `GPU=`; note two runs can share the box with `GPU=0` and `GPU=1`. "What is untested" keeps every item that Task 5 does not exercise and drops the ones it does.
- [ ] Step 3: `run_card.py`: `launch_line()` emits `A=.. B=.. bash local/train.sh`; the dpo/eval strings likewise. Run `.venv/bin/python -m pytest tests/test_run_card.py tests/test_dashboard.py -q`.
- [ ] Step 4: full suite `.venv/bin/python -m pytest tests -q`, count unchanged.
- [ ] Step 5: commit: "docs and run cards: slurm/conda references point at local/ and uv".

### Task 4: GPU smoke (main session, background commands)

Prerequisites: Tasks 1-3a done, `data/train.jsonl` exists.

- [ ] Step 1 HF-generate, no judge: `RUN_ARM=full RUN_SEED=42 RUN_STEPS=2 RUN_LIMIT=64 RUN_NUM_GENERATIONS=4 RUN_MAX_COMPLETION_TOKENS=64 RUN_EFFECTIVE_BATCH=8 RUN_NO_VETO=1 RUN_USE_VLLM=0 RUN_OUT_DIR=runs/smoke_hf GPU=0 bash local/train.sh`. Note: `local/train.sh` must let `RUN_USE_VLLM` be overridden (the sbatch hard-coded `=1`; make it `${RUN_USE_VLLM:-1}`). Pass = `### conpo_train DONE rc=0` in the log and `runs/smoke_hf/run_metadata.json` exists.
- [ ] Step 2 vLLM colocate, no judge: same with `RUN_USE_VLLM=1 RUN_OUT_DIR=runs/smoke_vllm`.
- [ ] Step 3 batched veto: same with `RUN_USE_VLLM=1 RUN_VETO_PERIOD=1 RUN_OUT_DIR=runs/smoke_veto` and `RUN_NO_VETO` unset. Pass = rc=0 and the log shows a veto burst.
- [ ] Step 4 eval: `EVAL_LIMIT=8 GPU=0 bash local/eval.sh` (base model, vllm backend). Pass = `### conpo_eval DONE rc_external=0 rc_internal=0`.
- [ ] Step 5: record durations and peak memory (`nvidia-smi` sampled) in this file; remove the now-verified items from README "What is untested".

### Task 5: Verify, review, record

- [ ] Step 1: fresh-context verifier (templates.md §6) over Tasks 2, 3a, 3b acceptance lines.
- [ ] Step 2: review leg (templates.md §5) over `git diff fa842e3..HEAD`.
- [ ] Step 3: `tasks/lessons.md` in ConPO: the fetch_data weights gap, the deepspeed-not-locked gap.
- [ ] Step 4: `/home/dev/project/ethan/SERVER.md`: add driver 580.126.20, network open to HF/PyPI/PyTorch, the D2/D3 cache decisions, changelog line. No sync push.
- [ ] Step 5: final commit; report to Ethan with the push left to him.

## Session log

- 2026-09-08: plan written; explored repo, caches, GPU, network (all facts above measured this session).
