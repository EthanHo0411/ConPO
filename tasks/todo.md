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
- D2 revised (2026-09-08 09:30): the venv itself moves to `/node-local-cache/cache/conpo-venv`. Measured: `uv sync` there 3.0 s (hardlinks from the node-local cache), `import transformers, torch` 9.7 s; from the NFS `.venv` the same import took 31 s warm earlier and 2m28s under NFS load, and pytest collection exceeded 7 min. `local/env.sh` sets `CONPO_VENV` and `UV_PROJECT_ENVIRONMENT` to the node-local path when it exists (fallback `.venv`). Cost: gone after a node move, rebuilt by one `uv sync`.
- D9 judge placement: the batched veto's `VLLMBatchedJudge` now spawns a child process per burst (commit after 615263a); `RUN_JUDGE_MODE=http` with `GPU=1 bash local/judge_server.sh` is the two-card alternative; `local/judge_server.sh` no longer aliases the model (HttpJudge sends the model id).

---

### Task 1: GPU uv environment (main session, commands only)

**Files:** none changed; creates `.venv/` (gitignored) and populates `/node-local-cache/cache/uv`.

- [ ] Step 1: `source local/env.sh` (sets `UV_CACHE_DIR`, `UV_LINK_MODE`).
- [ ] Step 2: `uv sync --frozen --extra cu130 --extra cluster --extra dev` (background; multi-GB download).
- [ ] Step 3 proving check: `uv run --frozen --extra cu130 --extra cluster --extra dev python -c "import torch, vllm; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count(), vllm.__version__)"` prints `2.11.0+cu130 True 2 0.26.0`.
- [ ] Step 4 regression: `.venv/bin/python -m pytest tests -q` (README baseline in the cpu venv: 882 passed, 14 skipped). Record the counts here.
- [ ] Step 5: measure `time .venv/bin/python -c "import torch, transformers, trl, vllm"` (NFS venv import cost) and record it below.

Result (2026-09-08): steps 1-3 and 5 done. `uv sync` installed 217 packages in 1m34s (cache on node-local ZFS); interpreter is /usr/bin/python3.12 (3.12.3). Proving check printed `2.11.0+cu130 True 2 0.26.0 5.14.1 1.9.2 0.20.0`. Import of torch+vllm+transformers+trl+peft: 66s cold, 31s warm (NFS venv; acceptable per-run cost, revisit with CONPO_VENV on node-local only if it hurts). Step 4 full suite: running in the background, counts pending.
Guard note: `source local/env.sh` from a Claude session is allowed only because env.sh is exports-only (commit 1adef22); the tee/cd prologue lives in local/lib.sh, which launchers source and sessions never do. Scripts passed to `bash` are not scanned by the guard.

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

Result (2026-09-08): staging ran twice. First run staged Qwen/Qwen3-4B (7.5 GB, 3 shards, dir `dev:dev 2755`) plus the IF datasets into the shared hub in 1m13s, then crashed on Qwen/Qwen3-14B: that entry is root-owned and its `.locks/` dir is not group-writable, so the online `snapshot_download` fails taking a lock even with nothing to download. Fix: offline lookup first (`local_files_only=True`), online only on `LocalEntryNotFoundError`. Second run: 47s, both models "already cached". Raw dataset files and processed arrow live in the in-boundary `HF_DATASETS_CACHE` (324 MB). Corpus build (offline): 3m26s, train 81897 / dev 1000 / eval_i 1500 / eval_ii 2952 / eval_iii 6947 rows.
Side finding: the first test run created `/home/dev/nltk_data` (159 files) outside the boundary. Cause: the vendored `instructions_util.py` calls `nltk.download()` at import time without a target, and nltk's module-level downloader snapshots `~/nltk_data` as its default at the first `import nltk` in the process. Fixed in both vendored modules (repo-root `nltk_data/`, explicit `download_dir=`) and `ensure_nltk` now stages all four packages; proven twice under a fake HOME. `/home/dev/nltk_data` itself is held for Ethan (never deleted from a session).

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

Results (2026-09-08):
- Step 1 HF path: `### conpo_train DONE rc=0`, 2 steps in 16m31s (495 s/step: HF generate runs without KV cache under gradient checkpointing, so this path is smoke-only), 56423 tokens generated, GPU 0 peak about 10.9 GB, `runs/smoke_hf/{adapter,checkpoint-2,run_metadata.json,mechanism.jsonl,family_stats.json,edges.jsonl}` written. Model loaded from the shared hub offline.
- Step 4 eval, first attempt: vLLM engine core failed: Triton JIT needs `Python.h`, and uv had picked the apt `/usr/bin/python3.12` (3.12.3, no headers). Fix: `.python-version` pinned to `3.12.11` (uv-managed build with headers, installed under `.claude-ethan/uv-python`), venv rebuilt. Side effect found: `uv python install` also wrote a `python3.12` symlink to `/opt/local/bin` (the container's `XDG_BIN_HOME`), outside the boundary; `UV_PYTHON_BIN_DIR` added to `.ethan-env`, symlink held for Ethan.
- TRL 1.9.2 warns it supports vLLM 0.17.0 to 0.25.1 while uv.lock pins 0.26.0; colocate smoke (step 2) is the test of that pairing.
- Step 2 vLLM colocate, first attempt: `TypeError: '>' not supported between 'str' and 'int'` inside vLLM input validation. Root cause in ConPO, not the pairing: `conpo/engine.py` passed `apply_chat_template(tokenize=True)` output to vLLM, which under transformers 5.14 is a `BatchEncoding` dict. Fixed with `return_dict=False` (commit 677cb89) plus a contract test in `tests/test_engine.py`.
- Step 2 retry: `### conpo_train DONE rc=0`, 2 steps in 1m17s (38.5 s/step), 12812 tokens, GPU 0 peak about 29 GB during engine start; `runs/smoke_vllm/` complete. TRL 1.9.2 + vLLM 0.26.0 colocate works despite TRL's version warning.
- Step 3 batched veto, first attempt: step 1 trained (20.8 s), then the veto burst built the judge's `vllm.LLM` inside the training process (TRL's external_launcher engine forces vLLM's in-process mode for the whole process, `InprocClient`), loaded Qwen3-14B (8 shards, about 55 s), and died in `profile_run` with `RuntimeError: CUDA driver error: invalid argument` next to the sleeping colocated engine (cumem allocator). Two vLLM engines in one process do not coexist. Fix in flight: `VLLMBatchedJudge` runs each burst in a spawned child process (fresh CUDA context). Alternative that needs no code: `RUN_JUDGE_MODE=http` with `GPU=1 bash local/judge_server.sh`, being smoked in parallel.
- Step 3b http judge (2026-09-08 09:32): `GPU=1 bash local/judge_server.sh` (after dropping the `judge` alias) plus `RUN_JUDGE_MODE=http RUN_VETO_PERIOD=1 ... GPU=0 bash local/train.sh` from the node-local venv: `### conpo_train DONE rc=0`, 2 steps, 8 judge calls answered 200, `runs/smoke_http3/{adapter,checkpoint-2,judge_replies.jsonl,family_stats.json,run_metadata.json}` written, about 2 min wall clock including imports. The same launch from the NFS venv sat 11 min in an NFS RPC wait before I killed it.
- Step 3 batched veto, child-process judge (09:45): first retry died in the child with `No available memory for the cache blocks` (judge default `gpu_memory_utilization=0.35` = 27.7 GiB, below the 27.5 GiB weights plus KV); raised to 0.6. Second retry: `### conpo_train DONE rc=0`, 2 steps in 4m00s (120 s per step, each with a 14B judge load in a fresh child), 8 verdicts parsed (`judge_replies.jsonl`), full artifact set in `runs/smoke_veto3/`. The default one-card design now runs on this box.
- Eval, second attempt (Python 3.12.11 venv): `eval_external.py --backend vllm` generated 8 rows and wrote `runs/eval_base_smoke_eval2/external.json`, then its process never exited (parent in `wait()`, vLLM EngineCore child in a futex) and held GPU 1 for 7 min until killed; `eval_internal.py` never started. Fix: `VLLM_ENABLE_V1_MULTIPROCESSING=0` in `local/env.sh` (in-process engine for one-shot scripts; `local/judge_server.sh` sets it back to 1). Third attempt running with it.

Final suite (node-local venv, after every fix): 888 passed, 14 skipped in 7m34s (2026-09-08 10:20, commit 8b9b2e9; round 3 changed comments, a docstring and an unused import only, and the 34 judge tests re-passed after it).

### Task 5: Verify, review, record

Round 1 (2026-09-08 09:55): verifier 10 of 10 PASS. Reviewer: spec PASS, quality FAIL on one defect (`_run_burst` used `Pool.apply` with no timeout, so a child killed by the OOM killer or a CUDA abort would hang the training step forever) plus five smaller findings (full-FT run card still said not implemented; three launcher headers still said never run; stale `_vllm_burst` comment; staging note; full-FT process count hardcoded). All six fixed: `apply_async(...).get(timeout=burst_timeout_s)` raising RuntimeError through the hook-restoring finally (test added), real launch line for `full_fullft_s42`, headers rewritten, comment updated, note added, `--num_processes` derived from the GPU list. 144 affected tests pass. Round 2 reviewer dispatched on the fix diff.
Round 2 (10:15): spec 6 of 6 PASS; quality FAIL on `Pool.__exit__` joining a wedged child without a bound (a driver-level hang would still block the step), plus a README count duplicate and unvalidated GPU lists in `train_fullft.sh`. Fixed: `_run_burst` now uses `Process` + `Pipe` with `poll(burst_timeout_s)`, EOF detection for a dead child, a capped `join` followed by `kill`, and the child's traceback re-raised as RuntimeError; four lifecycle tests with fake pipe ends; GPU list validated; README points at one count. Round 3 reviewer on the new diff and a GPU re-smoke of the batched veto follow.
GPU re-smoke on the Process+Pipe path (10:11): `### conpo_train DONE rc=0`, 2 steps in 1m50s (55 s per step), 8 of 8 verdicts parse_ok, no leftover child processes, both cards freed (`runs/smoke_veto4/`).
Round 3 (10:25): spec PASS; quality findings were a stale comment, the undocumented residual (a child stuck in an uninterruptible driver wait survives SIGKILL and is joined at interpreter exit; no user-space fix), an unused import, and README count wording. All four applied. Review loop closed at three rounds per docs/review-routing.md; the residual is documented in `_run_burst`'s docstring for Ethan.


- [ ] Step 1: fresh-context verifier (templates.md §6) over Tasks 2, 3a, 3b acceptance lines.
- [ ] Step 2: review leg (templates.md §5) over `git diff fa842e3..HEAD`.
- [ ] Step 3: `tasks/lessons.md` in ConPO: the fetch_data weights gap, the deepspeed-not-locked gap.
- [ ] Step 4: `/home/dev/project/ethan/SERVER.md`: add driver 580.126.20, network open to HF/PyPI/PyTorch, the D2/D3 cache decisions, changelog line. No sync push.
- [ ] Step 5: final commit; report to Ethan with the push left to him.

## Session log

- 2026-09-08: plan written; explored repo, caches, GPU, network (all facts above measured this session).

## 2026-09-11 takeover from the handoff: Ethan's decisions and the work plan

Ethan's decisions (2026-09-11, in his words, paraphrased):
1. Primary model is `Qwen/Qwen3-4B` non-thinking; the Instruct-2507 variant is retired completely.
2. The ConPO checkout on this devbox is the truth (`handoff/conpo` is a symlink to it).
3. M0: a 50-step production-shape probe first, then the real `full_s42`.
4. This session owns the handoff repo; the Windows session does nothing from now on.
5. Node-local disk is wiped about every 48 hours: immediate outputs and logs go to node-local; Ethan reviews and says when to move them into the repo.
6. Both GPUs are ours: never wait on jobs, always run evals, baselines or tests on a free card; one card for testing changes while the other runs evals.
7. HackMD: tokens and scripts move to our own config; this session publishes and updates from now on.
8. Stage `allenai/Llama-3.1-Tulu-3-8B-SFT` and `allenai/Olmo-3-7B-Instruct-SFT` into the shared hub now.
9. Full-FT point-check stays, scheduled last.
10. AlpacaEval 2 and the optional statistics stay on hold.
Also: audit IFBench's IF-RLVR GRPO pipeline (open source? cloneable?) as the strongest comparator; implement it if it is not.

Running as of 2026-09-11 (outputs under /node-local-cache/cache/conpo-runs):
- `probe_full_s42`: `RUN_ARM=full RUN_SEED=42 RUN_STEPS=50 GPU=0` at production knobs (M0 numbers).
- `eval_base_full`: `GPU=1 bash local/eval.sh` on the base model, full IFEval + IFBench + internal splits (the untrained floor).
- staging of the two ablation models into the shared hub; reference clones of allenai/open-instruct and allenai/IFBench under /home/dev/project/ethan/refs.

Plan (Sonnet 5 xhigh builders, xhigh reviewers, fresh verifier):
- [x] A. local/: (e6af63b) `CONPO_RUNS_DIR` (node-local) as the default output root for every launcher, logs beside it, venv self-heal after a wipe, `local/harvest.sh` to copy finished runs into `runs/`; fetch_data.py stages the ablation models; README updated.
- [x] B. handoff docs: (handoff 318efd7, 051d345) resolve the two flagged inconsistencies, retire 2507 everywhere it is called ConPO's model, fix the stale rows, details note scheduler/environment rows, publish scripts read the token from our config; resume point 2026-09-11.
- [x] C. IF-RLVR pipeline audit (handoff 4f5eb9e) written to handoff/ifrlvr-pipeline-audit.md.
- [ ] D. After the probe: record steps/hour, peak memory, tokens/step; launch `full_s42` on GPU 0; keep GPU 1 on evals (retention on base, then eval of every finished adapter).
- [ ] E. Full-FT point-check last; AlpacaEval 2 held.

### M0 probe results (2026-09-11, production knobs, one H100 each, outputs under /node-local-cache/cache/conpo-runs)
- `probe_full_s42` (`RUN_ARM=full`, GPU 0): steps 0-2 took 395, 400, 406 s each; 174,624 / 177,704 / 223,521 generated tokens per step (about 440 tokens/s against about 9,000 tokens/s from the standalone vLLM eval engine); 31 GB peak by nvidia-smi; degenerate group rate 0.81-0.88; forward repair at step 0: 388 no_improvement, 32 distance, 7 admitted; 0 edges in the buffer before the first veto burst (K=25). Crashed at step 3: `VLLMValidationError`, a 3,731-token prompt against `MAX_MODEL_LEN` 3584. One 15k pass at this rate would be about 4 days, against the 18 H100-hour pre-M0 estimate.
- `probe_ifrlvr_s42` (`RUN_ARM=ifrlvr`, GPU 1): 9 steps in 13m50s (about 92 s/step; 59 s at step 1, 108 s/it near the end); 32k-69k tokens per step (mean about 54k); 35 GB peak; degenerate rate 0.44-0.75. Crashed at step 9: a 6,971-token prompt against 3584. One pass would be about 24 h.
- Both crashes share one cause with the base-model internal eval crash (eval_ii, 40,961 tokens): prompts are never capped before reaching vLLM. The 15k subset holds 154 rows over 2,048 tokens, 116 over 2,560, 50 over 3,584, 17 over 8,192 (max 90,649); eval_i/ii/iii hold 23/31/75 rows over 2,048.
- Base-model floor (GPU 1): IFEval prompt-loose 0.815 (strict 0.778), IFBench prompt-loose 0.300 (strict 0.263), decoded at a 1,024 cap (protocol says 2,048; re-run pending the eval fix); retention GSM8K 0.915, MMLU 0.678, BBH 0.499 (BBH dyck_languages and web_of_lies 0.0, word_sorting 0.068: scorer format suspected, under investigation).
- Base-model floor at the locked protocol (2026-09-11 02:21, `eval_base_2048`, 2,048 new tokens, prompts left-truncated to 2,048): IFEval prompt-loose 0.8336 (strict 0.7967; instruction loose 0.8849, strict 0.8585), IFBench prompt-loose 0.3133 (strict 0.2767; instruction loose 0.3285, strict 0.2878); internal PAcc dev 0.264, eval_i 0.284, eval_ii 0.215, eval_iii 0.034 (IAcc 0.536 / 0.534 / 0.621 / 0.511); truncated prompts 0 / 0 external, 4 / 23 / 31 / 75 internal. Both eval scripts exit 0. Full CPU suite after commits e6af63b, ba943ec, 2bd582b: 901 passed, 14 skipped in 6m50s.
- `probe_solve_only_s42` (GPU 0): 5 steps at 92 s/step, rc 0, 32k-69k tokens/step; phase timing (new): TRL generation 24-59 s, optimiser update 32-115 s per step.

### 2026-09-11 03:45 UTC: M0 closed, headline runs launched
- Commits since the takeover block: e6af63b (launchers, node-local outputs, harvest.sh, ablation staging), ba943ec + 2649f46 + 649ad76 + eb08d11 (retention verdicts carry responses; BBH extractor rebuilt on real responses, no gold in extraction), 2bd582b (eval decoding 2048, eval-side prompt truncation), df14029 (load-time prompt cap, judge max_model_len 5120), 8d10b5c (phase timing in mechanism.jsonl, forward-repair proposals batched per repair round, T_c probe memory bounded), 10115bd (docstrings, launcher examples), 8695bc9 (SAVE_STEPS 25, save_total_limit 2). Full suite after the first three: 901 passed, 14 skipped.
- M0 (full arm, production knobs, one H100): 445 s/step before batching (perf_before, 3 steps, 578k tokens), 182 / 171 / 222 s after (perf_after; TRL generation 24-38 s, forward repair 62-72 s, optimiser update 85-116 s; peak 62-68 GB). Adversarial 16-prompt step with five 2,048-token prompts: rc 0, peak 77.6 GB (was OOM at 96.5 GB before the probe fix). One 15k pass is 937 steps of 16 prompts: about 50 h for full, about 24 h for ifrlvr (92 s/step), 92 s/step for solve_only. Estimates in hackmd/conpo-plan.md section 6 (18 and 6 H100-h) are 2.8x and 4x low; the notes need a rewrite once steady-state numbers exist.
- Base-model floor, locked protocol: IFEval prompt-loose 0.8336, IFBench prompt-loose 0.3133; PAcc dev 0.264 / eval_i 0.284 / eval_ii 0.215 / eval_iii 0.034; retention GSM8K 0.9151, MMLU 0.6779, BBH 0.5738 under the rebuilt extractor (word_sorting 0.196 and logical_deduction_seven_objects 0.076 remain low: ordering errors and the 512-token cap, both real).
- Launched at 03:45 with commit 8695bc9: `full_s42` (`RUN_ARM=full RUN_SEED=42 RUN_STEPS=937 GPU=0`) and `ifrlvr_s42` (`RUN_ARM=ifrlvr RUN_SEED=42 RUN_STEPS=937 GPU=1`), outputs under /node-local-cache/cache/conpo-runs, checkpoints every 25 steps. Each hourly loop tick harvests both with `KEEP_CHECKPOINTS=1 bash local/harvest.sh full_s42 ifrlvr_s42` into runs/ on NFS. Task I (RUN_RESUME with ConPO state persisted at every save) is being built so a wipe costs at most 25 steps.
- Incident: two builders hung 40 min on command-line writes to /node-local-cache (permission prompt with Ethan away); stopped, work recovered from a stash, lesson recorded in tasks/lessons.md.
