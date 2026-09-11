# ConPO on the devbox: durable knowledge

> Architecture, invariants, and decisions a new session needs before touching this repo. Line-level detail lives in the code; history in `tasks/todo.md`; machine facts in `/home/dev/project/ethan/SERVER.md`.

## Invariants

- The `RUN_*` environment contract (README) is the only way runs are configured. Launchers export env; they never retype values `scripts/train.py` reads from `conpo/config.py`.
- Launchers live in `local/`. Each one sources `local/env.sh` (exports only: caches, venv, `JOB_ID`, vLLM engine mode) and `local/lib.sh` (prologue: cd, tee to `logs/<NAME>_<JOB_ID>.out`, venv activation). `GPU=<idx>` selects the card. `env.sh` must stay free of `cd`, `tee`, `exec` and `${X:-default}` write targets: the devbox PreToolUse guard scans sourced files and refuses paths it cannot resolve (SERVER.md §3); scripts passed to `bash` are not scanned.
- Every launcher runs with `HF_HUB_OFFLINE=1`; `scripts/fetch_data.py` stages everything first (datasets, nltk data, policy and judge weights) and is stage-once, not sync.
- `VLLM_ENABLE_V1_MULTIPROCESSING=0` (in-process vLLM engine) for one-shot scripts and the batched judge; `local/judge_server.sh` sets it back to 1. Reason: the EngineCore subprocess left a finished script hanging in `wait()`.
- `VLLMBatchedJudge` runs each burst in a spawned child process (`_run_burst` -> `_burst_entry` -> `_vllm_burst`): two vLLM engines in one process crash CUDA because TRL colocate forces in-process mode. Waits are bounded (`burst_timeout_s`, capped join, SIGKILL); a child stuck in an uninterruptible driver wait is the documented residual. Default `gpu_memory_utilization=0.6` (Qwen3-14B bf16 weights are 27.5 GiB; the sleeping trainer holds about 12 GiB).
- `apply_chat_template(tokenize=True)` returns a dict under transformers 5; any caller wanting ids passes `return_dict=False`.
- Vendored IFBench nltk downloads go to the repo-root `nltk_data/` with an explicit `download_dir`; nltk's default dir is another account's home on this box.

## Placement (SERVER.md is canonical)

| What | Where |
|---|---|
| venv | `/node-local-cache/cache/conpo-venv` (`CONPO_VENV`, `UV_PROJECT_ENVIRONMENT`), Python 3.12.11 per `.python-version`; repo `.venv` is the fallback |
| uv cache | `/node-local-cache/cache/uv` |
| HF hub (models, datasets) | `/node-local-cache/huggingface-home/hub`, shared with other tenants; write only by Ethan's permission |
| processed datasets, xet | inside the boundary (`hf_home/`) |
| run outputs, logs | `runs/`, `logs/` (gitignored) |

## Verified on 2026-09-08 (2-step smokes, real Qwen3-4B)

HF generate (495 s/step), vLLM colocate (38 s/step), batched veto with the child judge (55 s/step at `RUN_VETO_PERIOD=1`), http judge against `local/judge_server.sh` on GPU 1, eval with the vLLM backend on the base model. Suite: 888 passed, 14 skipped. Not yet run: eval with an adapter, retention, DPO, two-GPU full-FT (FSDP config, `deepspeed` is not in `uv.lock`), production knobs, model-axis replications.

## Decisions (who, when)

- Ethan, 2026-09-08: all ConPO changes pre-approved; reversible writes under the two node-local dirs allowed; nothing pushed.
- Session, 2026-09-08 (documented in `tasks/todo.md` D1-D9): `slurm/` deleted; venv and cache on node-local disk; shared hub as `HF_HUB_CACHE`; FSDP instead of DeepSpeed; child-process judge; judge server serves the model id (no alias).
- Two colocate runs that initialise torch.distributed at the same moment collide on the default rendezvous port 29500 (`EADDRINUSE`, seen 2026-09-11); `local/lib.sh` exports `MASTER_PORT` as 29500 plus the first `CUDA_VISIBLE_DEVICES` index unless the caller sets it. Launches on different cards are safe at any spacing; two runs on the same card index need explicit distinct ports.
- Production shape (2026-09-11, after commit 8d10b5c): full arm about 190 s/step and 62-68 GB peak (77.6 GB on a batch with five 2,048-token prompts), ifrlvr and solve_only about 92 s/step; one 15k pass is 937 steps; checkpoints every 25 steps (`SAVE_STEPS`), last two kept; harvest to NFS every hour with `KEEP_CHECKPOINTS=1 bash local/harvest.sh <run>`.
- A fresh run refuses a `RUN_OUT_DIR` that already holds `checkpoint-*` (RUN_RESUME guard); `bash local/retire_run.sh <run>` moves a run aside with a timestamp suffix from inside a script, the only way to retire a node-local run directory without a guard prompt. Checkpoints written before commit 38703cb carry no `conpo_state.json` and cannot be resumed; `full_s42` and `ifrlvr_s42` were restarted on 2026-09-11 04:40 UTC for that reason.

- 2026-09-11 07:42 UTC, RUN_RESUME verified on a real run: `ifrlvr_s42` (GPU 1) was stopped with SIGTERM right after `checkpoint-150` finished writing (rc 143) and relaunched with `RUN_RESUME=1` and the same `RUN_OUT_DIR`; it was back at step 152 four minutes later. The checkpoint restored the token meter (7,698,585 at step 150), edge state and RNG exactly: the redone in-flight update produced the same 56,788 tokens as the discarded one, so `mechanism.jsonl` carries two identical step-150 records (expected: the update after the checkpoint is lost and redone) plus a `resume` event with `checkpoint` and `resumed_from_step`. Logs are appended, not overwritten. Cost of a restart: about four minutes of startup plus whatever steps were done since the last checkpoint (checkpoints every 25 steps).
