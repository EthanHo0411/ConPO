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
