#!/bin/bash
# Shared environment for every launcher in local/, and for a shell that wants
# the same uv/HF cache placement (`source local/env.sh` before `uv sync` or
# `python scripts/fetch_data.py`). Exports only; the launcher prologue that
# redirects output lives in local/lib.sh.
#
# Devbox facts (see /home/dev/project/ethan/SERVER.md): one node, 2x H100 80 GB,
# no job scheduler, 128 GB RAM cgroup cap. /home/dev is NFS (persists, 94% full).
# /node-local-cache is node-local ZFS (fast, 6.8 TB free, NOT synced between
# nodes: anything under it must be re-creatable).
#
# Decisions (2026-09-08, Ethan's permission for the two node-local dirs):
#   uv cache  -> /node-local-cache/cache/uv (disposable; .venv stays in the repo on NFS)
#   HF hub    -> /node-local-cache/huggingface-home/hub (FAR-wide shared model cache;
#                Qwen3-14B and Qwen3-8B were already there, Qwen3-4B and the IF
#                datasets are staged into it by scripts/fetch_data.py)
#   HF datasets cache (processed arrow files) stays inside the boundary (HF_HOME).
# Override any of these per shell with CONPO_UV_CACHE / CONPO_HF_HUB / CONPO_VENV.

# The devbox identity/cache isolation file; a plain shell may not have sourced it.
if [ -z "${UV_PYTHON_INSTALL_DIR:-}" ] && [ -f /home/dev/project/ethan/.ethan-env ]; then
    # shellcheck disable=SC1091
    source /home/dev/project/ethan/.ethan-env
fi

# readlink rather than "cd ... && pwd": the devbox PreToolUse guard scans
# sourced files and refuses a cd whose target is a command substitution.
PROJECT="${PROJECT:-$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")}"
export PROJECT

export UV_CACHE_DIR="${CONPO_UV_CACHE:-/node-local-cache/cache/uv}"
export UV_LINK_MODE=copy  # cache (ZFS) and .venv (NFS) are different filesystems

export HF_HUB_CACHE="${CONPO_HF_HUB:-/node-local-cache/huggingface-home/hub}"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"

export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# In-process vLLM engine for scripts that build a vllm.LLM themselves (eval,
# retention, the batched veto judge): with the default EngineCore subprocess
# the parent sat in wait() and the child in a futex after the outputs were
# written (2026-09-08 eval smoke), and a judge's GPU memory only returns when
# that child exits. TRL's colocate engine is in-process regardless.
# local/judge_server.sh sets this back to 1: a long-lived server wants the
# separate engine process.
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# Stands in for the scheduler's job id in log and output names.
JOB_ID="${JOB_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
export JOB_ID
