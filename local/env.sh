#!/bin/bash
# Shared environment for every launcher in local/. Source it; never run it.
#
# Devbox facts (see /home/dev/project/ethan/SERVER.md): one node, 2x H100 80 GB,
# no SLURM, 128 GB RAM cgroup cap. /home/dev is NFS (persists, 94% full).
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

# Stands in for SLURM_JOB_ID in log and output names.
JOB_ID="${JOB_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
export JOB_ID

# Every launcher calls this after setting NAME; it activates the venv and tees
# stdout+stderr into logs/<NAME>_<JOB_ID>.out (what SLURM's --output did).
conpo_launch_setup() {
    cd "${PROJECT}"
    mkdir -p "${PROJECT}/logs"
    exec > >(tee -a "${PROJECT}/logs/${NAME}_${JOB_ID}.out") 2>&1
    echo "node=$(hostname) date=$(date -Is) job=${JOB_ID} gpus=${CUDA_VISIBLE_DEVICES:-all}"
    nvidia-smi -L || echo "nvidia-smi -L failed"
    # shellcheck disable=SC1091
    source "${CONPO_VENV:-${PROJECT}/.venv}/bin/activate"
    echo "env=${VIRTUAL_ENV:-none} python=$(python -V 2>&1)"
}
