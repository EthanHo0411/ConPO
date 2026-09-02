#!/bin/bash
# UNTESTED: written on a machine with no GPU and no vllm install (dev conda
# env "conpo" is torch CPU only). This script has never been submitted or
# run. scripts/eval_external.py and scripts/eval_internal.py exist and have
# a "hf" backend exercised by tests/test_eval_scripts.py; their "vllm"
# backend (each script's _generate_vllm, lazily imported) has never run
# anywhere. Verify the gres line, the conda env name, and this launcher's
# --out paths on the mlp cluster before trusting it.
#
# Runs both eval scripts against one adapter directory, vLLM backend, on one
# H200 3g.71gb MIG slice: scripts/eval_external.py (IFEval + IFBench,
# vendored official scorers) and scripts/eval_internal.py (PAcc/IAcc over the
# four internal splits, conpo/metrics.py). No judge server is needed;
# neither script calls HttpJudge (conpo/metrics.py's task_relevance hook is
# left None here).
#
#   sbatch slurm/eval.sbatch                                # base model
#   sbatch slurm/eval.sbatch runs/full_s42/adapter           # one adapter
#   sbatch --export=ALL,EVAL_BACKEND=hf slurm/eval.sbatch runs/full_s42/adapter
#
# Env:
#   EVAL_BACKEND  "vllm" (default here) or "hf"
#   EVAL_OUT_DIR  results directory (default runs/eval_<adapter-or-base>_<jobid>)
#   EVAL_LIMIT    cap rows per dataset/split, for a smoke run (unset = full)
#
#SBATCH --job-name=conpo_eval
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT}"
mkdir -p logs

echo "node=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID}"
nvidia-smi -L || echo "nvidia-smi -L failed"

source "${CONDA_HOME:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate vllm
echo "env=${CONDA_DEFAULT_ENV:-none} python=$(python -V 2>&1)"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ADAPTER_DIR="${1:-}"
BACKEND="${EVAL_BACKEND:-vllm}"
ADAPTER_NAME="$(basename "${ADAPTER_DIR:-base}")"
OUT_DIR="${EVAL_OUT_DIR:-${PROJECT}/runs/eval_${ADAPTER_NAME}_${SLURM_JOB_ID}}"
mkdir -p "${OUT_DIR}"

echo "adapter=${ADAPTER_DIR:-<base model>} backend=${BACKEND} out=${OUT_DIR}"

# Both scripts take --model/--adapter/--backend/--out (a single JSON file,
# required) rather than an output directory; this launcher points each at
# its own file inside OUT_DIR. --model is left at each script's own
# default (config.POLICY_MODEL); pass EVAL_MODEL for a replication model.
COMMON_ARGS=(--backend "${BACKEND}")
if [ -n "${ADAPTER_DIR}" ]; then
    COMMON_ARGS+=(--adapter "${ADAPTER_DIR}")
fi
if [ -n "${EVAL_MODEL:-}" ]; then
    COMMON_ARGS+=(--model "${EVAL_MODEL}")
fi
if [ -n "${EVAL_LIMIT:-}" ]; then
    COMMON_ARGS+=(--limit "${EVAL_LIMIT}")
fi

# set +e around both scripts: with set -e, a nonzero exit from the first
# would skip the second and jump straight past the DONE line below.
set +e
python -u "${PROJECT}/scripts/eval_external.py" \
    "${COMMON_ARGS[@]}" --out "${OUT_DIR}/external.json"
rc_ext=$?

python -u "${PROJECT}/scripts/eval_internal.py" \
    "${COMMON_ARGS[@]}" --out "${OUT_DIR}/internal.json"
rc_int=$?
set -e

echo "### conpo_eval DONE rc_external=${rc_ext} rc_internal=${rc_int} $(date -Is)"
if [ "${rc_ext}" -ne 0 ] || [ "${rc_int}" -ne 0 ]; then
    exit 1
fi
