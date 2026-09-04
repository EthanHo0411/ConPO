#!/bin/bash
# UNTESTED: written on a machine with no GPU and no vllm install (dev conda
# env "conpo" is torch CPU only). This script has never been submitted or
# run. scripts/eval_retention.py exists and has a "hf" backend exercised by
# tests/test_retention.py (--score-only path only); its "vllm" backend
# (_generate_vllm, imported from scripts/eval_external.py) has never run
# anywhere. Verify the gres line, the conda env name, and this launcher's
# --out path on the mlp cluster before trusting it.
#
# Runs the retention suite (GSM8K, MMLU, BBH; scripts/eval_retention.py) for
# one adapter directory, vLLM backend, on one H100 (whichever card is free).
#
#   sbatch slurm/retention.sbatch                                # base model
#   sbatch slurm/retention.sbatch runs/full_s42/adapter           # one adapter
#   sbatch --export=ALL,EVAL_BACKEND=hf slurm/retention.sbatch runs/full_s42/adapter
#
# Env:
#   EVAL_BACKEND  "vllm" (default here) or "hf"
#   EVAL_OUT_DIR  results directory (default runs/retention_<adapter-or-base>_<jobid>)
#   EVAL_LIMIT    cap rows per benchmark, for a smoke run (unset = full)
#   EVAL_MODEL    replication model, overrides config.POLICY_MODEL
#
#SBATCH --job-name=conpo_retention
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
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
OUT_DIR="${EVAL_OUT_DIR:-${PROJECT}/runs/retention_${ADAPTER_NAME}_${SLURM_JOB_ID}}"
mkdir -p "${OUT_DIR}"

echo "adapter=${ADAPTER_DIR:-<base model>} backend=${BACKEND} out=${OUT_DIR}"

ARGS=(--backend "${BACKEND}" --out "${OUT_DIR}/retention.json")
if [ -n "${ADAPTER_DIR}" ]; then
    ARGS+=(--adapter "${ADAPTER_DIR}")
fi
if [ -n "${EVAL_MODEL:-}" ]; then
    ARGS+=(--model "${EVAL_MODEL}")
fi
if [ -n "${EVAL_LIMIT:-}" ]; then
    ARGS+=(--limit "${EVAL_LIMIT}")
fi

set +e
python -u "${PROJECT}/scripts/eval_retention.py" "${ARGS[@]}"
rc=$?
set -e

echo "### conpo_retention DONE rc=${rc} $(date -Is)"
exit "${rc}"
