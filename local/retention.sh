#!/bin/bash
# Local devbox, 2x H100 80 GB, no job scheduler: GPU=<idx> picks the card via
# CUDA_VISIBLE_DEVICES (default 0). This script has never been run:
# scripts/eval_retention.py exists and has a "hf" backend exercised by
# tests/test_retention.py (--score-only path only); its "vllm" backend
# (_generate_vllm, imported from scripts/eval_external.py) has never run
# anywhere.
#
# Runs the retention suite (GSM8K, MMLU, BBH; scripts/eval_retention.py) for
# one adapter directory, vLLM backend, on whichever GPU is selected.
#
#   bash local/retention.sh                                # base model
#   bash local/retention.sh "$CONPO_RUNS_DIR/full_s42/adapter"   # one adapter (runs/... after harvest)
#   EVAL_BACKEND=hf bash local/retention.sh "$CONPO_RUNS_DIR/full_s42/adapter"
#
# Env:
#   GPU           GPU index for CUDA_VISIBLE_DEVICES (default 0)
#   EVAL_BACKEND  "vllm" (default here) or "hf"
#   EVAL_OUT_DIR  results directory (default ${CONPO_RUNS_DIR}/retention_<adapter-or-base>_<jobid>)
#   EVAL_LIMIT    cap rows per benchmark, for a smoke run (unset = full)
#   EVAL_MODEL    replication model, overrides config.POLICY_MODEL

set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
source "$(dirname "$(readlink -f "$0")")/lib.sh"
NAME=conpo_retention
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
conpo_launch_setup

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ADAPTER_DIR="${1:-}"
BACKEND="${EVAL_BACKEND:-vllm}"
ADAPTER_NAME="$(basename "${ADAPTER_DIR:-base}")"
OUT_DIR="${EVAL_OUT_DIR:-${CONPO_RUNS_DIR}/retention_${ADAPTER_NAME}_${JOB_ID}}"
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
