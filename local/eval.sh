#!/bin/bash
# Local devbox, 2x H100 80 GB, no job scheduler: GPU=<idx> picks the card via
# CUDA_VISIBLE_DEVICES (default 0). Verified on this box on 2026-09-08 for
# the base model with the vllm backend (EVAL_LIMIT=8, both scripts wrote
# their JSON and exited cleanly); the adapter (LoRARequest) path and the
# hf backend's generation have not run, see README "What is verified on
# the devbox, and what is not".
#
# Runs both eval scripts against one adapter directory, vLLM backend, on
# whichever GPU is selected: scripts/eval_external.py (IFEval + IFBench,
# vendored official scorers) and scripts/eval_internal.py (PAcc/IAcc over the
# four internal splits, conpo/metrics.py). No judge server is needed;
# neither script calls HttpJudge (conpo/metrics.py's task_relevance hook is
# left None here).
#
#   bash local/eval.sh                                # base model
#   bash local/eval.sh runs/full_s42/adapter           # one adapter
#   EVAL_BACKEND=hf bash local/eval.sh runs/full_s42/adapter
#
# Env:
#   GPU           GPU index for CUDA_VISIBLE_DEVICES (default 0)
#   EVAL_BACKEND  "vllm" (default here) or "hf"
#   EVAL_OUT_DIR  results directory (default ${CONPO_RUNS_DIR}/eval_<adapter-or-base>_<jobid>)
#   EVAL_LIMIT    cap rows per dataset/split, for a smoke run (unset = full)

set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
source "$(dirname "$(readlink -f "$0")")/lib.sh"
NAME=conpo_eval
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
conpo_launch_setup

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ADAPTER_DIR="${1:-}"
BACKEND="${EVAL_BACKEND:-vllm}"
ADAPTER_NAME="$(basename "${ADAPTER_DIR:-base}")"
OUT_DIR="${EVAL_OUT_DIR:-${CONPO_RUNS_DIR}/eval_${ADAPTER_NAME}_${JOB_ID}}"
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
