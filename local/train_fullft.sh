#!/bin/bash
# Local devbox, 2x H100 80 GB, no job scheduler: CUDA_VISIBLE_DEVICES defaults to
# "0,1" (both cards, since this is the only two-GPU launcher); GPU=<idx> or
# GPU=<idx,idx> overrides it; the accelerate process count follows the
# number of listed cards, and the sharded full-FT needs two. This script
# has never been run. Two open
# questions beyond the usual GPU-untested caveat (local/train.sh's own
# header, RUN_JUDGE_MODE=http wait loop, RUN_OUT_DIR layout):
#   1. TRL's vllm_mode="colocate" generation under a two-process accelerate/
#      FSDP launch -- scripts/train.py's engine_factory always builds one
#      VLLMColocateEngine per process; whether that composes cleanly with
#      FSDP-sharded weights is unverified.
#   2. Optimizer sharding memory -- local/accelerate_fullft.yaml's FSDP
#      config (no param offload) is a guess at fitting fp32 AdamW state for
#      the full Qwen3-4B policy across two 80 GB H100s; may need offload or
#      a smaller per-device batch if it OOMs. deepspeed is not in uv.lock
#      (plan decision D5), so this path uses torch-native FSDP instead of
#      the DeepSpeed ZeRO-2 config this launcher used to specify.
#
# Trains the full-FT point-check (README.md's 21st run, LoRA-vs-full-FT
# comparison) across two H100 80 GB GPUs: RUN_FULL_FT=1 (scripts/train.py,
# peft_config=None) plus accelerate's FSDP launch to shard the resulting
# full optimizer state. Otherwise identical to local/train.sh's RUN_*
# contract (conpo/config.py, README.md); nothing here reads config.py, since
# scripts/train.py imports it directly.
#
#   RUN_ARM=full RUN_SEED=42 RUN_FULL_FT=1 RUN_OUT_DIR=runs/full_fullft_s42 \
#     bash local/train_fullft.sh
#
# (scripts/run_card.py's full_fullft_s42 launch line sets exactly these:
# RUN_ARM=full RUN_SEED=42 RUN_FULL_FT=1 RUN_OUT_DIR=runs/full_fullft_s42 --
# a plain runs/${RUN_ARM}_s${RUN_SEED} default would collide with the LoRA
# full_s42 run's own output directory.)
#
# Env: see local/train.sh's own Env block (TRAIN_SKIP_JUDGE,
# TRAIN_JUDGE_WAIT, TRAIN_JUDGE_ENDPOINT, RUN_SUBSET, RUN_JUDGE_MODE,
# RUN_VETO_PERIOD) -- unchanged here, plus:
#   GPU  GPU index/indices for CUDA_VISIBLE_DEVICES (default "0,1")
# RUN_FULL_FT and RUN_USE_VLLM are forced to "1" below rather than left
# overridable: this script exists only for the full-FT point-check.

set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
source "$(dirname "$(readlink -f "$0")")/lib.sh"
NAME=conpo_train_fullft
export CUDA_VISIBLE_DEVICES="${GPU:-0,1}"
conpo_launch_setup

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# RUN_* contract (conpo/config.py, README.md), same defaults as
# local/train.sh. RUN_FULL_FT/RUN_USE_VLLM are forced, not defaulted.
export RUN_ARM="${RUN_ARM:-full}"
export RUN_SEED="${RUN_SEED:-42}"
export RUN_STEPS="${RUN_STEPS:-500}"
export RUN_TOKEN_BUDGET="${RUN_TOKEN_BUDGET:-0}"
export RUN_FULL_FT=1
export RUN_USE_VLLM=1
export RUN_SUBSET="${RUN_SUBSET:-${PROJECT}/subsets/train_15000_seed42.keys}"
export RUN_OUT_DIR="${RUN_OUT_DIR:-${PROJECT}/runs/${RUN_ARM}_s${RUN_SEED}}"
mkdir -p "${RUN_OUT_DIR}"

echo "arm=${RUN_ARM} seed=${RUN_SEED} steps=${RUN_STEPS} token_budget=${RUN_TOKEN_BUDGET} subset=${RUN_SUBSET} out=${RUN_OUT_DIR}"

# Judge endpoint wait, ported unchanged from local/train.sh: only
# RUN_JUDGE_MODE=http needs it (the default "batched" never resolves
# RUN_JUDGE_ENDPOINT -- scripts/train.py's build_judge picks
# VLLMBatchedJudge/HFBatchedJudge instead).
if [ "${RUN_NO_VETO:-0}" = "1" ] || [ "${TRAIN_SKIP_JUDGE:-0}" = "1" ] || [ "${RUN_JUDGE_MODE:-batched}" != "http" ]; then
    echo "not waiting on the judge server (RUN_NO_VETO=${RUN_NO_VETO:-0}, TRAIN_SKIP_JUDGE=${TRAIN_SKIP_JUDGE:-0}, RUN_JUDGE_MODE=${RUN_JUDGE_MODE:-batched})"
else
    ENDPOINT_FILE="${TRAIN_JUDGE_ENDPOINT:-${PROJECT}/runs/serve/judge.endpoint}"
    WAIT="${TRAIN_JUDGE_WAIT:-3600}"
    echo "waiting up to ${WAIT}s for ${ENDPOINT_FILE}"
    END=$(( $(date +%s) + WAIT ))
    FOUND=0
    while [ "$(date +%s)" -lt "${END}" ]; do
        if [ -s "${ENDPOINT_FILE}" ]; then
            FOUND=1
            break
        fi
        sleep 5
    done
    if [ "${FOUND}" -ne 1 ]; then
        echo "no judge endpoint at ${ENDPOINT_FILE} after ${WAIT}s. Is local/judge_server.sh running?"
        exit 1
    fi
    read -r JHOST JPORT JJOB < "${ENDPOINT_FILE}"
    if ! [[ "${JPORT}" =~ ^[0-9]+$ ]]; then
        echo "malformed endpoint file ${ENDPOINT_FILE}: $(cat "${ENDPOINT_FILE}")"
        exit 1
    fi
    export RUN_JUDGE_ENDPOINT="http://${JHOST}:${JPORT}"
    echo "judge endpoint: ${RUN_JUDGE_ENDPOINT} (job ${JJOB})"
fi

# set +e around the trainer only: with set -e, a nonzero exit here would
# jump straight past the DONE line below instead of reporting it.
set +e
case "${CUDA_VISIBLE_DEVICES}" in
    *[!0-9,]*|*,,*|,*|*,|"")
        echo "GPU must be a comma-separated list of device indices, got '${CUDA_VISIBLE_DEVICES}'"
        exit 2
        ;;
esac
NPROC=$(( $(printf '%s' "${CUDA_VISIBLE_DEVICES}" | tr -cd ',' | wc -c) + 1 ))
accelerate launch --config_file "${PROJECT}/local/accelerate_fullft.yaml" --num_processes "${NPROC}" \
    "${PROJECT}/scripts/train.py"
rc=$?
set -e
echo "### conpo_train_fullft DONE rc=${rc} $(date -Is)"
exit ${rc}
