#!/bin/bash
# Local devbox, 2x H100 80 GB, no job scheduler: GPU=<idx> picks the card via
# CUDA_VISIBLE_DEVICES (default 0), so two such runs can share the box with
# GPU=0 and GPU=1. This script has never been run: scripts/train.py exists
# and its RUN_ARM/RUN_SEED/RUN_STEPS/RUN_USE_VLLM/RUN_OUT_DIR/RUN_NO_VETO/
# RUN_JUDGE_MODE/RUN_JUDGE_MODEL/RUN_VETO_PERIOD/RUN_MODEL/RUN_LIMIT reads
# are read directly from that file (below), but its real-model, real-GPU,
# RUN_USE_VLLM=1 path (as opposed to its --tiny CPU smoke path) has never
# run, and neither has VLLMBatchedJudge/HFBatchedJudge or the sleep/wake_up
# path.
#
# Trains one ConPO run on one H100 80 GB GPU: policy + LoRA + colocated vLLM
# generation (vllm_mode "colocate" -- ConPO-Build-Plan.md section 1). Launches
# scripts/train.py entirely through the RUN_* environment contract documented
# in conpo/config.py, conpo/arms.py and README.md; nothing here reads
# config.py, since scripts/train.py imports it directly.
#
# Batched veto is the default judge placement (USER DECISIONS 2026-09-02):
# RUN_JUDGE_MODE=batched (unset = batched) needs no judge server at all --
# scripts/train.py's build_judge loads/unloads a VLLMBatchedJudge/
# HFBatchedJudge in periodic bursts (ConPOTrainer.flush_pending_veto), so
# the judge never has to sit resident beside training. This script only
# waits for local/judge_server.sh's endpoint file when RUN_JUDGE_MODE=http
# is explicitly requested (the fallback placement, one card serialised
# against the other, kept for comparison/debugging).
#
#   RUN_ARM=full RUN_SEED=42 bash local/train.sh
#   RUN_ARM=ifrlvr RUN_SEED=43 bash local/train.sh
#
# Ablations and batched-veto knobs: any RUN_SELECTION / RUN_NO_VETO /
# RUN_NO_EXACTNESS / RUN_NO_DISTANCE / RUN_LAMBDA_P / RUN_JUDGE_MODE /
# RUN_JUDGE_MODEL / RUN_VETO_PERIOD / RUN_MODEL / RUN_LIMIT set in the
# calling shell's environment pass through unmodified (see the arm table in
# conpo/experiments.md); this script does not set or validate them.
#
# scripts/train.py's build_judge() picks the judge from RUN_NO_VETO /
# RUN_JUDGE_MODE / RUN_USE_VLLM: RUN_NO_VETO=1 -> StaticJudge; RUN_JUDGE_MODE=
# http -> HttpJudge (needs the server below); RUN_JUDGE_MODE=batched, the
# default -> VLLMBatchedJudge/HFBatchedJudge, no server needed. This script
# only waits for the judge-server endpoint when RUN_JUDGE_MODE=http is set;
# TRAIN_SKIP_JUDGE=1 is a manual escape hatch for that same case, e.g.
# RUN_JUDGE_ENDPOINT already set by hand:
#
#   RUN_ARM=full RUN_JUDGE_MODE=http bash local/train.sh
#
# Two-GPU variant: see local/train_fullft.sh, for the LoRA-vs-full-FT
# point-check and the 7B/8B model-ablation replications. This script stays
# single-GPU.
#
# Env:
#   GPU               GPU index for CUDA_VISIBLE_DEVICES (default 0)
#   TRAIN_SKIP_JUDGE  "1" to skip waiting on the judge endpoint (default "0",
#                     only relevant with RUN_JUDGE_MODE=http)
#   TRAIN_JUDGE_WAIT  seconds to wait for the endpoint file (default 3600)
#   TRAIN_JUDGE_ENDPOINT  endpoint file (default runs/serve/judge.endpoint)
#   RUN_SUBSET        key list every arm trains on (default
#                     subsets/train_15000_seed42.keys, the fixed 15k subset)
#   RUN_JUDGE_MODE    "batched" (default) or "http" (needs the judge server)
#   RUN_USE_VLLM      "1" (default) or "0" for the HF-generate smoke path
#   RUN_VETO_PERIOD   optimiser steps between veto bursts (default config.VETO_PERIOD_K)

set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
source "$(dirname "$(readlink -f "$0")")/lib.sh"
NAME=conpo_train
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
conpo_launch_setup

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# RUN_* contract (conpo/config.py, README.md). Every value below is
# overridable from the calling shell; defaults apply only when the caller
# left the variable unset.
export RUN_ARM="${RUN_ARM:-full}"
export RUN_SEED="${RUN_SEED:-42}"
export RUN_STEPS="${RUN_STEPS:-500}"
export RUN_TOKEN_BUDGET="${RUN_TOKEN_BUDGET:-0}"
export RUN_USE_VLLM="${RUN_USE_VLLM:-1}"
export RUN_SUBSET="${RUN_SUBSET:-${PROJECT}/subsets/train_15000_seed42.keys}"
export RUN_OUT_DIR="${RUN_OUT_DIR:-${PROJECT}/runs/${RUN_ARM}_s${RUN_SEED}}"
mkdir -p "${RUN_OUT_DIR}"

echo "arm=${RUN_ARM} seed=${RUN_SEED} steps=${RUN_STEPS} token_budget=${RUN_TOKEN_BUDGET} subset=${RUN_SUBSET} out=${RUN_OUT_DIR}"

# Wait for the judge server's endpoint file and turn it into RUN_JUDGE_ENDPOINT
# (conpo/judge.py:read_endpoint format: "hostname port jobid"). Only
# RUN_JUDGE_MODE=http needs this: the default (unset, "batched") never
# resolves RUN_JUDGE_ENDPOINT at all (scripts/train.py's build_judge picks
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
python -u "${PROJECT}/scripts/train.py"
rc=$?
set -e
echo "### conpo_train DONE rc=${rc} $(date -Is)"
exit ${rc}
