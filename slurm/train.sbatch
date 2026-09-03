#!/bin/bash
# UNTESTED: written on a machine with no GPU and no vllm install (dev conda
# env "conpo" is torch CPU only). This script has never been submitted or
# run. scripts/train.py exists and its RUN_ARM/RUN_SEED/RUN_STEPS/
# RUN_USE_VLLM/RUN_OUT_DIR/RUN_NO_VETO reads are read directly from that
# file (below), but its real-model, real-GPU, RUN_USE_VLLM=1 path (as
# opposed to its --tiny CPU smoke path) has never run. Verify the gres
# line and the conda env name on the mlp cluster before trusting it.
#
# Trains one ConPO run on one H100 80 GB GPU: policy + LoRA + colocated vLLM
# generation (vllm_mode "colocate" -- ConPO-Build-Plan.md section 1). Two
# such runs share the two-card estate. Waits for the judge server's
# endpoint file, then launches scripts/train.py entirely through the RUN_*
# environment contract documented in conpo/config.py, conpo/arms.py and
# README.md; nothing here reads config.py, since scripts/train.py imports
# it directly.
#
#   sbatch slurm/train.sbatch                              # RUN_ARM=full, seed 42
#   RUN_ARM=ifrlvr RUN_SEED=43 \
#     sbatch --export=ALL,RUN_ARM,RUN_SEED slurm/train.sbatch
#
# Ablations: any RUN_SELECTION / RUN_NO_VETO / RUN_NO_EXACTNESS /
# RUN_NO_DISTANCE / RUN_LAMBDA_P set in the submitting shell's environment
# pass through unmodified via --export=ALL (see the arm table in
# conpo/experiments.md); this script does not set or validate them.
#
# scripts/train.py builds an HttpJudge unconditionally, for every RUN_ARM,
# unless RUN_NO_VETO=1 (then it uses StaticJudge instead) -- construction
# happens before any forward/reverse call, so even RUN_ARM=solve_only
# still needs a resolvable RUN_JUDGE_ENDPOINT. This script therefore only
# skips the judge wait when RUN_NO_VETO=1; TRAIN_SKIP_JUDGE=1 is a manual
# escape hatch for other cases, e.g. RUN_JUDGE_ENDPOINT already set by hand:
#
#   RUN_ARM=full RUN_NO_VETO=1 \
#     sbatch --export=ALL,RUN_ARM,RUN_NO_VETO slurm/train.sbatch
#
# Two-GPU variant (comment only -- the live directive below stays one
# H100). Use for the LoRA-vs-full-FT point-check (~85 GB needed with fp32
# AdamW, over one 80 GB card; optimizer state sharded across both) and for
# the 7B/8B model-ablation replications (Qwen3-8B, Olmo-3-7B-Instruct-SFT)
# if their LoRA footprint does not fit one card. scripts/train.py derives
# gradient accumulation from the process count, so launch it under
# accelerate/torchrun with 2 processes:
#
#   sbatch --gres=gpu:h100:2 --export=ALL,RUN_ARM,RUN_SEED slurm/train.sbatch
#
# Env:
#   TRAIN_SKIP_JUDGE  "1" to skip waiting on the judge endpoint (default "0")
#   TRAIN_JUDGE_WAIT  seconds to wait for the endpoint file (default 3600)
#   TRAIN_JUDGE_ENDPOINT  endpoint file (default runs/serve/judge.endpoint)
#   RUN_SUBSET        key list every arm trains on (default
#                     subsets/train_15000_seed42.keys, the fixed 15k subset)
#
#SBATCH --job-name=conpo_train
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
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
# no_proxy landmine (README.md): the judge lives on a second node; without
# this, HTTP calls to it are handed to the site proxy and fail.
export no_proxy="localhost,127.0.0.1,::1,$(hostname),$(hostname -f)"
export NO_PROXY="${no_proxy}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# RUN_* contract (conpo/config.py, README.md). Every value below is
# overridable from the submitting shell via --export=ALL,<name>; defaults
# apply only when the caller left the variable unset.
export RUN_ARM="${RUN_ARM:-full}"
export RUN_SEED="${RUN_SEED:-42}"
export RUN_STEPS="${RUN_STEPS:-500}"
export RUN_TOKEN_BUDGET="${RUN_TOKEN_BUDGET:-0}"
export RUN_USE_VLLM=1
export RUN_SUBSET="${RUN_SUBSET:-${PROJECT}/subsets/train_15000_seed42.keys}"
export RUN_OUT_DIR="${PROJECT}/runs/${RUN_ARM}_s${RUN_SEED}"
mkdir -p "${RUN_OUT_DIR}"

echo "arm=${RUN_ARM} seed=${RUN_SEED} steps=${RUN_STEPS} token_budget=${RUN_TOKEN_BUDGET} subset=${RUN_SUBSET} out=${RUN_OUT_DIR}"

# Wait for the judge server's endpoint file and turn it into RUN_JUDGE_ENDPOINT
# (conpo/judge.py:read_endpoint format: "hostname port jobid"). Ported from
# the same polling contract as slurm/judge_server.sbatch's own health wait.
if [ "${RUN_NO_VETO:-0}" = "1" ] || [ "${TRAIN_SKIP_JUDGE:-0}" = "1" ]; then
    echo "not waiting on the judge server (RUN_NO_VETO=${RUN_NO_VETO:-0}, TRAIN_SKIP_JUDGE=${TRAIN_SKIP_JUDGE:-0})"
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
        echo "no judge endpoint at ${ENDPOINT_FILE} after ${WAIT}s. Is slurm/judge_server.sbatch running?"
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
