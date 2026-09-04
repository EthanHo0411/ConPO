#!/bin/bash
# UNTESTED: written on a machine with no GPU and no vllm/deepspeed install
# (dev conda env "conpo" is torch CPU only). This script has never been
# submitted or run. Two open questions beyond the usual GPU-untested caveat
# (slurm/train.sbatch's own header, RUN_JUDGE_MODE=http wait loop, RUN_OUT_DIR
# layout):
#   1. TRL's vllm_mode="colocate" generation under a two-process accelerate/
#      deepspeed launch -- scripts/train.py's engine_factory always builds
#      one VLLMColocateEngine per process; whether that composes cleanly
#      with ZeRO-2-sharded weights is unverified.
#   2. Optimizer sharding memory -- slurm/accelerate_fullft.yaml's ZeRO
#      stage 2 (no offload) is a guess at fitting fp32 AdamW state for the
#      full Qwen3-4B policy across two 80 GB H100s; may need offload or a
#      smaller per-device batch if it OOMs.
#
# Trains the full-FT point-check (README.md's 21st run, LoRA-vs-full-FT
# comparison) across two H100 80 GB GPUs: RUN_FULL_FT=1 (scripts/train.py,
# peft_config=None) plus accelerate's ZeRO-2 launch to shard the resulting
# full optimizer state. Otherwise identical to slurm/train.sbatch's RUN_*
# contract (conpo/config.py, README.md); nothing here reads config.py, since
# scripts/train.py imports it directly.
#
#   sbatch --export=ALL,RUN_ARM,RUN_SEED,RUN_FULL_FT,RUN_OUT_DIR \
#     slurm/train_fullft.sbatch
#
# (scripts/run_card.py's full_fullft_s42 launch line sets exactly these:
# RUN_ARM=full RUN_SEED=42 RUN_FULL_FT=1 RUN_OUT_DIR=runs/full_fullft_s42 --
# a plain runs/${RUN_ARM}_s${RUN_SEED} default would collide with the LoRA
# full_s42 run's own output directory.)
#
# Env: see slurm/train.sbatch's own Env block (TRAIN_SKIP_JUDGE,
# TRAIN_JUDGE_WAIT, TRAIN_JUDGE_ENDPOINT, RUN_SUBSET, RUN_JUDGE_MODE,
# RUN_VETO_PERIOD) -- unchanged here. RUN_FULL_FT and RUN_USE_VLLM are
# forced to "1" below rather than left overridable: this script exists
# only for the full-FT point-check.
#
#SBATCH --job-name=conpo_train_fullft
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=16:00:00
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

# RUN_* contract (conpo/config.py, README.md), same defaults as
# slurm/train.sbatch. RUN_FULL_FT/RUN_USE_VLLM are forced, not defaulted.
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

# Judge endpoint wait, ported unchanged from slurm/train.sbatch: only
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
accelerate launch --config_file "${PROJECT}/slurm/accelerate_fullft.yaml" --num_processes 2 \
    "${PROJECT}/scripts/train.py"
rc=$?
set -e
echo "### conpo_train_fullft DONE rc=${rc} $(date -Is)"
exit ${rc}
