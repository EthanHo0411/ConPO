#!/bin/bash
# Single-GPU launch of open-instruct's grpo_fast.py with ConPO's IF-RLVR settings
# (Qwen3-4B non-thinking, LoRA, ConPO sampling/optimiser knobs). Knob-by-knob mapping and
# the two open-instruct source patches this depends on: README.md.
#
# Usage:
#   GPU=1 bash train_ifrlvr.sh              # full 15,000-row subset, 937 steps
#   GPU=1 SMOKE=1 bash train_ifrlvr.sh       # 64-row smoke dataset, 3 steps
#   GPU=1 STEPS=50 bash train_ifrlvr.sh      # full dataset, a custom step count
#
# Run as `bash train_ifrlvr.sh` (not sourced): mkdir/log-redirect below happen from inside this
# script file, per the devbox rule that a node-local write typed on the command line hangs on an
# unanswerable "ask" prompt.
set -euo pipefail

GPU="${GPU:?set GPU=<index>, e.g. GPU=1 (never 0: it runs the live full_s42 training)}"
SMOKE="${SMOKE:-0}"
STEPS="${STEPS:-937}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.5}"

if [ -z "${UV_PYTHON_INSTALL_DIR:-}" ]; then
    # shellcheck disable=SC1091
    source /home/dev/project/ethan/.ethan-env
fi

CONPO_ROOT="/home/dev/project/ethan/ConPO"
OI_SRC="/node-local-cache/cache/open-instruct-src"
OI_VENV="/node-local-cache/cache/open-instruct-venv"
CONPO_RUNS_DIR="${CONPO_RUNS_DIR:-/node-local-cache/cache/conpo-runs}"

if [ "${SMOKE}" = "1" ]; then
    STEPS=3
    DATASET_PATH="${CONPO_ROOT}/openinstruct/data/ifrlvr_smoke_64.jsonl"
    RUN_NAME="openinstruct_ifrlvr_smoke"
    # The streaming dataloader needs >= async_steps * num_unique_prompts_rollout prompts just to
    # prefill (open_instruct/grpo_fast.py:2247-2252); the default async_steps=8 needs 128, more
    # than the 64-row smoke set holds. 2 needs only 32, comfortably under 64 even after the
    # length filter drops a few rows.
    ASYNC_STEPS="${ASYNC_STEPS:-2}"
else
    DATASET_PATH="${CONPO_ROOT}/openinstruct/data/ifrlvr_15000_seed42.jsonl"
    RUN_NAME="openinstruct_ifrlvr_s42"
    ASYNC_STEPS="${ASYNC_STEPS:-8}"
fi

if [ ! -f "${DATASET_PATH}" ]; then
    echo "missing ${DATASET_PATH}; run convert_subset.py first (see README.md)" >&2
    exit 1
fi
if [ ! -x "${OI_VENV}/bin/python" ]; then
    echo "missing ${OI_VENV}; run setup_venv.sh first" >&2
    exit 1
fi

# NUM_UNIQUE_PROMPTS_ROLLOUT * NUM_SAMPLES_PER_PROMPT_ROLLOUT = 128 = ConPO's EFFECTIVE_BATCH.
NUM_UNIQUE_PROMPTS_ROLLOUT=16
NUM_SAMPLES_PER_PROMPT_ROLLOUT=8
EFFECTIVE_BATCH=$((NUM_UNIQUE_PROMPTS_ROLLOUT * NUM_SAMPLES_PER_PROMPT_ROLLOUT))
TOTAL_EPISODES=$((STEPS * EFFECTIVE_BATCH))

OUTPUT_DIR="${CONPO_RUNS_DIR}/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${CONPO_RUNS_DIR}/logs"
LOG_FILE="${CONPO_RUNS_DIR}/logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).out"

export CUDA_VISIBLE_DEVICES="${GPU}"
export TORCH_COMPILE_DISABLE=1
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OPEN_INSTRUCT=0.0.0+conpo
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1
export HF_HUB_OFFLINE=1
export HF_HUB_CACHE="/node-local-cache/huggingface-home/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export NLTK_DATA="${CONPO_ROOT}/nltk_data"
export TOKENIZERS_PARALLELISM=false
# See setup_venv.sh: DeepSpeed's op_builder reads this directly and otherwise falls back to
# $HOME/.triton/autotune, outside the boundary.
export TRITON_CACHE_DIR="/node-local-cache/cache/triton"

{
echo "date=$(date -Is) gpu=${GPU} smoke=${SMOKE} steps=${STEPS} total_episodes=${TOTAL_EPISODES} dataset=${DATASET_PATH} output_dir=${OUTPUT_DIR}"
nvidia-smi -L
} | tee -a "${LOG_FILE}"

cd "${OI_SRC}"
"${OI_VENV}/bin/python" open_instruct/grpo_fast.py \
    --exp_name "${RUN_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_name_or_path Qwen/Qwen3-4B \
    --dataset_mixer_list "${DATASET_PATH}" 1.0 \
    --dataset_mixer_list_splits train \
    --enable_thinking False \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --num_unique_prompts_rollout "${NUM_UNIQUE_PROMPTS_ROLLOUT}" \
    --num_samples_per_prompt_rollout "${NUM_SAMPLES_PER_PROMPT_ROLLOUT}" \
    --async_steps "${ASYNC_STEPS}" \
    --max_prompt_token_length 2048 \
    --response_length 1024 \
    --pack_length 4096 \
    --temperature 0.8 \
    --vllm_top_p 0.95 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --beta 0.001 \
    --clip_lower 0.2 \
    --clip_higher 0.2 \
    --advantage_normalization_type standard \
    --total_episodes "${TOTAL_EPISODES}" \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --deepspeed_stage 2 \
    --single_gpu_mode \
    --vllm_tensor_parallel_size 1 \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEM_UTIL}" \
    --vllm_enforce_eager \
    --apply_verifiable_reward true \
    --seed 42 \
    --local_eval_every -1 \
    --push_to_hub false \
    2>&1 | tee -a "${LOG_FILE}"
