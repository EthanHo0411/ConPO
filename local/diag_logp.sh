#!/bin/bash
# Diagnosis-only launcher for scripts/diag_logp.py (brief_diag_logp.md).
# Mirrors local/train.sh's env/venv setup (conpo_launch_setup from lib.sh) so
# this runs under the identical CUDA/HF/vLLM environment as the live full_s42
# run, but hardcodes GPU 0 (the live ifrlvr_s42 run owns GPU 1, never touch
# it) and the diag script's own RUN_* contract instead of accepting arm/seed
# overrides from the caller.
#
#   bash local/diag_logp.sh
#   RUN_STEPS=80 bash local/diag_logp.sh   # brief's "run 5 steps" fallback

set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
source "$(dirname "$(readlink -f "$0")")/lib.sh"
NAME=conpo_diag_logp
export CUDA_VISIBLE_DEVICES=0
conpo_launch_setup

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export RUN_ARM=full
export RUN_SEED=42
export RUN_STEPS="${RUN_STEPS:-78}"
export RUN_RESUME=1
export RUN_OUT_DIR=/node-local-cache/cache/conpo-runs/diag_full_ckpt75
export RUN_USE_VLLM=1
export RUN_SUBSET="${PROJECT}/subsets/train_15000_seed42.keys"

echo "diag_logp: steps=${RUN_STEPS} out=${RUN_OUT_DIR}"
python -u "${PROJECT}/scripts/diag_logp.py"
rc=$?
echo "### conpo_diag_logp DONE rc=${rc} $(date -Is)"
exit ${rc}
