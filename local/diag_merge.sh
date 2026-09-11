#!/bin/bash
# Diagnosis-only launcher for scripts/diag_merge.py (brief_diag_merge.md).
# Mirrors local/diag_logp.sh's env/venv setup but never touches
# scripts/train.py or the GRPOTrainer: standalone HF+PEFT scoring, no vLLM.
# Hardcodes GPU 0 (GPU 1 owns the live ifrlvr_s42 run, never touch it).
#
#   bash local/diag_merge.sh
#   bash local/diag_merge.sh --num-prompts 2 --num-completions 1 --max-new-tokens 32   # smoke test

set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
source "$(dirname "$(readlink -f "$0")")/lib.sh"
NAME=conpo_diag_merge
export CUDA_VISIBLE_DEVICES=0
conpo_launch_setup

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUT_DIR="${DIAG_MERGE_OUT_DIR:-/home/dev/project/ethan/.cache/tmp/claude-1001/-home-dev-project-ethan-research/84bb2180-5c56-4110-9294-50df603afb3c/scratchpad/diag_merge_out}"
echo "diag_merge: out_dir=${OUT_DIR}"
python -u "${PROJECT}/scripts/diag_merge.py" --out-dir "${OUT_DIR}" "$@"
rc=$?
echo "### conpo_diag_merge DONE rc=${rc} $(date -Is)"
exit ${rc}
