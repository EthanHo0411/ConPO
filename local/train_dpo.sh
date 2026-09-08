#!/bin/bash
# Local devbox, 2x H100 80 GB, no job scheduler: GPU=<idx> picks the card via
# CUDA_VISIBLE_DEVICES (default 0). This script has never been run:
# scripts/build_pairs.py and scripts/train_dpo.py exist and their RUN_ARM/
# RUN_SEED/RUN_STEPS/RUN_OUT_DIR/RUN_PAIRS reads are read directly from
# scripts/train_dpo.py (see its docstring), but the real-model, real-GPU
# path (as opposed to train_dpo.py's --tiny CPU smoke path) has never run,
# and neither has build_pairs.py's --engine vllm path
# (_VLLMStandaloneEngine, lazily imported).
#
# Trains one DPO baseline on one H100 80 GB GPU (vf_dpo | edge_dpo, USER
# DECISIONS 2026-09-02(c)): step 1 builds RUN_PAIRS with
# scripts/build_pairs.py (skipped when a pairs file already exists there),
# step 2 trains on it with scripts/train_dpo.py. Both scripts read their
# configuration from the RUN_* environment contract (README.md's "The RUN_*
# contract"); nothing here reads conpo/config.py directly.
#
#   RUN_ARM=vf_dpo RUN_SEED=42 bash local/train_dpo.sh
#   RUN_ARM=edge_dpo RUN_SEED=42 RUN_EDGES=runs/full_s42/edges.jsonl \
#     bash local/train_dpo.sh
#
# scripts/run_card.py's status logic (derive_status, via load_metadata)
# reads RUN_OUT_DIR/metadata.json -- scripts/train_dpo.py's own artefact
# (not run_metadata.json, the GRPO/ConPOTrainer path's name only) -- to
# tell "planned" from "running"/"done"; a card for a DPO run stays
# "planned" until this script's step 2 has written that file.
#
# Env:
#   GPU         GPU index for CUDA_VISIBLE_DEVICES (default 0)
#   RUN_ARM     "vf_dpo" or "edge_dpo", required
#   RUN_SEED    default 42
#   RUN_STEPS   optimiser-step budget, default 500 (design choice, builder,
#               2026-09-04, matching local/train.sh's default)
#   RUN_OUT_DIR default runs/${RUN_ARM}_s${RUN_SEED}
#   RUN_PAIRS   default ${RUN_OUT_DIR}/pairs.jsonl; step 1 skips building
#               when a non-empty file already exists there
#   RUN_SUBSET  vf_dpo only, key list build_pairs.py --subset filters
#               --rows to, default subsets/train_15000_seed42.keys
#   RUN_EDGES   edge_dpo only, an existing edges.jsonl from a finished full
#               run, default runs/full_s${RUN_SEED}/edges.jsonl -- step 1
#               fails loud when this does not exist

set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
NAME=conpo_train_dpo
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
conpo_launch_setup

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# RUN_* contract (README.md). Every value below is overridable from the
# calling shell; defaults apply only when the caller left the variable
# unset.
export RUN_ARM="${RUN_ARM:?RUN_ARM is required (vf_dpo or edge_dpo)}"
case "${RUN_ARM}" in
    vf_dpo|edge_dpo) ;;
    *)
        echo "RUN_ARM=${RUN_ARM} is not a DPO arm (expected vf_dpo or edge_dpo)"
        exit 1
        ;;
esac
export RUN_SEED="${RUN_SEED:-42}"
export RUN_STEPS="${RUN_STEPS:-500}"
export RUN_OUT_DIR="${RUN_OUT_DIR:-${PROJECT}/runs/${RUN_ARM}_s${RUN_SEED}}"
export RUN_PAIRS="${RUN_PAIRS:-${RUN_OUT_DIR}/pairs.jsonl}"
export RUN_SUBSET="${RUN_SUBSET:-${PROJECT}/subsets/train_15000_seed42.keys}"
mkdir -p "${RUN_OUT_DIR}"

echo "arm=${RUN_ARM} seed=${RUN_SEED} steps=${RUN_STEPS} pairs=${RUN_PAIRS} out=${RUN_OUT_DIR}"

# Step 1: build RUN_PAIRS when it does not already exist (a rerun after an
# interrupted training step should not rebuild pairs it already has).
if [ -s "${RUN_PAIRS}" ]; then
    echo "pairs already exist at ${RUN_PAIRS}, skipping build_pairs.py"
elif [ "${RUN_ARM}" = "edge_dpo" ]; then
    RUN_EDGES="${RUN_EDGES:-${PROJECT}/runs/full_s${RUN_SEED}/edges.jsonl}"
    if [ ! -s "${RUN_EDGES}" ]; then
        echo "RUN_EDGES=${RUN_EDGES} does not exist (edge_dpo needs a finished full run's edges.jsonl)"
        exit 1
    fi
    echo "building pairs: --mode edges --edges ${RUN_EDGES}"
    python -u "${PROJECT}/scripts/build_pairs.py" --mode edges --edges "${RUN_EDGES}" --out "${RUN_PAIRS}"
else
    echo "building pairs: --mode vf --subset ${RUN_SUBSET} --engine vllm"
    python -u "${PROJECT}/scripts/build_pairs.py" --mode vf --subset "${RUN_SUBSET}" --engine vllm \
        --seed "${RUN_SEED}" --out "${RUN_PAIRS}"
fi

# set +e around the trainer only: with set -e, a nonzero exit here would
# jump straight past the DONE line below instead of reporting it.
set +e
python -u "${PROJECT}/scripts/train_dpo.py"
rc=$?
set -e
echo "### conpo_train_dpo DONE rc=${rc} $(date -Is)"
exit ${rc}
