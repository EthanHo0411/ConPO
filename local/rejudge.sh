#!/bin/bash
# Local devbox, 2x H100 80 GB, no job scheduler: GPU=<idx> picks the card via
# CUDA_VISIBLE_DEVICES (default 0).
#
# Offline A/B re-judge (scripts/rejudge.py) of one judge_replies.jsonl file
# against conpo/judge.py's vLLM offline path: the stored judge_request as
# sent (baseline) versus a named prompt variant. Training code and the
# prompt it uses are untouched; this only reads a replies file and writes
# its own output.
#
#   GPU=1 bash local/rejudge.sh <replies.jsonl> [--variant ignore_constraints] [--out PATH]
#
# Env:
#   GPU  GPU index for CUDA_VISIBLE_DEVICES (default 0)
#
# --out on the command line overrides the default output path below;
# --variant/--judge-model/--limit/--directions pass straight through to
# scripts/rejudge.py.

set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
source "$(dirname "$(readlink -f "$0")")/lib.sh"
NAME=conpo_rejudge
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
conpo_launch_setup

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

REPLIES="${1:-}"
if [ -z "${REPLIES}" ]; then
    echo "usage: GPU=<n> bash local/rejudge.sh <replies.jsonl> [--variant ignore_constraints] [--out PATH]"
    exit 1
fi
shift

REJUDGE_DIR="${CONPO_RUNS_DIR}/rejudge"
mkdir -p "${REJUDGE_DIR}"
DEFAULT_OUT="${REJUDGE_DIR}/${JOB_ID}.jsonl"

echo "replies=${REPLIES} default_out=${DEFAULT_OUT}"

# "$@" (the caller's own --variant/--out/--judge-model/--limit/--directions,
# if any) comes after --out here: argparse keeps the last value of a
# repeated flag, so an explicit --out on the command line overrides
# DEFAULT_OUT and every other flag passes through untouched.
python -u "${PROJECT}/scripts/rejudge.py" --replies "${REPLIES}" --out "${DEFAULT_OUT}" "$@"

echo "### conpo_rejudge DONE $(date -Is)"
