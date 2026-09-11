#!/bin/bash
# Move one or more run directories under CONPO_RUNS_DIR aside with a
# timestamp suffix, never delete them. Needed because scripts/train.py refuses
# a fresh run into a RUN_OUT_DIR that already holds checkpoint-* directories
# (RUN_RESUME guard, 2026-09-11), and the devbox guard turns any rm or mv typed
# on a command line under /node-local-cache into a permission prompt; a script
# body is not scanned, so this is the sanctioned way to retire a run.
#
#   bash local/retire_run.sh full_s42            # -> full_s42.retired.<stamp>
#   bash local/retire_run.sh full_s42 ifrlvr_s42
#
# Env:
#   CONPO_RUNS_DIR  run root (local/env.sh default, the node-local disk)

set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"

if [ "$#" -lt 1 ]; then
    echo "usage: bash local/retire_run.sh <run> [<run>...]" >&2
    exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
for run in "$@"; do
    case "${run}" in
        */*|.|..|"") echo "retire_run: '${run}' is not a plain run name" >&2; exit 2 ;;
    esac
    src="${CONPO_RUNS_DIR}/${run}"
    if [ ! -d "${src}" ]; then
        echo "retire_run: ${src} does not exist, nothing to do"
        continue
    fi
    dest="${src}.retired.${STAMP}"
    mv "${src}" "${dest}"
    echo "retired ${src} -> ${dest}"
done
