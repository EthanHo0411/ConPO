#!/bin/bash
# Submits slurm/eval.sbatch for the base model plus every runs/*/adapter
# directory, and (RETENTION=1) slurm/retention.sbatch for the base model
# and for each run id in RETENTION_RUNS. Plain bash, not an sbatch script
# itself -- run it from a login/head node with sbatch already on PATH.
#
#   bash slurm/eval_sweep.sh                              # eval.sbatch only
#   RETENTION=1 bash slurm/eval_sweep.sh                  # + retention.sbatch
#   DRY_RUN=1 bash slurm/eval_sweep.sh                     # print, do not submit
#   RETENTION=1 RETENTION_RUNS="full_s42 full_s43" bash slurm/eval_sweep.sh
#
# Env:
#   RETENTION       "1" to also submit slurm/retention.sbatch (being
#                   written by another builder in parallel; this script
#                   only assumes its CLI mirrors slurm/eval.sbatch's own
#                   "positional adapter dir, omit for base model")
#   RETENTION_RUNS  space-separated run ids, default "full_s42 full_s43
#                   full_s44 ifrlvr_s42 ifrlvr_s43 ifrlvr_s44" (USER
#                   DECISIONS retention set: the full-arm and IF-RLVR
#                   seeds retention.sbatch scores)
#   DRY_RUN         "1" to print the sbatch command lines instead of
#                   submitting them
#
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT}"

RETENTION="${RETENTION:-0}"
RETENTION_RUNS="${RETENTION_RUNS:-full_s42 full_s43 full_s44 ifrlvr_s42 ifrlvr_s43 ifrlvr_s44}"
DRY_RUN="${DRY_RUN:-0}"

submit() {
    if [ "${DRY_RUN}" = "1" ]; then
        echo "sbatch $*"
    else
        sbatch "$@"
    fi
}

# eval.sbatch: base model once, then every runs/*/adapter directory (the
# glob stays a literal, unmatched string when no runs/ exist yet -- the
# -d check below skips that case silently).
submit slurm/eval.sbatch
for adapter in runs/*/adapter; do
    [ -d "${adapter}" ] || continue
    submit slurm/eval.sbatch "${adapter}"
done

if [ "${RETENTION}" = "1" ]; then
    submit slurm/retention.sbatch
    for run_id in ${RETENTION_RUNS}; do
        submit slurm/retention.sbatch "runs/${run_id}/adapter"
    done
fi
