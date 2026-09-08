#!/bin/bash
# Runs local/eval.sh for the base model plus every runs/*/adapter directory,
# and (RETENTION=1) local/retention.sh for the base model and for each run
# id in RETENTION_RUNS. Sequential, in the foreground: each launcher owns
# whichever GPU it is given (GPU env, default 0) until it exits, since this
# devbox has no queue to serialise them for us.
#
#   bash local/eval_sweep.sh                              # eval.sh only
#   RETENTION=1 bash local/eval_sweep.sh                  # + retention.sh
#   DRY_RUN=1 bash local/eval_sweep.sh                     # print, do not run
#   RETENTION=1 RETENTION_RUNS="full_s42 full_s43" bash local/eval_sweep.sh
#
# Env:
#   RETENTION       "1" to also run local/retention.sh (this script only
#                   assumes its CLI mirrors local/eval.sh's own "positional
#                   adapter dir, omit for base model")
#   RETENTION_RUNS  space-separated run ids, default "full_s42 full_s43
#                   full_s44 ifrlvr_s42 ifrlvr_s43 ifrlvr_s44" (USER
#                   DECISIONS retention set: the full-arm and IF-RLVR
#                   seeds retention.sh scores)
#   DRY_RUN         "1" to print the launcher command lines instead of
#                   running them
#
set -euo pipefail

PROJECT="$(dirname "$(dirname "$(readlink -f "$0")")")"
cd "${PROJECT}"

RETENTION="${RETENTION:-0}"
RETENTION_RUNS="${RETENTION_RUNS:-full_s42 full_s43 full_s44 ifrlvr_s42 ifrlvr_s43 ifrlvr_s44}"
DRY_RUN="${DRY_RUN:-0}"

submit() {
    if [ "${DRY_RUN}" = "1" ]; then
        echo "bash $*"
    else
        bash "$@"
    fi
}

# eval.sh: base model once, then every runs/*/adapter directory (the
# glob stays a literal, unmatched string when no runs/ exist yet -- the
# -d check below skips that case silently).
submit local/eval.sh
for adapter in runs/*/adapter; do
    [ -d "${adapter}" ] || continue
    submit local/eval.sh "${adapter}"
done

if [ "${RETENTION}" = "1" ]; then
    submit local/retention.sh
    for run_id in ${RETENTION_RUNS}; do
        submit local/retention.sh "runs/${run_id}/adapter"
    done
fi
