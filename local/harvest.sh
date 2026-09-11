#!/bin/bash
# Copies a finished run out of the node-local ${CONPO_RUNS_DIR} into
# ${PROJECT}/runs/, the persisted (NFS) location: the node-local disk is
# wiped about every 48 hours (Ethan's decision 2026-09-11), so this is the
# step that moves a run somewhere that survives the wipe. Ethan decides when
# a run is finished enough to harvest; this script does not check status.
#
#   bash local/harvest.sh                              # every dir under CONPO_RUNS_DIR
#   bash local/harvest.sh probe_full_s42               # one named run only
#   KEEP_CHECKPOINTS=1 bash local/harvest.sh full_s42  # also copy checkpoint-*/
#   DRY_RUN=1 bash local/harvest.sh                    # print the rsync lines, copy nothing
#
# Env:
#   CONPO_RUNS_DIR    source directory (local/env.sh, node-local disk)
#   KEEP_CHECKPOINTS  "1" to also copy checkpoint-*/ directories (default
#                     "0": skipped, they are the largest and least useful
#                     part of a harvested run) -- REQUIRED for a resumable
#                     copy: RUN_RESUME (scripts/train.py) needs a
#                     checkpoint-N directory (weights, optimiser, and
#                     ConPO's own conpo_state.json/conpo_edges.jsonl) to
#                     resume from, and the default skips every one of them.
#   DRY_RUN           "1" to print the rsync command lines only
#
# Resuming a harvested run (RUN_RESUME, devbox 2026-09-11): the node-local
# disk is wiped about every 48 hours, so a run that outlives one wipe must
# restart from a KEEP_CHECKPOINTS=1 harvest, either copied back under
# CONPO_RUNS_DIR or pointed at directly with RUN_OUT_DIR=runs/<run>, then
# relaunched with RUN_RESUME=1 (local/train.sh). To see the newest
# checkpoint step a harvested run holds before resuming it:
#   ls -d runs/<run>/checkpoint-*/ | sed 's#.*checkpoint-##;s#/##' | sort -n | tail -1

set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"

DEST="${PROJECT}/runs"
mkdir -p "${DEST}"

EXCLUDE=()
if [ "${KEEP_CHECKPOINTS:-0}" != "1" ]; then
    EXCLUDE=(--exclude "checkpoint-*/")
fi

harvest_one() {
    local src="$1"
    local name
    name="$(basename "${src}")"
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "rsync -a ${EXCLUDE[*]} ${src}/ ${DEST}/${name}/"
        return
    fi
    echo "before: $(du -sh "${src}" 2>/dev/null | cut -f1)  ${name}"
    rsync -a "${EXCLUDE[@]}" "${src}/" "${DEST}/${name}/"
    echo "after:  $(du -sh "${DEST}/${name}" 2>/dev/null | cut -f1)  ${name}"
}

if [ $# -ge 1 ]; then
    harvest_one "${CONPO_RUNS_DIR}/$1"
else
    for src in "${CONPO_RUNS_DIR}"/*/; do
        [ -d "${src}" ] || continue
        harvest_one "${src%/}"
    done
fi
