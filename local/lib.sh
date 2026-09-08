#!/bin/bash
# Launcher prologue shared by local/*.sh. Sourced by launchers only, never by
# an interactive shell: it redirects the caller's stdout and stderr.
# Kept apart from env.sh so that `source local/env.sh` stays a pure export
# file: the devbox PreToolUse guard scans sourced files and refuses redirect
# targets it cannot resolve, such as the tee below.

# Call after NAME and CUDA_VISIBLE_DEVICES are set. Activates the venv and
# tees stdout+stderr into logs/<NAME>_<JOB_ID>.out (what the scheduler's
# --output option used to do).
conpo_launch_setup() {
    cd "${PROJECT}"
    mkdir -p "${PROJECT}/logs"
    exec > >(tee -a "${PROJECT}/logs/${NAME}_${JOB_ID}.out") 2>&1
    echo "node=$(hostname) date=$(date -Is) job=${JOB_ID} gpus=${CUDA_VISIBLE_DEVICES:-all}"
    nvidia-smi -L || echo "nvidia-smi -L failed"
    # shellcheck disable=SC1091
    source "${CONPO_VENV:-${PROJECT}/.venv}/bin/activate"
    echo "env=${VIRTUAL_ENV:-none} python=$(python -V 2>&1)"
}
