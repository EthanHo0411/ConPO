#!/bin/bash
# Creates the open-instruct venv for the IF-RLVR devbox adaptation.
#
# Run as `bash setup_venv.sh` (not sourced, not typed inline): every write under
# /node-local-cache below happens from inside this script file, per the devbox rule that a
# node-local write typed on the command line hangs on an unanswerable "ask" prompt.
#
# What it does:
#   1. Copies refs/open-instruct into a fresh working copy under /node-local-cache/cache
#      (refs/open-instruct itself is never modified -- see the patches/ README).
#   2. Applies the two patches in openinstruct/patches/ to that copy with `patch -p1`.
#   3. `uv sync --frozen` in the copy: installs open-instruct (editable) plus its pinned
#      dependencies (pyproject.toml's default groups already select torch 2.10.0+cu128 /
#      vllm 0.19.1, the pair this box's driver runs; see README.md "Environment").
#   4. Prints the installed torch/vllm/ray/deepspeed/peft versions as a sanity check.
set -euo pipefail

if [ -z "${UV_PYTHON_INSTALL_DIR:-}" ]; then
    # shellcheck disable=SC1091
    source /home/dev/project/ethan/.ethan-env
fi

CONPO_ROOT="/home/dev/project/ethan/ConPO"
OI_CLONE="/home/dev/project/ethan/refs/open-instruct"
OI_SRC="/node-local-cache/cache/open-instruct-src"
OI_VENV="/node-local-cache/cache/open-instruct-venv"

# uv cache is shared with ConPO's own venv (local/env.sh uses the same default path); the two
# projects pin different torch/vllm versions so cache reuse across them will be small, but it
# costs nothing to share the directory.
export UV_CACHE_DIR="/node-local-cache/cache/uv"
export UV_PROJECT_ENVIRONMENT="${OI_VENV}"
export UV_LINK_MODE=hardlink

# refs/open-instruct has no git tags (`git describe --tags` fails there); setuptools_scm would
# refuse the editable install without this, exactly as open-instruct's own Dockerfile and debug
# script set it.
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OPEN_INSTRUCT=0.0.0+conpo

# `import deepspeed` writes a small Triton autotune cache; DeepSpeed's op_builder reads
# TRITON_CACHE_DIR directly (not the XDG_CACHE_HOME .ethan-env already redirects) and falls back
# to $HOME/.triton/autotune otherwise -- $HOME is the account owner's /home/dev, outside our
# boundary (found 2026-09-11 when the version-check import below wrote 4 tiny files there before
# this line existed; see research/tasks/lessons.md). Keep this set in every script that imports
# deepspeed or torch on this box.
export TRITON_CACHE_DIR="/node-local-cache/cache/triton"

echo "== copying ${OI_CLONE} -> ${OI_SRC} =="
rm -rf "${OI_SRC}"
mkdir -p "$(dirname "${OI_SRC}")"
# -r, not -a: this node-local mount rejects preserving the source's exact permission bits
# (cp -a fails with "Operation not supported" on every file). Plain -r is enough for a Python
# source checkout; .git is dropped right after since we only need the working tree.
cp -r "${OI_CLONE}" "${OI_SRC}"
rm -rf "${OI_SRC}/.git"

echo "== applying patches =="
cd "${OI_SRC}"
patch -p1 --forward < "${CONPO_ROOT}/openinstruct/patches/lora_peft.patch"
patch -p1 --forward < "${CONPO_ROOT}/openinstruct/patches/enable_thinking_flag.patch"
patch -p1 --forward < "${CONPO_ROOT}/openinstruct/patches/local_dataset_tool_discovery.patch"
patch -p1 --forward < "${CONPO_ROOT}/openinstruct/patches/lora_vllm_weight_sync.patch"

echo "== uv sync (this downloads torch/vllm/ray/deepspeed/flash-attn; can take a while) =="
uv sync --frozen

echo "== installed versions =="
"${OI_VENV}/bin/python" "${CONPO_ROOT}/openinstruct/print_versions.py"

echo "== done: venv at ${OI_VENV}, patched source at ${OI_SRC} =="
