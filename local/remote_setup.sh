#!/bin/bash
# Remote-cluster bootstrap for ConPO (2026-09-15). Run from the repository root on the remote machine:
#   bash local/remote_setup.sh            # env + models + data check
#   bash local/remote_setup.sh --env-only # conda env only
# Idempotent: every step skips what already exists. Needs: conda (Anaconda or Miniconda), an NVIDIA driver >= 580
# (CUDA 13 wheels: torch 2.11.0+cu130, vllm 0.26.0), network access to PyPI, download.pytorch.org and huggingface.co.
#
# Env knobs (all optional):
#   CONPO_ENV      conda env name (default conpo)
#   HF_HOME        Hugging Face cache root (default $HOME/.cache/huggingface); models land in $HF_HOME/hub
#   HF_TOKEN       only needed for gated repos (Qwen3 is public)
#   CONPO_DATA_SRC rsync source for data/ and subsets/ (user@host:/path/to/ConPO), else data/ must already be present
#                  or is rebuilt with scripts/fetch_data.py (rebuild is NOT byte-identical to the devbox files; prefer rsync)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
ENV_NAME="${CONPO_ENV:-conpo}"
QWEN4B_REV=1cfa9a7208912126459214e8b04321603b3df60c
QWEN14B_REV=40c069824f4251a91eefaf281ebe4c544efd3e18

echo "== driver check"
if command -v nvidia-smi > /dev/null; then
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  echo "driver $DRV"
  MAJOR=${DRV%%.*}
  if [ "${MAJOR:-0}" -lt 580 ]; then
    echo "WARNING: driver $DRV is older than 580; the cu130 wheels (torch 2.11.0+cu130, vllm 0.26.0) need a CUDA 13 capable driver."
    echo "         Either update the driver or build the cu128 variant (uv extra cu128 plus a cu128 vllm wheel); do not continue blindly."
  fi
else
  echo "nvidia-smi not found; continuing without a driver check"
fi

echo "== conda"
install_miniforge() {
  echo "installing Miniforge (conda-forge only, no Anaconda Terms of Service) into $HOME/miniforge3"
  curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o "$HOME/miniforge-installer.sh"
  bash "$HOME/miniforge-installer.sh" -b -p "$HOME/miniforge3"
  rm -f "$HOME/miniforge-installer.sh"
  export PATH="$HOME/miniforge3/bin:$PATH"
}
if [ -x "$HOME/miniforge3/bin/conda" ]; then export PATH="$HOME/miniforge3/bin:$PATH"; fi
if ! command -v conda > /dev/null; then install_miniforge; fi
conda --version
echo "== conda env $ENV_NAME"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "env $ENV_NAME exists, skipping create"
elif ! conda env create -f environment.yml -n "$ENV_NAME"; then
  # An existing Anaconda/Miniconda install fails the Terms-of-Service check on its default channels (CondaToSNonInteractiveError);
  # fall back to a private Miniforge, which never consults those channels.
  echo "conda env create failed with the installed conda; retrying with a private Miniforge"
  install_miniforge
  conda env create -f environment.yml -n "$ENV_NAME"
fi
conda run -n "$ENV_NAME" python -c "import torch, vllm, trl, transformers, peft; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), '| vllm', vllm.__version__, '| trl', trl.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__)"
if [ "${1:-}" = "--env-only" ]; then exit 0; fi

echo "== models (pinned revisions, same as the devbox)"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
conda run -n "$ENV_NAME" python - <<'EOF'
import os
from huggingface_hub import snapshot_download
for repo, rev in (("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"), ("Qwen/Qwen3-14B", "40c069824f4251a91eefaf281ebe4c544efd3e18")):
    path = snapshot_download(repo, revision=rev)
    print(repo, rev[:12], "->", path)
EOF

echo "== data"
if [ -f data/train.jsonl ] && [ -f data/dev.jsonl ] && [ -f subsets/train_15000_seed42.keys ]; then
  echo "data/ and subsets/ present: $(wc -l < data/train.jsonl) train rows"
elif [ -n "${CONPO_DATA_SRC:-}" ]; then
  rsync -av "${CONPO_DATA_SRC}/data/" data/
  rsync -av "${CONPO_DATA_SRC}/subsets/" subsets/
else
  echo "data/ missing: set CONPO_DATA_SRC=user@host:/path/to/ConPO and rerun, or run scripts/fetch_data.py (not byte-identical to the devbox files)"
  exit 2
fi

echo "== smoke"
conda run -n "$ENV_NAME" python -c "import conpo, conpo.config as c; print('conpo import ok; policy', c.POLICY_MODEL, 'judge', c.JUDGE_MODEL)"
echo "remote setup done $(date -Is)"
