#!/bin/bash
# UNTESTED: written on a machine with no GPU and no vllm install (dev conda
# env "conpo" is torch CPU only). This script has never been submitted or
# run. Verify the gres line, the uv venv path, and the health-poll timing
# on the mlp cluster before trusting it.
#
# Serves config.JUDGE_MODEL (Qwen/Qwen3-14B) as a long-lived OpenAI-compatible
# vLLM server on the second H100, for conpo/judge.py's HttpJudge (the
# quality veto of ConPO.pdf section 2.3). RUN_JUDGE_MODE=http fallback
# placement only (USER DECISIONS 2026-09-02): the default RUN_JUDGE_MODE=
# batched needs no server -- scripts/train.py's build_judge scores pending
# pairs in periodic bursts with a colocated VLLMBatchedJudge/HFBatchedJudge
# instead (ConPOTrainer.flush_pending_veto). Run this only when explicitly
# opting into RUN_JUDGE_MODE=http (slurm/train.sbatch); while it holds a
# card in that mode, every training run serializes on the other one.
# Measured footprint for this judge: 27.5 GB (config.py "Compute notes").
#
# Endpoint-file handshake, ported from the pattern in
# stackelberg-if/slurm/corruptor_server.sbatch: this job writes
# "hostname port jobid" to the endpoint file once /health answers, and stays
# alive until TTL or cancellation. slurm/train.sbatch polls that file and
# builds RUN_JUDGE_ENDPOINT = http://<hostname>:<port> from it --
# conpo/judge.py:read_endpoint requires exactly that three-field format and
# fails closed on a malformed read.
#
#   sbatch slurm/judge_server.sbatch
#
# Env:
#   JS_PORT       port              (default derived from the job id)
#   JS_TTL        seconds to serve  (default 14400)
#   JS_POLLS      health polls at 10s each before giving up (default 360)
#   JS_ENDPOINT   endpoint file     (default runs/serve/judge.endpoint)
#
#SBATCH --job-name=judge_server
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT}"
mkdir -p logs

echo "node=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID}"
nvidia-smi -L || echo "nvidia-smi -L failed"

# ponytail (2026-09-06): switching from conda to a uv venv is the user's
# decision; CONPO_VENV/.venv naming is this session's choice.
source "${CONPO_VENV:-${PROJECT}/.venv}/bin/activate"
echo "env=${VIRTUAL_ENV:-none} python=$(python -V 2>&1)"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# no_proxy landmine (README.md): node-to-node HTTP must bypass the site
# proxy, or /health and chat calls get a 503 from a proxy that cannot
# connect back to a compute node.
export no_proxy="localhost,127.0.0.1,::1,$(hostname),$(hostname -f)"
export NO_PROXY="${no_proxy}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# config.JUDGE_MODEL, read by name rather than retyped. Assumes conpo is
# installed editable in the uv venv by `uv sync` (pyproject.toml).
MODEL="$(python -c 'from conpo import config; print(config.JUDGE_MODEL)')"
PORT="${JS_PORT:-$(( 20000 + SLURM_JOB_ID % 20000 ))}"
TTL="${JS_TTL:-14400}"
SERVE_DIR="${PROJECT}/runs/serve"
ENDPOINT="${JS_ENDPOINT:-${SERVE_DIR}/judge.endpoint}"
mkdir -p "${SERVE_DIR}"

# Remove any endpoint file from a dead predecessor before advertising this
# one, or a client polls forever against a port nothing is listening on.
rm -f "${ENDPOINT}"

cleanup() {
    echo "cleanup: removing ${ENDPOINT} and stopping the server"
    rm -f "${ENDPOINT}"
    if [ -n "${SRV_PID:-}" ]; then
        kill "${SRV_PID}" 2>/dev/null || true
        wait "${SRV_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "model=${MODEL} host=$(hostname) port=${PORT} ttl=${TTL}s"
echo "endpoint file=${ENDPOINT}"

# bf16, max-model-len 4096: a veto prompt is one task prompt plus two
# responses plus a fixed instruction (conpo/judge.py:build_veto_prompt), well
# under the policy's own 3584-token budget, so 4096 leaves margin without
# sizing the KV cache for Qwen3's declared 262k context. enforce-eager
# matches the trainer's proven vLLM settings (ConPO-Build-Plan.md section 1);
# drop it once M0 shows CUDA graphs help on the H100.
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --served-model-name judge \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --tensor-parallel-size 1 \
    --enforce-eager \
    --no-enable-log-requests &
SRV_PID=$!

# Advertise only once /health answers, so a client never mistakes "not
# ready yet" for "died during load".
JS_POLLS="${JS_POLLS:-360}"
echo "waiting for the engine to come up, budget $(( JS_POLLS * 10 ))s"
READY=0
for i in $(seq 1 "${JS_POLLS}"); do
    if ! kill -0 "${SRV_PID}" 2>/dev/null; then
        echo "server process exited during startup, see the log above"
        exit 1
    fi
    if curl -sf --noproxy '*' "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        READY=1
        echo "engine ready after ~$(( i * 10 ))s"
        break
    fi
    sleep 10
done
if [ "${READY}" -ne 1 ]; then
    echo "engine did not answer /health within $(( JS_POLLS * 10 ))s"
    exit 1
fi

printf '%s %s %s\n' "$(hostname)" "${PORT}" "${SLURM_JOB_ID}" > "${ENDPOINT}"
echo "advertised: $(cat "${ENDPOINT}")"
curl -s --noproxy '*' "http://127.0.0.1:${PORT}/v1/models" | head -c 400 || true
echo

echo "serving until TTL ${TTL}s elapses or the job is cancelled"
END=$(( $(date +%s) + TTL ))
while [ "$(date +%s)" -lt "${END}" ]; do
    if ! kill -0 "${SRV_PID}" 2>/dev/null; then
        echo "server process died at $(date -Is)"
        exit 1
    fi
    sleep 30
done
echo "TTL reached, shutting down at $(date -Is)"
