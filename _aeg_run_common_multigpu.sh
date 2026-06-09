#!/usr/bin/env bash
# _aeg_run_common_multigpu.sh — multi-GPU AEG runner.
#
# Starts one Ollama instance per GPU, distributes worker agents round-robin
# across them, then runs generator.py without the CEE proxy (timing study only).
#
# Source from a model-specific SLURM wrapper that sets:
#
#   AEG_MODEL        — Ollama model tag, e.g. "qwen2.5-coder:32b"
#   AEG_NUM_AGENTS   — number of parallel worker agents   (default: 4)
#   AEG_NUM_STEPS    — refinement steps per worker        (default: 2)
#   AEG_WORKFLOW_ID  — unique run identifier
#   AEG_NUM_GPUS     — GPUs to use; defaults to SLURM_GPUS_ON_NODE or 4
#
# Port layout (GPU index 0-3):
#   Ollama : 11434 / 11934 / 12434 / 12934  (base + idx*500)
#
# CEE is intentionally skipped — this script is designed for pure timing
# experiments.  Use the standard single-GPU script for CEE-instrumented runs.

set -euo pipefail

WORK=/work/hdd/bekn/kbateman
AEG_DIR="${WORK}/agent-error-generator"
AEG_VENV="${AEG_DIR}/.venv"

# ── Defaults ────────────────────────────────────────────────────────────────────
AEG_MODEL="${AEG_MODEL:-qwen2.5-coder:32b}"
AEG_NUM_AGENTS="${AEG_NUM_AGENTS:-4}"
AEG_NUM_STEPS="${AEG_NUM_STEPS:-2}"
AEG_EXTRA_ARGS="${AEG_EXTRA_ARGS:-}"
AEG_NUM_GPUS="${AEG_NUM_GPUS:-${SLURM_GPUS_ON_NODE:-4}}"

MODEL_SAFE="${AEG_MODEL//:/_}"
MODEL_SAFE="${MODEL_SAFE//\//_}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
AEG_WORKFLOW_ID="${AEG_WORKFLOW_ID:-aeg_mg_${MODEL_SAFE}_${TIMESTAMP}}"
AEG_RESULT_DIR="${AEG_RESULT_DIR:-${AEG_DIR}/results}"

OUTPUT_JSON="${AEG_RESULT_DIR}/${AEG_WORKFLOW_ID}/result.json"

mkdir -p "${AEG_RESULT_DIR}/${AEG_WORKFLOW_ID}" "${AEG_DIR}/logs"

echo "================================================================"
echo " agent-error-generator (multi-GPU) on $(hostname)"
echo " $(date)"
echo " Model        : ${AEG_MODEL}"
echo " Num agents   : ${AEG_NUM_AGENTS}"
echo " Num steps    : ${AEG_NUM_STEPS}"
echo " Num GPUs     : ${AEG_NUM_GPUS}"
echo " Workflow ID  : ${AEG_WORKFLOW_ID}"
echo " Result dir   : ${AEG_RESULT_DIR}/${AEG_WORKFLOW_ID}/"
echo "================================================================"

# ── Sanity checks ────────────────────────────────────────────────────────────────
if [[ ! -x "${AEG_VENV}/bin/python" ]]; then
  echo "ERROR: agent-error-generator venv not found at ${AEG_VENV}" >&2
  exit 1
fi

# ── Ollama environment ────────────────────────────────────────────────────────────
source "${WORK}/install/ollama_env.sh"
export OLLAMA_KEEP_ALIVE="30m"
export OLLAMA_LOAD_TIMEOUT="60m"
# Each GPU instance handles at most ceil(workers/gpus) simultaneous requests.
_PER_GPU_PARALLEL=$(( (AEG_NUM_AGENTS + AEG_NUM_GPUS - 1) / AEG_NUM_GPUS ))
# Cap at 2 to stay within VRAM budget (two KV caches per GPU).
[[ "$_PER_GPU_PARALLEL" -gt 2 ]] && _PER_GPU_PARALLEL=2
export OLLAMA_NUM_PARALLEL="${_PER_GPU_PARALLEL}"
echo "[ollama] OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL} per GPU instance"

_BASE_PORT=11434
_PORT_STEP=500

# Arrays tracking each GPU's Ollama process and URL.
declare -a _OLLAMA_PIDS=()
declare -a _OLLAMA_PORTS=()
declare -a _OLLAMA_URLS=()

# ── Cleanup trap ─────────────────────────────────────────────────────────────────
cleanup() {
  echo ""
  echo "[cleanup] Stopping all Ollama instances..."
  for pid in "${_OLLAMA_PIDS[@]:-}"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
  echo "[cleanup] Done."
}
trap cleanup EXIT INT TERM

# ── Start one Ollama per GPU ──────────────────────────────────────────────────────
for gpu_idx in $(seq 0 $((AEG_NUM_GPUS - 1))); do
  _port=$(( _BASE_PORT + gpu_idx * _PORT_STEP ))
  _url="http://127.0.0.1:${_port}"

  # Kill anything already holding this port.
  _stale=$(ss -tlnp "sport = :${_port}" 2>/dev/null | grep -oP 'pid=\K[0-9]+' || true)
  if [[ -n "$_stale" ]]; then
    echo "[ollama-${gpu_idx}] Killing stale process on port ${_port}: PID(s) $_stale"
    kill -9 $_stale 2>/dev/null || true
    sleep 2
  fi

  echo "[ollama-${gpu_idx}] Starting on port ${_port} (CUDA_VISIBLE_DEVICES=${gpu_idx})..."
  CUDA_VISIBLE_DEVICES=$gpu_idx \
  OLLAMA_HOST="http://127.0.0.1:${_port}" \
  OLLAMA_NUM_GPU=1 \
    ollama serve &
  _OLLAMA_PIDS+=($!)
  _OLLAMA_PORTS+=("$_port")
  _OLLAMA_URLS+=("$_url")
done

# ── Wait for all instances to respond ────────────────────────────────────────────
for gpu_idx in $(seq 0 $((AEG_NUM_GPUS - 1))); do
  _url="${_OLLAMA_URLS[$gpu_idx]}"
  _pid="${_OLLAMA_PIDS[$gpu_idx]}"
  echo -n "[ollama-${gpu_idx}] Waiting for ${_url}"
  _ready=false
  for i in $(seq 1 60); do
    sleep 2
    if curl -sf "${_url}/api/tags" >/dev/null 2>&1; then
      echo " ready (${i}×2 s)."
      _ready=true
      break
    fi
    if ! kill -0 "$_pid" 2>/dev/null; then
      echo ""
      echo "ERROR: ollama-${gpu_idx} (PID $_pid) died during startup." >&2
      exit 1
    fi
    echo -n "."
  done
  if ! $_ready; then
    echo ""
    echo "ERROR: ollama-${gpu_idx} did not become ready within 120 s." >&2
    exit 1
  fi
done

# ── Warm up the model on each GPU in parallel ─────────────────────────────────────
echo "[warmup] Loading ${AEG_MODEL} on all ${AEG_NUM_GPUS} GPUs in parallel..."
declare -a _WARMUP_PIDS=()
for gpu_idx in $(seq 0 $((AEG_NUM_GPUS - 1))); do
  _url="${_OLLAMA_URLS[$gpu_idx]}"
  (
    _t0=$(date +%s)
    curl -sf -X POST "${_url}/api/generate" \
      --max-time 4000 \
      -d "{\"model\":\"${AEG_MODEL}\",\"prompt\":\"ping\",\"stream\":false}" \
      > /dev/null
    echo "[warmup] GPU ${gpu_idx} ready in $(( $(date +%s) - _t0 ))s"
  ) &
  _WARMUP_PIDS+=($!)
done
for pid in "${_WARMUP_PIDS[@]}"; do wait "$pid"; done
echo "[warmup] All GPUs loaded."

# Verify all instances still healthy.
for gpu_idx in $(seq 0 $((AEG_NUM_GPUS - 1))); do
  _url="${_OLLAMA_URLS[$gpu_idx]}"
  if ! curl -sf "${_url}/api/tags" >/dev/null 2>&1; then
    echo "ERROR: ollama-${gpu_idx} failed health check after warm-up." >&2
    exit 1
  fi
done
echo "[ollama] All ${AEG_NUM_GPUS} instances healthy."

# ── Build --ollama-urls argument ──────────────────────────────────────────────────
_URLS_ARG=$(IFS=,; echo "${_OLLAMA_URLS[*]}")
echo ""
echo "──────────────────────────────────────────────────────"
echo "  Workflow ID  : ${AEG_WORKFLOW_ID}"
echo "  Ollama URLs  : ${_URLS_ARG}"
echo "  Workers/GPU  : ~$(( (AEG_NUM_AGENTS + AEG_NUM_GPUS - 1) / AEG_NUM_GPUS ))"
echo "──────────────────────────────────────────────────────"
echo ""

# ── Run generator.py ──────────────────────────────────────────────────────────────
echo "[generator] Launching workflow..."
AEG_PYTHON="${AEG_VENV}/bin/python"
# shellcheck disable=SC2086
"${AEG_PYTHON}" "${AEG_DIR}/generator.py" \
  --num-agents    "${AEG_NUM_AGENTS}" \
  --num-steps     "${AEG_NUM_STEPS}" \
  --model         "${AEG_MODEL}" \
  --workflow-id   "${AEG_WORKFLOW_ID}" \
  --ollama-urls   "${_URLS_ARG}" \
  --output        "${OUTPUT_JSON}" \
  ${AEG_EXTRA_ARGS}

GEN_EXIT=$?
echo ""
echo "[generator] Exited with code ${GEN_EXIT}."

echo ""
echo "================================================================"
echo " agent-error-generator (multi-GPU) run complete."
echo "  Result JSON  : ${OUTPUT_JSON}"
echo "  Workflow ID  : ${AEG_WORKFLOW_ID}"
echo "================================================================"

exit "$GEN_EXIT"
