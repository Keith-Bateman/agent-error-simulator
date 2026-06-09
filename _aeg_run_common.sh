#!/usr/bin/env bash
# _aeg_run_common.sh — shared CEE startup + agent-error-generator pipeline logic.
#
# DO NOT submit this script directly.  Source it from a model-specific wrapper
# that sets these variables BEFORE sourcing:
#
#   AEG_MODEL         — Ollama model tag, e.g. "qwen2.5-coder:32b" or "gemma4:27b"
#   AEG_NUM_AGENTS    — number of parallel worker agents    (default: 3)
#   AEG_NUM_STEPS     — refinement steps per worker         (default: 2)
#   AEG_WORKFLOW_ID   — unique identifier for this run      (default: aeg_${MODEL_SAFE}_${timestamp})
#   AEG_RESULT_DIR    — directory for output JSON + visuals (default: results)
#   AEG_EXTRA_ARGS    — extra flags forwarded verbatim to generator.py
#                       (e.g. "--inject format:worker:1:2 --inject tool_call:worker:0")
#   AEG_SKIP_EXPORT   — set to 1 to skip CEE visual export (default: 0)

set -euo pipefail

WORK=/work/hdd/bekn/kbateman
AEG_DIR="${WORK}/agent-error-generator"
AEG_VENV="${AEG_DIR}/.venv"
CEE_ROOT="${WORK}/clio-core"
CEE_VENV="${CEE_ROOT}/.venv"
VIS_DIR="${CEE_ROOT}/context-visualizer"
BUILD_DIR="${CEE_ROOT}/build"
EXPORT_SCRIPT="${WORK}/Kramabench/export_cee_visuals.py"
# Derive unique ports from SLURM_JOB_ID so that multiple jobs sharing the same
# multi-GPU node (ghx4 has 4 GH200s per node) don't collide on Ollama, Flask,
# or the Chimaera runtime. Each job gets a slot 0–3 based on (JOB_ID % 4);
# slots are 500 ports apart so there is no overlap. Falls back to defaults when
# running outside SLURM.
#
# Port layout (slot 0 / 1 / 2 / 3):
#   OLLAMA_PORT : 11434 / 11934 / 12434 / 12934
#   PROXY_PORT  :  9090 /  9590 / 10090 / 10590
#   CEE_PORT    :  9513 / 10013 / 10513 / 11013
_JOB_SLOT=$(( ${SLURM_JOB_ID:-0} % 4 ))
OLLAMA_PORT=$(( 11434 + _JOB_SLOT * 500 ))
PROXY_PORT=$(( 9090  + _JOB_SLOT * 500 ))
CEE_PORT=$(( 9513   + _JOB_SLOT * 500 ))

# ── Defaults ───────────────────────────────────────────────────────────────────
AEG_MODEL="${AEG_MODEL:-qwen2.5-coder:32b}"
AEG_NUM_AGENTS="${AEG_NUM_AGENTS:-3}"
AEG_NUM_STEPS="${AEG_NUM_STEPS:-2}"
AEG_EXTRA_ARGS="${AEG_EXTRA_ARGS:-}"
AEG_SKIP_EXPORT="${AEG_SKIP_EXPORT:-0}"

# Sanitise model tag for file paths
MODEL_SAFE="${AEG_MODEL//:/_}"
MODEL_SAFE="${MODEL_SAFE//\//_}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
AEG_WORKFLOW_ID="${AEG_WORKFLOW_ID:-aeg_${MODEL_SAFE}_${TIMESTAMP}}"
AEG_RESULT_DIR="${AEG_RESULT_DIR:-${AEG_DIR}/results}"

VISUALS_DIR="${AEG_RESULT_DIR}/${AEG_WORKFLOW_ID}/visuals"
OUTPUT_JSON="${AEG_RESULT_DIR}/${AEG_WORKFLOW_ID}/result.json"

mkdir -p "${AEG_RESULT_DIR}/${AEG_WORKFLOW_ID}" "${AEG_DIR}/logs" "${VISUALS_DIR}"

echo "================================================================"
echo " agent-error-generator + CEE job on $(hostname)"
echo " $(date)"
echo " Model        : ${AEG_MODEL}"
echo " Num agents   : ${AEG_NUM_AGENTS}"
echo " Num steps    : ${AEG_NUM_STEPS}"
echo " Workflow ID  : ${AEG_WORKFLOW_ID}"
echo " Result dir   : ${AEG_RESULT_DIR}/${AEG_WORKFLOW_ID}/"
echo "================================================================"

# ── Sanity checks ──────────────────────────────────────────────────────────────
if [[ ! -x "${AEG_VENV}/bin/python" ]]; then
  echo "ERROR: agent-error-generator venv not found at ${AEG_VENV}" >&2
  echo "       Run once:" >&2
  echo "         cd ${AEG_DIR}" >&2
  echo "         python3.11 -m venv .venv" >&2
  echo "         .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

if [[ ! -x "${CEE_VENV}/bin/python3" ]]; then
  echo "ERROR: CEE venv not found at ${CEE_VENV}" >&2
  exit 1
fi

EXT_SO=$(find "${BUILD_DIR}/bin" -name "chimaera_runtime_ext*.so" 2>/dev/null | head -1)
if [[ -z "$EXT_SO" ]]; then
  echo "ERROR: chimaera_runtime_ext.so missing from ${BUILD_DIR}/bin" >&2
  echo "       Build CEE first (see clio-core README)." >&2
  exit 1
fi
echo "[check] chimaera_runtime_ext : $EXT_SO"
echo "[check] CEE venv             : ${CEE_VENV}"
echo "[check] AEG venv             : ${AEG_VENV}"

# ── Ollama environment ─────────────────────────────────────────────────────────
# shellcheck source=/dev/null
source "${WORK}/install/ollama_env.sh"
# Re-export after sourcing: ollama_env.sh resets OLLAMA_HOST to the default
# port, so we override it here to ensure our job-specific port takes effect.
export OLLAMA_HOST="http://127.0.0.1:${OLLAMA_PORT}"
OLLAMA_URL="http://127.0.0.1:${OLLAMA_PORT}"
# Use a long keep-alive so that a slow first load (~300 s from cold NFS disk)
# does not exhaust the timer before CEE can issue its first request.
# KEEP_ALIVE in Ollama resets on every response, but the *first* response for a
# cold model arrives ~296 s after the request, so 5 m would leave only ~4 s of
# runway.  30 m gives ample margin regardless of disk cache state.
export OLLAMA_KEEP_ALIVE="30m"
# Allow up to 60 minutes for the runner to load from cold NFS disk.
# The default (5m) is too short; some nodes with saturated NFS have taken 30m+.
export OLLAMA_LOAD_TIMEOUT="60m"

# Allow concurrent requests for all worker agents + planner + aggregator.
# Cap at 5 to avoid CUDA OOM: KvSize = num_parallel × num_ctx.  With gemma4's
# 131072-token default context, PARALLEL=8 reserves 1,048,576 KV tokens and
# exceeds the 95.6 GiB GH200 VRAM budget.  All 8 sessions still run; Ollama
# just queues beyond the cap rather than pre-allocating slots for all of them.
export OLLAMA_NUM_GPU="${OLLAMA_NUM_GPU:-1}"
_OLLAMA_PARALLEL_CAP=5
_REQ_PARALLEL="${OLLAMA_NUM_PARALLEL:-${AEG_NUM_AGENTS}}"
if [[ "$_REQ_PARALLEL" -gt "$_OLLAMA_PARALLEL_CAP" ]]; then
  echo "[ollama] Capping OLLAMA_NUM_PARALLEL: ${_REQ_PARALLEL} → ${_OLLAMA_PARALLEL_CAP} (VRAM limit)"
  export OLLAMA_NUM_PARALLEL="${_OLLAMA_PARALLEL_CAP}"
else
  export OLLAMA_NUM_PARALLEL="${_REQ_PARALLEL}"
fi

# ── Start Ollama serve ─────────────────────────────────────────────────────────
# Kill any stale Ollama from a previous run that may still hold our port.
# Without this, our new `ollama serve` silently fails to bind, the startup
# health-check passes (old Ollama responds), and then the old process dies
# mid-run leaving the port unbound.
_STALE=$(ss -tlnp "sport = :${OLLAMA_PORT}" 2>/dev/null | grep -oP 'pid=\K[0-9]+' || true)
if [[ -n "$_STALE" ]]; then
  echo "[ollama] Killing stale Ollama on port ${OLLAMA_PORT}: PID(s) $_STALE"
  kill -9 $_STALE 2>/dev/null || true
  sleep 2
fi

echo "[ollama] Starting ollama serve..."
ollama serve &
OLLAMA_PID=$!

echo -n "[ollama] Waiting for server"
OLLAMA_READY=false
for i in $(seq 1 60); do
  sleep 2
  if curl -sf "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
    echo " ready (${i}×2 s)."
    OLLAMA_READY=true
    break
  fi
  if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
    echo ""
    echo "ERROR: ollama serve died on startup." >&2
    exit 1
  fi
  echo -n "."
done

if ! $OLLAMA_READY; then
  echo ""
  echo "ERROR: Ollama did not become ready within 120 s." >&2
  kill "$OLLAMA_PID" 2>/dev/null || true
  exit 1
fi

# Verify that $OLLAMA_PID is the process that actually owns our port.
# If a stale Ollama survived the pkill above and our new serve failed to bind,
# $OLLAMA_PID would point to a dead process while the stale one holds the port.
_PORT_OWNER=$(ss -tlnp "sport = :${OLLAMA_PORT}" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1 || true)
if [[ -z "$_PORT_OWNER" ]]; then
  echo "ERROR: Nothing is listening on port ${OLLAMA_PORT} despite health-check passing!" >&2
  exit 1
fi
if [[ "$_PORT_OWNER" != "$OLLAMA_PID" ]]; then
  echo "[ollama] WARNING: port ${OLLAMA_PORT} owned by PID $_PORT_OWNER, not our ollama PID ${OLLAMA_PID}." >&2
  echo "[ollama] WARNING: Our ollama serve failed to bind — adopting the existing process." >&2
  OLLAMA_PID="$_PORT_OWNER"
fi
echo "[ollama] Confirmed port ${OLLAMA_PORT} owned by PID ${OLLAMA_PID}."

# Pull model if not already cached
if ! ollama list 2>/dev/null | grep -qF "${AEG_MODEL}"; then
  echo "[ollama] Pulling ${AEG_MODEL} (first run only)..."
  ollama pull "${AEG_MODEL}"
  echo "[ollama] Pull complete."
else
  echo "[ollama] ${AEG_MODEL} already present."
fi

# Warm up model — load into GPU VRAM before the workflow starts so the first
# agent request doesn't hit the cold-load timeout (90 s is not enough for a
# 27B+ model loading from disk).
echo -n "[ollama] Warming up ${AEG_MODEL}..."
WARMUP_START=$(date +%s)
curl -sf -X POST "${OLLAMA_URL}/api/generate" \
  --max-time 4000 \
  -d "{\"model\":\"${AEG_MODEL}\",\"prompt\":\"ping\",\"stream\":false}" \
  > /dev/null
WARMUP_ELAPSED=$(( $(date +%s) - WARMUP_START ))
echo " done (${WARMUP_ELAPSED}s)."

# Verify Ollama is still alive immediately after warm-up.
# If KEEP_ALIVE is measured from request arrival (not completion), a slow first
# load (~296 s) consumes nearly all of the 5-minute budget.  A second ping
# resets the timer so the model stays loaded while CEE starts up.
echo "[ollama] Post-warm-up health check..."
if ! curl -sf "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
  echo "ERROR: Ollama /api/tags failed immediately after warm-up on port ${OLLAMA_PORT}" >&2
  echo "       Is Ollama still running? PID=${OLLAMA_PID}" >&2
  kill -0 "$OLLAMA_PID" 2>/dev/null && echo "       Process alive." >&2 || echo "       Process DEAD." >&2
  ss -tlnp 2>/dev/null | grep ":${OLLAMA_PORT}" || echo "       Port ${OLLAMA_PORT} not listening." >&2
  exit 1
fi
echo "[ollama] /api/tags OK on port ${OLLAMA_PORT}."

# Second ping — resets KEEP_ALIVE timer (warm-up may have used most of it).
echo -n "[ollama] Resetting KEEP_ALIVE timer..."
curl -sf -X POST "${OLLAMA_URL}/api/generate" \
  --max-time 60 \
  -d "{\"model\":\"${AEG_MODEL}\",\"prompt\":\"ping\",\"stream\":false}" \
  > /dev/null
echo " done."
echo "[ollama] Model loaded and ready on port ${OLLAMA_PORT}."

# Background health monitor: log Ollama's status every 5 s until cleanup fires.
# This tells us exactly when (and from whose perspective) Ollama becomes unavailable.
MONITOR_PID=""
_ollama_monitor() {
  local _port="$1" _pid="$2" _url="$3"
  local _i=0
  while true; do
    sleep 5
    _i=$(( _i + 1 ))
    _alive="alive"; kill -0 "$_pid" 2>/dev/null || _alive="DEAD"
    _bound=$(ss -tlnp "sport = :${_port}" 2>/dev/null | grep -c "${_port}" || true)
    if curl -sf "${_url}/api/tags" >/dev/null 2>&1; then
      _http="HTTP-OK"
    else
      _http="HTTP-FAIL"
    fi
    echo "[monitor+${_i}] ollama PID=${_pid} process=${_alive} port=${_port} bound=${_bound} ${_http}"
    # Stop if http is failing — print extra diagnostics
    if [[ "$_http" == "HTTP-FAIL" ]]; then
      echo "[monitor] OLLAMA UNAVAILABLE — port owner: $(ss -tlnp sport = :${_port} 2>/dev/null || echo none)" >&2
    fi
  done
}
_ollama_monitor "${OLLAMA_PORT}" "${OLLAMA_PID}" "${OLLAMA_URL}" &
MONITOR_PID=$!

# ── CEE environment ────────────────────────────────────────────────────────────
# build/bin must come BEFORE myenv/lib: myenv/lib has an older installed copy of
# libhermes_shm_host.so (built on a 4KB-page machine) that bakes kBackendHeaderSize=4096.
# On the GH200 (64KB pages) MAP_FIXED requires a 64KB-aligned offset, so it must be
# 65536. Putting build/bin first ensures the freshly rebuilt SOs win over the stale install.
# myenv/lib still supplies libyaml-cpp.so.0.8, libzmq, etc. that aren't in the system dirs.
export LD_LIBRARY_PATH="${BUILD_DIR}/bin:/u/kbateman/miniconda3/envs/myenv/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CHI_REPO_PATH="${BUILD_DIR}/bin"
export PYTHONPATH="${BUILD_DIR}/bin${PYTHONPATH:+:${PYTHONPATH}}"
# Prevent chimaera from treating this shell itself as an agent session
unset CLAUDECODE 2>/dev/null || true

VENV_PYTHON="${CEE_VENV}/bin/python3"

# Kill leftover CEE processes from a *previous* run of this same job slot.
# Use ss to find only the process holding OUR CEE_PORT — a broad pkill would
# kill sibling jobs' dt_demo_server processes on the same multi-GPU node.
_DT_STALE=$(ss -tlnp "sport = :${CEE_PORT}" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1 || true)
if [[ -n "$_DT_STALE" ]]; then
  echo "[cee] Killing stale dt_demo_server on CEE port ${CEE_PORT}: PID $_DT_STALE"
  kill -9 "$_DT_STALE" 2>/dev/null || true
  sleep 2
fi
pkill -9 -f "flask run.*${PROXY_PORT}"   2>/dev/null || true
# Remove only the IPC socket for our slot — not all chimaera files.
rm -f "/tmp/chimaera_$(whoami)/chimaera_${CEE_PORT}.ipc" 2>/dev/null || true
sleep 1

# Write a per-job copy of the CEE server config with job-specific ports.
# Must come AFTER the rm -f cleanup above, which would otherwise delete it.
# Two substitutions are applied:
#   1. Ollama upstream URL  — so dt_intercept_ollama reaches our Ollama instance
#   2. Chimaera networking port — so concurrent jobs on the same node each get
#      their own IPC socket (/tmp/chimaera_…/chimaera_<CEE_PORT>.ipc) and TCP
#      port, avoiding shared-memory collisions.
_CEE_CONF_TEMPLATE="${WORK}/Kramabench/cee_server_config.yaml"
_CEE_CONF_JOB="/tmp/aeg_cee_conf_${SLURM_JOB_ID:-local}.yaml"
sed \
  -e "s|http://127.0.0.1:11434|http://127.0.0.1:${OLLAMA_PORT}|g" \
  -e "s|port: 9513|port: ${CEE_PORT}|g" \
  "${_CEE_CONF_TEMPLATE}" > "${_CEE_CONF_JOB}"
echo "[cee] Using per-job config: ${_CEE_CONF_JOB} (ollama=127.0.0.1:${OLLAMA_PORT}, cee port=${CEE_PORT})"
export CHI_SERVER_CONF="${_CEE_CONF_JOB}"

# ── Cleanup trap — stops all background processes on exit ─────────────────────
FLASK_PID=""
SERVER_PID=""
cleanup() {
  echo ""
  echo "[cleanup] Stopping CEE and Ollama..."
  [[ -n "$MONITOR_PID" ]] && kill "$MONITOR_PID" 2>/dev/null || true
  [[ -n "$FLASK_PID"   ]] && kill "$FLASK_PID"   2>/dev/null || true
  [[ -n "$SERVER_PID"  ]] && kill "$SERVER_PID"  2>/dev/null || true
  [[ -n "$OLLAMA_PID"  ]] && kill "$OLLAMA_PID"  2>/dev/null || true
  rm -f "${_CEE_CONF_JOB}"                                          2>/dev/null || true
  rm -f "/tmp/chimaera_$(whoami)/chimaera_${CEE_PORT}.ipc"          2>/dev/null || true
  echo "[cleanup] Done."
}
trap cleanup EXIT INT TERM

# ── Start Chimaera runtime ─────────────────────────────────────────────────────
echo "[server] Starting Chimaera server..."
"${BUILD_DIR}/bin/dt_demo_server" &
SERVER_PID=$!
sleep 12

if ! kill -0 "$SERVER_PID" 2>/dev/null; then
  echo "ERROR: Chimaera server failed to start." >&2
  exit 1
fi
echo "[server] Running (PID=$SERVER_PID)"

# ── Start Flask proxy ──────────────────────────────────────────────────────────
echo "[flask] Starting Flask on port ${PROXY_PORT}..."
cd "$VIS_DIR"
# No semantic_checks YAML for the error-generator — semantic checking is optional.
# Unset to avoid picking up a stale value from the environment.
unset DTP_SEMANTIC_CHECKS 2>/dev/null || true
FLASK_APP=context_visualizer.app "$VENV_PYTHON" -m flask run \
  --host 0.0.0.0 --port "$PROXY_PORT" &
FLASK_PID=$!

echo -n "[flask] Waiting for Flask on port ${PROXY_PORT}"
FLASK_READY=false
for i in $(seq 1 30); do
  sleep 2
  if ! kill -0 "$FLASK_PID" 2>/dev/null; then
    echo ""
    echo "ERROR: Flask process died during startup." >&2
    exit 1
  fi
  if curl -sf "http://localhost:${PROXY_PORT}/provenance" >/dev/null 2>&1; then
    echo " ready (${i}×2 s)."
    FLASK_READY=true
    break
  fi
  echo -n "."
done

if ! $FLASK_READY; then
  echo ""
  echo "ERROR: Flask did not become ready on port ${PROXY_PORT} within 60 s." >&2
  kill "$FLASK_PID" 2>/dev/null || true
  exit 1
fi
echo "[flask] Running (PID=$FLASK_PID)"
cd "$AEG_DIR"

PROXY_SESSION_BASE="http://localhost:${PROXY_PORT}"

echo ""
echo "──────────────────────────────────────────────────────"
echo "  CEE Dashboard   : ${PROXY_SESSION_BASE}/provenance"
echo "  Workflow ID     : ${AEG_WORKFLOW_ID}"
echo "  Ollama URL      : ${OLLAMA_URL}  (proxied via CEE)"
echo "──────────────────────────────────────────────────────"
echo ""

# ── Run generator.py ───────────────────────────────────────────────────────────
echo "[generator] Launching workflow..."
echo "            Agents     : ${AEG_NUM_AGENTS}"
echo "            Steps/agent: ${AEG_NUM_STEPS}"
echo "            Model      : ${AEG_MODEL}"
echo "            Output     : ${OUTPUT_JSON}"
echo ""

AEG_PYTHON="${AEG_VENV}/bin/python"
# shellcheck disable=SC2086
"${AEG_PYTHON}" "${AEG_DIR}/generator.py" \
  --num-agents    "${AEG_NUM_AGENTS}" \
  --num-steps     "${AEG_NUM_STEPS}" \
  --model         "${AEG_MODEL}" \
  --workflow-id   "${AEG_WORKFLOW_ID}" \
  --enable-proxy \
  --proxy-host    "127.0.0.1" \
  --proxy-port    "${PROXY_PORT}" \
  --output        "${OUTPUT_JSON}" \
  ${AEG_EXTRA_ARGS}

GEN_EXIT=$?
echo ""
echo "[generator] Exited with code ${GEN_EXIT}."

# ── Export CEE visuals — all agent sessions combined into one HTML per task ────
if [[ "${AEG_SKIP_EXPORT}" -eq 0 && -x "${AEG_VENV}/bin/python" ]]; then
  echo ""
  echo "[export] Exporting CEE visuals to: ${VISUALS_DIR}"

  # Resolve which Python has requests (prefer the AEG venv, fall back to bench venv)
  EXPORT_PYTHON="${AEG_VENV}/bin/python"
  if ! "${EXPORT_PYTHON}" -c "import requests" 2>/dev/null; then
    EXPORT_PYTHON="${WORK}/Kramabench/.venv/bin/python"
  fi

  # Build the --session-id argument list: planner, all workers, aggregator.
  # Passing multiple --session-id flags triggers combined mode in export_cee_visuals.py,
  # which writes a single report_<ts>.html covering all agents.
  _EXPORT_ARGS=(
    --api-url    "${PROXY_SESSION_BASE}"
    --output-dir "${VISUALS_DIR}"
    --sut        "${AEG_WORKFLOW_ID}"
    --session-id "${AEG_WORKFLOW_ID}_planner_0"
  )
  for idx in $(seq 0 $((AEG_NUM_AGENTS - 1))); do
    _EXPORT_ARGS+=(--session-id "${AEG_WORKFLOW_ID}_worker_${idx}")
  done
  _EXPORT_ARGS+=(--session-id "${AEG_WORKFLOW_ID}_aggregator_0")

  "${EXPORT_PYTHON}" "${EXPORT_SCRIPT}" "${_EXPORT_ARGS[@]}" \
    || echo "[export] WARNING: combined visual export failed (non-fatal)." >&2

  echo "[export] Done.  Visuals in: ${VISUALS_DIR}"
else
  echo "[export] Skipped (AEG_SKIP_EXPORT=1 or venv missing)."
fi

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " agent-error-generator run complete."
echo "  Result JSON  : ${OUTPUT_JSON}"
echo "  Visuals      : ${VISUALS_DIR}"
echo "  Workflow ID  : ${AEG_WORKFLOW_ID}"
echo "================================================================"

exit "$GEN_EXIT"
