#!/usr/bin/env bash
# tests/submit_ctx512.sh — num_ctx=512 batch to trigger real context exhaustion.
#
# At ~200 tokens/turn, the 512-token window is exceeded around turn 3, causing
# Ollama to silently truncate oldest messages and the CEE to record a token-count
# dip in the context graph.  Six steps per run gives multiple exhaustion events.
#
# Test IDs: t35 – t38 (GROUP K)

set -euo pipefail

AEG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QWEN_SCRIPT="${AEG_DIR}/slurm_aeg_qwen25coder.sh"
GEMMA_SCRIPT="${AEG_DIR}/slurm_aeg_gemma4.sh"

sub() {
  local id="$1" script="$2" agents="$3" steps="$4" extra="$5"
  echo "[submit] ${id}  agents=${agents}  steps=${steps}  extra=\"${extra}\""
  sbatch \
    --export=ALL,AEG_NUM_AGENTS="${agents}",AEG_NUM_STEPS="${steps}",AEG_WORKFLOW_ID="${id}",AEG_EXTRA_ARGS="${extra}" \
    "${script}"
}

# K1 — clean baseline: only exhaustion signal, no injected errors
sub t35_ctx512_clean_qwen   "$QWEN_SCRIPT"  3 6 "--num-ctx 512"
sub t36_ctx512_clean_gemma4 "$GEMMA_SCRIPT" 3 6 "--num-ctx 512"

# K2 — format error on worker 0 step 2, plus context exhaustion:
#       two independent failure paths, exhaustion occurring on later turns
sub t37_ctx512_fmt_qwen     "$QWEN_SCRIPT"  3 6 "--num-ctx 512 --inject format:worker:0:2"

# K3 — tool_call error on worker 1 step 1, plus context exhaustion:
#       tests attribution of HTTP-error and context-exhaustion events in same session
sub t38_ctx512_toolcall_gemma4 "$GEMMA_SCRIPT" 3 6 "--num-ctx 512 --inject tool_call:worker:1:1"

echo ""
echo "All 4 ctx-512 tests submitted."
echo "Results will appear under: ${AEG_DIR}/results/"
echo "Monitor with: squeue -u \$USER"
