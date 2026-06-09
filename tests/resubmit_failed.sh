#!/usr/bin/env bash
# tests/resubmit_failed.sh — resubmit tests that failed in the 2240798-2240831 batch.
#
# Failures root-caused to:
#   - broad pkill killing sibling dt_demo_server processes (now fixed: port-targeted kill)
#   - OLLAMA_LOAD_TIMEOUT / warm-up curl timeout too short (now fixed: 30m / 2000s)
#   - CUDA OOM in scale_large (now fixed: OLLAMA_NUM_PARALLEL capped at 5)
#   - SLURM node congestion (just retry)

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

# GROUP C
sub t11_toolcall_worker_gemma4 "$GEMMA_SCRIPT" 3 2 "--inject tool_call:worker:2:1"

# GROUP D
sub t15_staggered_steps_qwen "$QWEN_SCRIPT" 3 3 \
  "--inject tool_call:worker:0:1 --inject format:worker:1:2 --inject logic:worker:2:3"

# GROUP E
sub t16_two_errors_gemma4 "$GEMMA_SCRIPT" 3 2 \
  "--inject format:worker:0:1 --inject tool_call:worker:2:1"
sub t17_all_types_gemma4 "$GEMMA_SCRIPT" 3 2 \
  "--inject format:worker:0:1 --inject logic:worker:1:1 --inject tool_call:worker:2:1"

# GROUP F
sub t18_scale_min_qwen "$QWEN_SCRIPT" 2 1 ""

# GROUP G
sub t20_scale_med_qwen "$QWEN_SCRIPT" 5 2 "--inject tool_call:worker:2:1"

# GROUP H (scale_large — OLLAMA_NUM_PARALLEL now capped at 5 to avoid CUDA OOM)
sub t22_scale_large_gemma4 "$GEMMA_SCRIPT" 8 2 \
  "--inject logic:worker:3:1 --inject format:worker:7:1"

# GROUP I
sub t25_deep_fmt_mid_qwen "$QWEN_SCRIPT" 3 6 "--inject format:worker:1:3"

# GROUP J
sub t29_ctx_exhaust_clean_qwen   "$QWEN_SCRIPT" 3 4 "--num-ctx 2048"
sub t31_ctx_exhaust_fmt_qwen     "$QWEN_SCRIPT" 3 4 "--num-ctx 2048 --inject format:worker:0:2"
sub t33_ctx_exhaust_toolcall_qwen "$QWEN_SCRIPT" 3 4 "--num-ctx 2048 --inject tool_call:worker:1:1"

echo ""
echo "All 11 failed tests resubmitted."
echo "Monitor with: squeue -u \$USER"
