#!/usr/bin/env bash
# tests/submit_compact.sh — explicit context-compaction batch (GROUP L).
#
# Uses --max-turns 1 so each worker trims its history after every step beyond
# the second, producing event_type=compression events in the CEE context graph.
# With 6 steps per run, compaction fires at steps 3-6 — 4 dips per worker.

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

# L1 — clean baseline: compaction only, no injected errors
sub t39_compact_clean_qwen   "$QWEN_SCRIPT"  3 6 "--max-turns 1"
sub t40_compact_clean_gemma4 "$GEMMA_SCRIPT" 3 6 "--max-turns 1"

# L2 — format error mid-session + compaction: two independent failure signals
sub t41_compact_fmt_qwen     "$QWEN_SCRIPT"  3 6 "--max-turns 1 --inject format:worker:0:2"

# L3 — logic error + compaction
sub t42_compact_logic_gemma4 "$GEMMA_SCRIPT" 3 6 "--max-turns 1 --inject logic:worker:1:3"

echo ""
echo "All 4 compaction tests submitted."
echo "Results will appear under: ${AEG_DIR}/results/"
echo "Monitor with: squeue -u \$USER"
