#!/usr/bin/env bash
# tests/submit_scaling_repeated.sh — 5-repeat scaling study (Group S).
#
# Submits 5 independent runs for each (model, worker-count) combination so
# that Fig 5 can be re-drawn with mean ± std error bars and phase-decomposed
# stacked bars (planner / workers / aggregator).
#
# Design:
#   Worker counts : 2, 3, 5, 8  (matching original groups F, A, G, H)
#   Repeats       : 5 per cell
#   Models        : qwen2.5-coder:32b, gemma4:27b
#   Steps/worker  : 2  (same as original scaling tests)
#   Error inject  : none (clean baseline)
#
#   Total jobs    : 4 × 5 × 2 = 40
#
# Workflow IDs follow the pattern:  sc_{model}_{N}w_r{rep}
#   e.g.  sc_qwen_2w_r1 … sc_gemma4_8w_r5
#
# Results land under:  results/sc_*/result.json
# Analysis:            python3 analysis/plot_scaling.py

set -euo pipefail

AEG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QWEN_SCRIPT="${AEG_DIR}/slurm_aeg_qwen25coder.sh"
GEMMA_SCRIPT="${AEG_DIR}/slurm_aeg_gemma4.sh"

STEPS=2
REPEATS=5
WORKER_COUNTS=(2 3 5 8)

submitted=0

for workers in "${WORKER_COUNTS[@]}"; do
  for rep in $(seq 1 "${REPEATS}"); do

    # ── qwen ───────────────────────────────────────────────────────────────
    id="sc_qwen_${workers}w_r${rep}"
    echo "[submit] ${id}  agents=${workers}  steps=${STEPS}"
    sbatch \
      --export=ALL,\
AEG_NUM_AGENTS="${workers}",\
AEG_NUM_STEPS="${STEPS}",\
AEG_WORKFLOW_ID="${id}",\
AEG_EXTRA_ARGS="" \
      "${QWEN_SCRIPT}"
    (( submitted++ )) || true

    # ── gemma4 ─────────────────────────────────────────────────────────────
    id="sc_gemma4_${workers}w_r${rep}"
    echo "[submit] ${id}  agents=${workers}  steps=${STEPS}"
    sbatch \
      --export=ALL,\
AEG_NUM_AGENTS="${workers}",\
AEG_NUM_STEPS="${STEPS}",\
AEG_WORKFLOW_ID="${id}",\
AEG_EXTRA_ARGS="" \
      "${GEMMA_SCRIPT}"
    (( submitted++ )) || true

  done
done

echo ""
echo "Submitted ${submitted} jobs (4 worker counts × ${REPEATS} repeats × 2 models)."
echo "Results will appear under: ${AEG_DIR}/results/sc_*/"
echo "Monitor with: squeue -u \$USER"
echo "Analyse with: python3 ${AEG_DIR}/analysis/plot_scaling.py"
