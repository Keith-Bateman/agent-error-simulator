#!/usr/bin/env bash
# tests/submit_scaling_multigpu.sh — multi-GPU scaling study (Group MG).
#
# Submits 3-repeat runs for each (model, worker-count) combination on a full
# 4-GPU node.  Workers are distributed round-robin across 4 independent Ollama
# instances (one per GH200), so up to 4 workers run truly in parallel.
#
# Design:
#   Worker counts : 1, 2, 4, 8
#                   1 = serial baseline (all requests hit GPU 0)
#                   2 = 2-parallel (GPUs 0-1)
#                   4 = fully parallel (one worker per GPU)
#                   8 = two workers per GPU, queuing within each instance
#   Repeats       : 3 per cell  (full-node jobs are expensive)
#   Models        : qwen2.5-coder:32b, gemma4:27b
#   Steps/worker  : 2  (same as Group S for direct comparison)
#   GPUs/job      : 4  (exclusive node allocation)
#
#   Total jobs    : 4 × 3 × 2 = 24  (each uses a full 4-GPU node)
#
# Workflow IDs follow the pattern:  mg_{model}_{N}w_4g_r{rep}
#   e.g.  mg_qwen_4w_4g_r1 … mg_gemma4_8w_4g_r3
#
# Results land under:  results/mg_*/result.json
# Analysis:            python3 analysis/plot_scaling.py

set -euo pipefail

AEG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QWEN_SCRIPT="${AEG_DIR}/slurm_aeg_multigpu_qwen.sh"
GEMMA_SCRIPT="${AEG_DIR}/slurm_aeg_multigpu_gemma4.sh"

STEPS=2
REPEATS=3
WORKER_COUNTS=(1 2 4 8)
NUM_GPUS=4

submitted=0

for workers in "${WORKER_COUNTS[@]}"; do
  for rep in $(seq 1 "${REPEATS}"); do

    # ── qwen ───────────────────────────────────────────────────────────────
    id="mg_qwen_${workers}w_${NUM_GPUS}g_r${rep}"
    echo "[submit] ${id}  agents=${workers}  steps=${STEPS}  gpus=${NUM_GPUS}"
    sbatch \
      --export=ALL,\
AEG_NUM_AGENTS="${workers}",\
AEG_NUM_STEPS="${STEPS}",\
AEG_WORKFLOW_ID="${id}",\
AEG_EXTRA_ARGS="" \
      "${QWEN_SCRIPT}"
    (( submitted++ )) || true

    # ── gemma4 ─────────────────────────────────────────────────────────────
    id="mg_gemma4_${workers}w_${NUM_GPUS}g_r${rep}"
    echo "[submit] ${id}  agents=${workers}  steps=${STEPS}  gpus=${NUM_GPUS}"
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
echo "Each job requests a full 4-GPU node — expect longer queue times."
echo "Results will appear under: ${AEG_DIR}/results/mg_*/"
echo "Monitor with: squeue -u \$USER"
echo "Analyse with: python3 ${AEG_DIR}/analysis/plot_scaling.py"
