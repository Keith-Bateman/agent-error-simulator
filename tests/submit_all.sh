#!/usr/bin/env bash
# tests/submit_all.sh — submit the full AEG test suite to SLURM.
#
# Usage:
#   bash tests/submit_all.sh [GROUP]
#
#   GROUP (optional): A | B | C | D | E | F | G | H | I | J
#   Omit to submit every group.
#
# Groups:
#   A — Baseline (no errors)
#   B — Error-type variety, qwen2.5-coder:32b
#   C — Error-type variety, gemma4:27b
#   D — Multiple simultaneous errors, qwen2.5-coder:32b
#   E — Multiple simultaneous errors, gemma4:27b
#   F — Agent-scale: minimal (2 workers, 1 step)
#   G — Agent-scale: medium (5 workers)
#   H — Agent-scale: large  (8 workers, gemma4)
#   I — Context compaction: deep sessions (6+ steps, no forced exhaustion)
#   J — Context compaction: forced context exhaustion (num_ctx=2048)

set -euo pipefail

AEG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QWEN_SCRIPT="${AEG_DIR}/slurm_aeg_qwen25coder.sh"
GEMMA_SCRIPT="${AEG_DIR}/slurm_aeg_gemma4.sh"
FILTER="${1:-ALL}"

# ── helper ────────────────────────────────────────────────────────────────────
submit() {
  local group="$1" id="$2" script="$3" agents="$4" steps="$5" extra="$6"
  if [[ "$FILTER" != "ALL" && "$FILTER" != "$group" ]]; then
    return 0
  fi
  echo "[submit] ${id}  agents=${agents}  steps=${steps}  extra=\"${extra}\""
  sbatch \
    --export=ALL,AEG_NUM_AGENTS="${agents}",AEG_NUM_STEPS="${steps}",AEG_WORKFLOW_ID="${id}",AEG_EXTRA_ARGS="${extra}" \
    "${script}"
}

# ─────────────────────────────────────────────────────────────────────────────
# GROUP A — Baseline: no errors
# Purpose: establish a clean reference trace in the CEE for both models.
# CEE features exercised: session recording, context diff, auto-checkpoint.
# ─────────────────────────────────────────────────────────────────────────────
submit A  t01_baseline_qwen   "$QWEN_SCRIPT"  3 2 ""
submit A  t02_baseline_gemma4 "$GEMMA_SCRIPT" 3 2 ""

# ─────────────────────────────────────────────────────────────────────────────
# GROUP B — Error-type variety, qwen2.5-coder:32b, 3 agents, 2 steps
# Purpose: one test per error type and per agent role to give the CEE a clean
#          sample of each failure signature.
# CEE features exercised: failure detection, recovery-event emission, scoring.
# ─────────────────────────────────────────────────────────────────────────────

# B1 — worker FORMAT: agent returns plain text instead of JSON
submit B  t03_fmt_worker_qwen     "$QWEN_SCRIPT" 3 2 "--inject format:worker:1:1"

# B2 — worker LOGIC: agent returns JSON with result × (−1) (sign flip)
submit B  t04_logic_worker_qwen   "$QWEN_SCRIPT" 3 2 "--inject logic:worker:0:1"

# B3 — worker TOOL_CALL: compute_riemann returns HTTP-404 error body
submit B  t05_toolcall_worker_qwen "$QWEN_SCRIPT" 3 2 "--inject tool_call:worker:2:1"

# B4 — planner FORMAT: planner returns prose instead of task JSON
submit B  t06_fmt_planner_qwen    "$QWEN_SCRIPT" 3 2 "--inject format:planner"

# B5 — aggregator FORMAT: aggregator returns prose instead of report JSON
submit B  t07_fmt_aggregator_qwen "$QWEN_SCRIPT" 3 2 "--inject format:aggregator"

# B6 — aggregator LOGIC: aggregator returns wrong sum (result × 0.5)
submit B  t08_logic_aggregator_qwen "$QWEN_SCRIPT" 3 2 "--inject logic:aggregator"

# ─────────────────────────────────────────────────────────────────────────────
# GROUP C — Error-type variety, gemma4:27b, 3 agents, 2 steps
# Purpose: same error taxonomy as Group B but under gemma4 to reveal
#          model-specific differences in failure signatures and recovery.
# ─────────────────────────────────────────────────────────────────────────────
submit C  t09_fmt_worker_gemma4     "$GEMMA_SCRIPT" 3 2 "--inject format:worker:1:1"
submit C  t10_logic_worker_gemma4   "$GEMMA_SCRIPT" 3 2 "--inject logic:worker:0:1"
submit C  t11_toolcall_worker_gemma4 "$GEMMA_SCRIPT" 3 2 "--inject tool_call:worker:2:1"

# ─────────────────────────────────────────────────────────────────────────────
# GROUP D — Multiple simultaneous errors, qwen2.5-coder:32b
# Purpose: stress the CEE's ability to record and attribute several independent
#          failure events within a single workflow run.
# ─────────────────────────────────────────────────────────────────────────────

# D1 — two error types on two different workers, same step
submit D  t12_two_errors_qwen \
  "$QWEN_SCRIPT" 3 2 \
  "--inject format:worker:0:1 --inject logic:worker:1:1"

# D2 — all three error types, one per worker, same step
submit D  t13_all_types_qwen \
  "$QWEN_SCRIPT" 3 2 \
  "--inject format:worker:0:1 --inject logic:worker:1:1 --inject tool_call:worker:2:1"

# D3 — cross-agent errors: planner + one worker
submit D  t14_cross_agent_qwen \
  "$QWEN_SCRIPT" 3 2 \
  "--inject format:planner --inject logic:worker:0:1"

# D4 — staggered-step errors: each worker fails on a different step
#       (requires 3 steps so step 3 can be targeted)
submit D  t15_staggered_steps_qwen \
  "$QWEN_SCRIPT" 3 3 \
  "--inject tool_call:worker:0:1 --inject format:worker:1:2 --inject logic:worker:2:3"

# ─────────────────────────────────────────────────────────────────────────────
# GROUP E — Multiple simultaneous errors, gemma4:27b
# Purpose: same multi-error scenarios under gemma4 for model comparison.
# ─────────────────────────────────────────────────────────────────────────────
submit E  t16_two_errors_gemma4 \
  "$GEMMA_SCRIPT" 3 2 \
  "--inject format:worker:0:1 --inject tool_call:worker:2:1"

submit E  t17_all_types_gemma4 \
  "$GEMMA_SCRIPT" 3 2 \
  "--inject format:worker:0:1 --inject logic:worker:1:1 --inject tool_call:worker:2:1"

# ─────────────────────────────────────────────────────────────────────────────
# GROUP F — Agent-scale: minimal (2 workers, 1 refinement step)
# Purpose: smallest possible workflow — verifies CEE handles single-step
#          sessions with only planner + 2 workers + aggregator.
# ─────────────────────────────────────────────────────────────────────────────
submit F  t18_scale_min_qwen   "$QWEN_SCRIPT"  2 1 ""
submit F  t19_scale_min_gemma4 "$GEMMA_SCRIPT" 2 1 ""

# ─────────────────────────────────────────────────────────────────────────────
# GROUP G — Agent-scale: medium (5 workers, 2 steps)
# Purpose: more concurrent sessions to stress the CEE's parallel tracking.
#          One error per run so both clean and failed sessions are present.
# ─────────────────────────────────────────────────────────────────────────────
submit G  t20_scale_med_qwen   "$QWEN_SCRIPT"  5 2 "--inject tool_call:worker:2:1"
submit G  t21_scale_med_gemma4 "$GEMMA_SCRIPT" 5 2 "--inject format:worker:3:1"

# ─────────────────────────────────────────────────────────────────────────────
# GROUP H — Agent-scale: large (8 workers, 2 steps, gemma4 only)
# Purpose: maximum agent count to stress CEE session management and export.
#          Two injected errors at non-adjacent workers.
# Note: OLLAMA_NUM_PARALLEL is capped at 5 in _aeg_run_common.sh to prevent
#       CUDA OOM (KvSize=1,048,576 at PARALLEL=8 × 131072 ctx exceeds GH200 VRAM).
# ─────────────────────────────────────────────────────────────────────────────
submit H  t22_scale_large_gemma4 \
  "$GEMMA_SCRIPT" 8 2 \
  "--inject logic:worker:3:1 --inject format:worker:7:1"

# ─────────────────────────────────────────────────────────────────────────────
# GROUP I — Context compaction: deep sessions (6 steps, no context cap)
# Purpose: each worker accumulates 25+ conversation turns, driving the CEE's
#          auto-checkpoint thread to create many intermediate restore points
#          and the context-diff tracker to build a deep Ctx_graph.
#          No num_ctx cap — tests checkpoint density, not exhaustion.
# ─────────────────────────────────────────────────────────────────────────────

# I1 — clean deep session: baseline for checkpoint density
submit I  t23_deep_clean_qwen   "$QWEN_SCRIPT"  3 6 ""
submit I  t24_deep_clean_gemma4 "$GEMMA_SCRIPT" 3 6 ""

# I2 — error injected mid-session (step 3 of 6): tests recovery-event
#       generation partway through a long context graph
submit I  t25_deep_fmt_mid_qwen \
  "$QWEN_SCRIPT" 3 6 \
  "--inject format:worker:1:3"

submit I  t26_deep_toolcall_mid_gemma4 \
  "$GEMMA_SCRIPT" 3 6 \
  "--inject tool_call:worker:0:4"

# I3 — late-step errors (step 5 of 6): recovery event near end of session
submit I  t27_deep_logic_late_qwen \
  "$QWEN_SCRIPT" 3 6 \
  "--inject logic:worker:2:5"

# I4 — multi-error across steps in a deep session
submit I  t28_deep_multi_err_gemma4 \
  "$GEMMA_SCRIPT" 5 6 \
  "--inject logic:worker:1:3 --inject format:worker:3:5"

# ─────────────────────────────────────────────────────────────────────────────
# GROUP J — Context compaction: forced context exhaustion (num_ctx=2048)
# Purpose: cap the Ollama context window so that worker sessions exceed it
#          around step 4, triggering stop_reason=max_tokens → Failure Detector
#          emits context_exhausted recovery events → CheckpointManager builds
#          a rollback plan → Replay Injector re-injects error context on the
#          next request.  This is the primary CEE compaction pathway.
# ─────────────────────────────────────────────────────────────────────────────

# J1 — clean (no injected errors): context exhaustion is the only failure signal
submit J  t29_ctx_exhaust_clean_qwen \
  "$QWEN_SCRIPT" 3 4 \
  "--num-ctx 2048"

submit J  t30_ctx_exhaust_clean_gemma4 \
  "$GEMMA_SCRIPT" 3 4 \
  "--num-ctx 2048"

# J2 — format error on worker 0, step 2 + context exhaustion from small window:
#       two independent failure paths for the CEE to track simultaneously
submit J  t31_ctx_exhaust_fmt_qwen \
  "$QWEN_SCRIPT" 3 4 \
  "--num-ctx 2048 --inject format:worker:0:2"

submit J  t32_ctx_exhaust_fmt_gemma4 \
  "$GEMMA_SCRIPT" 3 4 \
  "--num-ctx 2048 --inject format:worker:0:2"

# J3 — tool_call error on step 1, then context exhaustion on later steps:
#       tests that the CEE correctly attributes two different recovery events
#       (HTTP-error recovery vs context-exhaustion recovery) to the same session
submit J  t33_ctx_exhaust_toolcall_qwen \
  "$QWEN_SCRIPT" 3 4 \
  "--num-ctx 2048 --inject tool_call:worker:1:1"

# J4 — 5 agents with capped context: broader fan-out of exhausted sessions
submit J  t34_ctx_exhaust_scale_gemma4 \
  "$GEMMA_SCRIPT" 5 4 \
  "--num-ctx 2048 --inject logic:worker:2:2"

echo ""
echo "All requested tests submitted."
echo "Results will appear under: ${AEG_DIR}/results/"
echo "Monitor with: squeue -u \$USER"
