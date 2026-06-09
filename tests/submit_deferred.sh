#!/usr/bin/env bash
# tests/submit_deferred.sh -- submit the deferred-detection test suite (GROUP K).
#
# Usage:
#   bash tests/submit_deferred.sh [SUBGROUP]
#
#   SUBGROUP (optional): K1 | K2 | K3 | K4 | K5
#   Omit to submit every subgroup.
#
# Subgroups:
#   K1 -- Within-worker single-step lag (inject at step 1, detect at step 2)
#   K2 -- Within-worker multi-step lag  (inject at step 1, detect at step 3+)
#   K3 -- Aggregator-phase detection    (worker stays silent; coordinator reports)
#   K4 -- Silent propagation            (detect_phase=none; error never flagged)
#   K5 -- Mixed: deferred + immediate in the same run (comparison baselines)
#
# Inject format: TYPE:ROLE[:IDX[:INJECT_STEP[:DETECT_STEP[:DETECT_PHASE]]]]
#
# All tests use 3 workers / 4 steps (enough room to vary detection lag)
# unless a subgroup needs a different scale.

set -euo pipefail

AEG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QWEN_SCRIPT="${AEG_DIR}/slurm_aeg_qwen25coder.sh"
GEMMA_SCRIPT="${AEG_DIR}/slurm_aeg_gemma4.sh"
FILTER="${1:-ALL}"

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

# =============================================================================
# K1 -- Within-worker, single-step detection lag
#
# Purpose: establish the smallest detectable lag (1 step).  Error fires at
#          step 1; the worker does not flag it until step 2.  The CEE should
#          record an injection event at step 1 and a detection event at step 2.
#
# Detection benchmark: lag = detect_step - inject_step = 1 step.
# Comparison baseline: equivalent run with immediate detection (no detect_step).
# =============================================================================

# K1a -- logic error, 1-step lag, qwen
#   inject logic at step 1; detect at step 2 (lag=1)
submit K1  t50_defer_logic_lag1_qwen \
  "$QWEN_SCRIPT" 3 4 \
  "--inject logic:worker:0:1:2:workers"

# K1b -- same scenario, gemma4 (model comparison)
submit K1  t51_defer_logic_lag1_gemma4 \
  "$GEMMA_SCRIPT" 3 4 \
  "--inject logic:worker:0:1:2:workers"

# K1c -- tool_call error, 1-step lag, qwen
#   inject tool_call at step 1; detect at step 2
submit K1  t52_defer_toolcall_lag1_qwen \
  "$QWEN_SCRIPT" 3 4 \
  "--inject tool_call:worker:1:1:2:workers"

# K1d -- format error, 1-step lag, qwen
submit K1  t53_defer_format_lag1_qwen \
  "$QWEN_SCRIPT" 3 4 \
  "--inject format:worker:2:1:2:workers"

# =============================================================================
# K2 -- Within-worker, multi-step detection lag
#
# Purpose: test larger detection lags (2 and 3 steps).  Longer lags give the
#          CEE more conversation turns between injection and detection, making
#          it harder to correctly attribute the error.
#
# Detection benchmark: lag = 2 steps (inject=1, detect=3) and
#                      lag = 3 steps (inject=1, detect=4).
# =============================================================================

# K2a -- logic error, 2-step lag (inject step 1, detect step 3), qwen
submit K2  t54_defer_logic_lag2_qwen \
  "$QWEN_SCRIPT" 3 4 \
  "--inject logic:worker:0:1:3:workers"

# K2b -- logic error, 2-step lag, gemma4
submit K2  t55_defer_logic_lag2_gemma4 \
  "$GEMMA_SCRIPT" 3 4 \
  "--inject logic:worker:0:1:3:workers"

# K2c -- logic error, 3-step lag (inject step 1, detect step 4), qwen
#   detect_step equals the last step -- detection fires right at run end
submit K2  t56_defer_logic_lag3_qwen \
  "$QWEN_SCRIPT" 3 4 \
  "--inject logic:worker:1:1:4:workers"

# K2d -- tool_call + format, different lags on different workers, qwen
#   worker 0: tool_call at step 1, detected at step 2 (lag=1)
#   worker 2: format    at step 2, detected at step 4 (lag=2)
submit K2  t57_defer_multi_lag_qwen \
  "$QWEN_SCRIPT" 3 4 \
  "--inject tool_call:worker:0:1:2:workers --inject format:worker:2:2:4:workers"

# K2e -- deep session: logic error injected at step 2, detected at step 5 (lag=3)
#   6 steps gives the CEE a long graph before and after the detection point
submit K2  t58_defer_logic_deep_lag3_qwen \
  "$QWEN_SCRIPT" 3 6 \
  "--inject logic:worker:1:2:5:workers"

# K2f -- detect_step beyond num_steps: inject at step 1, detect_step=8, num_steps=4
#   Should be caught by the run-end drain and reported with lag annotation
submit K2  t59_defer_detect_beyond_end_qwen \
  "$QWEN_SCRIPT" 3 4 \
  "--inject logic:worker:0:1:8:workers"

# =============================================================================
# K3 -- Aggregator-phase detection
#
# Purpose: error fires during the worker phase but the worker does not self-
#          report it.  Detection is logged by the coordinator after the
#          aggregator LLM call completes.  The wrong value still propagates to
#          the aggregator's input; the aggregator simply does not know it is
#          seeing a bad number unless it detects the outlier itself.
#
# CEE features: the coordinator emits a detection event in the aggregator-phase
#               timeline even though no worker session recorded an error.
#               final_report["aggregator_detected_errors"] carries the record.
# =============================================================================

# K3a -- logic error, aggregator-phase detection, qwen
submit K3  t60_defer_agg_logic_qwen \
  "$QWEN_SCRIPT" 3 3 \
  "--inject logic:worker:0:1::aggregator"

# K3b -- same, gemma4
submit K3  t61_defer_agg_logic_gemma4 \
  "$GEMMA_SCRIPT" 3 3 \
  "--inject logic:worker:0:1::aggregator"

# K3c -- tool_call error, aggregator-phase detection, qwen
submit K3  t62_defer_agg_toolcall_qwen \
  "$QWEN_SCRIPT" 3 3 \
  "--inject tool_call:worker:1:1::aggregator"

# K3d -- two workers with aggregator-phase detection, qwen
#   worker 0: logic error; worker 2: format error; both silent at worker phase
submit K3  t63_defer_agg_two_workers_qwen \
  "$QWEN_SCRIPT" 3 3 \
  "--inject logic:worker:0:1::aggregator --inject format:worker:2:1::aggregator"

# K3e -- mixed: worker 0 detected immediately, worker 1 detected at aggregator
#   Gives the CEE one normal failure event and one deferred-to-aggregator event
submit K3  t64_defer_agg_mixed_qwen \
  "$QWEN_SCRIPT" 3 3 \
  "--inject logic:worker:0:1 --inject tool_call:worker:1:1::aggregator"

# =============================================================================
# K4 -- Silent propagation (detect_phase=none)
#
# Purpose: error is injected but NEVER flagged anywhere.  The wrong value
#          propagates silently to the aggregator and final answer.  This
#          scenario tests the CEE's ability to detect anomalies without an
#          explicit injection signal -- the only observable effect is the
#          elevated absolute_error in the final report.
#
# CEE features: no failure events are emitted; the injections[] list in the
#               worker output still records the injection for offline analysis,
#               but errors[] is empty.
# =============================================================================

# K4a -- logic error, silent, qwen
submit K4  t65_silent_logic_qwen \
  "$QWEN_SCRIPT" 3 3 \
  "--inject logic:worker:0:1::none"

# K4b -- logic error, silent, gemma4
submit K4  t66_silent_logic_gemma4 \
  "$GEMMA_SCRIPT" 3 3 \
  "--inject logic:worker:0:1::none"

# K4c -- format error, silent (tests CEE's ability to observe parse failures
#         without a matching injection event in the session record)
submit K4  t67_silent_format_qwen \
  "$QWEN_SCRIPT" 3 3 \
  "--inject format:worker:1:1::none"

# K4d -- two silent errors on two workers
submit K4  t68_silent_two_workers_qwen \
  "$QWEN_SCRIPT" 3 3 \
  "--inject logic:worker:0:1::none --inject logic:worker:2:1::none"

# =============================================================================
# K5 -- Mixed detection modes (benchmarking comparisons)
#
# Purpose: run pairs of scenarios that differ only in detect_phase or
#          detect_step so the CEE can compute apples-to-apples comparisons
#          of how detection lag affects downstream metrics (absolute_error,
#          recovery-event timeline, checkpoint scoring).
#
# Each K5 pair uses the same injection spec; only detection differs.
# =============================================================================

# K5a -- immediate vs 2-step lag on the same error type/location
#   Pair A: immediate (no detect_step)
submit K5  t69_cmp_immediate_qwen \
  "$QWEN_SCRIPT" 3 4 \
  "--inject logic:worker:1:2"

#   Pair B: 2-step lag (inject=2, detect=4)
submit K5  t70_cmp_lag2_qwen \
  "$QWEN_SCRIPT" 3 4 \
  "--inject logic:worker:1:2:4:workers"

# K5b -- immediate vs aggregator-phase for a tool_call error (gemma4)
#   Pair A: immediate
submit K5  t71_cmp_toolcall_immediate_gemma4 \
  "$GEMMA_SCRIPT" 3 3 \
  "--inject tool_call:worker:2:1"

#   Pair B: aggregator-phase
submit K5  t72_cmp_toolcall_agg_gemma4 \
  "$GEMMA_SCRIPT" 3 3 \
  "--inject tool_call:worker:2:1::aggregator"

# K5c -- three-way comparison: immediate / deferred(lag=2) / silent
#   All inject logic on worker 0 at step 1; only detection mode differs.
#   t04_logic_worker_qwen (Group B) serves as the immediate baseline.
#   Pair B: lag=2
submit K5  t73_cmp3_lag2_qwen \
  "$QWEN_SCRIPT" 3 4 \
  "--inject logic:worker:0:1:3:workers"

#   Pair C: silent
submit K5  t74_cmp3_silent_qwen \
  "$QWEN_SCRIPT" 3 4 \
  "--inject logic:worker:0:1::none"

echo ""
echo "All requested Group K (deferred-detection) tests submitted."
echo "Results will appear under: ${AEG_DIR}/results/"
echo "Monitor with: squeue -u \$USER"
