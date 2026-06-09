# agent-error-simulator
Overview
--------
The test suite uses a declarative matrix (tests/test_matrix.yaml) and a Python
submission script (tests/submit_matrix.py) to generate and submit SLURM jobs.
Each job runs one workflow configuration on one model.

Current matrix: 144 jobs across 10 groups, targeting gemma4/mistral/granite4.
Full 4-model backup (including qwen): tests/full-test-backup.yaml (137 jobs —
the qwen set was run first and kept in a separate batch; qwen results are in
results/*_qwen_*).


SLURM Scripts
-------------
Each model has its own SLURM wrapper that sets AEG_MODEL and sources the shared
_aeg_run_common.sh:

  slurm_aeg_gemma4.sh      — gemma4:27b       (ghx4 partition, 1 GH200, 60G)
  slurm_aeg_mistral.sh     — mistral:latest   (ghx4 partition, 1 GH200, 60G)
  slurm_aeg_granite4.sh    — granite4:latest  (ghx4 partition, 1 GH200, 60G)
  slurm_aeg_qwen25coder.sh — qwen2.5-coder:32b (ghx4, OLLAMA_NUM_PARALLEL=1)

Multi-GPU variants (scaling studies, not in the main matrix):
  slurm_aeg_multigpu_gemma4.sh
  slurm_aeg_multigpu_qwen.sh  (sources _aeg_run_common_multigpu.sh)

_aeg_run_common.sh sequence
----------------------------
1. Sanity-checks .venv, CEE build, and chimaera_runtime_ext.so
2. Sources ollama_env.sh; starts ollama serve; polls until ready; pulls model
3. Clears stale Chimaera IPC; starts dt_demo_server; starts Flask proxy (port 9090)
4. Runs generator.py with --enable-proxy so every agent call routes through
   http://localhost:9090/_session/{workflow_id}_{role}_{index}/api/chat (recorded in CTE)
5. Exports CEE visuals per session → results/{workflow_id}/visuals/{role}/
6. Cleanup trap kills Flask, Chimaera, and Ollama on any exit

Each SLURM job binds to its own Ollama port (base 11434 + slot×500) so
concurrent jobs do not collide.


Setup (run once on a login node)
---------------------------------
cd /work/hdd/bekn/kbateman/agent-error-generator
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt


Submitting Jobs
---------------
Submit the full current matrix (144 jobs):
  python tests/submit_matrix.py

Submit specific groups only:
  python tests/submit_matrix.py A B K

Dry-run (print sbatch commands without submitting):
  python tests/submit_matrix.py --dry-run

List all jobs with their inject args:
  python tests/submit_matrix.py --list

Use a different matrix file:
  python tests/submit_matrix.py --matrix tests/full-test-backup.yaml

Submit a single ad-hoc job (bypass the matrix):
  sbatch --export=ALL,AEG_EXTRA_ARGS="--inject logic:worker:0:1 --inject format:aggregator" \
         slurm_aeg_gemma4.sh


Test Groups
-----------
Group A  — Baseline (no errors; 2 steps; 3 workers)
           Establishes per-model performance baselines.
           3 jobs: gemma4 / mistral / granite4

Group B  — Error-type variety (2 steps; 3 workers)
           One scenario per error type (format, logic, tool_call) per agent role
           (worker, planner, aggregator).
           21 jobs: worker errors × 3 models; planner/agg errors × 3 models

Group D  — Multiple simultaneous errors (2–3 steps; 3 workers)
           Two workers with different error types; all three types; cross-agent
           (planner+worker); staggered steps (each worker fails at a different step).
           12 jobs: 4 scenarios × 3 models

Group F  — Scale — minimal (1 step; 2 workers)
           Minimum viable worker count with no injections.
           3 jobs: 3 models

Group G  — Scale — medium (2 steps; 5 workers; 1 tool_call injection)
           Aggregator arithmetic under moderate scale.
           3 jobs: 3 models  [note: granite4 job failed to produce result]

Group H  — Scale — large (2 steps; 8 workers; 2 injections)
           Gemma4 only (resource-intensive).
           1 job: gemma4

Group I  — Deep sessions (6 steps; 3 workers, or 5 for multi_err)
           Context compaction; errors at mid/late steps.
           15 jobs: 5 scenarios × 3 models  [note: granite4 5-worker job failed]

Group J  — Context exhaustion (4–6 steps; 3–5 workers)
           --num-ctx 2048 (exhaustion ~step 4) and --num-ctx 512 (aggressive
           exhaustion ~step 3).
           21 jobs: 7 scenarios × 3 models  [several granite4 jobs failed]

Group K  — Deferred detection (3–6 steps; 3 workers)
           Tests inject_step / detect_step / detect_phase mechanics.
           K1: single-step detection lag (detect = inject+1)
           K2: multi-step lag (detect = inject+2 or +3); deep session; beyond-end
           K3: aggregator-phase detection (worker stays silent)
           K4: silent propagation (detect_phase: none)
           K5: pairwise comparison benchmarks (only detection mode varies)
           ~60 jobs across subgroups  [several granite4/gemma4 jobs failed]

Group L  — Explicit compaction (--max-turns 1; 6 steps; 3 workers)
           Forces context compaction at every turn.
           9 jobs: 3 scenarios × 3 models  [granite4 logic job failed]


Inject Argument Syntax
-----------------------
--inject <error_type>:<agent_type>[:<agent_index>[:<inject_step>[:<detect_step>[:<detect_phase>]]]]

  error_type:   format | logic | tool_call
  agent_type:   worker | planner | aggregator
  agent_index:  integer (0-based); omit or use empty for auto-assign
  inject_step:  step number (1-based) to inject the error
  detect_step:  step at which the error appears in errors[]; omit = same as inject_step
  detect_phase: workers | aggregator | none

Examples:
  --inject format:worker:0:1          # format error on worker 0 at step 1
  --inject logic:worker:0:1:3:workers # logic error at step 1, detected at step 3
  --inject tool_call:worker:0:1::aggregator  # error reported only at aggregator phase
  --inject logic:worker:0:1::none     # silent — never flagged


Results
-------
Each completed job writes to:
  results/{workflow_id}/
    result.json          — structured output (abs_err, worker finals, injections)
    visuals/{role}/      — CEE HTML + JSON per agent session

Analysis outputs:
  analysis/qwen_analysis.md         — per-experiment table + findings for 50 qwen jobs
  analysis/cross_model_analysis.md  — 134-job cross-model analysis (gemma4/mistral/granite4)
  analysis/results_table.csv        — raw results CSV


Model Registry (test_matrix.yaml)
----------------------------------
  qwen:     qwen2.5-coder:32b  — script: slurm_aeg_qwen25coder.sh
  gemma4:   gemma4:27b         — script: slurm_aeg_gemma4.sh
  mistral:  mistral:latest     — script: slurm_aeg_mistral.sh
  granite4: granite4:latest    — script: slurm_aeg_granite4.sh

Current defaults (test_matrix.yaml): models=[gemma4, mistral, granite4], num_steps=3,
agent_counts=[3]. To run qwen tests, use tests/full-test-backup.yaml or override
models per scenario.
