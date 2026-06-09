#!/usr/bin/env bash
# agent-error-generator — gemma4:27b, multi-GPU (4× GH200)
#
# Requests the full 4-GPU node.  Workers are distributed round-robin across
# four independent Ollama instances (one per GPU) so they run truly in parallel.
# CEE instrumentation is intentionally disabled — this script is for timing only.
#
# Submit:
#   sbatch slurm_aeg_multigpu_gemma4.sh
#
# Override worker count / repeat ID:
#   sbatch --export=ALL,AEG_NUM_AGENTS=8,AEG_WORKFLOW_ID=mg_gemma4_8w_4g_r1 \
#          slurm_aeg_multigpu_gemma4.sh

#SBATCH --job-name=aeg-mg-gemma4
#SBATCH --partition=ghx4
#SBATCH --account=bekn-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/work/hdd/bekn/kbateman/agent-error-generator/logs/aeg-mg-gemma4-%j.out
#SBATCH --error=/work/hdd/bekn/kbateman/agent-error-generator/logs/aeg-mg-gemma4-%j.err

mkdir -p /work/hdd/bekn/kbateman/agent-error-generator/logs

export AEG_MODEL=gemma4:latest
export AEG_NUM_AGENTS="${AEG_NUM_AGENTS:-4}"
export AEG_NUM_STEPS="${AEG_NUM_STEPS:-2}"
export AEG_NUM_GPUS=4
export AEG_EXTRA_ARGS="${AEG_EXTRA_ARGS:-}"

source "/work/hdd/bekn/kbateman/agent-error-generator/_aeg_run_common_multigpu.sh"
