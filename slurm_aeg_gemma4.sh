#!/usr/bin/env bash
# agent-error-generator — gemma4:27b + CEE
#
# Prerequisites (run once before submitting):
#   cd /work/hdd/bekn/kbateman/agent-error-generator
#   python3.11 -m venv .venv
#   .venv/bin/pip install -r requirements.txt
#
# Submit:
#   sbatch /work/hdd/bekn/kbateman/agent-error-generator/slurm_aeg_gemma4.sh
#
# To add error injections, pass them via AEG_EXTRA_ARGS:
#   sbatch --export=ALL,AEG_EXTRA_ARGS="--inject logic:worker:0:1 --inject format:aggregator" \
#          /work/hdd/bekn/kbateman/agent-error-generator/slurm_aeg_gemma4.sh

#SBATCH --job-name=aeg-gemma4
#SBATCH --partition=ghx4
#SBATCH --account=bekn-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1          # GH200 (120 GB HBM) — sufficient for gemma4:27b
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=/work/hdd/bekn/kbateman/agent-error-generator/logs/aeg-gemma4-%j.out
#SBATCH --error=/work/hdd/bekn/kbateman/agent-error-generator/logs/aeg-gemma4-%j.err

mkdir -p /work/hdd/bekn/kbateman/agent-error-generator/logs

export AEG_MODEL=gemma4:latest
export AEG_NUM_AGENTS="${AEG_NUM_AGENTS:-3}"
export AEG_NUM_STEPS="${AEG_NUM_STEPS:-2}"
export AEG_EXTRA_ARGS="${AEG_EXTRA_ARGS:-}"

# gemma4:27b is smaller than qwen2.5-coder:32b; allow all worker agents to
# send requests in parallel — the 120 GB GH200 should handle the KV-cache
# allocation for all concurrent sessions comfortably.
export OLLAMA_NUM_PARALLEL="${AEG_NUM_AGENTS}"

source "/work/hdd/bekn/kbateman/agent-error-generator/_aeg_run_common.sh"
