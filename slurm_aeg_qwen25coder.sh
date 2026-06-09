#!/usr/bin/env bash
# agent-error-generator — qwen2.5-coder:32b + CEE
#
# Prerequisites (run once before submitting):
#   cd /work/hdd/bekn/kbateman/agent-error-generator
#   python3.11 -m venv .venv
#   .venv/bin/pip install -r requirements.txt
#
# Submit:
#   sbatch /work/hdd/bekn/kbateman/agent-error-generator/slurm_aeg_qwen25coder.sh
#
# To add error injections, pass them via AEG_EXTRA_ARGS:
#   sbatch --export=ALL,AEG_EXTRA_ARGS="--inject format:worker:1:2 --inject tool_call:worker:0" \
#          /work/hdd/bekn/kbateman/agent-error-generator/slurm_aeg_qwen25coder.sh

#SBATCH --job-name=aeg-qwen25coder
#SBATCH --partition=ghx4
#SBATCH --account=bekn-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1          # GH200 (120 GB HBM) — sufficient for qwen2.5-coder:32b
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=/work/hdd/bekn/kbateman/agent-error-generator/logs/aeg-qwen25coder-%j.out
#SBATCH --error=/work/hdd/bekn/kbateman/agent-error-generator/logs/aeg-qwen25coder-%j.err

mkdir -p /work/hdd/bekn/kbateman/agent-error-generator/logs

export AEG_MODEL=qwen2.5-coder:32b
export AEG_NUM_AGENTS="${AEG_NUM_AGENTS:-3}"
export AEG_NUM_STEPS="${AEG_NUM_STEPS:-2}"
export AEG_EXTRA_ARGS="${AEG_EXTRA_ARGS:-}"

# qwen2.5-coder:32b is large; keep Ollama to one parallel slot per request to
# avoid OOM from simultaneous KV-cache allocations across all worker agents.
# Increase to AEG_NUM_AGENTS if the 120 GB HBM proves sufficient under load.
export OLLAMA_NUM_PARALLEL=1

source "/work/hdd/bekn/kbateman/agent-error-generator/_aeg_run_common.sh"
