#!/usr/bin/env bash
# agent-error-generator — granite4:latest + CEE
#
# Prerequisites (run once before submitting):
#   cd /work/hdd/bekn/kbateman/agent-error-generator
#   python3.11 -m venv .venv
#   .venv/bin/pip install -r requirements.txt
#
# Submit:
#   sbatch /work/hdd/bekn/kbateman/agent-error-generator/slurm_aeg_granite4.sh
#
# To add error injections, pass them via AEG_EXTRA_ARGS:
#   sbatch --export=ALL,AEG_EXTRA_ARGS="--inject logic:worker:0:1 --inject format:aggregator" \
#          /work/hdd/bekn/kbateman/agent-error-generator/slurm_aeg_granite4.sh

#SBATCH --job-name=aeg-granite4
#SBATCH --partition=ghx4
#SBATCH --account=bekn-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1          # GH200 (120 GB HBM) — ample for granite4:latest (~2.1 GB)
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=/work/hdd/bekn/kbateman/agent-error-generator/logs/aeg-granite4-%j.out
#SBATCH --error=/work/hdd/bekn/kbateman/agent-error-generator/logs/aeg-granite4-%j.err

mkdir -p /work/hdd/bekn/kbateman/agent-error-generator/logs

export AEG_MODEL=granite4:latest
export AEG_NUM_AGENTS="${AEG_NUM_AGENTS:-3}"
export AEG_NUM_STEPS="${AEG_NUM_STEPS:-2}"
export AEG_EXTRA_ARGS="${AEG_EXTRA_ARGS:-}"

# granite4:latest is a compact model (~2 GB); allow all worker agents to run
# in parallel — KV-cache pressure is negligible on the 120 GB GH200.
export OLLAMA_NUM_PARALLEL="${AEG_NUM_AGENTS}"

source "/work/hdd/bekn/kbateman/agent-error-generator/_aeg_run_common.sh"
