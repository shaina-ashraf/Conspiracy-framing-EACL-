#!/bin/bash
#SBATCH --job-name=annotate_unseen
#SBATCH --partition=owner_fb12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/annotate_unseen_%j.out
#SBATCH --error=logs/annotate_unseen_%j.err
#SBATCH --export=ALL

set -euo pipefail

echo "========================================="
echo "Job started at: $(date)"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Partition: ${SLURM_JOB_PARTITION:-N/A}"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "========================================="

# Optional: load your environment here
# module load anaconda/2024.06
# source ~/.bashrc
# conda activate conspiracy-annotations

PROJECT_ROOT="/home/ashrafs/projects/Conspiracy-Proj"
cd "$PROJECT_ROOT"

# Hugging Face cache locations (override as needed)
export HF_HOME="${HF_HOME:-/home/ashrafs/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/home/ashrafs/.cache/huggingface}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Activate virtual environment if available
if [[ -z "${VIRTUAL_ENV:-}" && -d ".venv" ]]; then
  source .venv/bin/activate
fi

mkdir -p logs outputs
nvidia-smi || true

PROMPT_PATH="${PROMPT_PATH:-Data/prompts/llama_few_shot_prompt_unseen.txt}"
CSV_PATH="${CSV_PATH:-Data/unseen/un-labelled.csv}"
# Default to 8B; override with MODEL_ID="meta-llama/Meta-Llama-3.3-70B-Instruct" (or similar) for 70B runs.
MODEL_ID="${MODEL_ID:-meta-llama/Meta-Llama-3-8B-Instruct}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
LIMIT="${LIMIT:-None}"

LIMIT_ARG=()
if [[ "${LIMIT}" != "None" ]]; then
  LIMIT_ARG=(--limit "${LIMIT}")
fi

echo ""
echo "Running unseen annotation with:"
echo "  PROMPT_PATH: ${PROMPT_PATH}"
echo "  CSV_PATH: ${CSV_PATH}"
echo "  MODEL_ID: ${MODEL_ID}"
echo "  MAX_NEW_TOKENS: ${MAX_NEW_TOKENS}"
if [[ ${#LIMIT_ARG[@]} -gt 0 ]]; then
  echo "  LIMIT: ${LIMIT}"
else
  echo "  LIMIT: all samples"
fi
echo ""

srun python llm_annotate_unseen.py \
  --prompt "${PROMPT_PATH}" \
  --csv "${CSV_PATH}" \
  --model "${MODEL_ID}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  "${LIMIT_ARG[@]}"

echo ""
echo "Job finished at: $(date)"
echo "========================================="


