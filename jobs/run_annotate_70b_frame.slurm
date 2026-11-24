#!/bin/bash
#SBATCH --job-name=annotate_70b_frame
#SBATCH --output=logs/annotate_70b_frame_%j.out
#SBATCH --error=logs/annotate_70b_frame_%j.err
#SBATCH --partition=owner_fb12
#SBATCH --gres=gpu:a100:3
#SBATCH --time=12:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16

set -euo pipefail

echo "========================================="
echo "Job started at: $(date)"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Job ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "========================================="

cd /home/ashrafs/projects/Conspiracy-Proj

export HF_HOME=/home/ashrafs/.cache/huggingface
export TRANSFORMERS_CACHE=/home/ashrafs/.cache/huggingface
export TOKENIZERS_PARALLELISM=false

source .venv/bin/activate

mkdir -p logs

nvidia-smi || true

# Model path: Using snapshot path directly
MODEL_PATH="${LLAMA_70B_MODEL_PATH:-/home/ashrafs/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b}"

# Using stable llm_annotate.py with 4-bit quantization (works, proven)
python -u llm_annotate.py \
  --prompt Data/prompts/llama_few_shot_prompt_frame.txt \
  --csv Data/all_classifications.csv \
  --update_csv_out outputs/llama-3.3-70B-frame_all_classifications_raw.csv \
  --model "$MODEL_PATH" \
  --max_new_tokens 512

echo "========================================="
echo "Job finished at: $(date)"
echo "========================================="


