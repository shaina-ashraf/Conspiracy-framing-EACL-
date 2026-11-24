#!/bin/bash
#SBATCH --job-name=annotate_8b_frame
#SBATCH --output=logs/annotate_8b_frame_%j.out
#SBATCH --error=logs/annotate_8b_frame_%j.err
#SBATCH --partition=owner_fb12
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4

set -euo pipefail

echo "========================================="
echo "Job started at: $(date)"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Job ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "========================================="

cd /home/ashrafs/projects/Conspiracy-Proj

# Optional: set caches to shared storage
export HF_HOME=/home/ashrafs/.cache/huggingface
export TRANSFORMERS_CACHE=/home/ashrafs/.cache/huggingface
export TOKENIZERS_PARALLELISM=false

# venv
source .venv/bin/activate

mkdir -p logs

nvidia-smi || true

python -u llm_annotate.py \
  --prompt Data/prompts/llama_few_shot_prompt_frame.txt \
  --csv Data/all_classifications.csv \
  --update_csv_out outputs/llama8b-frame_all_classifications_raw.csv \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --max_new_tokens 512

echo "========================================="
echo "Job finished at: $(date)"
echo "========================================="


