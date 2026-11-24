#!/bin/bash
#SBATCH --job-name=70b_fewshot_vllm
#SBATCH --output=logs/annotate_70b_fewshot_vllm_%j.out
#SBATCH --error=logs/annotate_70b_fewshot_vllm_%j.err
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
export TOKENIZERS_PARALLELISM=false

source .venv/bin/activate

mkdir -p logs outputs

nvidia-smi || true

# Model path
MODEL_PATH="${LLAMA_70B_MODEL_PATH:-/home/ashrafs/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b}"

# vLLM Configuration
# - tensor_parallel_size=3: Use all 3 GPUs
# - gpu_memory_utilization=0.9: Use 90% of GPU memory
# - dtype=bfloat16: Balance between speed and precision
# - max_model_len=4096: Maximum sequence length
# - continuous batching: Automatically handles variable batch sizes
python -u llm_annotate_vllm.py \
  --prompt Data/prompts/llama_few_shot_prompt.txt \
  --csv Data/all_classifications.csv \
  --model "$MODEL_PATH" \
  --max_new_tokens 256 \
  --tensor_parallel_size 1 \
  --pipeline_parallel_size 3 \
  --gpu_memory_utilization 0.75 \
  --dtype bfloat16

echo "========================================="
echo "Job finished at: $(date)"
echo "========================================="

