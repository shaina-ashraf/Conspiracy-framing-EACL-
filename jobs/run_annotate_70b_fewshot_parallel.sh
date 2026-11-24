#!/bin/bash
# Script to run 70B few-shot annotation in parallel jobs
# This splits the dataset into N parts and runs them in parallel

TOTAL_TEXTS=2077  # Adjust based on actual count
NUM_JOBS=4        # Number of parallel jobs

# Calculate chunk size
CHUNK_SIZE=$((TOTAL_TEXTS / NUM_JOBS))

echo "Splitting ${TOTAL_TEXTS} texts into ${NUM_JOBS} jobs of ~${CHUNK_SIZE} texts each"
echo "Using FEW-SHOT prompt for 70B model"
echo "Using OPTIMIZED version with batch processing"

# Submit parallel jobs
for i in $(seq 0 $((NUM_JOBS - 1))); do
    SKIP=$((i * CHUNK_SIZE))
    TAKE=$CHUNK_SIZE
    
    # Last job takes remaining texts
    if [ $i -eq $((NUM_JOBS - 1)) ]; then
        TAKE=$((TOTAL_TEXTS - SKIP))
    fi
    
    echo "Submitting job $i: skip=${SKIP}, take=${TAKE}"
    
    # Create temporary SLURM script
    cat > /tmp/annotate_70b_fewshot_job${i}.slurm <<EOF
#!/bin/bash
#SBATCH --job-name=70b_fewshot_job${i}
#SBATCH --output=logs/annotate_70b_fewshot_job${i}_%j.out
#SBATCH --error=logs/annotate_70b_fewshot_job${i}_%j.err
#SBATCH --partition=owner_fb12
#SBATCH --gres=gpu:a100:3
#SBATCH --time=12:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=24

cd /home/ashrafs/projects/Conspiracy-Proj
export HF_HOME=/home/ashrafs/.cache/huggingface
export TRANSFORMERS_CACHE=/home/ashrafs/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
source .venv/bin/activate

MODEL_PATH="${LLAMA_70B_MODEL_PATH:-/home/ashrafs/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b}"

# OPTIMIZATION: Batch processing + reduced tokens + Flash Attention 2
python -u llm_annotate_optimized.py \
  --prompt Data/prompts/llama_few_shot_prompt.txt \
  --csv Data/all_classifications.csv \
  --model "\$MODEL_PATH" \
  --max_new_tokens 256 \
  --batch_size 4 \
  --skip_index ${SKIP} \
  --take_index ${TAKE} \
  --update_csv_out outputs/llama-3.3-70B-fewshot_job${i}_all_classifications_raw.csv \
  --fail_log outputs/llama-3.3-70B-fewshot_job${i}_annotation_failures.jsonl

echo "Job ${i} completed"
EOF
    
    sbatch /tmp/annotate_70b_fewshot_job${i}.slurm
done

echo "All ${NUM_JOBS} jobs submitted for 70B few-shot experiment"
echo "Monitor with: squeue -u ashrafs"

