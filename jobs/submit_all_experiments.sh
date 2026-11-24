#!/bin/bash
# Submit all 6 annotation experiments for seen data
# Usage: ./submit_all_experiments.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "========================================="
echo "Submitting all 6 annotation experiments"
echo "========================================="
echo ""

# 8B experiments
echo "Submitting 8B experiments..."
JOB_8B_ZERO=$(sbatch jobs/run_annotate_8b.slurm | awk '{print $4}')
echo "  - 8B Zero-shot: Job ID $JOB_8B_ZERO"

JOB_8B_FEW=$(sbatch jobs/run_annotate_8b_fewshot.slurm | awk '{print $4}')
echo "  - 8B Few-shot: Job ID $JOB_8B_FEW"

JOB_8B_FRAME=$(sbatch jobs/run_annotate_8b_frame.slurm | awk '{print $4}')
echo "  - 8B Frame: Job ID $JOB_8B_FRAME"

echo ""

# 70B experiments
echo "Submitting 70B experiments..."
JOB_70B_ZERO=$(sbatch jobs/run_annotate_70b_zeroshot.slurm | awk '{print $4}')
echo "  - 70B Zero-shot: Job ID $JOB_70B_ZERO"

JOB_70B_FEW=$(sbatch jobs/run_annotate_70b_fewshot.slurm | awk '{print $4}')
echo "  - 70B Few-shot: Job ID $JOB_70B_FEW"

JOB_70B_FRAME=$(sbatch jobs/run_annotate_70b_frame.slurm | awk '{print $4}')
echo "  - 70B Frame: Job ID $JOB_70B_FRAME"

echo ""
echo "========================================="
echo "All jobs submitted!"
echo "========================================="
echo ""
echo "Job IDs:"
echo "  8B Zero-shot: $JOB_8B_ZERO"
echo "  8B Few-shot:  $JOB_8B_FEW"
echo "  8B Frame:     $JOB_8B_FRAME"
echo "  70B Zero-shot: $JOB_70B_ZERO"
echo "  70B Few-shot:  $JOB_70B_FEW"
echo "  70B Frame:     $JOB_70B_FRAME"
echo ""
echo "Monitor jobs with: squeue -u $USER"
echo "Check logs in: logs/"


