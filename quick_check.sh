#!/bin/bash
# Quick one-time status check
# Usage: ./quick_check.sh [JOB_ID]

JOB_ID=${1:-8269908}

echo "Quick Status Check for Job $JOB_ID"
echo "=========================================="
echo ""

# Job status
echo "Queue Status:"
squeue -j $JOB_ID --format="%.18i %.9P %.20j %.8u %.8T %.10M %.9l %.6D" 2>/dev/null || echo "Job not in queue"

echo ""
echo "Detailed Info:"
scontrol show job $JOB_ID 2>/dev/null | grep -E "JobState|StartTime|RunTime|Reason" || echo "Job not found"

echo ""
OUTPUT="logs/annotate_8b_${JOB_ID}.out"
if [ -f "$OUTPUT" ]; then
    echo "Output file exists. Last 10 lines:"
    tail -n 10 "$OUTPUT"
else
    echo "Output file not created yet (job may not have started)"
fi









