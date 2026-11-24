#!/bin/bash
# Monitor SLURM job status and output
# Usage: ./monitor_job.sh [JOB_ID] [INTERVAL_SECONDS]

JOB_ID=${1:-8269908}
INTERVAL=${2:-30}  # Default: check every 30 seconds
OUTPUT_FILE="logs/annotate_8b_${JOB_ID}.out"
ERROR_FILE="logs/annotate_8b_${JOB_ID}.err"

echo "=========================================="
echo "Monitoring Job: $JOB_ID"
echo "Check interval: ${INTERVAL} seconds"
echo "Press Ctrl+C to stop"
echo "=========================================="
echo ""

# Function to check job status
check_status() {
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
    
    # Get job status
    STATUS=$(squeue -j $JOB_ID -h -o "%T %R" 2>/dev/null)
    if [ -z "$STATUS" ]; then
        echo "Job $JOB_ID not found in queue (may have completed or failed)"
        if [ -f "$OUTPUT_FILE" ]; then
            echo ""
            echo "Last 10 lines of output:"
            tail -n 10 "$OUTPUT_FILE"
        fi
        return 1
    else
        echo "Status: $STATUS"
    fi
    
    # Show job details if running
    if echo "$STATUS" | grep -q "RUNNING"; then
        echo ""
        # Show progress from output file if it exists
        if [ -f "$OUTPUT_FILE" ]; then
            echo "Recent progress:"
            tail -n 5 "$OUTPUT_FILE" | grep -E "\[|Processing|Done|total|row=" || echo "  (no progress lines yet)"
            
            # Count processed samples if available
            if grep -q "row=" "$OUTPUT_FILE"; then
                LAST_ROW=$(grep "row=" "$OUTPUT_FILE" | tail -1 | grep -oE "row=[0-9]+" | cut -d= -f2)
                if [ ! -z "$LAST_ROW" ]; then
                    echo "  Progress: ~$LAST_ROW samples processed"
                fi
            fi
        fi
        
        # Show GPU utilization if job is running
        NODE=$(squeue -j $JOB_ID -h -o "%N" 2>/dev/null)
        if [ ! -z "$NODE" ] && [ "$NODE" != "N/A" ]; then
            echo ""
            echo "GPU utilization (node: $NODE):"
            ssh -o ConnectTimeout=2 $NODE "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader" 2>/dev/null || echo "  (cannot access node)"
        fi
    fi
    
    # Show errors if any
    if [ -f "$ERROR_FILE" ] && [ -s "$ERROR_FILE" ]; then
        ERR_COUNT=$(wc -l < "$ERROR_FILE")
        if [ $ERR_COUNT -gt 0 ]; then
            echo ""
            echo "ERRORS (last 3 lines):"
            tail -n 3 "$ERROR_FILE"
        fi
    fi
    
    echo ""
    return 0
}

# Main monitoring loop
while true; do
    check_status
    if [ $? -ne 0 ]; then
        echo "Job monitoring ended."
        break
    fi
    sleep $INTERVAL
done









