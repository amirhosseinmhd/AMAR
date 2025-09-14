#!/bin/bash
# Quick script to check the status of submitted jobs

echo "Checking job status at $(date)"
echo "=============================="

# Read job IDs from the most recent tracking file
LATEST_TRACKING=$(ls -t submitted_jobs_*.txt 2>/dev/null | head -1)

if [ -z "$LATEST_TRACKING" ]; then
    echo "No job tracking file found. Please check manually with: squeue -u \$USER"
    exit 1
fi

echo "Using tracking file: $LATEST_TRACKING"
echo ""

# Extract job IDs from the tracking file
JOB_IDS=($(grep "Job ID:" "$LATEST_TRACKING" | grep -o '[0-9]*' | head -20))

if [ ${#JOB_IDS[@]} -eq 0 ]; then
    echo "No job IDs found in tracking file."
    exit 1
fi

echo "Checking ${#JOB_IDS[@]} jobs..."
echo ""

# Check each job
RUNNING=0
COMPLETED=0
PENDING=0
OTHER=0

echo "Current job status:"
printf "%-12s %-10s %-20s %-10s %-10s\n" "JOB_ID" "STATE" "NAME" "TIME" "NODELIST"
echo "----------------------------------------------------------------"

for job_id in "${JOB_IDS[@]}"; do
    job_info=$(squeue -j $job_id --noheader --format="%.12i %.10T %.20j %.10M %.20N" 2>/dev/null)
    if [ ! -z "$job_info" ]; then
        echo "$job_info"
        state=$(echo "$job_info" | awk '{print $2}')
        case $state in
            "RUNNING") ((RUNNING++));;
            "PENDING"|"PD") ((PENDING++));;
            *) ((OTHER++));;
        esac
    else
        printf "%-12s %-10s %-20s %-10s %-10s\n" "$job_id" "COMPLETED" "N/A" "N/A" "N/A"
        ((COMPLETED++))
    fi
done

echo ""
echo "Summary:"
echo "- Running: $RUNNING"
echo "- Pending: $PENDING" 
echo "- Completed: $COMPLETED"
echo "- Other: $OTHER"
echo ""

if [ $((RUNNING + PENDING + OTHER)) -eq 0 ]; then
    echo "🎉 All jobs completed! You can now run './sync_wandb_runs.sh' to sync results."
else
    echo "⏳ Jobs still running/pending. Check again later."
fi
