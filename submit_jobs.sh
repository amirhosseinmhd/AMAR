#!/bin/bash

# Script to submit multiple experiments with different environments and models
# This script submits jobs and exits - you can logout after running this

# Define arrays for different configurations
ENVIRONMENTS=("classroom" "meeting_room" "empty_room")
MODELS=("DETR_RVQ")

echo "Starting multi-experiment job submission at $(date)"
echo "Will submit ${#ENVIRONMENTS[@]} environments × ${#MODELS[@]} models = $((${#ENVIRONMENTS[@]} * ${#MODELS[@]})) total experiments"

# Store job IDs for tracking
JOB_IDS=()
JOB_DETAILS=()

# Create a job tracking file to save job information
JOB_TRACKING_FILE="submitted_jobs_$(date +%Y%m%d_%H%M%S).txt"

echo "Experiment Job Submission - $(date)" > "$JOB_TRACKING_FILE"
echo "=====================================" >> "$JOB_TRACKING_FILE"
echo "" >> "$JOB_TRACKING_FILE"

# Loop through all combinations
for env in "${ENVIRONMENTS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo ""
        echo "========================================="
        echo "Submitting job for Environment: $env, Model: $model"
        echo "========================================="
        
        # Create a custom job name for this experiment
        JOB_NAME="csi_${env}_${model}"
        
        # Submit the job with environment variables set
        JOB_ID=$(sbatch --job-name="$JOB_NAME" \
                       --export="ENVIRONMENTS_EXP=$env,MODEL_TYPE=$model,WANDB_MODE=offline" \
                       cc-job.sh | grep -o '[0-9]*')
        
        if [ ! -z "$JOB_ID" ]; then
            JOB_IDS+=($JOB_ID)
            JOB_DETAILS+=("$JOB_ID:$env:$model")
            echo "Job submitted with ID: $JOB_ID"
            echo "Environment: $env, Model: $model"
            
            # Save to tracking file
            echo "Job ID: $JOB_ID - Environment: $env, Model: $model" >> "$JOB_TRACKING_FILE"
        else
            echo "Error: Failed to submit job for $env - $model"
            echo "ERROR: Failed to submit job for $env - $model" >> "$JOB_TRACKING_FILE"
        fi
        
        # Small delay to avoid overwhelming the scheduler
        sleep 2
    done
done

echo ""
echo "========================================="
echo "All jobs submitted!"
echo "Job IDs: ${JOB_IDS[@]}"
echo "========================================="

# Save detailed job information
echo "" >> "$JOB_TRACKING_FILE"
echo "All Job IDs: ${JOB_IDS[@]}" >> "$JOB_TRACKING_FILE"
echo "" >> "$JOB_TRACKING_FILE"
echo "Commands to check job status:" >> "$JOB_TRACKING_FILE"
echo "squeue -u \$USER" >> "$JOB_TRACKING_FILE"
echo "squeue -j ${JOB_IDS[*]// /,}" >> "$JOB_TRACKING_FILE"
echo "" >> "$JOB_TRACKING_FILE"
echo "To check specific jobs:" >> "$JOB_TRACKING_FILE"
for job_id in "${JOB_IDS[@]}"; do
    echo "squeue -j $job_id" >> "$JOB_TRACKING_FILE"
done

echo "" >> "$JOB_TRACKING_FILE"
echo "To cancel all jobs if needed:" >> "$JOB_TRACKING_FILE"
echo "scancel ${JOB_IDS[*]}" >> "$JOB_TRACKING_FILE"

# Create a simple job status check script
cat > "check_jobs.sh" << 'EOF'
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
EOF

chmod +x check_jobs.sh

echo ""
echo "🚀 Job submission completed!"
echo ""
echo "📁 Job tracking saved to: $JOB_TRACKING_FILE"
echo "📊 Quick status check: ./check_jobs.sh"
echo ""
echo "💡 You can now safely logout from Compute Canada."
echo "   When you return, use these commands:"
echo "   - Check job status: ./check_jobs.sh"
echo "   - Check your jobs: squeue -u \$USER"
echo "   - Sync wandb runs: ./sync_wandb_runs.sh (after jobs complete)"
echo ""
echo "🔧 If you need to cancel jobs: scancel ${JOB_IDS[*]}"
echo ""

# Display the current job status one time before exiting
echo "Current status snapshot:"
./check_jobs.sh

echo ""
echo "✅ Safe to logout! Jobs are running in the background."