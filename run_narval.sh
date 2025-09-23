#!/bin/bash

# Script to run multiple experiments with different environments and models
# Runs wandb in offline mode during computation, then syncecho ""
echo "========================================="
echo "Multi-experiment run completed at $(date)"
echo "Summary:"
echo "- Environments tested: ${ENVIRONMENTS[@]}"
echo "- Models tested: ${MODELS[@]}"
echo "- NUM_CODES values tested: ${NUM_CODES_VALUES[@]}"
echo "- NUM_RVQ_LAYERS values tested: ${NUM_RVQ_LAYERS_VALUES[@]}"
echo "- Total experiments: $((${#ENVIRONMENTS[@]} * ${#MODELS[@]} * ${#NUM_CODES_VALUES[@]} * ${#NUM_RVQ_LAYERS_VALUES[@]}))"
echo "- Job IDs: ${JOB_IDS[@]}"
echo "========================================"
# Define arrays for different configurations
ENVIRONMENTS=("empty_room") # ("empty_room")
MODELS=("DETR_RVQ") # ("DETR_RVQ") #"")
NUM_CODES_VALUES=(8 16 32 64 128 256)
NUM_RVQ_LAYERS_VALUES=(1 2 4 6 8)

echo "Starting multi-experiment run at $(date)"
echo "Will run ${#ENVIRONMENTS[@]} environments × ${#MODELS[@]} models × ${#NUM_CODES_VALUES[@]} num_codes × ${#NUM_RVQ_LAYERS_VALUES[@]} num_rvq_layers = $((${#ENVIRONMENTS[@]} * ${#MODELS[@]} * ${#NUM_CODES_VALUES[@]} * ${#NUM_RVQ_LAYERS_VALUES[@]})) total experiments"

# Store job IDs for monitoring
JOB_IDS=()

# Loop through all combinations
for env in "${ENVIRONMENTS[@]}"; do
    for model in "${MODELS[@]}"; do
        for num_codes in "${NUM_CODES_VALUES[@]}"; do
            for num_rvq_layers in "${NUM_RVQ_LAYERS_VALUES[@]}"; do
                echo ""
                echo "========================================="
                echo "Submitting job for Environment: $env, Model: $model, Codes: $num_codes, RVQ Layers: $num_rvq_layers"
                echo "========================================="
                
                # Create a custom job name for this experiment
                JOB_NAME="csi_${env}_${model}_c${num_codes}_l${num_rvq_layers}"
                
                # Submit the job with environment variables set
                # We'll modify the cc-job.sh temporarily for each run
                JOB_ID=$(sbatch --job-name="$JOB_NAME" \
                               --export="ENVIRONMENTS_EXP=$env,MODEL_TYPE=$model,WANDB_MODE=offline,NUM_CODES=$num_codes,NUM_RVQ_LAYERS=$num_rvq_layers" \
                               cc-job.sh | grep -o '[0-9]*')
                
                if [ ! -z "$JOB_ID" ]; then
                    JOB_IDS+=($JOB_ID)
                    echo "Job submitted with ID: $JOB_ID"
                    echo "Environment: $env, Model: $model, Codes: $num_codes, RVQ Layers: $num_rvq_layers"
                else
                    echo "Error: Failed to submit job for $env - $model - $num_codes - $num_rvq_layers"
                fi
                
                # Small delay to avoid overwhelming the scheduler
                sleep 2
            done
        done
    done
done

echo ""
echo "========================================="
echo "All jobs submitted. Job IDs: ${JOB_IDS[@]}"
echo "========================================="

# Function to check if all jobs are completed
check_jobs_completion() {
    local all_completed=true
    for job_id in "${JOB_IDS[@]}"; do
        # Check job status using squeue with specific format
        local job_status=$(squeue -j $job_id --noheader --format="%T" 2>/dev/null)
        
        # If job_status is not empty, job is still in queue/running
        if [ ! -z "$job_status" ]; then
            all_completed=false
            break
        fi
    done
    echo $all_completed
}

# Wait for all jobs to complete
echo "Monitoring job completion..."

# Set a maximum wait time (e.g., 24 hours = 1440 minutes)
MAX_WAIT_MINUTES=1440
wait_minutes=0

while [ "$(check_jobs_completion)" = "false" ]; do
    echo "$(date): Some jobs still running. Waiting... (${wait_minutes}/${MAX_WAIT_MINUTES} minutes)"
    
    # Check if we've reached the maximum wait time
    if [ $wait_minutes -ge $MAX_WAIT_MINUTES ]; then
        echo "WARNING: Maximum wait time reached ($MAX_WAIT_MINUTES minutes)."
        echo "Some jobs may still be running. You can:"
        echo "1. Check job status manually with: squeue -u \$USER"
        echo "2. Run wandb sync manually later from /home/amirmhd/scratch/wandb/offline-run-* directories"
        echo "3. Cancel remaining jobs with: scancel <job_id>"
        break
    fi
    
    # Show current status of our jobs
    echo "Current job status:"
    for job_id in "${JOB_IDS[@]}"; do
        local job_info=$(squeue -j $job_id --noheader --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" 2>/dev/null)
        if [ ! -z "$job_info" ]; then
            echo "$job_info"
        else
            echo "Job $job_id: COMPLETED"
        fi
    done
    
    sleep 60  # Check every minute
    ((wait_minutes++))
done

echo ""
echo "========================================="
echo "All jobs completed at $(date)"
echo "========================================="

# Now sync all wandb offline runs to online
echo "Starting wandb sync process..."

# Find all wandb offline run directories
WANDB_DIR="/home/amirmhd/scratch/wandb"
if [ -d "$WANDB_DIR" ]; then
    echo "Syncing wandb offline runs to online..."
    
    # Find all offline run directories
    OFFLINE_RUNS=$(find $WANDB_DIR -name "offline-run-*" -type d)
    
    if [ ! -z "$OFFLINE_RUNS" ]; then
        echo "Found offline runs to sync:"
        echo "$OFFLINE_RUNS"
        
        # Sync each offline run
        for run_dir in $OFFLINE_RUNS; do
            echo "Syncing: $run_dir"
            wandb sync "$run_dir"
            
            if [ $? -eq 0 ]; then
                echo "Successfully synced: $run_dir"
            else
                echo "Failed to sync: $run_dir"
            fi
        done
        
        echo "All wandb syncing completed!"
    else
        echo "No offline wandb runs found to sync."
    fi
else
    echo "No wandb directory found in scratch. No runs to sync."
fi

echo ""
echo "========================================="
echo "Multi-experiment run completed at $(date)"
echo "Summary:"
echo "- Environments tested: ${ENVIRONMENTS[@]}"
echo "- Models tested: ${MODELS[@]}"
echo "- Total experiments: $((${#ENVIRONMENTS[@]} * ${#MODELS[@]}))"
echo "- Job IDs: ${JOB_IDS[@]}"
echo "========================================="

# Optional: Generate a summary report
echo "Generating experiment summary..."
SUMMARY_FILE="experiment_summary_$(date +%Y%m%d_%H%M%S).txt"

cat > "$SUMMARY_FILE" << EOF
Experiment Summary - $(date)
============================

Configuration:
- Environments: ${ENVIRONMENTS[@]}
- Models: ${MODELS[@]}
- NUM_CODES values: ${NUM_CODES_VALUES[@]}
- NUM_RVQ_LAYERS values: ${NUM_RVQ_LAYERS_VALUES[@]}
- Total experiments: $((${#ENVIRONMENTS[@]} * ${#MODELS[@]} * ${#NUM_CODES_VALUES[@]} * ${#NUM_RVQ_LAYERS_VALUES[@]}))

Job Details:
- Job IDs: ${JOB_IDS[@]}
- Started: $(date)

Results Location:
- Check experiment_results/ directory for output files
- Wandb runs have been synced online

Individual Experiments:
EOF

# Add details for each combination
for env in "${ENVIRONMENTS[@]}"; do
    for model in "${MODELS[@]}"; do
        for num_codes in "${NUM_CODES_VALUES[@]}"; do
            for num_rvq_layers in "${NUM_RVQ_LAYERS_VALUES[@]}"; do
                echo "- Environment: $env, Model: $model, Codes: $num_codes, RVQ Layers: $num_rvq_layers" >> "$SUMMARY_FILE"
            done
        done
    done
done

echo "Summary saved to: $SUMMARY_FILE"
echo "Script completed successfully!"
