#!/bin/bash

# Script to sync wandb offline runs to online after jobs complete
# Run this when you return and jobs are finished

echo "🔄 Wandb Sync Script"
echo "===================="
echo "Run this script after your jobs have completed to sync offline wandb runs to the cloud."
echo ""

# First check if there are any jobs still running
echo "🔍 Checking for running jobs..."
if command -v squeue >/dev/null 2>&1; then
    RUNNING_JOBS=$(squeue -u $USER --noheader 2>/dev/null | grep -c "csi_")
    if [ $RUNNING_JOBS -gt 0 ]; then
        echo "⚠️  WARNING: Found $RUNNING_JOBS CSI-related jobs still running:"
        squeue -u $USER --format="%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R" | grep csi_
        echo ""
        read -p "Do you want to continue syncing anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "❌ Sync cancelled. Wait for jobs to complete first."
            exit 0
        fi
    else
        echo "✅ No CSI jobs currently running."
    fi
else
    echo "⚠️  Cannot check job status (squeue not available). Proceeding with sync."
fi

echo ""

# Check for wandb directory
WANDB_DIR="/home/amirmhd/scratch/wandb"
if [ ! -d "$WANDB_DIR" ]; then
    echo "❌ Error: wandb directory not found in scratch directory: $WANDB_DIR"
    echo "Wandb files should be saved to scratch directory by the compute jobs"
    exit 1
fi

echo "📁 Found wandb directory: $WANDB_DIR"

# Find all offline run directories
echo "🔍 Searching for offline runs..."
OFFLINE_RUNS=$(find $WANDB_DIR -name "offline-run-*" -type d | sort)

if [ -z "$OFFLINE_RUNS" ]; then
    echo "❌ No offline wandb runs found to sync."
    echo "Searched in: $WANDB_DIR"
    echo ""
    echo "This could mean:"
    echo "1. Jobs haven't created wandb runs yet"
    echo "2. Jobs are still running"
    echo "3. Wandb was disabled"
    echo "4. You're in the wrong directory"
    exit 0
fi

echo "✅ Found offline runs to sync:"
echo "$OFFLINE_RUNS" | sed 's/^/   /'
echo ""

# Count total runs
TOTAL_RUNS=$(echo "$OFFLINE_RUNS" | wc -l)
echo "📊 Total offline runs to sync: $TOTAL_RUNS"
echo ""

# Show run details
echo "📋 Run details:"
for run_dir in $OFFLINE_RUNS; do
    if [ -f "$run_dir/wandb-metadata.json" ]; then
        run_name=$(grep -o '"name": *"[^"]*"' "$run_dir/wandb-metadata.json" 2>/dev/null | cut -d'"' -f4)
        if [ ! -z "$run_name" ]; then
            echo "   $(basename $run_dir) -> $run_name"
        else
            echo "   $(basename $run_dir)"
        fi
    else
        echo "   $(basename $run_dir)"
    fi
done
echo ""

# Ask for confirmation
read -p "🚀 Do you want to sync all these runs to wandb cloud? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Sync cancelled."
    exit 0
fi

echo ""
echo "🔄 Starting sync process..."
echo ""

# Check wandb login status
echo "🔐 Checking wandb authentication..."
if ! wandb status >/dev/null 2>&1; then
    echo "❌ Not logged into wandb. Please login first:"
    echo "   wandb login"
    exit 1
else
    echo "✅ Wandb authentication OK"
fi

echo ""

# Sync each offline run
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_RUNS=()

for run_dir in $OFFLINE_RUNS; do
    echo "📤 Syncing: $(basename $run_dir)"
    
    if wandb sync "$run_dir" --include-globs="*" 2>&1; then
        echo "   ✅ Successfully synced: $(basename $run_dir)"
        ((SUCCESS_COUNT++))
    else
        echo "   ❌ Failed to sync: $(basename $run_dir)"
        FAILED_RUNS+=("$run_dir")
        ((FAIL_COUNT++))
    fi
    echo ""
done

echo "========================================="
echo "🎯 Sync Summary:"
echo "- Total runs: $TOTAL_RUNS"
echo "- Successfully synced: $SUCCESS_COUNT"
echo "- Failed: $FAIL_COUNT"
echo "========================================="

if [ $FAIL_COUNT -gt 0 ]; then
    echo ""
    echo "⚠️  Some runs failed to sync. Failed runs:"
    for failed_run in "${FAILED_RUNS[@]}"; do
        echo "   - $failed_run"
    done
    echo ""
    echo "💡 You can try syncing failed runs individually:"
    echo "   wandb sync <run_directory>"
    echo ""
    echo "   Or retry with more verbose output:"
    echo "   wandb sync <run_directory> --include-globs='*' --verbose"
fi

if [ $SUCCESS_COUNT -gt 0 ]; then
    echo ""
    echo "🎉 Successfully synced $SUCCESS_COUNT wandb runs!"
    echo "   Visit https://wandb.ai to view your experiments"
fi

# Check for experiment results
echo ""
echo "📁 Checking for experiment results..."
if [ -d "experiment_results" ]; then
    RESULT_FILES=$(ls -1 experiment_results/prev_model-* 2>/dev/null | wc -l)
    if [ $RESULT_FILES -gt 0 ]; then
        echo "✅ Found $RESULT_FILES experiment result files in experiment_results/"
        echo "   Latest results:"
        ls -lt experiment_results/prev_model-* | head -5 | sed 's/^/   /'
    else
        echo "⚠️  No experiment result files found in experiment_results/"
    fi
else
    echo "⚠️  experiment_results directory not found"
fi

echo ""
echo "✅ Wandb sync process completed!"

# Optional: Create a sync summary report
SYNC_SUMMARY="sync_summary_$(date +%Y%m%d_%H%M%S).txt"
cat > "$SYNC_SUMMARY" << EOF
Wandb Sync Summary - $(date)
============================

Sync Results:
- Total runs found: $TOTAL_RUNS
- Successfully synced: $SUCCESS_COUNT
- Failed to sync: $FAIL_COUNT

Synced Runs:
EOF

for run_dir in $OFFLINE_RUNS; do
    if [[ ! " ${FAILED_RUNS[@]} " =~ " ${run_dir} " ]]; then
        echo "✅ $run_dir" >> "$SYNC_SUMMARY"
    fi
done

if [ $FAIL_COUNT -gt 0 ]; then
    echo "" >> "$SYNC_SUMMARY"
    echo "Failed Runs:" >> "$SYNC_SUMMARY"
    for failed_run in "${FAILED_RUNS[@]}"; do
        echo "❌ $failed_run" >> "$SYNC_SUMMARY"
    done
fi

echo "" >> "$SYNC_SUMMARY"
echo "Sync completed at: $(date)" >> "$SYNC_SUMMARY"

echo "📄 Sync summary saved to: $SYNC_SUMMARY"