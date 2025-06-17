#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=1:30:0
#SBATCH --mail-user=mdi.amirhossein@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --gpus-per-node=1
#SBATCH --job-name=jepa_prototype
#SBATCH --output=experiment_results/jepa_prototype-%j.out

# Create output directory
mkdir -p experiment_results

# Define directories
PROJECT_DIR=/home/amirmhd/projects/def-hinat/amirmhd/multi_modal_CSI
CODE_DIR=benchmark/wifi_csi

echo "Starting JEPA prototype job at $(date)"
echo "Job ID: ${SLURM_JOB_ID}"

# Copy code to temporary directory
echo "Copying code to temporary directory..."
mkdir -p $SLURM_TMPDIR/$CODE_DIR
cp -r $PROJECT_DIR/$CODE_DIR/* $SLURM_TMPDIR/$CODE_DIR/
cd $SLURM_TMPDIR/$CODE_DIR

# Load modules and activate environment
module purge
module load python/3.11.5 scipy-stack
source ~/py311/bin/activate

echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"

# Check if this is a resume run (look for existing checkpoint)
CHECKPOINT_DIR="$PROJECT_DIR/experiment_results/jepa_checkpoints"
LATEST_CHECKPOINT=""

if [ -d "$CHECKPOINT_DIR" ]; then
    LATEST_CHECKPOINT=$(find $CHECKPOINT_DIR -name "best_model.pth" -type f | head -1)
fi

# Run JEPA training
echo "Starting JEPA training..."
if [ -n "$LATEST_CHECKPOINT" ] && [ -f "$LATEST_CHECKPOINT" ]; then
    echo "Found existing checkpoint: $LATEST_CHECKPOINT"
    echo "Resuming training from checkpoint..."
    python -c "
import sys
sys.path.append('.')
sys.path.append('model')
from JEPA import resume_jepa_training

# Resume training for 2 more epochs
state_dict = resume_jepa_training(
    checkpoint_path='$LATEST_CHECKPOINT',
    environments=['meeting_room', 'empty_room'],
    num_epochs=4,  # Will continue from where it left off
    batch_size=8
)
print('Resume training completed successfully!')
"
else
    echo "No existing checkpoint found. Starting fresh training..."
    python -c "
import sys
sys.path.append('.')
sys.path.append('model')
from JEPA import run_jepa

# Run initial training for 2 epochs
state_dict = run_jepa(
    environments=['meeting_room', 'empty_room'],
    num_epochs=2,
    batch_size=8
)
print('Initial training completed successfully!')
"
fi

# Copy results back to project directory
if [ -d "saved_models" ]; then
    echo "Copying saved models back to project directory..."
    cp -r saved_models $PROJECT_DIR/experiment_results/jepa_models_${SLURM_JOB_ID}
fi

# Copy any checkpoint directories
if ls jepa_ssl_* 1> /dev/null 2>&1; then
    echo "Copying checkpoint directories back..."
    cp -r jepa_ssl_* $PROJECT_DIR/experiment_results/
fi

echo "JEPA prototype job finished at $(date)"
echo "Results saved to: $PROJECT_DIR/experiment_results/"