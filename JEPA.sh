#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=08:30:0
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
source ~/py311_g/bin/activate

echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"

# Define checkpoint directory for reference
CHECKPOINT_DIR="$PROJECT_DIR/experiment_results/jepa_checkpoints"


export DECAY_TEACHER=0.9997 
# Now we change the preset accordingly
python config_modifier.py preset.py modified_preset.py
# just to verify:
cat modified_preset.py
rm preset.py
mv modified_preset.py preset.py
echo "starting the main script"

# OPTION 1: Fresh training (default)
# Run with 2 environments, for 2 epochs, with batch size 8
python -m model.JEPA --envs meeting_room empty_room classroom --epochs 400 --batch-size 64
# OPTION 2: Resume training (commented by default)
# Uncomment the following lines and comment the above python command to resume training
# LATEST_CHECKPOINT=$(find $CHECKPOINT_DIR -name "best_model.pth" -type f | head -1)
# if [ -n "$LATEST_CHECKPOINT" ] && [ -f "$LATEST_CHECKPOINT" ]; then
#     echo "Found existing checkpoint: $LATEST_CHECKPOINT"
#     echo "Resuming training from checkpoint..."
#     python -m model.JEPA --resume $LATEST_CHECKPOINT --envs meeting_room empty_room --epochs 4 --batch-size 8
# else
#     echo "No checkpoint found. Starting fresh training..."
#     python -m model.JEPA --envs meeting_room empty_room --epochs 2 --batch-size 8
# fi

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
