#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=50:0:0
#SBATCH --gpus-per-node=a100:1  # or whatever GPU type is available
#SBATCH --mail-user=mdi.amirhossein@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=csi_job
#SBATCH --output=experiment_results/prev_model-%j.out

mkdir -p $PROJECT_DIR/experiment_results

export OUTFILE_NAME="experiment_results/prev_model_-${SLURM_JOB_ID}"

export OUTFILE_NAME="prev_model_-${SLURM_JOB_ID}"

# Define important directories
PROJECT_DIR=/home/amirmhd/projects/def-hinat/amirmhd/multi_modal_CSI
CODE_DIR=benchmark/wifi_csi
# DATA_DIR=dataset

# Create directory structure in SLURM_TMPDIR
echo "Copying code and data to temporary directory..."
mkdir -p $SLURM_TMPDIR/$CODE_DIR
# mkdir -p $SLURM_TMPDIR/$DATA_DIR

# Copy data to temporary directory for faster execution
# cp -r $PROJECT_DIR/$DATA_DIR/* $SLURM_TMPDIR/$DATA_DIR/
# echo "Data copied to temporary directory."
# Copy code to temporary directory for faster execution
# echo "Copying code to temporary directory..."
cp -r $PROJECT_DIR/$CODE_DIR/* $SLURM_TMPDIR/$CODE_DIR/
cd $SLURM_TMPDIR/$CODE_DIR

module purge
module load python/3.11.5 scipy-stack
source ~/py311/bin/activate

echo "Starting job at $(date)"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"


# Update your data paths in the Python script to use SLURM_TMPDIR
# export DATA_PATH=$SLURM_TMPDIR/$DATA_DIR
# export AUX_LOSS=0.5
# export NUM_QUERIES=5
# export ENVIRONMENTS_EXP=classroom
# export MODEL_TYPE=DETR
# export WANDB_MODE=offline
export ENVIRONMENTS_EXP=${ENVIRONMENTS_EXP:-classroom}
export MODEL_TYPE=${MODEL_TYPE:-DETR}
export WANDB_MODE=${WANDB_MODE:-offline}
# export NUM_EPOCHS=20

# export TOKEN_LENGTH=100
# export EMBEDDING_DIM=16
# export LABEL_SMOOTHING=0.2 
# export LEARNING_RATE=0.0007
# export WANDB_NAME_=T200_d16_SM02
# export MODEL_TYPE=THAT_COUNT 
# Now we change the preset accordingly
python config_modifier.py preset.py modified_preset.py
# just to verify:
cat modified_preset.py
rm preset.py
mv modified_preset.py preset.py
echo "starting the main script"
python run_main.py

echo "Copying results back to project directory..."
# Copy results directory if it exists
if [ -d "results" ]; then
    echo "Copying results directory..."
    cp -r results $PROJECT_DIR/$OUTFILE_NAME
fi

# Copy wandb offline runs back to project directory
if [ -d "wandb" ]; then
    echo "Copying wandb offline runs..."
    mkdir -p $PROJECT_DIR/wandb
    cp -r wandb/* $PROJECT_DIR/wandb/
    echo "Wandb files copied to $PROJECT_DIR/wandb/"
else
    echo "No wandb directory found to copy"
fi

echo "Job finished at $(date)"  
