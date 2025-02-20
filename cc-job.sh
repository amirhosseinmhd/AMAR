#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=30:30:0
#SBATCH --mail-user=mdi.amirhossein@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --gpus-per-node=1
#SBATCH --job-name=csi_job
#SBATCH --output=experiment_results/SLURM_LOG-%j.out

mkdir -p $PROJECT_DIR/experiment_results

export OUTFILE_NAME="experiment_results/timeStream_decoder_-${SLURM_JOB_ID}"

export OUTFILE_NAME="timeStream_decoder_-${SLURM_JOB_ID}"

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
# export NUM_DECODER_LAYERS=6

export TOKEN_LENGTH=100
export EMBEDDING_DIM=10
export NUM_ATTENTION_HEAD=4

# export MODEL_TYPE=THAT_COUNT
# Now we change the preset accordingly
python config_modifier.py preset.py modified_preset.py
# just to verify:
cat modified_preset.py
rm preset.py
mv modified_preset.py preset.py
echo "starting the main script"
python run_main.py


if [ -d "results" ]; then
    echo "Copying results back to project directory..."
    cp -r results $PROJECT_DIR/$OUTFILE_NAME
fi

echo "Job finished at $(date)"