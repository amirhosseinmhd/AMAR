# AMAR: Efficient Attention-Based Multi-User Activity Recognition from Wi-Fi CSI

This repository contains the implementation of AMAR model submitted to IEEE TNNLS

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Running Experiments](#running-experiments)
- [Configuration](#configuration)
- [Available Models](#available-models)

## Project Structure

```
multi_modal_CSI/
├── configs/                    # Configuration files
│   ├── preset.py              # Main configuration file
│   ├── modified_preset.py     # Modified presets
│   └── config_modifier.py     # Config modification utilities
│
├── src/                        # Source code
│   ├── models/                # Model implementations
│   │   ├── baseline models/   # All models impelmented in the paper including AMAR
│   │   ├── losses/            # Hungarian Matching Loss implementation
│   │   └── modules/           # Reusable components in models
│   │       ├── elements.py    # Smaller modules (DSC, AC, MemoryPositionalEncoding)
│   │       ├── molecules.py   # Larger components (RVQ, Backbone, Encoder, ...)
│   │       └── helper.py      # helper functions (checkpoint management, ...)
│   │
│   ├── data/                  # Data handling
│   │   ├── load_data.py       # Data loading functions
│   │   └── preprocess.py      # Preprocessing utilities
│   │
│   ├── utils.py               # functions inside this utility are mainly for evaluation and vizualization
│   ├── train.py               # training function for benchmarks and also AMAR_WO_RVQ
│   ├── train_joint.py         # training function for MultiSenseX
│   └── train_rvq.py           # AMAR training function
│
├── scripts/                     # Executable scripts
│   ├── run_main.py              # Main experiment runner
│   └── run_main_jointActLoc.py  # For running MultiSenseX 
│
│
├── experiments/               # Experimental data
│   └── data_splits/          # Train/test split definitions
│
├── results/                   # Output artifacts
│   ├── checkpoints/          # Saved model weights
│   ├── metrics/              # Performance metrics (JSON)
│   └── figures/              # Visualization outputs
│
├── dataset/                   # Dataset directory (configure path in preset.py)
├── environment.yaml           # Conda environment specification
└── README.md                 # This file
```

## Installation

### 1. Create Conda Environment

```bash
conda env create -f environment.yaml
conda activate AMAR
```

### 2. Verify Installation

```bash
python -c "import torch; print(torch.__version__)"
python -c "import wandb; print('wandb installed')"
```

## Quick Start

### Basic Usage

Run a single experiment with default settings:

```bash
conda activate AMAR
python scripts/run_main.py
```

### With Custom Parameters

```bash
python scripts/run_main.py --model AMAR --task activity --repeat 3 --users "0,1,2,3,4,5"
```

### Available Arguments

- `--model`: Model name (default: from preset.py)
- `--repeat`: Number of experiment repetitions (default: from preset.py)
- `--users`: Comma-separated list of user IDs (default: "0,1,2,3,4,5")

## Running Experiments

### 1. Configure Experiment Settings

Edit `configs/preset.py` to set:

```python
preset = {
    "model": "AMAR",              # Choose your model
    "task": "activity",                # activity, identity, or location
    "repeat": 8,                       # Number of runs
    "path": {
        "data_x": "/path/to/wifi_csi/amp",        # CSI amplitude data
        "data_y": "/path/to/annotation.csv",       # Annotations
        "save": "results/result.json"              # Results output
    },
    "data": {
        "num_users": ["0","1","2","3","4","5"],   # User selection
        "wifi_band": ["5"],                         # WiFi band (2.4 or 5 GHz)
        "environment": ["empty_room"],              # Environment type
        "length": 3000,                             # CSI sequence length
    },
    "nn": {
        "lr": 5e-4,                    # Learning rate
        "epoch": 300,                    # Training epochs
        "batch_size": 16,              # Batch size
        # ... more hyperparameters
    }
}
```

### 2. Run Experiments

**Single model experiment:**
```bash
python scripts/run_main.py --model AMAR 
```





## Configuration

### Dataset Configuration

Update paths in `configs/preset.py`:

```python
"path": {
    "data_x": "/path/to/your/wifi_csi/amp",
    "data_y": "/path/to/your/annotation.csv",
    "save": "results/result.json"
}
```

### Environment Selection

Choose your experimental environment:

```python
"data": {
    "environment": ["empty_room"],      # Options: "empty_room", "classroom", "meeting_room"
    "wifi_band": ["5"],                 
    "num_users": ["0","1","2","3","4","5"]
}
```

### Model Hyperparameters

Adjust neural network settings in the `nn` section:

```python
"nn": {
    "lr": 5e-4,                        # Learning rate
    "epoch": 300,                        # Number of epochs
    "batch_size": 16,                  # Batch size
    "num_obj_queries": 6,              # Number of object queries (for AMAR models)
    "num_decoder_layers": 6,           # Decoder layers
    "d_embedding": 64,                 # Embedding dimension
    "n_layers_encoder": 4,             # Encoder layers
    "n_attention_heads": 4,            # Attention heads
    # ... more parameters
}
```

## Available Models

### Baseline Models
- `ABLSTM`: Attention-based LSTM
- `DEM_ABLSTM`: ABLSTM for DEM
- `THAT`: Transformer-based model
- `DEM_THAT`: THAT DEM
- `AMAR`: AMAR with Residual Vector Quantization

## Project Organization

### Source Code (`src/`)
- **models/**: All model implementations with their forward passes and training functions
- **data/**: Data loading, preprocessing, and dataset creation utilities
- **utils.py**: Helper functions for metrics, visualization, and data manipulation
- **train*.py**: Training loop implementations for different model types

### Scripts (`scripts/`)
- Executable entry points for running experiments
- Import from `src/` modules
- Can be run from project root

### Configuration (`configs/`)
- All experimental settings and hyperparameters
- Modify `preset.py` for different experiments
- No hard-coded paths in source code

### Results (`results/`)
- **checkpoints/**: Trained model weights (.pth files)
- **metrics/**: Performance metrics in JSON format
- **figures/**: Generated visualizations and plots

## Monitoring with Weights & Biases

The project uses Weights & Biases (wandb) for experiment tracking:

1. Login to wandb (first time only):
```bash
wandb login
```

2. Experiments are automatically logged during training
3. View results at: https://wandb.ai/your-username/your-project

## Output

After running an experiment, you'll see:
- Model architecture summary
- Training progress with loss/metrics
- Final results (Precision, Recall, F1-Score, Accuracy)
- Results saved to `results/metrics/result.json`
- Model checkpoints in `results/checkpoints/` (if `save_model: True`)

