# AMAR: Efficient Attention-Based Multi-User Activity Recognition from Wi-Fi CSI

## Introduction

AMAR is a transformer-based framework for recognizing multiple concurrent human activities from Wi-Fi Channel State Information (CSI). It formulates multi-user HAR as a set prediction problem, eliminating the need for prior occupancy knowledge or auxiliary user/location annotations.

**Key Features:**
- **Set prediction** with learnable query embeddings for concurrent activity detection
- **Edge-cloud architecture**: Lightweight backbone (0.11M params) + RVQ (99.2% bandwidth reduction)
- **State-of-the-art performance**: 53.4% F₁-score, 1.72× improvement in perfect prediction rate, 74% lower occupancy error

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
│   └── preset.py              # Main configuration file
│
├── src/                        # Source code
│   ├── models/                # Model implementations
│   │   ├── *.py               # Implementation of models including AMAR, AMAR_WO_RVQ, DEM and 
│   │   │                      # BCE based methods 
│   │   ├── losses/            # Hungarian Matching Loss implementation
│   │   └── modules/           # Reusable components in models
│   │       ├── elements.py    # Smaller modules (DSC, AC, MemoryPositionalEncoding)
│   │       ├── molecules.py   # Larger components (RVQ, Backbone, Encoder, ...)
│   │       └── helper.py      # helper functions (checkpoint management, ...)
│   │
│   ├── data/                  # Data handling specific to WiMANS dataset
│   │   ├── load_data.py       # Data loading functions
│   │   └── preprocess.py      # Preprocessing utilities
│   │
│   ├── utils.py               # functions inside this utility are mainly for evaluation and viz
│   ├── train.py               # training function for benchmarks and also AMAR_WO_RVQ
│   ├── train_joint.py         # training function for MultiSenseX
│   └── train_rvq.py           # AMAR training function
│
├── scripts/                     # Executable scripts
│   ├── run_main.py              # Main experiment runner
│   └── run_main_jointActLoc.py  # For running MultiSenseX 
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

 Create Conda Environment

```bash
conda env create -f environment.yaml
conda activate AMAR
```
## Dataset Setup

Download the WiMANS dataset from [Kaggle](https://www.kaggle.com/datasets/shuokanghuang/wimans) and extract it to the project root. Update the paths in `configs/preset.py`:
```python
"path": {
    "data_x": "dataset/wifi_csi/amp",
    "data_y": "dataset/annotation.csv",
    "save": "results/result.json"
}
```

**Sample Data:**

<table align="center">
  <tr align="center">
    <td rowspan="2"><b>Sample includes these activities:<br>Walking, Sitting Down, and Picking Up</b></td>
    <td>WiFi CSI</td>
    <td>Synchronized Video</td>
  </tr>
  <tr align="center">
    <td><img src="visualizations/wifi_csi_act_30_25.gif" height="188"/></td>
    <td><img src="visualizations/video_act_30_25.gif" height="188"></td>
  </tr>
</table>

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

## Available Models

| Model | Type | Description | Reference |
|-------|------|-------------|-----------|
| `AMAR` | Proposed | Attention-based multi-user HAR with RVQ | This work |
| `AMAR_WO_RVQ` | Proposed | AMAR without quantization | This work |
| `BCE_ABLSTM` | Baseline | Attention-based Bi-directional LSTM + BCE loss | [Chen et al., TMC'19](https://doi.org/10.1109/TMC.2018.2878233) |
| `BCE_THAT` | Baseline | Two-stream Transformer + BCE loss | [Li et al., AAAI'21](https://ojs.aaai.org/index.php/AAAI/article/view/16103) |
| `DEM_ABLSTM` | Baseline | Attention-based Bi-directional LSTM + Smooth L1 | [Chen et al., TMC'19](https://doi.org/10.1109/TMC.2018.2878233) |
| `DEM_THAT` | Baseline | Two-stream Transformer + Smooth L1 | [Li et al., AAAI'21](https://ojs.aaai.org/index.php/AAAI/article/view/16103) |

**Note:** BCE variants follow the WiMANS dataset formulation [[Huang et al., arXiv'24]](https://arxiv.org/abs/2402.09430) for multi-label classification. DEM variants use regression to predict activity counts directly.




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


## Project Organization

### Source Code (`src/`)
- **models/**: All model implementations with their forward passes and training functions
- **data/**: Data loading, preprocessing, and dataset creation utilities
- **utils.py**: Helper functions for metrics, visualization, and data manipulation
- **train\*.py**: Training loop implementations for different model types

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

