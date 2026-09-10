# Installation

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (recommended for training)

## Setup

```bash
# Clone the repository
git clone https://github.com/shrutij01/ssae.git
cd ssae

# Create environment
conda create -n ssae python=3.10
conda activate ssae

# Install the package and optional identifiability metrics
pip install -e ".[metrics]"
```

## Verify Installation

```python
import torch
import cooper
from ssae.model import DictLinearAE

model = DictLinearAE(rep_dim=512, hid=64, norm_type="ln")
x = torch.randn(4, 512)
x_hat, h = model(x)
print(f"Input: {x.shape}, Reconstruction: {x_hat.shape}, Codes: {h.shape}")
```

## Supported Models

The framework has been tested with the following language models:

| Model | Hidden Dim | Recommended Layer |
|-------|------------|-------------------|
| EleutherAI/pythia-70m | 512 | 5 |
| EleutherAI/pythia-160m | 768 | 8 |
| EleutherAI/pythia-410m | 1024 | 12 |
| google/gemma-2-2b-it | 2304 | 12-18 |
| meta-llama/Llama-3.1-8b | 4096 | 16-24 |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SSAE_DATA_ROOT` | required for data-backed runs | Base directory for embedding `.h5` files, checkpoints, and YAML configs |
| `SSAE_REPO_ROOT` | repository checkout | Used by configurable cluster launchers |
| `SSAE_VENV` | required for cluster jobs | Path to the environment activation script, for example `/path/to/venv/bin/activate` |
| `WANDB_ENTITY` | — | Weights & Biases entity for experiment tracking |
| `WANDB_PROJECT` | — | Weights & Biases project name |
