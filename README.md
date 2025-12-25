# U-Net Semantic Segmentation

PyTorch implementation of U-Net for semantic segmentation on the Oxford-IIIT Pet dataset.

## Overview

This project implements the U-Net architecture for pixel-wise segmentation of pet images into 3 classes:
- Background
- Foreground (pet)
- Border

## Installation

Requires Python 3.12+

```bash
uv sync
```

## Usage

### Training

```bash
# Train with default settings
uv run python run.py --train

# Train with custom hyperparameters
uv run python run.py --train --epochs 50 --lr 1e-4 --batch_size 16
```

### Testing

```bash
# Test model (loads checkpoints/best.pt)
uv run python run.py
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--train` | False | Enable training mode |
| `--epochs` | 100 | Number of training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--weight_decay` | 1e-5 | Weight decay |
| `--save_dir` | checkpoints | Directory for saving models |

## Configuration

Model and dataset settings are in `config.toml`:

```toml
input_size = 572
output_size = 388
in_channels = 3
out_channels = 3
num_classes = 3
```

## Project Structure

```
unet/
├── model.py          # U-Net architecture (Encoder, Latent, Decoder)
├── dataset.py        # Dataset loading and transforms
├── run.py            # Training and testing scripts
├── config.toml       # Configuration
└── notebooks/        # Jupyter notebooks for exploration
```

## Model Architecture

Standard U-Net with:
- 4-level encoder with max pooling
- Bottleneck (latent) layer
- 4-level decoder with transposed convolutions
- Skip connections via concatenation
- Input: 572x572 RGB → Output: 388x388 segmentation map
