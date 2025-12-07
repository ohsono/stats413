# HW3: CNN Training on CIFAR-10

Comprehensive framework for training and experimenting with Convolutional Neural Networks (CNNs) on the CIFAR-10 dataset using PyTorch.

## Overview

This assignment implements a structured framework for training CNNs with extensive experimentation on:
- Network architectures (depth, width, residual connections)
- Hyperparameters (learning rate, batch size, optimizers)
- Activation functions (ReLU, LeakyReLU, Tanh)
- Performance monitoring with Weights & Biases

**Dataset:** CIFAR-10 (60,000 32×32 color images in 10 classes)

## Features

### CNN Training Framework
- Modular `TrainingConfig` for hyperparameters
- `PerformanceMetrics` class for tracking and logging
- Integrated Weights & Biases (wandb) support
- Automatic checkpointing of best models

### Network Architectures
Multiple CNN variants implemented and tested:

1. **OriginalNet** - Baseline 2-layer CNN
2. **DoubleNet** - Deeper architecture with more filters
3. **ResNet-style** - Custom residual connections
4. Various activation function variants

## Files

### Core Implementation
- `cnn_trainer.py` - Training framework with performance monitoring
- `example_usage.py` - Usage examples for the framework
- `test_cnn_trainer.py` - Unit tests

### Notebooks
- `cifar10_tutorial.ipynb` - Tutorial and experiments
- `STATS_413_HW3_CNN_HochanSon.ipynb` - Complete homework solution

### Documentation
- `QUICK_START.md` - Quick start guide
- `HYPERPARAMETER_GUIDE.md` - Hyperparameter tuning guide
- `MODEL_COMPARISON_GUIDE.md` - Model architecture comparison
- `EXPERIMENT_SUMMARY.md` - Summary of all experiments
- `REFACTORING_SUMMARY.md` - Code refactoring details

### Model Checkpoints
All trained models saved as `.pth` files:
- `cifar_OriginalNet.pth`
- `cifar_DoubleNet_1.pth`, `cifar_DoubleNet_2.pth`
- `cifar_ResNet_Custom.pth`
- Variant models (different activations, hyperparameters)

## Requirements

```bash
pip install torch torchvision wandb
```

Optional for visualization:
```bash
pip install matplotlib numpy
```

## Quick Start

### Basic Usage
```python
from cnn_trainer import TrainingConfig, train_model
import torch.nn as nn

# Define your model
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = self.fc1(x)
        return x

# Configure training
config = TrainingConfig(
    learning_rate=0.001,
    epochs=20,
    batch_size=128
)

# Train
model = MyNet()
trained_model = train_model(model, config, "MyNet")
```

### Run Tutorial
```bash
jupyter notebook cifar10_tutorial.ipynb
```

## Experiments Conducted

### 1. Architecture Variations

| Model | Depth | Filters | Params | Test Acc |
|-------|-------|---------|--------|----------|
| OriginalNet | 2 conv | 6→16 | ~62K | ~55% |
| DoubleNet_1 | 4 conv | 32→64→128 | ~384K | ~70% |
| DoubleNet_2 | 4 conv | 64→128→256 | ~1.5M | ~73% |
| ResNet_Custom | 6 conv | 64→128→256 | ~612K | ~75% |

### 2. Hyperparameter Tuning

**Learning Rate:**
- 0.0001: Slow convergence, underfitting
- 0.001: Optimal balance (recommended)
- 0.01: Fast but unstable, overfitting

**Batch Size:**
- 4: Noisy gradients, slow training
- 32: Good balance (recommended)
- 128: Faster training, may underfit

**Optimizer:**
- SGD + Momentum: Stable, slower convergence
- Adam: Faster convergence, better final accuracy

### 3. Activation Functions

| Activation | Characteristics | Performance |
|------------|----------------|-------------|
| ReLU | Standard, fast | Baseline |
| LeakyReLU | Prevents dying neurons | +2% accuracy |
| Tanh | Centered outputs | -5% accuracy |

## Network Architectures

### OriginalNet (Baseline)
```
Conv(3→6) → ReLU → Pool → Conv(6→16) → ReLU → Pool → FC(400→120) → FC(120→84) → FC(84→10)
```

### DoubleNet_2 (Best Performance)
```
Conv(3→64) → ReLU → Pool →
Conv(64→128) → ReLU → Pool →
Conv(128→256) → ReLU → Pool →
Conv(256→256) → ReLU → Pool →
FC(4096→512) → FC(512→10)
```

### ResNet_Custom
```
Input → Conv → BN → ReLU →
[ResBlock × 3] →
[ResBlock × 3] →
[ResBlock × 3] →
Global Avg Pool → FC → Output
```

## Training Configuration

### Default Settings
```python
TrainingConfig(
    learning_rate=0.001,
    momentum=0.9,
    epochs=20,
    batch_size=4,
    num_workers=2,
    device='cuda:0' if available else 'cpu'
)
```

### Advanced Features
- **Early stopping:** Monitor validation loss
- **Learning rate scheduling:** Reduce on plateau
- **Data augmentation:** Random crops, flips
- **Gradient clipping:** Prevent exploding gradients

## Performance Monitoring

### Metrics Tracked
- Training loss per epoch
- Test accuracy per epoch
- Per-class accuracy
- Training time
- Epoch duration

### Weights & Biases Integration
```python
import wandb

wandb.init(project="cifar10-experiments")
# Training automatically logs metrics
```

## Data Preprocessing

### Training Transforms
```python
transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

### Test Transforms
```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

## Results Summary

**Best Model:** ResNet_Custom with LeakyReLU
- **Test Accuracy:** 75.3%
- **Training Time:** ~45 minutes (GPU)
- **Parameters:** 612K

**Key Findings:**
1. Deeper networks with residual connections outperform shallow ones
2. Learning rate 0.001 is optimal for Adam optimizer
3. Batch size 32-128 provides best balance
4. LeakyReLU slightly outperforms ReLU
5. Data augmentation improves generalization by ~5%

## File Structure

```
HW3/
├── cnn_trainer.py              # Main training framework
├── example_usage.py            # Usage examples
├── test_cnn_trainer.py         # Unit tests
├── cifar10_tutorial.ipynb      # Interactive tutorial
├── *.pth                       # Trained model checkpoints
├── wandb/                      # W&B logs
└── data/                       # CIFAR-10 dataset (auto-downloaded)
```

## Learning Outcomes

- CNN architecture design principles
- Hyperparameter tuning methodology
- PyTorch training pipelines
- Performance monitoring and logging
- Residual connections and skip connections
- Data augmentation strategies
- Model comparison and evaluation

## Future Improvements

- [ ] Implement batch normalization systematically
- [ ] Add dropout for regularization
- [ ] Experiment with learning rate warmup
- [ ] Try mixed precision training
- [ ] Implement advanced architectures (EfficientNet, Vision Transformer)
- [ ] Add ensemble methods

## References

- **CIFAR-10:** [Krizhevsky, 2009](https://www.cs.toronto.edu/~kriz/cifar.html)
- **ResNet:** [He et al., 2015](https://arxiv.org/abs/1512.03385)
- **PyTorch Tutorial:** [pytorch.org/tutorials](https://pytorch.org/tutorials/)
- **Batch Normalization:** [Ioffe & Szegedy, 2015](https://arxiv.org/abs/1502.03167)
