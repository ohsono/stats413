# HW7: Toy Diffusion Model - Swiss Roll

A minimal implementation of a diffusion model for generating 2D data following a Swiss roll distribution.

## Overview

This project demonstrates the core concepts of **Denoising Diffusion Probabilistic Models (DDPM)** using a simple 2D dataset. It's an educational implementation that makes the mathematics and mechanics of diffusion models easy to understand before tackling more complex applications like image generation.

## Requirements

```bash
pip install torch matplotlib seaborn scikit-learn
```

**Note**: The code assumes CUDA is available. To run on CPU, modify line 121:
```python
device = "cpu"  # or use: device = "cuda" if torch.cuda.is_available() else "cpu"
```

## What's Implemented

### 1. Data Generation
- Generates 100,000 samples from a Swiss roll distribution
- Reduces to 2D for visualization
- Normalizes to zero mean and unit standard deviation

### 2. Diffusion Process
- **40 diffusion steps** for the forward noising process
- **Cosine noise schedule** (Nichol & Dariwal, 2021)
- Forward diffusion: `x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε`

### 3. Neural Network
- Simple feedforward network with 4 residual-like blocks
- **Input**: Noised data (2D) + timestep (1D)
- **Output**: Predicted noise (2D)
- **Architecture**: Input → 64 units → 4 blocks → Output

### 4. Training
- **100 epochs**, batch size 2048
- **Loss**: MSE between predicted and actual noise
- **Optimizer**: Adam with learning rate decay (1.0 → 0.01)

### 5. Sampling Algorithms

#### DDPM Sampler (Classic)
The original algorithm from Ho et al. (2020):
- Starts from pure Gaussian noise
- Iteratively denoises over 40 steps
- Adds controlled noise at each step

#### DDPM-x0 Sampler (Alternative)
Used in HuggingFace Diffusers:
- First predicts the original data `x_0`
- Then uses `x_0` to predict `x_{t-1}`
- More flexible for different prediction targets

## Usage

Run the entire script:
```bash
python toy_diffuser_swiss_roll.py
```

This will:
1. Generate and visualize the Swiss roll dataset
2. Train the diffusion model
3. Generate 10,000 new samples
4. Create an animation (`swissroll_generation.mp4`) showing the denoising process

## Key Visualizations

- **Original vs Noised Data**: Shows how data becomes pure noise
- **Original vs Generated Data**: Validates the model learned the distribution
- **Denoising Animation**: 40 frames showing the reverse diffusion process

## Mathematical Foundation

### Forward Diffusion (Training)
```
q(x_t | x_0) = N(x_t; √ᾱ_t * x_0, (1-ᾱ_t) * I)
```

### Reverse Diffusion (Sampling)
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

Where the mean is computed as:
```
μ_θ(x_t, t) = 1/√α_t * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ(x_t, t))
```

## References

- **DDPM Paper**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - Ho et al., 2020
- **Improved DDPM**: [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) - Nichol & Dhariwal, 2021

## Code Structure

```
toy_diffuser_swiss_roll.py
├── Data Loading (lines 19-28)
├── Diffusion Hyperparameters (lines 37-52)
├── Noising Function (lines 56-59)
├── Neural Network Architecture (lines 90-116)
│   ├── DiffusionBlock
│   └── DiffusionModel
├── Training Loop (lines 130-150)
├── DDPM Sampler (lines 157-173)
├── DDPM-x0 Sampler (lines 213-236)
└── Visualization & Animation (lines 180-209)
```

## Known Issues

1. **CUDA requirement**: Hardcoded to use CUDA (line 121)
2. **No model saving**: Trained model is not persisted
3. **Missing scheduler step**: Learning rate scheduler created but not used
4. **No validation set**: All data used for training

## Learning Outcomes

This implementation teaches:
- How diffusion models gradually add and remove noise
- The role of timestep conditioning in neural networks
- The mathematics behind DDPM sampling
- The difference between noise prediction and data prediction
- How to visualize the denoising process

## Next Steps

To extend this project:
- Add model checkpointing
- Implement DDIM sampler (faster sampling)
- Try different noise schedules (linear, quadratic)
- Scale to higher-dimensional data or images
- Add classifier-free guidance for conditional generation
