# HW1: Neural Network From Scratch

Implementation of neural networks from scratch using NumPy, comparing different optimization algorithms for binary classification.

## Overview

This assignment implements two-layer neural networks to learn a circular decision boundary, classifying points in [0,1]² based on whether they fall inside or outside the unit circle: `x₁² + x₂² < 1`.

## Implementations

### 1. NeuralNetworkAdam
Full-batch Adam optimizer with detailed mathematical documentation.

**Features:**
- Full-batch training on entire dataset per iteration
- Adam optimizer (adaptive learning rates)
- Binary cross-entropy loss
- He initialization for ReLU networks

### 2. NeuralNetworkAdamFullyOptimized
Mini-batch Adam optimizer with advanced training features.

**Features:**
- Mini-batch training for scalability
- Early stopping to prevent overfitting
- Learning rate scheduling
- Gradient clipping for stability
- Faster convergence

### 3. NeuralNetworkSGDMomentum
SGD with momentum optimizer for comparison.

**Features:**
- Mini-batch stochastic gradient descent
- Momentum term for smoother convergence
- Traditional optimization approach

## Architecture

```
Input Layer (2D) → Hidden Layer (d units) → ReLU → Output Layer (1 unit) → Sigmoid
```

**Forward Pass:**
1. Hidden: `s = α₀ + αᵀx`
2. ReLU: `h = max(0, s)`
3. Output: `z = β₀ + βᵀh`
4. Sigmoid: `p = 1/(1 + e^(-z))`

**Loss Function:** Binary cross-entropy (negative log-likelihood)

## Files

- `hw1.py` - Main script comparing all three optimizers
- `NeuralNetworkAdam.py` - Full-batch Adam implementation
- `NeuralNetworkAdamFullyOptimized.py` - Mini-batch Adam with enhancements
- `NeuralNetworkSGDMomentum.py` - SGD with momentum implementation
- `test_hw1.py` - Unit tests
- `hw1_results.png` - Visualization of training curves and decision boundaries

## Requirements

```bash
pip install numpy matplotlib seaborn
```

## Usage

Run the comparison script:
```bash
python hw1.py
```

This will:
1. Generate training and validation data
2. Train all three models
3. Compare performance metrics
4. Generate visualization of decision boundaries and learning curves
5. Save results to `hw1_results.png`

## Key Hyperparameters

- **Hidden units (d):** 50-100 (adjustable)
- **Learning rate:** 0.001-0.01
- **Adam β₁:** 0.9 (momentum term)
- **Adam β₂:** 0.999 (RMSprop term)
- **Training epochs:** 1000-50000 (depends on variant)
- **Batch size:** 32-512 (mini-batch variants)

## Mathematical Details

### Adam Update Rule
```
m_t = β₁·m_{t-1} + (1-β₁)·g_t
v_t = β₂·v_{t-1} + (1-β₂)·g_t²
θ_t = θ_{t-1} - α·m_t / (√v_t + ε)
```

### Gradient Computation
The elegant property of cross-entropy + sigmoid:
```
∂J/∂z = p - y
```
This simplifies backpropagation significantly.

### He Initialization
```
W ~ N(0, √(2/n_in))
```
Recommended for ReLU activations to maintain gradient variance.

## Results

The script generates visualizations showing:
- **Training curves:** Loss over epochs for each optimizer
- **Decision boundaries:** Learned boundaries overlaid on data
- **Performance metrics:** Final accuracy and convergence speed

## Performance Comparison

| Optimizer | Convergence Speed | Generalization | Stability |
|-----------|------------------|----------------|-----------|
| Adam (full-batch) | Fast | Good | Very stable |
| Adam (mini-batch) | Fastest | Best | Stable with clipping |
| SGD + Momentum | Slower | Good | Less stable |

## Learning Outcomes

- Neural network implementation from scratch
- Understanding gradient descent variants
- Adam optimizer mechanics
- ReLU activation and initialization
- Binary classification with neural networks
- Overfitting prevention strategies

## References

- **Adam Optimizer:** [Kingma & Ba, 2014](https://arxiv.org/abs/1412.6980)
- **He Initialization:** [He et al., 2015](https://arxiv.org/abs/1502.01852)
- **Binary Cross-Entropy:** Standard loss for binary classification
