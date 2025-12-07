# HW2: Multi-Layer Perceptron with Flexible Architecture

Implementation of a flexible Multi-Layer Perceptron (MLP) with configurable architecture and Adam optimizer.

## Overview

This assignment extends HW1 by implementing a fully flexible MLP that supports arbitrary depth and width configurations. The network is applied to the same circular decision boundary problem from HW1, but with an element-wise implementation that allows for experimentation with different architectures.

**Problem:** Binary classification with circular boundary `x₁² + x₂² < 1`

## Implementation

### MLPAdam Class
A flexible, element-wise neural network implementation with:
- **Configurable architecture:** Specify any number of hidden layers and units
- **Adam optimizer:** Adaptive learning rates with momentum
- **Batch processing:** Efficient mini-batch training
- **Modern design:** Clean, extensible object-oriented structure

## Architecture

```
Input (2D) → Hidden Layer 1 (n₁ units) → ReLU → ... → Hidden Layer L → ReLU → Output (1) → Sigmoid
```

**Flexible configuration:**
```python
# Example: 2 → 64 → 32 → 16 → 1
model = MLPAdam(
    layer_sizes=[2, 64, 32, 16, 1],
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999
)
```

## Features

### Element-Wise Implementation
- Forward and backward passes operate on individual elements
- More interpretable than vectorized operations
- Easier to debug and understand

### Adam Optimizer
- First moment estimates (momentum)
- Second moment estimates (RMSprop)
- Bias correction for early iterations
- Adaptive per-parameter learning rates

### Training Capabilities
- Mini-batch gradient descent
- Automatic batch splitting
- Loss tracking per epoch
- Probability predictions for decision boundaries

## Files

- `ElementWiseNNAdam.py` - Flexible MLP implementation with Adam
- `hw2.py` - Demo script comparing different architectures
- `test_mlp_adam.py` - Unit tests for the MLP class
- `quick_test.py` - Quick validation script
- `hw2_problem2_results.png` - Architecture comparison results
- `hw2_problem2_loss_curves.png` - Training loss visualization
- `hw2_problem2_boundaries.png` - Decision boundary plots

## Requirements

```bash
pip install numpy matplotlib
```

## Usage

### Basic Training
```python
from ElementWiseNNAdam import MLPAdam

# Generate data
X_train, y_train = generate_data(10000)
X_val, y_val = generate_data(2000)

# Create and train model
model = MLPAdam(layer_sizes=[2, 64, 32, 1])
model.fit(X_train, y_train, epochs=1000, batch_size=128)

# Evaluate
y_pred = model.predict(X_val)
accuracy = (y_pred == y_val).mean()
```

### Run Demo
```bash
python hw2.py
```

This compares different architectures:
1. Shallow network: `[2, 32, 1]`
2. Medium network: `[2, 64, 32, 1]`
3. Deep network: `[2, 128, 64, 32, 16, 1]`

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `layer_sizes` | - | List of layer dimensions |
| `learning_rate` | 0.001 | Adam learning rate |
| `beta1` | 0.9 | First moment decay |
| `beta2` | 0.999 | Second moment decay |
| `epsilon` | 1e-8 | Numerical stability constant |
| `batch_size` | 128 | Mini-batch size |
| `epochs` | 1000 | Training iterations |

## Key Methods

### fit(X, y, epochs, batch_size)
Train the network on data.
```python
model.fit(X_train, y_train, epochs=1000, batch_size=128)
```

### predict(X)
Get binary predictions.
```python
y_pred = model.predict(X_test)  # Returns 0 or 1
```

### predict_proba(X)
Get probability predictions.
```python
probs = model.predict_proba(X_test)  # Returns values in [0, 1]
```

## Architecture Experiments

The demo script tests:

### 1. Shallow Network [2, 32, 1]
- **Pros:** Fast training, low overfitting risk
- **Cons:** Limited capacity for complex boundaries
- **Best for:** Simple decision boundaries

### 2. Medium Network [2, 64, 32, 1]
- **Pros:** Good balance of capacity and speed
- **Cons:** May underfit very complex patterns
- **Best for:** Most practical problems (recommended)

### 3. Deep Network [2, 128, 64, 32, 16, 1]
- **Pros:** High representational capacity
- **Cons:** Slower, higher overfitting risk
- **Best for:** Complex decision boundaries

## Implementation Details

### Forward Pass (per layer)
```
z^[l] = W^[l]·a^[l-1] + b^[l]
a^[l] = ReLU(z^[l])  (or sigmoid for output)
```

### Backward Pass
```
δ^[L] = a^[L] - y  (output layer)
δ^[l] = (W^[l+1])ᵀ·δ^[l+1] ⊙ ReLU'(z^[l])  (hidden layers)
```

### Adam Updates
```
m_t = β₁·m_{t-1} + (1-β₁)·∇θ
v_t = β₂·v_{t-1} + (1-β₂)·∇θ²
θ_t = θ_{t-1} - α·m̂_t/(√v̂_t + ε)
```

Where `m̂` and `v̂` are bias-corrected estimates.

## Visualizations

The demo generates three plots:

1. **Loss Curves:** Training loss over epochs for each architecture
2. **Decision Boundaries:** Learned boundaries for each model
3. **Combined Results:** Side-by-side comparison with metrics

## Learning Outcomes

- Flexible MLP architecture design
- Element-wise neural network operations
- Adam optimizer implementation
- Architecture selection trade-offs
- Mini-batch training
- Model comparison methodology

## Comparison with HW1

| Aspect | HW1 | HW2 |
|--------|-----|-----|
| Architecture | Fixed 2-layer | Flexible depth |
| Implementation | Vectorized | Element-wise |
| Experimentation | Limited | High flexibility |
| Code structure | Functional | Object-oriented |

## References

- **Adam Optimizer:** [Kingma & Ba, 2014](https://arxiv.org/abs/1412.6980)
- **Deep Learning:** [Goodfellow et al., 2016](https://www.deeplearningbook.org/)
