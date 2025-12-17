# Chapter 1: Fundamentals of Machine Learning - Complete Summary

## Core Learning Concepts

### 1. **Simple Linear Model (Foundation)**
The simplest learner model is a linear predictor:
$$s_i = x_i\beta$$

Where:
- $s_i$ = predicted score for sample $i$
- $x_i$ = input feature(s)
- $\beta$ = learnable parameter(s)

### 2. **Error and Residuals**
The error (residual) is the difference between actual and predicted values:
$$e_i = y_i - s_i$$

Where:
- $y_i$ = true label
- $s_i$ = predicted value
- $e_i$ = prediction error

---

## Loss Functions and Optimization

### 3. **Loss Function: Mean Squared Error (MSE)**
MSE penalizes prediction errors quadratically:
$$L(\beta) = \frac{1}{2}\sum_{i=1}^n e_i^2 = \frac{1}{2}\sum_{i=1}^n(y_i - x_i\beta)^2$$

**Alternative Loss Functions:**
- **MSE (Mean Squared Error)**: For regression tasks
- **Cross-Entropy Loss**: For classification tasks (with sigmoid or softmax outputs)

### 4. **Finding Minimum Loss**
Two approaches to minimize loss:

**A. Closed-Form Solution (Analytical)**
- Directly solve for optimal parameters
- Works well when solution is computable

**B. Gradient Descent (Iterative)**
- Iteratively update parameters by moving opposite to gradient
- Update rule: $\beta_{t+1} = \beta_t - \eta \nabla L(\beta_t)$
- $\eta$ = learning rate
- More flexible, works for complex models

### 5. **Maximum Likelihood Estimation (MLE)**
- Probabilistic framework for parameter estimation
- Choose parameters that maximize likelihood of observed data
- Connection: MLE with MSE loss = Gaussian distribution assumption
- Related to cross-entropy loss for classification

---

## Key Activation Functions

### 6. **Sigmoid Activation**
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

- Maps output to (0, 1) range
- Useful for binary classification
- Provides probabilistic interpretation

### 7. **ReLU (Rectified Linear Unit)**
$$\text{ReLU}(z) = \max(0, z)$$

- Piecewise linear function
- Modern standard in deep learning
- Introduces non-linearity without saturation issues

---

## Vector Representation and Geometry

### 8. **Vector Notation**
- Aggregate all samples: $X$ (design matrix), $\boldsymbol{\beta}$ (parameter vector), $\mathbf{y}$ (label vector)
- Vectorized prediction: $\mathbf{s} = X\hat{\boldsymbol{\beta}}$

### 9. **Geometric Interpretation**
- View $X\hat{\boldsymbol{\beta}}$ as projection of $\mathbf{y}$ onto the column space of $X$
- Linear regression finds the "closest" point in the subspace spanned by features
- Residuals are orthogonal to feature space (in optimal solution)

---

## Statistical Concepts

### 10. **Regression to the Mean**
Mean and variance of data:
$$\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$$
$$\text{Var}(y) = \frac{1}{n}\sum_{i=1}^n (y_i - \bar{y})^2$$

- Regression coefficients relate how features predict labels
- Extreme values tend to move toward average in predictions

---

## Model Complexity and Generalization

### 11. **Overfitting**
- Model memorizes training data instead of learning generalizable patterns
- High training loss but poor test performance
- Results from model being too complex for the amount of data

### 12. **Piecewise Linear Regression**
- Combine multiple linear segments (using ReLU activations)
- Increases model flexibility and expressiveness
- Example of increasing model capacity

### 13. **Implicit Regularization**
- Training procedure (e.g., early stopping, SGD) naturally prevents overfitting
- Model implicitly learns to generalize without explicit penalty terms
- Important for understanding why neural networks generalize despite overparameterization

### 14. **Over-parameterization**
- Model has more parameters than training samples
- Appears to be doomed to overfit
- Yet empirically generalizes well in deep learning

---

## Neural Network Architecture

### 15. **Neural Network Layers**
Standard architecture:
```
Input Layer (x_i)
    ↓
Hidden Layers (h = ReLU(Xw + b))
    ↓
Output Layer (y = h·w_out + b_out)
```

- **Input Layer**: Raw features
- **Hidden Layers**: ReLU non-linearities (piecewise linear)
- **Output Layer**: Final prediction (linear or sigmoid for probability)
- Each connection weighted by learnable parameters

---

## Generalization and the Double Descent Phenomenon

### 16. **Memorization vs. Generalization Balance**
- **Memorization**: Exact fitting of training data
- **Generalization**: Learning patterns that work on unseen data
- Goal: Find the sweet spot between both

### 17. **Training vs. Test Error**
- **Training Error**: Performance on data used to fit model
- **Test Error**: Performance on held-out, unseen data
- Overfitting: Training error ↓, Test error ↑

### 18. **Classical Regime (Underfitting)**
- Model capacity < data complexity
- Both training and test error are high
- Not enough model expressiveness

### 19. **Bias-Variance Tradeoff Region**
- Model capacity ≈ data complexity
- Risk of overfitting increases
- Test error can increase even as training error decreases

### 20. **Double Descent Regime**
- Model capacity >> data complexity (heavily over-parameterized)
- **Surprising finding**: Test error decreases again despite massive overfitting in training
- Empirically observed in deep neural networks
- Indicates generalization can improve in the over-parameterized regime

---

## Classification and Logistic Regression

### 21. **Logistic Regression**
For binary classification with probability output:
$$P(y=1|x) = \sigma(x\boldsymbol{\beta}) = \frac{1}{1 + e^{-x\boldsymbol{\beta}}}$$

Where $\sigma$ is the sigmoid function.

### 22. **Likelihood and Log-Likelihood**
- **Likelihood**: Probability of observed data given parameters
- **Log-Likelihood**: $\ell(\boldsymbol{\beta}) = \sum_i [y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)]$
- Maximizing log-likelihood = minimizing cross-entropy loss

### 23. **Gradients for Classification**
- Gradient of log-likelihood with respect to parameters
- Used in gradient descent optimization
- Automatically computed in modern frameworks

---

## Optimization with Momentum

### 24. **Gradient Descent with Momentum**
Standard SGD update rule:
$$\boldsymbol{\beta}_{t+1} = \boldsymbol{\beta}_t - \eta \nabla L(\boldsymbol{\beta}_t)$$

**With Momentum:**
$$v_{t+1} = \gamma v_t - \eta \nabla L(\boldsymbol{\beta}_t)$$
$$\boldsymbol{\beta}_{t+1} = \boldsymbol{\beta}_t + v_{t+1}$$

Where:
- $\gamma$ = momentum coefficient (typically 0.9)
- $v_t$ = velocity/accumulated gradient
- Accelerates convergence in consistent directions
- Dampens oscillations in noisy gradient estimates

---

## Deep Learning Overview

### 25. **Deep Learning**
- Stacking multiple layers with non-linear activations
- Learn hierarchical representations
- Each layer transforms input into more useful feature space
- Trade-off: Increased complexity but better expressiveness

**Key Properties:**
- Non-linearity from activation functions (ReLU)
- Composable representations
- Parameter sharing (convolutions)
- Empirically generalizes well despite overparameterization

---

## Summary: The Learning Process

1. **Setup**: Define model (e.g., $s = x\boldsymbol{\beta}$), choose loss function (MSE or cross-entropy)
2. **Optimization**: Use gradient descent to minimize loss
3. **Regularization**: Implicit through training dynamics or explicit penalties
4. **Evaluation**: Monitor training vs. test error to detect overfitting
5. **Inference**: Make predictions on new data using learned parameters

---

## Key Takeaways

✓ Linear models are simple but limited in expressiveness
✓ Loss functions quantify prediction quality
✓ Gradient descent is the workhorse of modern ML optimization
✓ Neural networks combine simple linear operations with non-linearities
✓ Generalization (not memorization) is the real goal
✓ Over-parameterization doesn't necessarily cause overfitting (double descent)
✓ Activation functions (sigmoid, ReLU) enable learning complex patterns
✓ Momentum accelerates optimization convergence
