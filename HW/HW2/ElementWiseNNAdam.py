import numpy as np
from typing import Dict, List

class MLPAdam:
    """
    Multi-layer perceptron (MLP) with flexible architecture and Adam optimizer

    Architecture:
        Input → Hidden Layer 1 → ReLU → ... → Hidden Layer L-1 → ReLU → Output → Sigmoid

    Allows flexible number of layers and nodes per layer.
    """

    def __init__(self, layer_dims: List[int],
                 learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        """
        Initialize network parameters and Adam optimizer state

        Args:
            layer_dims: List of layer dimensions [input_dim, hidden1, hidden2, ..., output_dim]
                       Example: [2, 64, 32, 1] creates 2→64→32→1 network
            learning_rate: Adam learning rate (η)
            beta1: Exponential decay rate for first moment (β₁)
            beta2: Exponential decay rate for second moment (β₂)
            epsilon: Small constant for numerical stability (ε)
        """
        if len(layer_dims) < 2:
            raise ValueError("layer_dims must have at least 2 elements (input and output)")

        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims) - 1  # L (number of weight matrices)
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # ============================================
        # Initialize Network Parameters
        # ============================================
        # W[l] is the weight matrix for layer l (l = 1, ..., L)
        # b[l] is the bias vector for layer l
        self.W = {}  # Weight matrices
        self.b = {}  # Bias vectors

        for l in range(1, self.num_layers + 1):
            # He initialization for ReLU (Xavier for last layer)
            if l < self.num_layers:  # Hidden layers
                self.W[l] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2.0 / layer_dims[l-1])
            else:  # Output layer
                self.W[l] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(1.0 / layer_dims[l-1])

            self.b[l] = np.zeros(layer_dims[l])

        # ============================================
        # Initialize Adam Optimizer State
        # ============================================
        # First moment (mean) - m
        self.m_W = {l: np.zeros_like(self.W[l]) for l in range(1, self.num_layers + 1)}
        self.m_b = {l: np.zeros_like(self.b[l]) for l in range(1, self.num_layers + 1)}

        # Second moment (uncentered variance) - v
        self.v_W = {l: np.zeros_like(self.W[l]) for l in range(1, self.num_layers + 1)}
        self.v_b = {l: np.zeros_like(self.b[l]) for l in range(1, self.num_layers + 1)}

        # Time step for bias correction
        self.t = 0

        # Cache for backpropagation
        self.cache = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the L-layer network

        Mathematical operations for each layer l = 1, ..., L:
            s^(l) = W^(l) @ h^(l-1) + b^(l)    [pre-activation]
            h^(l) = ReLU(s^(l))                 [activation, except last layer]

        For the last layer L:
            s^(L) = W^(L) @ h^(L-1) + b^(L)
            h^(L) = sigmoid(s^(L))              [output probability]

        Args:
            X: Input data of shape (n, p) where n is batch size, p is input dimension

        Returns:
            prob: Probabilities of shape (n,) where prob[i] = P(y=1 | x_i)
        """
        n = X.shape[0]

        # Initialize cache to store activations for backprop
        s = {}  # Pre-activations
        h = {0: X.T}  # Activations (h[0] = input, shape (p, n))

        # Forward pass through all layers
        for l in range(1, self.num_layers + 1):
            # Pre-activation: s^(l) = W^(l) @ h^(l-1) + b^(l)
            s[l] = self.W[l] @ h[l-1] + self.b[l][:, np.newaxis]  # (d_l, n)

            # Activation
            if l < self.num_layers:
                # Hidden layers: ReLU activation
                h[l] = np.maximum(0, s[l])  # (d_l, n)
            else:
                # Output layer: Sigmoid activation
                s[l] = np.clip(s[l], -500, 500)  # Numerical stability
                h[l] = 1.0 / (1.0 + np.exp(-s[l]))  # (output_dim, n)

        # Get output probabilities (flatten if output_dim = 1)
        prob = h[self.num_layers].flatten()  # (n,)

        # Cache intermediate values for backpropagation
        self.cache = {
            'X': X,
            's': s,  # All pre-activations
            'h': h,  # All activations (including input)
            'n': n
        }

        return prob

    def loss_fn(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Binary cross-entropy loss (negative log-likelihood)

        Mathematical formula:
            J = -(1/n) Σᵢ [yᵢ·log(pᵢ) + (1-yᵢ)·log(1-pᵢ)]

        Interpretation:
            - Penalizes confident wrong predictions heavily
            - Rewards confident correct predictions
            - Maximum likelihood estimation for Bernoulli distribution

        Args:
            y_true: True labels of shape (n,)
            y_pred: Predicted probabilities of shape (n,)

        Returns:
            loss: Scalar binary cross-entropy loss
        """
        eps = 1e-15  # Small constant to prevent log(0)

        # Clip probabilities to avoid numerical issues
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)

        # Binary cross-entropy
        loss = -np.mean(
            y_true * np.log(y_pred_clipped) +
            (1 - y_true) * np.log(1 - y_pred_clipped)
        )

        return loss

    def backward(self, y_true: np.ndarray) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Backpropagation: Compute gradients using chain rule for L-layer network

        Mathematical derivation:

        Step 1: Initialize top layer error
            δ^(L) = ∂J/∂s^(L) = y - σ(s^(L)) = y - p

        Step 2: For each layer l = L, L-1, ..., 1:
            a) Compute parameter gradients:
               ∂J/∂W^(l) = δ^(l) @ (h^(l-1))^T / n
               ∂J/∂b^(l) = mean(δ^(l), axis=1)

            b) Backpropagate error (if l > 1):
               δ^(l-1) = (W^(l))^T @ δ^(l) ⊙ σ'(s^(l-1))
               where σ'(z) = 𝟙(z > 0) for ReLU

        Args:
            y_true: True labels of shape (n,)

        Returns:
            gradients: Dictionary mapping layer index to {'W': grad_W, 'b': grad_b}
        """
        # Retrieve cached values from forward pass
        s = self.cache['s']
        h = self.cache['h']
        n = self.cache['n']

        # Initialize gradient storage
        gradients = {}

        # ============================================
        # Step 1: Initialize error at output layer L
        # ============================================
        # For binary cross-entropy with sigmoid:
        # δ^(L) = ∂J/∂s^(L) = y - p
        prob = h[self.num_layers].flatten()  # (n,)
        delta = (y_true - prob)[:, np.newaxis].T  # (1, n) - note negative for gradient descent

        # ============================================
        # Step 2: Backward pass through all layers
        # ============================================
        for l in range(self.num_layers, 0, -1):
            # Compute gradients for layer l
            # ∂J/∂W^(l) = δ^(l) @ (h^(l-1))^T / n
            grad_W = delta @ h[l-1].T / n  # (d_l, d_{l-1})

            # ∂J/∂b^(l) = mean(δ^(l), axis=1)
            grad_b = np.mean(delta, axis=1)  # (d_l,)

            # Store gradients
            gradients[l] = {
                'W': -grad_W,  # Negative because we computed y - p instead of p - y
                'b': -grad_b   # Negative because we computed y - p instead of p - y
            }

            # Backpropagate error to previous layer (if not at input)
            if l > 1:
                # δ^(l-1) = (W^(l))^T @ δ^(l) ⊙ ReLU'(s^(l-1))
                # where ReLU'(z) = 𝟙(z > 0)
                delta = self.W[l].T @ delta  # (d_{l-1}, n)
                relu_derivative = (s[l-1] > 0).astype(float)  # (d_{l-1}, n)
                delta = delta * relu_derivative  # Element-wise product

        return gradients

    def update_parameters(self, gradients: Dict[int, Dict[str, np.ndarray]]) -> None:
        """
        Update parameters using Adam optimizer for all L layers

        Adam Algorithm (Adaptive Moment Estimation):

        For each parameter θ and its gradient g:
            1. t = t + 1  (increment time step)
            2. m = β₁·m + (1-β₁)·g     (update biased first moment)
            3. v = β₂·v + (1-β₂)·g²    (update biased second moment)
            4. m̂ = m/(1-β₁^t)          (bias-corrected first moment)
            5. v̂ = v/(1-β₂^t)          (bias-corrected second moment)
            6. θ = θ - η·m̂/(√v̂ + ε)    (parameter update)

        Intuition:
            - m tracks the exponentially weighted average of gradients (momentum)
            - v tracks the exponentially weighted average of squared gradients (adaptive LR)
            - Bias correction accounts for initialization at zero
            - Each parameter gets its own adaptive learning rate

        Args:
            gradients: Dictionary mapping layer index to {'W': grad_W, 'b': grad_b}
        """
        # Increment time step
        self.t += 1

        # Bias correction factors (computed once per update)
        bias_correction1 = 1 - self.beta1 ** self.t
        bias_correction2 = 1 - self.beta2 ** self.t

        # Update parameters for all layers
        for l in range(1, self.num_layers + 1):
            grad_W = gradients[l]['W']
            grad_b = gradients[l]['b']

            # ============================================
            # Update W^(l) (weight matrix for layer l)
            # ============================================
            # First moment: m = β₁·m + (1-β₁)·g
            self.m_W[l] = self.beta1 * self.m_W[l] + (1 - self.beta1) * grad_W

            # Second moment: v = β₂·v + (1-β₂)·g²
            self.v_W[l] = self.beta2 * self.v_W[l] + (1 - self.beta2) * (grad_W ** 2)

            # Bias-corrected moments
            m_hat_W = self.m_W[l] / bias_correction1
            v_hat_W = self.v_W[l] / bias_correction2

            # Parameter update: θ = θ - η·m̂/(√v̂ + ε)
            self.W[l] -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + self.epsilon)

            # ============================================
            # Update b^(l) (bias vector for layer l)
            # ============================================
            self.m_b[l] = self.beta1 * self.m_b[l] + (1 - self.beta1) * grad_b
            self.v_b[l] = self.beta2 * self.v_b[l] + (1 - self.beta2) * (grad_b ** 2)

            m_hat_b = self.m_b[l] / bias_correction1
            v_hat_b = self.v_b[l] / bias_correction2

            self.b[l] -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 500000, print_every: int = 10000,
              verbose: bool = True) -> Dict[str, Dict[int, float]]:
        """
        Training loop with validation monitoring

        Algorithm:
            For each epoch:
                1. Forward pass: compute predictions
                2. Compute loss: measure error
                3. Backward pass: compute gradients
                4. Update parameters: apply Adam optimizer
                5. (Optional) Validate and log progress

        Args:
            X_train: Training inputs of shape (n_train, p)
            y_train: Training labels of shape (n_train,)
            X_val: Validation inputs of shape (n_val, p) [optional]
            y_val: Validation labels of shape (n_val,) [optional]
            epochs: Number of training iterations
            print_every: Frequency of progress logging
            verbose: Whether to print progress

        Returns:
            history: Dictionary containing training and validation losses
        """
        history = {
            'train_loss': {},
            'val_loss': {}
        }

        for epoch in range(epochs):
            # ============================================
            # Training Step
            # ============================================
            # 1. Forward pass
            y_train_pred = self.forward(X_train)

            # 2. Compute loss
            train_loss = self.loss_fn(y_train, y_train_pred)
            history['train_loss'][epoch] = train_loss

            # 3. Backward pass
            gradients = self.backward(y_train)

            # 4. Update parameters
            self.update_parameters(gradients)

            # ============================================
            # Validation Step (if validation data provided)
            # ============================================
            if X_val is not None and y_val is not None:
                y_val_pred = self.forward(X_val)
                val_loss = self.loss_fn(y_val, y_val_pred)
                history['val_loss'][epoch] = val_loss
            else:
                val_loss = None

            # ============================================
            # Logging
            # ============================================
            if verbose and (epoch % print_every == 0 or epoch == epochs - 1):
                log_str = f"Epoch {epoch:07d}/{epochs:07d} | Train Loss: {train_loss:.5f}"
                if val_loss is not None:
                    log_str += f" | Val Loss: {val_loss:.5f}"

                    # Compute accuracy
                    train_acc = np.mean((y_train_pred > 0.5) == y_train)
                    val_acc = np.mean((y_val_pred > 0.5) == y_val)
                    log_str += f" | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"

                print(log_str)

        return history

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Make binary predictions

        Mathematical operation:
            1. prob = forward(X)  → compute P(y=1 | x)
            2. prediction = 𝟙(prob > threshold)

        Args:
            X: Input data of shape (n, p)
            threshold: Decision threshold (default 0.5)

        Returns:
            predictions: Binary predictions of shape (n,)
        """
        prob = self.forward(X)
        predictions = (prob > threshold).astype(int)
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities P(y=1 | x)

        Args:
            X: Input data of shape (n, p)

        Returns:
            probabilities: Probabilities of shape (n,)
        """
        return self.forward(X)
