import numpy as np
from typing import Dict

class NeuralNetworkSGDMomentum:
    """
    Two-layer neural network with ReLU activation and SGD with Momentum optimizer

    Architecture:
        Input(2) → Hidden(d) → ReLU → Output(1) → Sigmoid

    Mathematical formulation:
        s_ik = α_k0 + Σⱼ α_kj·x_ij  (hidden layer)
        h_ik = max(0, s_ik)         (ReLU)
        s_i = β_0 + Σₖ β_k·h_ik     (output layer)
        p_i = σ(s_i) = 1/(1+e^(-s_i)) (sigmoid)
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 128,
                 learning_rate: float = 0.01, momentum: float = 0.9,
                 batch_size: int = 256):
        """
        Initialize network parameters and SGD with Momentum optimizer state

        Args:
            input_dim: Number of input features (p)
            hidden_dim: Number of hidden units (d)
            learning_rate: Learning rate (η)
            momentum: Momentum coefficient (γ), typically 0.9
            batch_size: Size of mini-batches for training
        """
        self.p = input_dim
        self.d = hidden_dim
        self.lr = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size

        # ============================================
        # Initialize Network Parameters
        # ============================================
        # Hidden layer (α parameters)
        self.alpha = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / input_dim)
        self.alpha0 = np.zeros(hidden_dim)

        # Output layer (β parameters)
        self.beta = np.random.randn(hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.beta0 = 0.0

        # ============================================
        # Initialize Momentum State
        # ============================================
        # Velocity (momentum buffer) - v
        self.v_alpha = np.zeros_like(self.alpha)
        self.v_alpha0 = np.zeros_like(self.alpha0)
        self.v_beta = np.zeros_like(self.beta)
        self.v_beta0 = 0.0

        # Cache for backpropagation
        self.cache = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network

        Mathematical operations:
            1. s_hidden = α₀ + α @ X^T        (hidden pre-activation)
            2. h = max(0, s_hidden)            (ReLU activation)
            3. s_output = β₀ + β^T @ h         (output pre-activation)
            4. prob = 1 / (1 + exp(-s_output)) (sigmoid activation)

        Args:
            X: Input data of shape (n, p)

        Returns:
            prob: Probabilities of shape (n,) where prob[i] = P(y=1 | x_i)
        """
        n = X.shape[0]

        # Step 1: Hidden layer pre-activation
        s_hidden = self.alpha0[:, np.newaxis] + self.alpha @ X.T  # (d, n)

        # Step 2: ReLU activation
        h = np.maximum(0, s_hidden)  # (d, n)

        # Step 3: Output layer pre-activation
        s_output = self.beta0 + self.beta @ h  # (n,)

        # Step 4: Sigmoid activation
        s_output_clipped = np.clip(s_output, -500, 500)  # Numerical stability
        prob = 1.0 / (1.0 + np.exp(-s_output_clipped))  # (n,)

        # Cache intermediate values for backpropagation
        self.cache = {
            'X': X,
            's_hidden': s_hidden,
            'h': h,
            's_output': s_output,
            'prob': prob,
            'n': n
        }

        return prob

    def loss_fn(self, y_true: np.ndarray, y_pred: np.ndarray = None) -> float:
        """
        Binary cross-entropy loss (negative log-likelihood)

        Mathematical formula:
            J = -(1/n) Σᵢ [yᵢ·log(pᵢ) + (1-yᵢ)·log(1-pᵢ)]

        Args:
            y_true: True labels of shape (n,)
            y_pred: Predicted probabilities of shape (n,) [optional, uses cache if None]

        Returns:
            loss: Scalar binary cross-entropy loss
        """
        if y_pred is None:
            # Use numerically stable version from logits
            s = self.cache['s_output']
            loss = np.mean(
                np.maximum(0, s) - s * y_true + np.log(1 + np.exp(-np.abs(s)))
            )
        else:
            eps = 1e-15
            y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
            loss = -np.mean(
                y_true * np.log(y_pred_clipped) +
                (1 - y_true) * np.log(1 - y_pred_clipped)
            )
        return loss

    def backward(self, y_true: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Backpropagation: Compute gradients using chain rule

        Mathematical derivation:

        Step 1: Gradient at output (sigmoid + cross-entropy)
            ∂J/∂s_output = p - y

        Step 2: Gradients for β parameters
            ∂J/∂β_k = (1/n) Σᵢ (pᵢ - yᵢ)·h_ik
            ∂J/∂β₀ = (1/n) Σᵢ (pᵢ - yᵢ)

        Step 3: Backprop through output layer
            ∂J/∂h_ik = (pᵢ - yᵢ)·β_k

        Step 4: Gradient through ReLU
            ∂J/∂s_ik = ∂J/∂h_ik · 𝟙(s_ik > 0)

        Step 5: Gradients for α parameters
            ∂J/∂α_kj = (1/n) Σᵢ (∂J/∂s_ik)·x_ij
            ∂J/∂α_k0 = (1/n) Σᵢ (∂J/∂s_ik)

        Args:
            y_true: True labels of shape (n,)

        Returns:
            gradients: Dictionary containing all parameter gradients
        """
        # Retrieve cached values from forward pass
        X = self.cache['X']
        s_hidden = self.cache['s_hidden']
        h = self.cache['h']
        prob = self.cache['prob']
        n = self.cache['n']

        # Step 1: Gradient at output layer
        dJ_ds_output = prob - y_true  # (n,)

        # Step 2: Gradients for β parameters
        grad_beta = (h @ dJ_ds_output) / n  # (d,)
        grad_beta0 = np.mean(dJ_ds_output)  # scalar

        # Step 3: Backprop through output layer
        dJ_dh = np.outer(self.beta, dJ_ds_output)  # (d, n)

        # Step 4: Gradient through ReLU
        dh_ds_hidden = (s_hidden > 0).astype(float)  # (d, n)
        dJ_ds_hidden = dJ_dh * dh_ds_hidden  # (d, n)

        # Step 5: Gradients for α parameters
        grad_alpha = (dJ_ds_hidden @ X) / n  # (d, p)
        grad_alpha0 = np.mean(dJ_ds_hidden, axis=1)  # (d,)

        # Return all gradients as dictionary
        gradients = {
            'alpha': grad_alpha,
            'alpha0': grad_alpha0,
            'beta': grad_beta,
            'beta0': grad_beta0
        }

        return gradients

    def update_parameters(self, gradients: Dict[str, np.ndarray]) -> None:
        """
        Update parameters using SGD with Momentum optimizer

        Momentum Algorithm:

        For each parameter θ and its gradient g:
            1. v = γ·v + η·g     (update velocity with momentum)
            2. θ = θ - v         (parameter update)

        Intuition:
            - v (velocity) accumulates gradient history
            - γ (momentum) determines how much past gradients influence current update
            - Helps accelerate SGD in relevant directions and dampen oscillations
            - Typical values: γ ∈ [0.5, 0.9, 0.99]

        Mathematical Note:
            Momentum can be viewed as exponentially weighted moving average:
            v_t = γ·v_{t-1} + η·g_t
                = η·Σᵢ γⁱ·g_{t-i}

        Args:
            gradients: Dictionary of gradients from backward pass
        """
        # Extract gradients
        grad_alpha = gradients['alpha']
        grad_alpha0 = gradients['alpha0']
        grad_beta = gradients['beta']
        grad_beta0 = gradients['beta0']

        # ============================================
        # Update α parameters (hidden layer weights)
        # ============================================
        # Update velocity: v = γ·v + η·g
        self.v_alpha = self.momentum * self.v_alpha + self.lr * grad_alpha

        # Update parameter: θ = θ - v
        self.alpha -= self.v_alpha

        # ============================================
        # Update α₀ parameters (hidden layer biases)
        # ============================================
        self.v_alpha0 = self.momentum * self.v_alpha0 + self.lr * grad_alpha0
        self.alpha0 -= self.v_alpha0

        # ============================================
        # Update β parameters (output layer weights)
        # ============================================
        self.v_beta = self.momentum * self.v_beta + self.lr * grad_beta
        self.beta -= self.v_beta

        # ============================================
        # Update β₀ parameter (output layer bias)
        # ============================================
        self.v_beta0 = self.momentum * self.v_beta0 + self.lr * grad_beta0
        self.beta0 -= self.v_beta0

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 50000, print_every: int = 1000,
              verbose: bool = True) -> Dict[str, Dict[int, float]]:
        """
        Training loop with mini-batch SGD and momentum

        Algorithm:
            For each epoch:
                1. Shuffle training data
                2. For each mini-batch:
                    a. Forward pass: compute predictions
                    b. Compute loss: measure error
                    c. Backward pass: compute gradients
                    d. Update parameters: apply SGD with momentum
                3. (Optional) Validate and log progress

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
        n_samples = X_train.shape[0]
        n_batches = int(np.ceil(n_samples / self.batch_size))

        history = {
            'train_loss': {},
            'val_loss': {}
        }

        for epoch in range(epochs):
            # Shuffle data for stochastic gradient descent
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_losses = []

            # ============================================
            # Mini-batch Training
            # ============================================
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, n_samples)

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # 1. Forward pass
                y_pred = self.forward(X_batch)

                # 2. Compute loss
                batch_loss = self.loss_fn(y_batch)
                epoch_losses.append(batch_loss)

                # 3. Backward pass
                gradients = self.backward(y_batch)

                # 4. Update parameters
                self.update_parameters(gradients)

            # Average loss for the epoch
            train_loss = np.mean(epoch_losses)
            history['train_loss'][epoch] = train_loss

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
                log_str = f"Epoch {epoch:05d}/{epochs:05d} | Train Loss: {train_loss:.5f}"
                if val_loss is not None:
                    log_str += f" | Val Loss: {val_loss:.5f}"

                    # Compute accuracy
                    train_acc = np.mean((self.forward(X_train) > 0.5) == y_train)
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
