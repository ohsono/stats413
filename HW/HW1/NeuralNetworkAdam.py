import numpy as np
from typing import Dict

class NeuralNetworkAdam:
    """
    Two-layer neural network with ReLU activation and Adam optimizer

    Architecture:
        Input(2) → Hidden(d) → ReLU → Output(1) → Sigmoid

    Mathematical formulation:
        s_ik = α_k0 + Σⱼ α_kj·x_ij  (hidden layer)
        h_ik = max(0, s_ik)         (ReLU)
        s_i = β_0 + Σₖ β_k·h_ik     (output layer)
        p_i = σ(s_i) = 1/(1+e^(-s_i)) (sigmoid)
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 128,
                 learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        """
        Initialize network parameters and Adam optimizer state

        Args:
            input_dim: Number of input features (p)
            hidden_dim: Number of hidden units (d)
            learning_rate: Adam learning rate (η)
            beta1: Exponential decay rate for first moment (β₁)
            beta2: Exponential decay rate for second moment (β₂)
            epsilon: Small constant for numerical stability (ε)
        """
        self.p = input_dim
        self.d = hidden_dim
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

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
        # Initialize Adam Optimizer State
        # ============================================
        # First moment (mean) - m
        self.m_alpha = np.zeros_like(self.alpha)
        self.m_alpha0 = np.zeros_like(self.alpha0)
        self.m_beta = np.zeros_like(self.beta)
        self.m_beta0 = 0.0

        # Second moment (uncentered variance) - v
        self.v_alpha = np.zeros_like(self.alpha)
        self.v_alpha0 = np.zeros_like(self.alpha0)
        self.v_beta = np.zeros_like(self.beta)
        self.v_beta0 = 0.0

        # Time step for bias correction
        self.t = 0

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
        # s_ik = α_k0 + Σⱼ α_kj·x_ij
        # Matrix form: S = α₀ ⊗ 1^T + α @ X^T
        s_hidden = self.alpha0[:, np.newaxis] + self.alpha @ X.T  # (d, n)

        # Step 2: ReLU activation
        # h_ik = max(0, s_ik)
        h = np.maximum(0, s_hidden)  # (d, n)

        # Step 3: Output layer pre-activation
        # s_i = β₀ + Σₖ β_k·h_ik
        s_output = self.beta0 + self.beta @ h  # (n,)

        # Step 4: Sigmoid activation
        # σ(s) = 1 / (1 + e^(-s))
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

    def backward(self, y_true: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Backpropagation: Compute gradients using chain rule

        Mathematical derivation:

        Step 1: Gradient at output (sigmoid + cross-entropy)
            ∂J/∂s_output = p - y  (beautiful simplification!)

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

        # ============================================
        # Step 1: Gradient at output layer
        # ============================================
        # ∂J/∂s_output = p - y
        dJ_ds_output = prob - y_true  # (n,)

        # ============================================
        # Step 2: Gradients for β parameters
        # ============================================
        # ∂J/∂β_k = (1/n) Σᵢ (pᵢ - yᵢ)·h_ik
        grad_beta = (h @ dJ_ds_output) / n  # (d,)

        # ∂J/∂β₀ = (1/n) Σᵢ (pᵢ - yᵢ)
        grad_beta0 = np.mean(dJ_ds_output)  # scalar

        # ============================================
        # Step 3: Backprop through output layer
        # ============================================
        # ∂J/∂h_ik = (pᵢ - yᵢ)·β_k
        dJ_dh = np.outer(self.beta, dJ_ds_output)  # (d, n)

        # ============================================
        # Step 4: Gradient through ReLU
        # ============================================
        # ∂h/∂s_hidden = 𝟙(s_hidden > 0)
        dh_ds_hidden = (s_hidden > 0).astype(float)  # (d, n)

        # Chain rule: ∂J/∂s_hidden = ∂J/∂h · ∂h/∂s_hidden
        dJ_ds_hidden = dJ_dh * dh_ds_hidden  # (d, n)

        # ============================================
        # Step 5: Gradients for α parameters
        # ============================================
        # ∂J/∂α_kj = (1/n) Σᵢ (∂J/∂s_ik)·x_ij
        grad_alpha = (dJ_ds_hidden @ X) / n  # (d, p)

        # ∂J/∂α_k0 = (1/n) Σᵢ (∂J/∂s_ik)
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
        Update parameters using Adam optimizer

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
            gradients: Dictionary of gradients from backward pass
        """
        # Increment time step
        self.t += 1

        # Extract gradients
        grad_alpha = gradients['alpha']
        grad_alpha0 = gradients['alpha0']
        grad_beta = gradients['beta']
        grad_beta0 = gradients['beta0']

        # Bias correction factors (computed once per update)
        bias_correction1 = 1 - self.beta1 ** self.t
        bias_correction2 = 1 - self.beta2 ** self.t

        # ============================================
        # Update α parameters (hidden layer weights)
        # ============================================
        # First moment: m = β₁·m + (1-β₁)·g
        self.m_alpha = self.beta1 * self.m_alpha + (1 - self.beta1) * grad_alpha

        # Second moment: v = β₂·v + (1-β₂)·g²
        self.v_alpha = self.beta2 * self.v_alpha + (1 - self.beta2) * (grad_alpha ** 2)

        # Bias-corrected moments
        m_hat_alpha = self.m_alpha / bias_correction1
        v_hat_alpha = self.v_alpha / bias_correction2

        # Parameter update: θ = θ - η·m̂/(√v̂ + ε)
        self.alpha -= self.lr * m_hat_alpha / (np.sqrt(v_hat_alpha) + self.epsilon)

        # ============================================
        # Update α₀ parameters (hidden layer biases)
        # ============================================
        self.m_alpha0 = self.beta1 * self.m_alpha0 + (1 - self.beta1) * grad_alpha0
        self.v_alpha0 = self.beta2 * self.v_alpha0 + (1 - self.beta2) * (grad_alpha0 ** 2)

        m_hat_alpha0 = self.m_alpha0 / bias_correction1
        v_hat_alpha0 = self.v_alpha0 / bias_correction2

        self.alpha0 -= self.lr * m_hat_alpha0 / (np.sqrt(v_hat_alpha0) + self.epsilon)

        # ============================================
        # Update β parameters (output layer weights)
        # ============================================
        self.m_beta = self.beta1 * self.m_beta + (1 - self.beta1) * grad_beta
        self.v_beta = self.beta2 * self.v_beta + (1 - self.beta2) * (grad_beta ** 2)

        m_hat_beta = self.m_beta / bias_correction1
        v_hat_beta = self.v_beta / bias_correction2

        self.beta -= self.lr * m_hat_beta / (np.sqrt(v_hat_beta) + self.epsilon)

        # ============================================
        # Update β₀ parameter (output layer bias)
        # ============================================
        self.m_beta0 = self.beta1 * self.m_beta0 + (1 - self.beta1) * grad_beta0
        self.v_beta0 = self.beta2 * self.v_beta0 + (1 - self.beta2) * (grad_beta0 ** 2)

        m_hat_beta0 = self.m_beta0 / bias_correction1
        v_hat_beta0 = self.v_beta0 / bias_correction2

        self.beta0 -= self.lr * m_hat_beta0 / (np.sqrt(v_hat_beta0) + self.epsilon)

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
