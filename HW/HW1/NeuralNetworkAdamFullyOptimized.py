import numpy as np
from typing import Dict

class NeuralNetworkAdamFullyOptimized:
    """
    Fully optimized neural network with:
        1. Mini-batch training
        2. Early stopping
        3. Learning rate scheduling
        4. Gradient clipping
        5. Optimized operations
        6. Better initialization
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 128,
                 learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8,
                 batch_size: int = 256):

        self.p = input_dim
        self.d = hidden_dim
        self.lr = learning_rate
        self.initial_lr = learning_rate  # Save for scheduling
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.batch_size = batch_size

        # Optimized initialization (He initialization for ReLU)
        self.alpha = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / input_dim)
        self.alpha0 = np.zeros(hidden_dim)
        self.beta = np.random.randn(hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.beta0 = 0.0

        # Adam state
        self.m_alpha = np.zeros_like(self.alpha)
        self.m_alpha0 = np.zeros_like(self.alpha0)
        self.m_beta = np.zeros_like(self.beta)
        self.m_beta0 = 0.0

        self.v_alpha = np.zeros_like(self.alpha)
        self.v_alpha0 = np.zeros_like(self.alpha0)
        self.v_beta = np.zeros_like(self.beta)
        self.v_beta0 = 0.0

        self.t = 0
        self.cache = {}

        # Pre-allocate buffers
        self._s_hidden_buffer = None
        self._h_buffer = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Optimized forward pass"""
        n = X.shape[0]

        # Allocate buffers if needed
        if self._s_hidden_buffer is None or self._s_hidden_buffer.shape[1] != n:
            self._s_hidden_buffer = np.empty((self.d, n))
            self._h_buffer = np.empty((self.d, n))

        # Hidden layer (in-place operations)
        np.dot(self.alpha, X.T, out=self._s_hidden_buffer)
        self._s_hidden_buffer += self.alpha0[:, np.newaxis]

        # ReLU (in-place)
        np.maximum(self._s_hidden_buffer, 0, out=self._h_buffer)

        # Output layer
        s_output = self.beta0 + np.dot(self.beta, self._h_buffer)

        # Sigmoid
        s_output_clipped = np.clip(s_output, -500, 500)
        prob = 1.0 / (1.0 + np.exp(-s_output_clipped))

        self.cache = {
            'X': X, 's_hidden': self._s_hidden_buffer,
            'h': self._h_buffer, 's_output': s_output,
            'prob': prob, 'n': n
        }

        return prob

    def loss_fn(self, y_true: np.ndarray, y_pred: np.ndarray = None) -> float:
        """Numerically stable BCE using logits"""
        s = self.cache['s_output']
        loss = np.mean(
            np.maximum(0, s) - s * y_true + np.log(1 + np.exp(-np.abs(s)))
        )
        return loss

    def backward(self, y_true: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass (same as before)"""
        X = self.cache['X']
        s_hidden = self.cache['s_hidden']
        h = self.cache['h']
        prob = self.cache['prob']
        n = self.cache['n']

        dJ_ds_output = prob - y_true
        grad_beta = (h @ dJ_ds_output) / n
        grad_beta0 = np.mean(dJ_ds_output)

        dJ_dh = np.outer(self.beta, dJ_ds_output)
        dh_ds_hidden = (s_hidden > 0).astype(float)
        dJ_ds_hidden = dJ_dh * dh_ds_hidden

        grad_alpha = (dJ_ds_hidden @ X) / n
        grad_alpha0 = np.mean(dJ_ds_hidden, axis=1)

        return {
            'alpha': grad_alpha, 'alpha0': grad_alpha0,
            'beta': grad_beta, 'beta0': grad_beta0
        }

    def clip_gradients(self, gradients: Dict, max_norm: float = 1.0) -> Dict:
        """Gradient clipping by global norm"""
        total_norm = np.sqrt(sum(np.sum(g**2) for g in gradients.values()))
        clip_coef = max_norm / (total_norm + 1e-6)

        if clip_coef < 1:
            for key in gradients:
                gradients[key] *= clip_coef

        return gradients

    def update_parameters(self, gradients: Dict) -> None:
        """Adam update (same as before)"""
        self.t += 1
        grad_alpha = gradients['alpha']
        grad_alpha0 = gradients['alpha0']
        grad_beta = gradients['beta']
        grad_beta0 = gradients['beta0']

        bias_correction1 = 1 - self.beta1 ** self.t
        bias_correction2 = 1 - self.beta2 ** self.t

        # Update alpha
        self.m_alpha = self.beta1 * self.m_alpha + (1 - self.beta1) * grad_alpha
        self.v_alpha = self.beta2 * self.v_alpha + (1 - self.beta2) * (grad_alpha ** 2)
        m_hat = self.m_alpha / bias_correction1
        v_hat = self.v_alpha / bias_correction2
        self.alpha -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        # Update alpha0
        self.m_alpha0 = self.beta1 * self.m_alpha0 + (1 - self.beta1) * grad_alpha0
        self.v_alpha0 = self.beta2 * self.v_alpha0 + (1 - self.beta2) * (grad_alpha0 ** 2)
        m_hat = self.m_alpha0 / bias_correction1
        v_hat = self.v_alpha0 / bias_correction2
        self.alpha0 -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        # Update beta
        self.m_beta = self.beta1 * self.m_beta + (1 - self.beta1) * grad_beta
        self.v_beta = self.beta2 * self.v_beta + (1 - self.beta2) * (grad_beta ** 2)
        m_hat = self.m_beta / bias_correction1
        v_hat = self.v_beta / bias_correction2
        self.beta -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        # Update beta0
        self.m_beta0 = self.beta1 * self.m_beta0 + (1 - self.beta1) * grad_beta0
        self.v_beta0 = self.beta2 * self.v_beta0 + (1 - self.beta2) * (grad_beta0 ** 2)
        m_hat = self.m_beta0 / bias_correction1
        v_hat = self.v_beta0 / bias_correction2
        self.beta0 -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 50000, print_every: int = 1000,
              early_stopping_patience: int = 20,
              lr_schedule: str = 'cosine',
              gradient_clip: float = 1.0,
              verbose: bool = True) -> Dict:
        """
        Fully optimized training with all features

        Features:
            1. Mini-batch training
            2. Early stopping
            3. Learning rate scheduling
            4. Gradient clipping
        """
        n_samples = X_train.shape[0]
        n_batches = int(np.ceil(n_samples / self.batch_size))

        history = {'train_loss': {}, 'val_loss': {}}

        # Early stopping
        best_val_loss = np.inf
        patience_counter = 0
        best_params = None

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_losses = []

            # Mini-batch training
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, n_samples)

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Forward
                y_pred = self.forward(X_batch)
                batch_loss = self.loss_fn(y_batch)
                epoch_losses.append(batch_loss)

                # Backward
                gradients = self.backward(y_batch)

                # Gradient clipping
                if gradient_clip > 0:
                    gradients = self.clip_gradients(gradients, gradient_clip)

                # Update
                self.update_parameters(gradients)

            train_loss = np.mean(epoch_losses)
            history['train_loss'][epoch] = train_loss

            # Validation
            if X_val is not None:
                y_val_pred = self.forward(X_val)
                val_loss = self.loss_fn(y_val)
                history['val_loss'][epoch] = val_loss

                # Early stopping check
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_params = {
                        'alpha': self.alpha.copy(),
                        'alpha0': self.alpha0.copy(),
                        'beta': self.beta.copy(),
                        'beta0': self.beta0
                    }
                else:
                    patience_counter += 1

                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"\nEarly stopping at epoch {epoch}")
                            print(f"Best val loss: {best_val_loss:.5f}")

                        # Restore best params
                        self.alpha = best_params['alpha']
                        self.alpha0 = best_params['alpha0']
                        self.beta = best_params['beta']
                        self.beta0 = best_params['beta0']
                        break

            # Learning rate scheduling
            if lr_schedule == 'cosine':
                self.lr = self.initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / epochs))
            elif lr_schedule == 'step':
                if epoch > 0 and epoch % 10000 == 0:
                    self.lr *= 0.5

            # Logging
            if verbose and (epoch % print_every == 0 or epoch == epochs - 1):
                log = f"Epoch {epoch:05d} | Train: {train_loss:.5f}"
                if X_val is not None:
                    log += f" | Val: {val_loss:.5f}"
                log += f" | LR: {self.lr:.6f}"
                print(log)

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        prob = self.forward(X)
        return (prob > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probabilities"""
        return self.forward(X)
