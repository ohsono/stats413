"""
STATS 413 - Homework 2, Problem 2
Multi-Layer Perceptron with Adam Optimizer

Apply flexible MLP architecture to HW1 classification problem:
- Task: Learn circular decision boundary (x₁² + x₂² < 1)
- Demonstrates flexible architecture with different layer configurations
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from ElementWiseNNAdam import MLPAdam


def generate_data(n_samples):
    """
    Generate training data: uniform in [0,1]^2, labeled by unit circle

    Args:
        n_samples: Number of samples to generate

    Returns:
        X: Input features of shape (n_samples, 2)
        y: Binary labels of shape (n_samples,)
    """
    X = np.random.uniform(0, 1, size=(n_samples, 2))
    y = (X[:, 0]**2 + X[:, 1]**2 < 1).astype(int)
    return X, y


def plot_decision_boundary(ax, model, X_val, y_val, title, val_acc, architecture_str):
    """
    Plot decision boundary for a trained model

    Args:
        ax: Matplotlib axis object
        model: Trained neural network model
        X_val: Validation data
        y_val: Validation labels
        title: Plot title
        val_acc: Validation accuracy
        architecture_str: String describing architecture
    """
    # Create meshgrid for decision boundary
    x1_min, x1_max = -0.1, 1.1
    x2_min, x2_max = -0.1, 1.1
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 300),
                           np.linspace(x2_min, x2_max, 300))
    X_grid = np.c_[xx1.ravel(), xx2.ravel()]

    # Get model predictions
    Z = model.predict_proba(X_grid).reshape(xx1.shape)

    # Plot filled contour
    contourf = ax.contourf(xx1, xx2, Z, levels=20, cmap='RdYlBu', alpha=0.7)
    ax.contour(xx1, xx2, Z, levels=[0.5], colors='black', linewidths=3)

    # Sample validation points for visualization
    sample_size = min(500, len(y_val))
    sample_idx = np.random.choice(len(y_val), sample_size, replace=False)

    ax.scatter(X_val[sample_idx][y_val[sample_idx]==1, 0],
              X_val[sample_idx][y_val[sample_idx]==1, 1],
              c='blue', marker='o', edgecolors='k', s=15, alpha=0.6,
              label='Inside (y=1)')
    ax.scatter(X_val[sample_idx][y_val[sample_idx]==0, 0],
              X_val[sample_idx][y_val[sample_idx]==0, 1],
              c='red', marker='s', edgecolors='k', s=15, alpha=0.6,
              label='Outside (y=0)')

    # Plot true boundary
    theta = np.linspace(0, np.pi/2, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'g--', linewidth=3, label='True boundary')

    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title(f'{title}\n{architecture_str}\n(Val Acc: {val_acc:.4f})',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    return contourf


def count_parameters(model):
    """Count total number of parameters in the model"""
    total = 0
    for l in range(1, model.num_layers + 1):
        total += model.W[l].size + model.b[l].size
    return total


def main():
    """Main training and evaluation function"""

    # Set random seed for reproducibility
    np.random.seed(2024)

    print("=" * 80)
    print("STATS 413 - HW2 Problem 2: Multi-Layer Perceptron with Adam")
    print("Task: Learn quarter circle decision boundary")
    print("=" * 80)
    print()

    # ============================================
    # Generate Data
    # ============================================
    print("Generating data...")
    N_train = 10000
    N_val = 5000

    X_train, y_train = generate_data(N_train)
    X_val, y_val = generate_data(N_val)

    print(f"Training samples: {N_train}")
    print(f"Validation samples: {N_val}")
    print(f"Positive samples (inside circle): {np.sum(y_train)}/{N_train} = {np.mean(y_train):.2%}")
    print()

    # ============================================
    # Define Different Architectures to Test
    # ============================================
    architectures = [
        ([2, 64, 1], "2-Layer: 2→64→1"),
        ([2, 128, 1], "2-Layer: 2→128→1"),
        ([2, 64, 32, 1], "3-Layer: 2→64→32→1"),
    ]

    models = []
    results = []

    # ============================================
    # Train Each Architecture
    # ============================================
    for i, (layer_dims, arch_str) in enumerate(architectures):
        print("=" * 80)
        print(f"MODEL {i+1}: {arch_str}")
        print("=" * 80)

        # Create model
        model = MLPAdam(
            layer_dims=layer_dims,
            learning_rate=0.001,
            beta1=0.9,
            beta2=0.999
        )

        n_params = count_parameters(model)
        print(f"Architecture: {arch_str}")
        print(f"Number of layers (L): {model.num_layers}")
        print(f"Layer dimensions: {layer_dims}")
        print(f"Total parameters: {n_params}")
        print(f"Optimizer: Adam (lr={model.lr}, β₁={model.beta1}, β₂={model.beta2})")
        print()

        # Train model
        print(f"Training Model {i+1}...")
        start_time = time.time()
        history = model.train(
            X_train, y_train, X_val, y_val,
            epochs=2000,
            print_every=500,
            verbose=True
        )
        training_time = time.time() - start_time

        # Evaluate model
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        train_acc = np.mean(y_train_pred == y_train)
        val_acc = np.mean(y_val_pred == y_val)
        final_train_loss = history['train_loss'][max(history['train_loss'].keys())]
        final_val_loss = history['val_loss'][max(history['val_loss'].keys())]

        print()
        print(f"Model {i+1} Results:")
        print(f"  Training Time: {training_time:.2f}s")
        print(f"  Training Loss: {final_train_loss:.5f} | Accuracy: {train_acc:.4f}")
        print(f"  Validation Loss: {final_val_loss:.5f} | Accuracy: {val_acc:.4f}")
        print()

        # Store results
        models.append(model)
        results.append({
            'arch_str': arch_str,
            'layer_dims': layer_dims,
            'n_params': n_params,
            'history': history,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_loss': final_train_loss,
            'val_loss': final_val_loss,
            'time': training_time
        })

    # ============================================
    # Performance Comparison Summary
    # ============================================
    print("=" * 80)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Model':<25} {'Params':<10} {'Time (s)':<12} {'Train Acc':<12} {'Val Acc':<12} {'Val Loss':<12}")
    print("-" * 90)
    for i, res in enumerate(results):
        print(f"{res['arch_str']:<25} {res['n_params']:<10} {res['time']:>10.2f}  {res['train_acc']:>10.4f}  {res['val_acc']:>10.4f}  {res['val_loss']:>10.5f}")
    print()

    # ============================================
    # Visualization
    # ============================================
    print("Creating comprehensive visualizations...")

    n_models = len(models)
    fig = plt.figure(figsize=(20, 4 * ((n_models + 2) // 3)))

    # Create grid for decision boundaries
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols

    for i, (model, res) in enumerate(zip(models, results)):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        plot_decision_boundary(ax, model, X_val, y_val,
                              f"Model {i+1}", res['val_acc'], res['arch_str'])

    plt.suptitle('Multi-Layer Perceptron with Adam: Different Architectures',
                fontsize=16, fontweight='bold', y=0.998)
    plt.tight_layout()

    # Save figure
    plt.savefig('hw2_problem2_results.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'hw2_problem2_results.png'")
    print()

    # ============================================
    # Loss Curves Comparison
    # ============================================
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, res in enumerate(results):
        epochs = list(res['history']['train_loss'].keys())
        ax1.plot(epochs, list(res['history']['train_loss'].values()),
                label=res['arch_str'], linewidth=2, color=colors[i % len(colors)])
        ax2.plot(epochs, list(res['history']['val_loss'].values()),
                label=res['arch_str'], linewidth=2, color=colors[i % len(colors)])

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('hw2_problem2_loss_curves.png', dpi=150, bbox_inches='tight')
    print("Loss curves saved as 'hw2_problem2_loss_curves.png'")
    print()

    print("=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
