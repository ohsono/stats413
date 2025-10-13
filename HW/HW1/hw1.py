"""
STATS 413 - Homework 1
Neural Network Implementation Comparison

Compares three optimization algorithms:
1. Adam (full-batch)
2. Adam (mini-batch with optimizations)
3. SGD with Momentum (mini-batch)

Task: Learn circular decision boundary (x₁² + x₂² < 1)
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from NeuralNetworkAdam import NeuralNetworkAdam
from NeuralNetworkAdamFullyOptimized import NeuralNetworkAdamFullyOptimized
from NeuralNetworkSGDMomentum import NeuralNetworkSGDMomentum


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


def plot_decision_boundary(ax, model, X_val, y_val, title, val_acc):
    """
    Plot decision boundary for a trained model

    Args:
        ax: Matplotlib axis object
        model: Trained neural network model
        X_val: Validation data
        y_val: Validation labels
        title: Plot title
        val_acc: Validation accuracy
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
    ax.set_title(f'{title}\n(Val Acc: {val_acc:.4f})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    return contourf


def main():
    """Main training and evaluation function"""

    # Set random seed for reproducibility
    np.random.seed(2024)

    print("=" * 80)
    print("STATS 413 - HW1: Neural Network Optimizer Comparison")
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
    # Model 1: Adam (Full-Batch)
    # ============================================
    print("=" * 80)
    print("MODEL 1: Adam Optimizer (Full-Batch)")
    print("=" * 80)

    model1 = NeuralNetworkAdam(
        input_dim=2,
        hidden_dim=128,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999
    )

    print(f"Architecture: Input(2) → Hidden(128) → ReLU → Output(1) → Sigmoid")
    print(f"Optimizer: Adam (lr={model1.lr}, β₁={model1.beta1}, β₂={model1.beta2})")
    print(f"Total parameters: {model1.alpha.size + model1.alpha0.size + model1.beta.size + 1}")
    print()

    print("Training Model 1...")
    start_time1 = time.time()
    history1 = model1.train(
        X_train, y_train, X_val, y_val,
        epochs=5000,
        print_every=1000,
        verbose=True
    )
    time1 = time.time() - start_time1

    # Evaluate Model 1
    y_train_pred1 = model1.predict(X_train)
    y_val_pred1 = model1.predict(X_val)
    train_acc1 = np.mean(y_train_pred1 == y_train)
    val_acc1 = np.mean(y_val_pred1 == y_val)
    final_train_loss1 = history1['train_loss'][max(history1['train_loss'].keys())]
    final_val_loss1 = history1['val_loss'][max(history1['val_loss'].keys())]

    print()
    print("Model 1 Results:")
    print(f"  Training Time: {time1:.2f}s")
    print(f"  Training Loss: {final_train_loss1:.5f} | Accuracy: {train_acc1:.4f}")
    print(f"  Validation Loss: {final_val_loss1:.5f} | Accuracy: {val_acc1:.4f}")
    print()

    # ============================================
    # Model 2: Adam Fully Optimized (Mini-Batch)
    # ============================================
    print("=" * 80)
    print("MODEL 2: Adam Optimizer (Mini-Batch + Optimizations)")
    print("=" * 80)

    model2 = NeuralNetworkAdamFullyOptimized(
        input_dim=2,
        hidden_dim=128,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        batch_size=256
    )

    print(f"Architecture: Input(2) → Hidden(128) → ReLU → Output(1) → Sigmoid")
    print(f"Optimizer: Adam (lr={model2.lr}, β₁={model2.beta1}, β₂={model2.beta2})")
    print(f"Batch Size: {model2.batch_size}")
    print(f"Features: Mini-batch, Early Stopping, LR Scheduling, Gradient Clipping")
    print(f"Total parameters: {model2.alpha.size + model2.alpha0.size + model2.beta.size + 1}")
    print()

    print("Training Model 2...")
    start_time2 = time.time()
    history2 = model2.train(
        X_train, y_train, X_val, y_val,
        epochs=5000,
        print_every=1000,
        early_stopping_patience=50,
        lr_schedule='cosine',
        gradient_clip=1.0,
        verbose=True
    )
    time2 = time.time() - start_time2

    # Evaluate Model 2
    y_train_pred2 = model2.predict(X_train)
    y_val_pred2 = model2.predict(X_val)
    train_acc2 = np.mean(y_train_pred2 == y_train)
    val_acc2 = np.mean(y_val_pred2 == y_val)
    final_train_loss2 = history2['train_loss'][max(history2['train_loss'].keys())]
    final_val_loss2 = history2['val_loss'][max(history2['val_loss'].keys())]

    print()
    print("Model 2 Results:")
    print(f"  Training Time: {time2:.2f}s")
    print(f"  Training Loss: {final_train_loss2:.5f} | Accuracy: {train_acc2:.4f}")
    print(f"  Validation Loss: {final_val_loss2:.5f} | Accuracy: {val_acc2:.4f}")
    print()

    # ============================================
    # Model 3: SGD with Momentum (Mini-Batch)
    # ============================================
    print("=" * 80)
    print("MODEL 3: SGD with Momentum (Mini-Batch)")
    print("=" * 80)

    model3 = NeuralNetworkSGDMomentum(
        input_dim=2,
        hidden_dim=128,
        learning_rate=0.01,
        momentum=0.9,
        batch_size=256
    )

    print(f"Architecture: Input(2) → Hidden(128) → ReLU → Output(1) → Sigmoid")
    print(f"Optimizer: SGD with Momentum (lr={model3.lr}, γ={model3.momentum})")
    print(f"Batch Size: {model3.batch_size}")
    print(f"Total parameters: {model3.alpha.size + model3.alpha0.size + model3.beta.size + 1}")
    print()

    print("Training Model 3...")
    start_time3 = time.time()
    history3 = model3.train(
        X_train, y_train, X_val, y_val,
        epochs=5000,
        print_every=1000,
        verbose=True
    )
    time3 = time.time() - start_time3

    # Evaluate Model 3
    y_train_pred3 = model3.predict(X_train)
    y_val_pred3 = model3.predict(X_val)
    train_acc3 = np.mean(y_train_pred3 == y_train)
    val_acc3 = np.mean(y_val_pred3 == y_val)
    final_train_loss3 = history3['train_loss'][max(history3['train_loss'].keys())]
    final_val_loss3 = history3['val_loss'][max(history3['val_loss'].keys())]

    print()
    print("Model 3 Results:")
    print(f"  Training Time: {time3:.2f}s")
    print(f"  Training Loss: {final_train_loss3:.5f} | Accuracy: {train_acc3:.4f}")
    print(f"  Validation Loss: {final_val_loss3:.5f} | Accuracy: {val_acc3:.4f}")
    print()

    # ============================================
    # Performance Comparison Summary
    # ============================================
    print("=" * 80)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Metric':<30} {'Model 1 (Adam FB)':<20} {'Model 2 (Adam MB)':<20} {'Model 3 (SGD+Mom)':<20}")
    print("-" * 90)
    print(f"{'Training Time':<30} {time1:>15.2f}s     {time2:>15.2f}s     {time3:>15.2f}s")
    print(f"{'Epochs Completed':<30} {max(history1['train_loss'].keys())+1:>20} {max(history2['train_loss'].keys())+1:>20} {max(history3['train_loss'].keys())+1:>20}")
    print(f"{'Final Train Loss':<30} {final_train_loss1:>20.5f} {final_train_loss2:>20.5f} {final_train_loss3:>20.5f}")
    print(f"{'Final Val Loss':<30} {final_val_loss1:>20.5f} {final_val_loss2:>20.5f} {final_val_loss3:>20.5f}")
    print(f"{'Train Accuracy':<30} {train_acc1:>20.4f} {train_acc2:>20.4f} {train_acc3:>20.4f}")
    print(f"{'Val Accuracy':<30} {val_acc1:>20.4f} {val_acc2:>20.4f} {val_acc3:>20.4f}")
    print()

    # ============================================
    # Visualization
    # ============================================
    print("Creating comprehensive visualizations...")

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

    # Row 1: Loss curves
    ax1 = fig.add_subplot(gs[0, :2])
    epochs1 = list(history1['train_loss'].keys())
    epochs2 = list(history2['train_loss'].keys())
    epochs3 = list(history3['train_loss'].keys())

    # Training losses
    ax1.plot(epochs1, list(history1['train_loss'].values()),
             label='Model 1 Train (Adam FB)', linewidth=2, linestyle='-', alpha=0.8, color='#1f77b4')
    ax1.plot(epochs2, list(history2['train_loss'].values()),
             label='Model 2 Train (Adam MB)', linewidth=2, linestyle='-', alpha=0.8, color='#ff7f0e')
    ax1.plot(epochs3, list(history3['train_loss'].values()),
             label='Model 3 Train (SGD+Mom)', linewidth=2, linestyle='-', alpha=0.8, color='#2ca02c')

    # Validation losses
    ax1.plot(epochs1, list(history1['val_loss'].values()),
             label='Model 1 Val', linewidth=2, linestyle='--', alpha=0.8, color='#1f77b4')
    ax1.plot(epochs2, list(history2['val_loss'].values()),
             label='Model 2 Val', linewidth=2, linestyle='--', alpha=0.8, color='#ff7f0e')
    ax1.plot(epochs3, list(history3['val_loss'].values()),
             label='Model 3 Val', linewidth=2, linestyle='--', alpha=0.8, color='#2ca02c')

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Binary Cross-Entropy Loss', fontsize=12)
    ax1.set_title('Training Progress: All Models Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right', ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Performance metrics table
    ax_table = fig.add_subplot(gs[0, 2])
    ax_table.axis('off')

    table_data = [
        ['Metric', 'M1', 'M2', 'M3'],
        ['Time (s)', f'{time1:.1f}', f'{time2:.1f}', f'{time3:.1f}'],
        ['Epochs', f'{max(epochs1)+1}', f'{max(epochs2)+1}', f'{max(epochs3)+1}'],
        ['Train Acc', f'{train_acc1:.4f}', f'{train_acc2:.4f}', f'{train_acc3:.4f}'],
        ['Val Acc', f'{val_acc1:.4f}', f'{val_acc2:.4f}', f'{val_acc3:.4f}'],
        ['Train Loss', f'{final_train_loss1:.4f}', f'{final_train_loss2:.4f}', f'{final_train_loss3:.4f}'],
        ['Val Loss', f'{final_val_loss1:.4f}', f'{final_val_loss2:.4f}', f'{final_val_loss3:.4f}']
    ]

    table = ax_table.table(cellText=table_data, cellLoc='center', loc='center',
                           colWidths=[0.35, 0.22, 0.22, 0.22])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Header styling
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    ax_table.set_title('Performance Metrics', fontsize=12, fontweight='bold', pad=20)

    # Row 2: Decision boundaries
    ax2 = fig.add_subplot(gs[1, 0])
    plot_decision_boundary(ax2, model1, X_val, y_val, 'Model 1: Adam (Full-Batch)', val_acc1)

    ax3 = fig.add_subplot(gs[1, 1])
    plot_decision_boundary(ax3, model2, X_val, y_val, 'Model 2: Adam (Mini-Batch)', val_acc2)

    ax4 = fig.add_subplot(gs[1, 2])
    plot_decision_boundary(ax4, model3, X_val, y_val, 'Model 3: SGD + Momentum', val_acc3)

    # Row 3: Validation loss comparison (zoomed)
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(epochs1, list(history1['val_loss'].values()),
             label='Adam (Full-Batch)', linewidth=2, color='#1f77b4')
    ax5.plot(epochs2, list(history2['val_loss'].values()),
             label='Adam (Mini-Batch)', linewidth=2, color='#ff7f0e')
    ax5.plot(epochs3, list(history3['val_loss'].values()),
             label='SGD + Momentum', linewidth=2, color='#2ca02c')
    ax5.set_xlabel('Epoch', fontsize=11)
    ax5.set_ylabel('Validation Loss', fontsize=11)
    ax5.set_title('Validation Loss Comparison', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    # Training loss comparison (zoomed)
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(epochs1, list(history1['train_loss'].values()),
             label='Adam (Full-Batch)', linewidth=2, color='#1f77b4')
    ax6.plot(epochs2, list(history2['train_loss'].values()),
             label='Adam (Mini-Batch)', linewidth=2, color='#ff7f0e')
    ax6.plot(epochs3, list(history3['train_loss'].values()),
             label='SGD + Momentum', linewidth=2, color='#2ca02c')
    ax6.set_xlabel('Epoch', fontsize=11)
    ax6.set_ylabel('Training Loss', fontsize=11)
    ax6.set_title('Training Loss Comparison', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)

    # Accuracy comparison
    ax7 = fig.add_subplot(gs[2, 2])
    models = ['Adam\n(Full-Batch)', 'Adam\n(Mini-Batch)', 'SGD +\nMomentum']
    train_accs = [train_acc1, train_acc2, train_acc3]
    val_accs = [val_acc1, val_acc2, val_acc3]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax7.bar(x - width/2, train_accs, width, label='Train Accuracy',
                    color='#1f77b4', alpha=0.8)
    bars2 = ax7.bar(x + width/2, val_accs, width, label='Val Accuracy',
                    color='#ff7f0e', alpha=0.8)

    ax7.set_ylabel('Accuracy', fontsize=11)
    ax7.set_title('Final Accuracy Comparison', fontsize=12, fontweight='bold')
    ax7.set_xticks(x)
    ax7.set_xticklabels(models, fontsize=9)
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3, axis='y')
    ax7.set_ylim([0.8, 1.0])

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax7.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.suptitle('Neural Network Optimizer Comparison: Adam vs SGD with Momentum',
                fontsize=16, fontweight='bold', y=0.995)

    # Save figure
    plt.savefig('hw1_results.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'hw1_results.png'")
    print()

    print("=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
