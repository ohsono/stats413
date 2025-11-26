"""Quick test of MLP implementation"""

import numpy as np
from ElementWiseNNAdam import MLPAdam

# Set random seed
np.random.seed(2024)

# Generate small dataset
print("Generating data...")
X_train = np.random.uniform(0, 1, size=(1000, 2))
y_train = (X_train[:, 0]**2 + X_train[:, 1]**2 < 1).astype(int)

X_val = np.random.uniform(0, 1, size=(500, 2))
y_val = (X_val[:, 0]**2 + X_val[:, 1]**2 < 1).astype(int)

print(f"Train: {X_train.shape}, Val: {X_val.shape}")
print(f"Positive ratio: {np.mean(y_train):.2%}")

# Test different architectures
architectures = [
    [2, 32, 1],
    [2, 32, 16, 1],
]

for arch in architectures:
    print("\n" + "="*60)
    print(f"Testing architecture: {arch}")
    print("="*60)

    model = MLPAdam(layer_dims=arch, learning_rate=0.01)

    print(f"Number of layers: {model.num_layers}")
    print(f"Layer dimensions: {model.layer_dims}")

    # Quick training
    print("\nTraining for 100 epochs...")
    history = model.train(
        X_train, y_train, X_val, y_val,
        epochs=100,
        print_every=25,
        verbose=True
    )

    # Evaluate
    y_pred = model.predict(X_val)
    acc = np.mean(y_pred == y_val)
    print(f"\nFinal validation accuracy: {acc:.4f}")

print("\n" + "="*60)
print("Quick test complete!")
print("="*60)
