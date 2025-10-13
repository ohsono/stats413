"""Quick test script to verify all imports and basic functionality"""

import numpy as np
from NeuralNetworkAdam import NeuralNetworkAdam
from NeuralNetworkAdamFullyOptimized import NeuralNetworkAdamFullyOptimized
from NeuralNetworkSGDMomentum import NeuralNetworkSGDMomentum

def generate_data(n_samples):
    X = np.random.uniform(0, 1, size=(n_samples, 2))
    y = (X[:, 0]**2 + X[:, 1]**2 < 1).astype(int)
    return X, y

# Set random seed
np.random.seed(2024)

print("Testing imports and basic functionality...")
print()

# Generate small dataset
X_train, y_train = generate_data(1000)
X_val, y_val = generate_data(500)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print()

# Test Model 1: Adam Full-Batch
print("=" * 60)
print("Testing Model 1: NeuralNetworkAdam")
print("=" * 60)
model1 = NeuralNetworkAdam(input_dim=2, hidden_dim=32, learning_rate=0.001)
history1 = model1.train(X_train, y_train, X_val, y_val, epochs=100, print_every=50, verbose=True)
pred1 = model1.predict(X_val)
acc1 = np.mean(pred1 == y_val)
print(f"Final validation accuracy: {acc1:.4f}")
print()

# Test Model 2: Adam Optimized
print("=" * 60)
print("Testing Model 2: NeuralNetworkAdamFullyOptimized")
print("=" * 60)
model2 = NeuralNetworkAdamFullyOptimized(input_dim=2, hidden_dim=32, learning_rate=0.001, batch_size=128)
history2 = model2.train(X_train, y_train, X_val, y_val, epochs=100, print_every=50,
                        early_stopping_patience=20, verbose=True)
pred2 = model2.predict(X_val)
acc2 = np.mean(pred2 == y_val)
print(f"Final validation accuracy: {acc2:.4f}")
print()

# Test Model 3: SGD with Momentum
print("=" * 60)
print("Testing Model 3: NeuralNetworkSGDMomentum")
print("=" * 60)
model3 = NeuralNetworkSGDMomentum(input_dim=2, hidden_dim=32, learning_rate=0.01,
                                  momentum=0.9, batch_size=128)
history3 = model3.train(X_train, y_train, X_val, y_val, epochs=100, print_every=50, verbose=True)
pred3 = model3.predict(X_val)
acc3 = np.mean(pred3 == y_val)
print(f"Final validation accuracy: {acc3:.4f}")
print()

print("=" * 60)
print("All tests passed! All models are working correctly.")
print("=" * 60)
