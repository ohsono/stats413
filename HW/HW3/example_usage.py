"""
Example usage of the refactored CNN training framework.
This script demonstrates various ways to use the framework.
"""

from cnn_trainer import (
    OriginalNet, DoubleNet_1, DoubleNet_2,
    TrainingConfig, prepare_data, CNNModel
)
import torch
import torch.nn as nn
import torch.nn.functional as F


def example_1_train_single_model():
    """Example 1: Train a single model with custom configuration."""
    print("\n" + "="*60)
    print("Example 1: Training a Single Model")
    print("="*60)

    # Custom configuration
    config = TrainingConfig(
        learning_rate=0.002,
        epochs=5,
        batch_size=8
    )

    # Prepare data
    trainloader, testloader, classes = prepare_data(config)

    # Create and train model
    model = OriginalNet()
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining {model.model_name} for {config.epochs} epochs...")
    model.train_model(trainloader, config, criterion)

    # Evaluate
    model.evaluate_model(testloader, classes, config)

    # Save model
    model.save_model('./data/example_model.pth')


def example_2_compare_models():
    """Example 2: Compare multiple models with same configuration."""
    print("\n" + "="*60)
    print("Example 2: Comparing Multiple Models")
    print("="*60)

    # Shared configuration
    config = TrainingConfig(epochs=3, batch_size=4)
    trainloader, testloader, classes = prepare_data(config)
    criterion = nn.CrossEntropyLoss()

    # Train multiple models
    models = [OriginalNet(), DoubleNet_1(), DoubleNet_2()]

    results = {}
    for model in models:
        print(f"\n{'='*60}")
        print(f"Training {model.model_name}")
        print(f"{'='*60}")

        model.train_model(trainloader, config, criterion)
        model.evaluate_model(testloader, classes, config)

        results[model.model_name] = {
            'accuracy': model.metrics.test_accuracy,
            'training_time': model.metrics.total_training_time
        }

    # Compare results
    print("\n" + "="*60)
    print("Comparison Summary")
    print("="*60)
    for name, metrics in results.items():
        print(f"{name:20s} | Accuracy: {metrics['accuracy']:5.2f}% | "
              f"Time: {metrics['training_time']:6.2f}s")


def example_3_custom_model():
    """Example 3: Create and train a custom model."""
    print("\n" + "="*60)
    print("Example 3: Custom Model Architecture")
    print("="*60)

    class CustomNet(CNNModel):
        """A custom CNN with batch normalization."""

        def __init__(self):
            super().__init__(model_name="CustomNet_BatchNorm")
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(32 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Train the custom model
    config = TrainingConfig(epochs=3, batch_size=8)
    trainloader, testloader, classes = prepare_data(config)
    criterion = nn.CrossEntropyLoss()

    model = CustomNet()
    print(f"Training custom model: {model.model_name}")
    model.train_model(trainloader, config, criterion)
    model.evaluate_model(testloader, classes, config)


def example_4_access_metrics():
    """Example 4: Access and analyze metrics after training."""
    print("\n" + "="*60)
    print("Example 4: Accessing Performance Metrics")
    print("="*60)

    config = TrainingConfig(epochs=2, batch_size=4)
    trainloader, testloader, classes = prepare_data(config)
    criterion = nn.CrossEntropyLoss()

    model = OriginalNet()
    model.train_model(trainloader, config, criterion)
    model.evaluate_model(testloader, classes, config)

    # Access various metrics
    print("\nDetailed Metrics:")
    print(f"Model Name: {model.metrics.model_name}")
    print(f"Total Training Time: {model.metrics.total_training_time:.2f}s")
    print(f"Number of Epochs: {len(model.metrics.epoch_times)}")
    print(f"Average Epoch Time: {sum(model.metrics.epoch_times)/len(model.metrics.epoch_times):.2f}s")
    print(f"Number of Loss Recordings: {len(model.metrics.training_losses)}")

    if model.metrics.training_losses:
        print(f"Average Training Loss: {sum(model.metrics.training_losses)/len(model.metrics.training_losses):.3f}")
        print(f"Final Training Loss: {model.metrics.training_losses[-1]:.3f}")

    print(f"\nTest Accuracy: {model.metrics.test_accuracy:.2f}%")
    print("\nPer-Class Accuracies:")
    for classname, acc in model.metrics.class_accuracies.items():
        print(f"  {classname:10s}: {acc:5.1f}%")


def example_5_load_and_evaluate():
    """Example 5: Load a saved model and evaluate."""
    print("\n" + "="*60)
    print("Example 5: Load and Evaluate Saved Model")
    print("="*60)

    # First, train and save a model
    config = TrainingConfig(epochs=1, batch_size=4)
    trainloader, testloader, classes = prepare_data(config)
    criterion = nn.CrossEntropyLoss()

    model = OriginalNet()
    model.train_model(trainloader, config, criterion)
    save_path = './data/example_saved_model.pth'
    model.save_model(save_path)

    # Now load it in a new instance
    print("\nLoading saved model...")
    loaded_model = OriginalNet()
    loaded_model.load_state_dict(torch.load(save_path, weights_only=True))

    # Evaluate the loaded model
    print("Evaluating loaded model...")
    loaded_model.evaluate_model(testloader, classes, config)


if __name__ == "__main__":
    import sys

    examples = {
        '1': example_1_train_single_model,
        '2': example_2_compare_models,
        '3': example_3_custom_model,
        '4': example_4_access_metrics,
        '5': example_5_load_and_evaluate,
    }

    print("\n" + "#"*60)
    print("CNN Training Framework - Usage Examples")
    print("#"*60)
    print("\nAvailable examples:")
    print("  1. Train a single model with custom configuration")
    print("  2. Compare multiple models")
    print("  3. Create and train a custom model")
    print("  4. Access and analyze metrics")
    print("  5. Load and evaluate a saved model")
    print("  all. Run all examples")

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nEnter example number (1-5 or 'all'): ").strip()

    if choice == 'all':
        for example_func in examples.values():
            example_func()
    elif choice in examples:
        examples[choice]()
    else:
        print(f"Invalid choice: {choice}")
        print("Usage: python example_usage.py [1-5|all]")
