"""
CNN Training Framework with Performance Monitoring
Author: Hochan Son
Course: STATS 413 HW3

This module provides a structured framework for training CNN models on CIFAR-10
with built-in performance metrics monitoring and logging to Weights & Biases.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass, field
import wandb


@dataclass
class TrainingConfig:
    """Configuration class for training hyperparameters."""
    learning_rate: float = 0.001
    momentum: float = 0.9
    epochs: int = 20
    batch_size: int = 4
    num_workers: int = 2
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    log_interval: int = 2000


class PerformanceMetrics:
    """Base class for monitoring and logging performance metrics during training."""

    def __init__(self, model_name: str):
        """
        Initialize performance metrics tracker.

        Args:
            model_name: Name identifier for the model
        """
        self.model_name = model_name
        self.training_losses: List[float] = []
        self.epoch_times: List[float] = []
        self.test_accuracy: Optional[float] = None
        self.class_accuracies: Dict[str, float] = {}
        self.start_time: Optional[float] = None
        self.total_training_time: float = 0.0

    def start_epoch_timer(self):
        """Start timer for epoch duration tracking."""
        self.start_time = time.time()

    def end_epoch_timer(self) -> float:
        """End timer and record epoch duration."""
        if self.start_time is None:
            return 0.0
        elapsed = time.time() - self.start_time
        self.epoch_times.append(elapsed)
        self.total_training_time += elapsed
        self.start_time = None
        return elapsed

    def log_training_loss(self, loss: float, step: int):
        """
        Log training loss.

        Args:
            loss: Loss value to log
            step: Current training step
        """
        self.training_losses.append(loss)
        if wandb.run is not None:
            wandb.log({f"{self.model_name}/training_loss": loss}, step=step)

    def log_test_results(self, overall_accuracy: float, class_accuracies: Dict[str, float]):
        """
        Log test evaluation results.

        Args:
            overall_accuracy: Overall test accuracy percentage
            class_accuracies: Per-class accuracy percentages
        """
        self.test_accuracy = overall_accuracy
        self.class_accuracies = class_accuracies

        if wandb.run is not None:
            wandb.log({
                f"{self.model_name}/test_overall_accuracy": overall_accuracy,
                f"{self.model_name}/total_training_time": self.total_training_time
            })

            for classname, accuracy in class_accuracies.items():
                wandb.log({f"{self.model_name}/test_accuracy_{classname}": accuracy})

    def print_summary(self):
        """Print a summary of all collected metrics."""
        print(f"\n{'='*60}")
        print(f"Performance Summary for {self.model_name}")
        print(f"{'='*60}")
        print(f"Total Training Time: {self.total_training_time:.2f} seconds")
        print(f"Average Epoch Time: {sum(self.epoch_times)/len(self.epoch_times):.2f} seconds")
        print(f"Overall Test Accuracy: {self.test_accuracy:.2f}%")
        print(f"\nPer-Class Accuracies:")
        for classname, accuracy in self.class_accuracies.items():
            print(f"  {classname:10s}: {accuracy:5.1f}%")
        print(f"{'='*60}\n")


class CNNModel(nn.Module):
    """Base class for CNN models with integrated performance monitoring."""

    def __init__(self, model_name: str):
        """
        Initialize CNN model.

        Args:
            model_name: Name identifier for the model
        """
        super().__init__()
        self.model_name = model_name
        self.metrics = PerformanceMetrics(model_name)

    def get_optimizer(self, config: TrainingConfig) -> optim.Optimizer:
        """
        Get optimizer for training. Can be overridden by subclasses.

        Args:
            config: Training configuration

        Returns:
            PyTorch optimizer instance
        """
        return optim.SGD(self.parameters(), lr=config.learning_rate, momentum=config.momentum)

    def train_model(self, trainloader, config: TrainingConfig, criterion):
        """
        Train the model on the training dataset.

        Args:
            trainloader: DataLoader for training data
            config: Training configuration
            criterion: Loss function
        """
        optimizer = self.get_optimizer(config)
        device = torch.device(config.device)
        self.to(device)

        print(f"\nTraining {self.model_name}...")
        print(f"{'='*60}")

        for epoch in range(config.epochs):
            self.metrics.start_epoch_timer()
            running_loss = 0.0

            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i % config.log_interval == (config.log_interval - 1):
                    avg_loss = running_loss / config.log_interval
                    step = epoch * len(trainloader) + i
                    print(f'[Epoch {epoch + 1:2d}, Batch {i + 1:5d}] loss: {avg_loss:.3f}')
                    self.metrics.log_training_loss(avg_loss, step)
                    running_loss = 0.0

            epoch_time = self.metrics.end_epoch_timer()
            print(f'Epoch {epoch + 1} completed in {epoch_time:.2f} seconds')

        print(f'Finished Training {self.model_name}')

    def evaluate_model(self, testloader, classes: Tuple[str, ...], config: TrainingConfig):
        """
        Evaluate the model on the test dataset.

        Args:
            testloader: DataLoader for test data
            classes: Tuple of class names
            config: Training configuration
        """
        device = torch.device(config.device)
        self.to(device)
        self.eval()

        print(f"\nEvaluating {self.model_name}...")

        # Overall accuracy
        correct = 0
        total = 0

        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        overall_accuracy = 100 * correct / total

        # Per-class accuracy
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = self(images)
                _, predictions = torch.max(outputs, 1)

                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

        class_accuracies = {
            classname: 100 * float(correct_pred[classname]) / total_pred[classname]
            for classname in classes
        }

        self.metrics.log_test_results(overall_accuracy, class_accuracies)
        self.metrics.print_summary()

    def save_model(self, path: str):
        """
        Save model state dictionary.

        Args:
            path: File path to save the model
        """
        torch.save(self.state_dict(), path)
        print(f"Model {self.model_name} saved to {path}")


class OriginalNet(CNNModel):
    """Original baseline CNN architecture."""

    def __init__(self):
        super().__init__(model_name="OriginalNet")
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DoubleNet_1(CNNModel):
    """CNN with doubled channel dimensions."""

    def __init__(self):
        super().__init__(model_name="DoubleNet_1")
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 240)
        self.fc2 = nn.Linear(240, 168)
        self.fc3 = nn.Linear(168, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DoubleNet_2(CNNModel):
    """CNN with doubled channels and additional convolutional layer."""

    def __init__(self):
        super().__init__(model_name="DoubleNet_2")
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 5)
        # After conv1 (32->28) + pool (28->14)
        # After conv2 (14->10) + pool (10->5)
        # Shape is 32 * 5 * 5 = 800
        self.fc1 = nn.Linear(32 * 5 * 5, 240)
        self.fc2 = nn.Linear(240, 168)
        self.fc3 = nn.Linear(168, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Note: conv3 is defined but not used in forward pass to match original notebook
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_optimizer(self, config: TrainingConfig) -> optim.Optimizer:
        """Override to use AdamW optimizer instead of SGD."""
        return optim.AdamW(self.parameters(), lr=config.learning_rate)


def prepare_data(config: TrainingConfig) -> Tuple[torch.utils.data.DataLoader,
                                                    torch.utils.data.DataLoader,
                                                    Tuple[str, ...]]:
    """
    Prepare CIFAR-10 data loaders and class names.

    Args:
        config: Training configuration

    Returns:
        Tuple of (trainloader, testloader, classes)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


def main():
    """Main function to orchestrate model training and evaluation."""

    # Initialize configuration
    config = TrainingConfig(
        learning_rate=0.001,
        momentum=0.9,
        epochs=20,
        batch_size=4,
        num_workers=2
    )

    # Initialize wandb
    wandb.init(
        project="STATS413_HW3",
        entity="ohsono-ucla",
        config={
            "learning_rate": config.learning_rate,
            "momentum": config.momentum,
            "epochs": config.epochs,
            "batch_size": config.batch_size
        }
    )

    # Check device
    device = torch.device(config.device)
    print(f"Using device: {device}")

    # Prepare data
    print("Preparing data...")
    trainloader, testloader, classes = prepare_data(config)
    print(f"Data loaded successfully. Training samples: {len(trainloader.dataset)}, "
          f"Test samples: {len(testloader.dataset)}")

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Initialize models
    models = [
        OriginalNet(),
        DoubleNet_1(),
        DoubleNet_2()
    ]

    # Train and evaluate each model
    for model in models:
        print(f"\n{'#'*60}")
        print(f"Processing Model: {model.model_name}")
        print(f"{'#'*60}")

        # Train
        model.train_model(trainloader, config, criterion)

        # Save model
        save_path = f'./data/cifar_{model.model_name}.pth'
        model.save_model(save_path)

        # Evaluate
        model.evaluate_model(testloader, classes, config)

    # Final summary
    print(f"\n{'#'*60}")
    print("All models trained and evaluated successfully!")
    print(f"{'#'*60}")

    # Close wandb
    wandb.finish()


if __name__ == "__main__":
    main()
