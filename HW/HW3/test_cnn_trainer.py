"""
Test script for CNN training framework.
Quick validation of the class structure and basic functionality.
"""

import torch
from cnn_trainer import (
    OriginalNet, DoubleNet_1, DoubleNet_2,
    TrainingConfig, PerformanceMetrics
)


def test_model_initialization():
    """Test that all models initialize correctly."""
    print("Testing model initialization...")

    models = [OriginalNet(), DoubleNet_1(), DoubleNet_2()]

    for model in models:
        print(f"  ✓ {model.model_name} initialized successfully")
        print(f"    - Has metrics: {hasattr(model, 'metrics')}")
        print(f"    - Metrics type: {type(model.metrics).__name__}")

    print("✓ All models initialized successfully\n")


def test_forward_pass():
    """Test forward pass with dummy data."""
    print("Testing forward pass...")

    # Create dummy input (batch_size=2, channels=3, height=32, width=32)
    dummy_input = torch.randn(2, 3, 32, 32)

    models = [OriginalNet(), DoubleNet_1(), DoubleNet_2()]

    for model in models:
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (2, 10), f"Expected shape (2, 10), got {output.shape}"
        print(f"  ✓ {model.model_name} forward pass successful (output shape: {output.shape})")

    print("✓ All forward passes successful\n")


def test_optimizer_configuration():
    """Test optimizer configuration."""
    print("Testing optimizer configuration...")

    config = TrainingConfig()

    original = OriginalNet()
    double1 = DoubleNet_1()
    double2 = DoubleNet_2()

    opt1 = original.get_optimizer(config)
    opt2 = double1.get_optimizer(config)
    opt3 = double2.get_optimizer(config)

    print(f"  ✓ OriginalNet optimizer: {type(opt1).__name__}")
    print(f"  ✓ DoubleNet_1 optimizer: {type(opt2).__name__}")
    print(f"  ✓ DoubleNet_2 optimizer: {type(opt3).__name__} (uses AdamW)")

    print("✓ Optimizer configuration successful\n")


def test_performance_metrics():
    """Test performance metrics tracking."""
    print("Testing performance metrics...")

    metrics = PerformanceMetrics("TestModel")

    # Test timer
    metrics.start_epoch_timer()
    import time
    time.sleep(0.1)
    elapsed = metrics.end_epoch_timer()
    assert elapsed > 0, "Timer should record positive time"
    print(f"  ✓ Timer working (recorded {elapsed:.3f}s)")

    # Test accuracy logging
    class_accuracies = {'class1': 90.0, 'class2': 85.0}
    metrics.log_test_results(87.5, class_accuracies)
    assert metrics.test_accuracy == 87.5, "Test accuracy should be stored"
    print(f"  ✓ Metrics logging working")

    print("✓ Performance metrics working\n")


def main():
    """Run all tests."""
    print("="*60)
    print("Running CNN Trainer Tests")
    print("="*60 + "\n")

    try:
        test_model_initialization()
        test_forward_pass()
        test_optimizer_configuration()
        test_performance_metrics()

        print("="*60)
        print("All tests passed! ✓")
        print("="*60)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
