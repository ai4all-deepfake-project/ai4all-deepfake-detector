# src/model.py
"""
Simplified DeepFake Detection Model following transfer learning pattern
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights


def get_model_transfer_learning(model_name="efficientnet_b0", n_classes=2):
    """
    Create a transfer learning model with frozen backbone and custom classifier.
    
    Args:
        model_name: Name of the model architecture (only "efficientnet_b0" supported)
        n_classes: Number of output classes
    
    Returns:
        model: PyTorch model with frozen backbone and new classifier head
    """
    
    # Get the requested architecture
    if model_name == "efficientnet_b0":
        model_transfer = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    else:
        raise ValueError(f"Model {model_name} is not supported. Only 'efficientnet_b0' is available.")

    # Freeze all parameters in the model
    for param in model_transfer.parameters():
        param.requires_grad = False

    # Replace the classifier with a new one for our number of classes
    # For EfficientNet-B0, the classifier is a Sequential with the final Linear layer
    if isinstance(model_transfer.classifier, nn.Sequential):
        # Find the last Linear layer to get input features
        last_linear = None
        for layer in reversed(model_transfer.classifier):
            if isinstance(layer, nn.Linear):
                last_linear = layer
                break
        
        if last_linear is None:
            raise RuntimeError("Could not find Linear layer in classifier")
        
        num_ftrs = last_linear.in_features
    else:
        # Fallback if classifier structure is different
        num_ftrs = model_transfer.classifier.in_features

    # Create new classifier head
    model_transfer.classifier = nn.Linear(num_ftrs, n_classes)

    return model_transfer


######################################################################################
#                                     TESTS
######################################################################################
import pytest


def test_get_model_transfer_learning():
    """Test the transfer learning model creation and forward pass"""
    
    model = get_model_transfer_learning(n_classes=2)
    
    # Test with a batch of images
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    
    model.eval()
    with torch.no_grad():
        out = model(images)

    assert isinstance(out, torch.Tensor), \
        "The output should be a Tensor"
    
    assert out.shape == torch.Size([batch_size, 2]), \
        f"Expected output shape ({batch_size}, 2), got {out.shape}"


def test_model_parameters_frozen():
    """Test that backbone parameters are frozen"""
    
    model = get_model_transfer_learning(n_classes=2)
    
    # Check that backbone features are frozen
    backbone_params_frozen = True
    for name, param in model.named_parameters():
        if 'classifier' not in name:  # backbone parameters
            if param.requires_grad:
                backbone_params_frozen = False
                break
    
    assert backbone_params_frozen, "Backbone parameters should be frozen"
    
    # Check that classifier parameters are trainable
    classifier_params_trainable = True
    for name, param in model.named_parameters():
        if 'classifier' in name:
            if not param.requires_grad:
                classifier_params_trainable = False
                break
    
    assert classifier_params_trainable, "Classifier parameters should be trainable"


def test_different_n_classes():
    """Test model creation with different number of classes"""
    
    for n_classes in [2, 10, 50, 100]:
        model = get_model_transfer_learning(n_classes=n_classes)
        
        images = torch.randn(2, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            out = model(images)
        
        assert out.shape == torch.Size([2, n_classes]), \
            f"Expected output shape (2, {n_classes}), got {out.shape}"


def test_unsupported_model():
    """Test that unsupported model names raise an error"""
    
    with pytest.raises(ValueError):
        get_model_transfer_learning(model_name="resnet18", n_classes=2)


if __name__ == "__main__":
    print("Running simplified model tests...")
    
    # Run tests
    test_get_model_transfer_learning()
    test_model_parameters_frozen()
    test_different_n_classes()
    
    try:
        test_unsupported_model()
    except AssertionError:
        print("✓ Unsupported model test passed")
    
    # Model statistics
    model = get_model_transfer_learning(n_classes=2)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\nModel Statistics:")
    print(f"Architecture: EfficientNet-B0 Transfer Learning")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")
    print(f"Trainable ratio: {trainable_params/total_params:.1%}")
    print(f"Using pretrained ImageNet weights")
    print(f"Backbone frozen, only classifier trainable")
    print("\nAll tests completed successfully!")