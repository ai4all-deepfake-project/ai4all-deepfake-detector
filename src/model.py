# src/model.py
"""
Simplified DeepFake Detection Model following transfer learning pattern
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


def get_model_transfer_learning(model_name="resnet18", n_classes=2):
    """
    Create a transfer learning model with frozen backbone and custom classifier.
    
    Args:
        model_name: Name of the model architecture (supports 'resnet18')
        n_classes: Number of output classes
    
    Returns:
        model: PyTorch model with frozen backbone and new classifier head
    """
    
    # Get the requested architecture
    if model_name == "resnet18":
        model_transfer = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    else:
        raise ValueError(f"Model {model_name} is not supported. Only 'resnet18' is available.")

    # Freeze all backbone parameters
    for param in model_transfer.parameters():
        param.requires_grad = False

    # Replace the fully connected layer for our number of classes
    num_ftrs = model_transfer.fc.in_features
    model_transfer.fc = nn.Linear(num_ftrs, n_classes)

    return model_transfer


######################################################################################
#                                     TESTS
######################################################################################
import pytest


def test_get_model_transfer_learning():
    """Test the transfer learning model creation and forward pass"""
    
    model = get_model_transfer_learning(n_classes=2)
    
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
    
    backbone_params_frozen = True
    for name, param in model.named_parameters():
        if 'fc' not in name:  # backbone params
            if param.requires_grad:
                backbone_params_frozen = False
                break
    
    assert backbone_params_frozen, "Backbone parameters should be frozen"
    
    classifier_params_trainable = True
    for name, param in model.named_parameters():
        if 'fc' in name:
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
        get_model_transfer_learning(model_name="efficientnet_b0", n_classes=2)
