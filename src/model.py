# src/model.py
"""
This module defines the deep learning model architecture used for detecting deepfakes.

TODO:

1. Define the model architecture
   - Use a custom CNN or a pretrained model (e.g., ResNet18).
   - Modify the final layer for binary classification.

2. Add get_model() function
   - Accepts parameters like model_name, num_classes, pretrained.
   - Returns a compiled model ready for training.

3. Inline test:
   - Write a test function at the bottom of this script.
   - Validate model instantiation and forward pass.
   - Check that the output shape matches the expected (e.g., [batch_size, 2]).
   - Test runs only if the script is executed directly.

Note: Keep model.py focused on architecture and inline testing. Do not include training or evaluation logic here.
"""

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, EfficientNet_B0_Weights, resnet18, efficientnet_b0

def get_model(model_name="efficientnet_b0", num_classes=2, pretrained=True):
    '''
    Load and return a deepfake detection model.

    Args:
      model_name (str): model architecture to use (default: "efficientnet_b0")
      num_classes (int): number of output classes (default: 2)
      pretrained (bool): whether to load pretained weights (ImageNet)
    '''
    model_name = model_name.lower() # Get model name
    
    if model_name == "resnet18":
        # Load ResNet18 with optional pretrained ImageNet weights
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)

        # Replace final fully connected layer for binary classification
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet_b0":
        # Load EfficientNet-B0 with optional pretrained ImageNet weights
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)

        # Replace final fully connected layer for binary classification
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    
    else:
        # Raise error if an unsupported model name is passed
        raise ValueError(f":Unsupported model name: {model_name}. Choose 'resnet18' or 'efficientnet_b0'.")
    
    return model
######################################################################################
#                                     TESTS
######################################################################################
import pytest

def test_model_output_shape():
    '''
    Test whether both model architectures return correct output shape
    '''

    for model_name in ["resnet18", "efficientnet_b0"]:
        model = get_model(model_name=model_name)
        dummy_input = torch.randn(4, 3, 224, 224)
        output = model(dummy_input)
        assert output.shape == (4, 2), f"{model_name} output shape failed: got {output.shape}"