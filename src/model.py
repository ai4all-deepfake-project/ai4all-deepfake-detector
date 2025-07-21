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


######################################################################################
#                                     TESTS
######################################################################################
import pytest