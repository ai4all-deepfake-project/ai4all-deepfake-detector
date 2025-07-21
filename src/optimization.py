# src/optimization.py
"""
This module is responsible for setting up the training components such as:
- Optimizer
- Loss function
- Learning rate scheduler

TODO:

1. Define get_loss_function()
   - Returns appropriate loss function for classification (e.g., BCEWithLogitsLoss or CrossEntropyLoss).

2. Define get_optimizer(model, lr, weight_decay)
   - Returns optimizer (e.g., AdamW, SGD) initialized with model parameters.

3. Define get_scheduler(optimizer)
   - Optional: return LR scheduler (e.g., StepLR, CosineAnnealingLR) if used.

Keep this file focused on components required to optimize the model during training.
"""


######################################################################################
#                                     TESTS
######################################################################################
import pytest