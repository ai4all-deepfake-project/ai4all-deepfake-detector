# src/optimization.py
"""
This module is responsible for setting up the training components such as:
- Optimizer
- Loss function
- Learning rate scheduler
Keep this file focused on components required to optimize the model during training.
"""

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

def get_loss_function():
   """ 
   Returns the loss function for binary classification tasks.
   """
   return BCEWithLogitsLoss()
 
def get_optimizer(model, learning_rate=1e-3, weight_decay=0.01):
   """
   Returns an AdamW optimizer initialized with the model parameters.
   
   Args:
      model: The model to optimize.
      learning_rate (float): Learning rate for the optimizer.
      weight_decay (float): Weight decay for regularization.
   
   Returns:
      torch.optim.Optimizer: Configured optimizer.
   """
   return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def get_scheduler(optimizer, scheduler_type='step', step_size=10, gamma=0.1, T_max=50):
    """
    Returns a learning rate scheduler.
    
    Args:
        optimizer (torch.optim.Optimizer): Optimizer to schedule
        scheduler_type (str): 'step' or 'cosine'
        step_size (int): Step size for StepLR
        gamma (float): Multiplicative factor for StepLR
        T_max (int): Max iterations for CosineAnnealingLR
    
    Returns:
        torch.optim.lr_scheduler._LRScheduler: Configured scheduler
    """
    if scheduler_type == 'step':
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=T_max)
    else:
        raise ValueError("Invalid scheduler_type.")

######################################################################################
#                                     TESTS
######################################################################################
if __name__ == "__main__":
    print("Testing get_loss_function()...")
    loss_fn = get_loss_function()
    print(f"Loss function created: {loss_fn.__class__.__name__}\n")

    print("Testing get_optimizer()...")
    dummy_model = torch.nn.Linear(10, 1)
    optimizer = get_optimizer(dummy_model, learning_rate=0.01, weight_decay=0.001)
    print(f"Optimizer created: {type(optimizer).__name__}")
    print(f" - Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f" - Weight decay: {optimizer.param_groups[0]['weight_decay']}\n")

    print("Testing get_scheduler() [step]...")
    step_scheduler = get_scheduler(optimizer, scheduler_type='step', step_size=5, gamma=0.5)
    print(f"StepLR created: step_size={step_scheduler.step_size}, gamma={step_scheduler.gamma}\n")

    print("Testing get_scheduler() [cosine]...")
    cosine_scheduler = get_scheduler(optimizer, scheduler_type='cosine', T_max=25)
    print(f"CosineAnnealingLR created: T_max={cosine_scheduler.T_max}\n")

    try:
        print("Testing get_scheduler() with invalid scheduler_type...")
        get_scheduler(optimizer, scheduler_type='invalid')
    except ValueError as e:
        print(f"Caught expected exception: {e}")