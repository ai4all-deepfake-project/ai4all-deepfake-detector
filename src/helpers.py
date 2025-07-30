## src/helpers.py

"""
This module contains lightweight, reusable utilities to support training, evaluation, and reproducibility.

1. set_seed(seed: int)
   - Ensure reproducibility across NumPy, Python, and PyTorch.
   - Keeps experiments consistent for debugging and comparison.

2. get_device()
   - Returns 'cuda' if GPU is available, else 'cpu'.
   - Centralizes device detection logic so it's not repeated in training scripts.

3. calculate_metrics(y_true, y_pred)
   - Returns accuracy, precision, recall, and F1-score using sklearn.
   - Can be reused in training, validation, or final evaluation phases.

4. print_metrics(metrics: dict)
   - Nicely formats and prints evaluation metrics.
   - Keeps output clean and consistent across runs.
"""
import numpy as np
import torch
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def set_seed(seed: int):
   random.seed(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   if torch.cuda.is_available():
       torch.cuda.manual_seed_all(seed)
       
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   
def get_device():
   return 'cuda' if torch.cuda.is_available() else 'cpu'

def calculate_metrics(y_true, y_pred):
   accuracy = accuracy_score(y_true, y_pred)
   precision = precision_score(y_true, y_pred, average='weighted', zero_division=0) # Change to other average methods if needed
   recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
   f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
   
   return {
       'accuracy': accuracy,
       'precision': precision,
       'recall': recall,
       'f1_score': f1
   }
   
def print_metrics(metrics: dict):
   print("Evaluation Metrics:")
   for key, value in metrics.items():
       print(f"{key.capitalize()}: {value:.4f}")
   print("\n")
