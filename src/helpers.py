## src/helpers.py

"""
This module contains lightweight, reusable utilities to support training, evaluation, and reproducibility.

TODO:

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
