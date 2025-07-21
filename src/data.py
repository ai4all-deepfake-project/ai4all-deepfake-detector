# data.py

"""
TODO:
1. Load dataset:
   - Read file paths and labels from the SDFVD 2.0 directory or metadata file (e.g., CSV or JSON).
   - Label: 0 = Real, 1 = Fake

2. Organize data:
   - Group or tag augmented samples if needed (optional).
   - Ensure no data leakage (e.g., same original video in train and test).

3. Implement dataset splitting:
   - Stratified train/val/test split (e.g., 80/10/10)
   - Use a random seed for reproducibility
   - Optionally save split lists (e.g., as JSON or CSV)

4. Create PyTorch-style Dataset class:
   - Load images or frames
   - Return (image_path, label) or (image_tensor, label) depending on design

5. Return DataLoaders:
   - Basic batching (no transforms)
   - Shuffling only for training set
"""




######################################################################################
#                                     TESTS
######################################################################################