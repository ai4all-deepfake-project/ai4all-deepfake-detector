# data.py
import os 
import glob 
import random
from sklearn.model_selection import train_test_split

# Step 1: Load all image paths and label them
def load_image_paths(base_dir="frames"):
    data = [] # will hold tuples: (image_path, label)
    label_map = {"Real": 0, "Fake": 1}

    for label_name, label in label_map.items():
        folder = os.path.join(base_dir, label_name) # e.g frames/Real or frames/Fake
        image_paths = glob.glob(os.path.join(folder, "*.jpg")) # list of all .jpg files in the folder
        for path in image_paths:
            data.append((path, label)) # append (image_oath, label) to data list

# Step 2: Shuffle and split data into train, val test sets
def split_data(data, seed=42, train_ratio=0.8, val_ratio=0.1):
    random.seed(seed) # set seed
    random.shuffle(data) # shuffle data list in-place

    total = len(data) # total num of samples
    train_end = int(train_ratio * total) # index cuttoff for train split
    val_end = int((train_ratio, + val_ratio) * total) # index cutoff for val split

    train = data[:train_end] # first chunk = train
    val = data[train_end:val_end] # next chunk = val
    test = data[val_end:] # remaining = test

    return train, val, test


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