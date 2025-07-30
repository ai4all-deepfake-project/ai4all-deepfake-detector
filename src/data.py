# data.py
import os 
import glob 
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Step 1: Load all image paths and label them
def load_image_paths(base_dir="frames"):
    data = [] # will hold tuples: (image_path, label)
    label_map = {"Real": 0, "Fake": 1}

    for label_name, label in label_map.items():
        folder = os.path.join(base_dir, label_name) # e.g frames/Real or frames/Fake
        image_paths = glob.glob(os.path.join(folder, "*.jpg")) # list of all .jpg files in the folder
        for path in image_paths:
            data.append((path, label)) # append (image_oath, label) to data list

# Step 2/3: Shuffle and split data into train, val test sets
# def split_data(data, seed=42, train_ratio=0.8, val_ratio=0.1):
#     random.seed(seed) # set seed
#     random.shuffle(data) # shuffle data list in-place

#     total = len(data) # total num of samples
#     train_end = int(train_ratio * total) # index cuttoff for train split
#     val_end = int((train_ratio, + val_ratio) * total) # index cutoff for val split

#     train = data[:train_end] # first chunk = train
#     val = data[train_end:val_end] # next chunk = val
#     test = data[val_end:] # remaining = test

#     return train, val, test

def stratified_split(data, seed=42, train_ratio=0.8, val_ratio=0.1):
    paths = [item[0] for item in data]
    labels = [item[1] for item in data]

    # First split: train vs temp (val + test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=(1 - train_ratio), stratify=labels, random_state=seed
    )

    # Calculate relative val/test split
    val_size = val_ratio / (1 - train_ratio)

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=(1 - val_size), stratify=temp_labels, random_state=seed
    )

    # Zip them back into (path, label) tuples
    train = list(zip(train_paths, train_labels))
    val = list(zip(val_paths, val_labels))
    test = list(zip(test_paths, test_labels))

    return train, val, test


# Step 4:  Create custom Dataset class for loading frames
class DeepFakeFrameDataset(Dataset):
    def __init__(self, data, transform=None):
        '''
        Args:
           data: list of tuples (image_path, label)
           transform: optional torchvision transforms
        '''
        self.data = data
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(), # Converts [0, 255] to [0.0, 1.0]
        ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, label
    
# Step 5: Create DotaLoaders
def create_dataloaders(train_data, val_data, test_data, batch_size=32):
    train_dataset = DeepFakeFrameDataset(train_data)
    val_dataset = DeepFakeFrameDataset(val_data)
    test_dataset = DeepFakeFrameDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

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
if __name__ == "__main__":
    data = load_image_paths("frames")
    print(f"Total samples: {len(data)}")

    train, val, test = stratified_split(data)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    train_loader, val_loader, test_loader = create_dataloaders(train, val, test)

    for batch_images, batch_labels in train_loader:
        print(f"Batch shape: {batch_images.shape}, Labels: {batch_labels}")
        break  # Just show one batch
