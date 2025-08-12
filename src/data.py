# src/data.py
"""
Optimized data handling for SDFVD 2.0 deepfake dataset.
Uses helpers.py functions instead of nested functions for better modularity.
"""

import os
import glob
import cv2
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

# Import from helpers.py instead of defining locally
from src.helpers import (
    parse_original_from_video_stem,
    label_from_name, 
    dir_has_videos,
    find_label_dirs,
    groups_to_indices,
    safe_group_stratified_split
)

# ========================= CONSTANTS & TRANSFORMS =========================

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_TRAIN_TFMS = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
    transforms.RandomCrop(224), transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(), transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
])

_EVAL_TFMS = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
])

# ========================== FRAME EXTRACTION =============================

def extract_frames_from_folder(input_folder: str, output_folder: str, frame_interval=5, 
                               video_exts=(".mp4", ".mov", ".mkv", ".avi"), limit_per_video=None):
    """Extract frames from all videos in folder."""
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        video_files = [f for f in os.listdir(input_folder) 
                      if any(f.lower().endswith(e) for e in video_exts)]
    except FileNotFoundError:
        print(f"[warn] Missing folder: {input_folder}")
        return
    
    if not video_files:
        print(f"[warn] No videos in: {input_folder}")
        return
    
    for video_file in video_files:
        in_path = os.path.join(input_folder, video_file)
        stem = os.path.splitext(video_file)[0]
        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            continue
        
        frame_idx = saved = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            if frame_idx % frame_interval == 0:
                out_path = os.path.join(output_folder, f"{stem}_frame_{saved:03d}.jpg")
                cv2.imwrite(out_path, frame)
                saved += 1
                if limit_per_video and saved >= limit_per_video: break
            frame_idx += 1
        cap.release()
        print(f"[ok] {video_file}: saved {saved} frames")

def extract_all_frames_sdfvd(dataset_root: str, output_root="frames", frame_interval=5, 
                             limit_per_video=None, search_depth=2):
    """Extract frames from SDFVD dataset structure using helpers.py functions."""
    os.makedirs(output_root, exist_ok=True)
    found = find_label_dirs(dataset_root, max_depth=search_depth)
    
    if not found["real"] and not found["fake"]:
        print(f"[warn] No real/fake folders found under: {dataset_root}")
        return
    
    for real_dir in found["real"]:
        extract_frames_from_folder(real_dir, os.path.join(output_root, "Real"), 
                                 frame_interval, limit_per_video=limit_per_video)
    for fake_dir in found["fake"]:
        extract_frames_from_folder(fake_dir, os.path.join(output_root, "Fake"), 
                                 frame_interval, limit_per_video=limit_per_video)

# ============================ DATA LOADING ===============================

def load_image_paths_with_groups(base_dir="frames", extensions=(".jpg", ".jpeg", ".png")):
    """Load frame paths with group information using helpers.py parsing."""
    items = []
    for sub, dir_label in [("Real", 0), ("Fake", 1)]:
        folder = os.path.join(base_dir, sub)
        if not os.path.isdir(folder): continue
        for ext in extensions:
            for path in glob.glob(os.path.join(folder, f"*{ext}")):
                stem = Path(path).stem
                original_id, label_from_name = parse_original_from_video_stem(stem)
                use_label = dir_label if label_from_name == -1 else label_from_name
                items.append((path, use_label, original_id))
    return items

def stratified_group_split(data, seed=42, train_ratio=0.8, val_ratio=0.1):
    """Group-aware split using helpers.py functions."""
    if not data: raise ValueError("No data to split.")
    
    # Group mapping
    group_to_label = {}
    for _, lbl, grp in data:
        group_to_label.setdefault(grp, lbl)
    
    groups = list(group_to_label.keys())
    y = [group_to_label[g] for g in groups]
    
    # Use helpers.py function for stratified splitting
    train_groups, val_groups, test_groups = safe_group_stratified_split(
        groups, y, train_ratio, val_ratio, seed
    )
    
    # Map back to items using helpers.py function
    train_idx = groups_to_indices(data, train_groups)
    val_idx = groups_to_indices(data, val_groups)
    test_idx = groups_to_indices(data, test_groups)
    
    return ([(data[i][0], data[i][1]) for i in train_idx],
            [(data[i][0], data[i][1]) for i in val_idx],
            [(data[i][0], data[i][1]) for i in test_idx])

# ========================== DATASET & LOADERS ============================

class DeepFakeFrameDataset(Dataset):
    def __init__(self, data: List[Tuple[str, int]], transform=None):
        self.data = data
        self.transform = transform or _EVAL_TFMS
    
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        path, label = self.data[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

def create_dataloaders(train_data, val_data, test_data, batch_size=32, num_workers=0, pin_memory=False):
    """Create train/val/test dataloaders."""
    datasets = [
        DeepFakeFrameDataset(train_data, _TRAIN_TFMS),
        DeepFakeFrameDataset(val_data, _EVAL_TFMS),
        DeepFakeFrameDataset(test_data, _EVAL_TFMS)
    ]
    
    return tuple(DataLoader(ds, batch_size=batch_size, shuffle=(i == 0), 
                           num_workers=num_workers, pin_memory=pin_memory) 
                 for i, ds in enumerate(datasets))

def get_data_loaders(batch_size=32, valid_size=0.1, seed=42, num_workers=0, base_dir="frames", limit=None):
    """Main interface for getting dataloaders."""
    data_wg = load_image_paths_with_groups(base_dir=base_dir)
    if limit and limit > 0:
        random.Random(seed).shuffle(data_wg)
        data_wg = data_wg[:limit]
    
    train, val, test = stratified_group_split(data_wg, seed=seed, train_ratio=0.8, val_ratio=valid_size)
    tl, vl, te = create_dataloaders(train, val, test, batch_size=batch_size, num_workers=num_workers)
    return {"train": tl, "valid": vl, "test": te}

def get_dataloaders(config: Dict):
    """Config-friendly wrapper for train.py usage."""
    data_wg = load_image_paths_with_groups(config.get("base_dir", "frames"))
    if config.get("limit"):
        random.Random(config.get("seed", 42)).shuffle(data_wg)
        data_wg = data_wg[:int(config.get("limit"))]
    
    train, val, test = stratified_group_split(
        data_wg, 
        seed=config.get("seed", 42),
        train_ratio=config.get("train_ratio", 0.8),
        val_ratio=config.get("val_ratio", 0.1)
    )
    return create_dataloaders(train, val, test, 
                             batch_size=config.get("batch_size", 32), 
                             num_workers=config.get("num_workers", 0))

def visualize_one_batch(data_loaders: Dict[str, DataLoader], max_n=5):
    """Visualize sample batch from training data."""
    import matplotlib.pyplot as plt
    
    if "train" not in data_loaders:
        raise KeyError("data_loaders must contain a 'train' DataLoader")
    
    images, labels = next(iter(data_loaders["train"]))
    
    # Denormalize for display
    inv = transforms.Compose([
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1/s for s in _IMAGENET_STD]),
        transforms.Normalize(mean=[-m for m in _IMAGENET_MEAN], std=[1.0, 1.0, 1.0])
    ])
    images = inv(images).clamp(0, 1).permute(0, 2, 3, 1).cpu()
    
    n = min(max_n, images.shape[0])
    fig = plt.figure(figsize=(3*n, 3))
    for i in range(n):
        ax = fig.add_subplot(1, n, i + 1, xticks=[], yticks=[])
        ax.imshow(images[i].numpy())
        ax.set_title(["Real", "Fake"][int(labels[i])])
    plt.tight_layout()
    plt.show()

# =============================================================================
#                                   TESTS
# =============================================================================

import pytest

def _make_tiny_sdfvd_video(path: str, nframes: int = 8, w: int = 64, h: int = 48):
    """Create a small synthetic SDFVD-style video for testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, 8.0, (w, h))
    assert vw.isOpened()
    for i in range(nframes):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(img, (5 + 3 * i, 5), (15 + 3 * i, 15), (0, 255, 0), -1)
        img[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)
        vw.write(img)
    vw.release()
    assert os.path.exists(path)

def test_extraction_and_group_split(tmp_path):
    """Test the complete pipeline: extraction → grouping → splitting."""
    # Create SDFVD-like tree with fuzzy-named dirs
    ds = tmp_path / "SDFVD2.0"
    (ds / "SDFVD2.0_real").mkdir(parents=True)
    (ds / "SDFVD2.0_fake").mkdir(parents=True)
    
    # Two originals each, multiple augment indices
    vids = [
        ("real", "origA", [0, 1]),
        ("real", "origB", [0]),
        ("fake", "origX", [0, 1]),
        ("fake", "origY", [0]),
    ]
    for prefix, orig, aug_ids in vids:
        for a in aug_ids:
            _make_tiny_sdfvd_video(str(ds / f"SDFVD2.0_{prefix}" / f"{prefix}_{orig}_aug_{a}.mp4"))
    
    out = tmp_path / "frames"
    extract_all_frames_sdfvd(str(ds), output_root=str(out), frame_interval=2, limit_per_video=2)
    
    items = load_image_paths_with_groups(base_dir=str(out))
    assert len(items) > 0
    
    # Ensure both labels present
    labels = {lbl for _, lbl, _ in items}
    assert labels == {0, 1}
    
    # Ensure no group leakage across splits
    train, val, test = stratified_group_split(items, seed=7, train_ratio=0.8, val_ratio=0.1)
    
    # Extract groups from items and splits to verify no leakage
    p2g = {p: g for p, _, g in items}
    g_train = {p2g[p] for p, _ in train}
    g_val = {p2g[p] for p, _ in val}
    g_test = {p2g[p] for p, _ in test}
    
    assert g_train.isdisjoint(g_val)
    assert g_train.isdisjoint(g_test)
    assert g_val.isdisjoint(g_test)

def test_loaders_shapes(tmp_path):
    """Test dataloader creation and tensor shapes."""
    # Minimal Real/Fake frame dirs with small fake images
    base = tmp_path / "frames"
    (base / "Real").mkdir(parents=True)
    (base / "Fake").mkdir(parents=True)
    
    arr = np.zeros((40, 40, 3), dtype=np.uint8)
    for i in range(6):
        cv2.imwrite(str(base / "Real" / f"real_origA_aug_0_frame_{i:03d}.jpg"), arr)
        cv2.imwrite(str(base / "Fake" / f"fake_origX_aug_1_frame_{i:03d}.jpg"), arr)
    
    loaders = get_data_loaders(batch_size=4, valid_size=0.2, seed=3, num_workers=0, base_dir=str(base))
    xb, yb = next(iter(loaders["train"]))
    assert xb.shape[1:] == (3, 224, 224)
    assert yb.ndim == 1

def test_helpers_integration():
    """Test that helpers.py functions work correctly."""
    # Test parse_original_from_video_stem from helpers
    assert parse_original_from_video_stem("real_someOriginal_aug_4_frame_001") == ("someOriginal", 0)
    assert parse_original_from_video_stem("fake_foo_aug_0") == ("foo", 1)
    assert parse_original_from_video_stem("unknown_stem") == ("unknown_stem", -1)
    
    # Test label_from_name from helpers
    assert label_from_name("SDFVD2.0_real") == "real"
    assert label_from_name("fake_videos") == "fake"
    assert label_from_name("random_dir") is None