# src/data.py
"""
Optimized data handling for SDFVD 2.0 deepfake dataset.
- Extracts frames (optionally face-cropped to 224x224)
- Uses helpers.py for parsing and stratified group splits
- Clean train/val/test DataLoaders with ImageNet normalization
"""

import os
import glob
import cv2
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

# Import from helpers.py
from src.helpers import (
    parse_original_from_video_stem,
    label_from_name,
    dir_has_videos,
    find_label_dirs,
    groups_to_indices,
    safe_group_stratified_split,
)

# ========================= CONSTANTS & TRANSFORMS =========================

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_TRAIN_TFMS = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

_EVAL_TFMS = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

# ====================== FACE DETECTION (MTCNN → Haar) =====================

try:
    from facenet_pytorch import MTCNN
    _HAS_MTCNN = True
except Exception:
    _HAS_MTCNN = False

_HAAR = None
def _init_haar():
    """Initialize Haar cascade once."""
    global _HAAR
    if _HAAR is None:
        _HAAR = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        )

def crop_face(
    bgr_img: np.ndarray,
    target_size: int = 224,
    margin_ratio: float = 0.2,
) -> Optional[Image.Image]:
    """
    Return PIL.Image (target_size x target_size) of detected face, or None if not found.
    Tries MTCNN first (if installed), then Haar cascade.
    """
    h, w = bgr_img.shape[:2]

    # --- Try MTCNN ---
    if _HAS_MTCNN:
        if not hasattr(crop_face, "_mtcnn"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            crop_face._mtcnn = MTCNN(keep_all=False, device=device)
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        box, prob = crop_face._mtcnn.detect(pil)
        if box is not None:
            x1, y1, x2, y2 = box[0]
            w0, h0 = x2 - x1, y2 - y1
            side = max(w0, h0)
            m = margin_ratio * side
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            side = int(side + 2 * m)
            x1s = int(max(0, cx - side / 2))
            y1s = int(max(0, cy - side / 2))
            x2s = int(min(w, x1s + side))
            y2s = int(min(h, y1s + side))
            face = pil.crop((x1s, y1s, x2s, y2s)).resize((target_size, target_size), Image.BILINEAR)
            return face

    # --- Fallback: Haar ---
    _init_haar()
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    faces = _HAAR.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w0, h0 = max(faces, key=lambda f: f[2] * f[3])  # largest
        rgb = cv2.cvtColor(bgr_img[y:y+h0, x:x+w0], cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((target_size, target_size), Image.BILINEAR)
        return pil

    return None

def _save_face_or_fallback(frame_bgr: np.ndarray, out_path: str, target_size: int = 224):
    """Save a detected face crop; if none found, save a safe center square crop."""
    face_img = crop_face(frame_bgr, target_size=target_size)
    if face_img is None:
        h, w = frame_bgr.shape[:2]
        s = min(h, w)
        y0 = (h - s) // 2
        x0 = (w - s) // 2
        rgb = cv2.cvtColor(frame_bgr[y0:y0+s, x0:x0+s], cv2.COLOR_BGR2RGB)
        face_img = Image.fromarray(rgb).resize((target_size, target_size), Image.BILINEAR)
    face_img.save(out_path)

# ========================== FRAME EXTRACTION =============================

def extract_frames_from_folder(
    input_folder: str,
    output_folder: str,
    frame_interval: int = 5,
    video_exts: Tuple[str, ...] = (".mp4", ".mov", ".mkv", ".avi"),
    limit_per_video: Optional[int] = None,
    use_face_detection: bool = True,
):
    """Extract frames from all videos in folder (optionally face-cropped)."""
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
            print(f"[warn] Could not open video: {in_path}")
            continue

        frame_idx = 0
        saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % frame_interval == 0:
                out_path = os.path.join(output_folder, f"{stem}_frame_{saved:03d}.jpg")
                if use_face_detection:
                    _save_face_or_fallback(frame, out_path, target_size=224)
                else:
                    cv2.imwrite(out_path, frame)
                saved += 1
                if limit_per_video and saved >= limit_per_video:
                    break
            frame_idx += 1
        cap.release()
        print(f"[ok] {video_file}: saved {saved} frames → {output_folder}")

def extract_all_frames_sdfvd(
    dataset_root: str,
    output_root: str = "frames",
    frame_interval: int = 5,
    limit_per_video: Optional[int] = None,
    search_depth: int = 2,
    use_face_detection: bool = True,
):
    """Extract frames from SDFVD dataset using helpers.py discovery."""
    os.makedirs(output_root, exist_ok=True)
    found = find_label_dirs(dataset_root, max_depth=search_depth)

    if not found["real"] and not found["fake"]:
        print(f"[warn] No real/fake folders found under: {dataset_root}")
        return

    for real_dir in found["real"]:
        extract_frames_from_folder(
            real_dir,
            os.path.join(output_root, "Real"),
            frame_interval,
            limit_per_video=limit_per_video,
            use_face_detection=use_face_detection,
        )
    for fake_dir in found["fake"]:
        extract_frames_from_folder(
            fake_dir,
            os.path.join(output_root, "Fake"),
            frame_interval,
            limit_per_video=limit_per_video,
            use_face_detection=use_face_detection,
        )

# ============================ DATA LOADING ===============================

def load_image_paths_with_groups(
    base_dir: str = "frames",
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
):
    """Load frame paths with group information using helpers.py parsing."""
    items: List[Tuple[str, int, str]] = []
    for sub, dir_label in [("Real", 0), ("Fake", 1)]:
        folder = os.path.join(base_dir, sub)
        if not os.path.isdir(folder):
            continue
        for ext in extensions:
            for path in glob.glob(os.path.join(folder, f"*{ext}")):
                stem = Path(path).stem
                original_id, lbl_from_name = parse_original_from_video_stem(stem)
                use_label = dir_label if lbl_from_name == -1 else lbl_from_name
                items.append((path, use_label, original_id))
    return items

def stratified_group_split(
    data: List[Tuple[str, int, str]],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
):
    """Group-aware split using helpers.py functions."""
    if not data:
        raise ValueError("No data to split.")

    # Group mapping
    group_to_label: Dict[str, int] = {}
    for _, lbl, grp in data:
        group_to_label.setdefault(grp, lbl)

    groups = list(group_to_label.keys())
    y = [group_to_label[g] for g in groups]

    # Stratified split on groups
    train_groups, val_groups, test_groups = safe_group_stratified_split(
        groups, y, train_ratio, val_ratio, seed
    )

    # Map back to items
    train_idx = groups_to_indices(data, train_groups)
    val_idx = groups_to_indices(data, val_groups)
    test_idx = groups_to_indices(data, test_groups)

    return (
        [(data[i][0], data[i][1]) for i in train_idx],
        [(data[i][0], data[i][1]) for i in val_idx],
        [(data[i][0], data[i][1]) for i in test_idx],
    )

# ========================== DATASET & LOADERS ============================

class DeepFakeFrameDataset(Dataset):
    def __init__(self, data: List[Tuple[str, int]], transform=None):
        self.data = data
        self.transform = transform or _EVAL_TFMS

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        path, label = self.data[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

def create_dataloaders(
    train_data,
    val_data,
    test_data,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    """Create train/val/test dataloaders."""
    datasets = [
        DeepFakeFrameDataset(train_data, _TRAIN_TFMS),
        DeepFakeFrameDataset(val_data, _EVAL_TFMS),
        DeepFakeFrameDataset(test_data, _EVAL_TFMS),
    ]

    return tuple(
        DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(i == 0),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for i, ds in enumerate(datasets)
    )

def get_data_loaders(
    batch_size: int = 32,
    valid_size: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    base_dir: str = "frames",
    limit: Optional[int] = None,
):
    """Main interface for getting dataloaders."""
    data_wg = load_image_paths_with_groups(base_dir=base_dir)
    if limit and limit > 0:
        rnd = random.Random(seed)
        rnd.shuffle(data_wg)
        data_wg = data_wg[:limit]

    train, val, test = stratified_group_split(
        data_wg, seed=seed, train_ratio=0.8, val_ratio=valid_size
    )
    tl, vl, te = create_dataloaders(
        train, val, test, batch_size=batch_size, num_workers=num_workers
    )
    return {"train": tl, "valid": vl, "test": te}

def get_dataloaders(config: Dict):
    """Config-friendly wrapper for train.py usage."""
    data_wg = load_image_paths_with_groups(config.get("base_dir", "frames"))
    if config.get("limit"):
        rnd = random.Random(config.get("seed", 42))
        rnd.shuffle(data_wg)
        data_wg = data_wg[:int(config.get("limit"))]

    train, val, test = stratified_group_split(
        data_wg,
        seed=config.get("seed", 42),
        train_ratio=config.get("train_ratio", 0.8),
        val_ratio=config.get("val_ratio", 0.1),
    )
    return create_dataloaders(
        train,
        val,
        test,
        batch_size=config.get("batch_size", 32),
        num_workers=config.get("num_workers", 0),
    )

def visualize_one_batch(data_loaders: Dict[str, DataLoader], max_n: int = 5):
    """Visualize sample batch from training data."""
    import matplotlib.pyplot as plt

    if "train" not in data_loaders:
        raise KeyError("data_loaders must contain a 'train' DataLoader")

    images, labels = next(iter(data_loaders["train"]))

    # Denormalize for display
    inv = transforms.Compose([
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1 / s for s in _IMAGENET_STD]),
        transforms.Normalize(mean=[-m for m in _IMAGENET_MEAN], std=[1.0, 1.0, 1.0]),
    ])
    images = inv(images).clamp(0, 1).permute(0, 2, 3, 1).cpu()

    n = min(max_n, images.shape[0])
    fig = plt.figure(figsize=(3 * n, 3))
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
    """Test pipeline: extraction → grouping → splitting (with face fallback)."""
    ds = tmp_path / "SDFVD2.0"
    (ds / "SDFVD2.0_real").mkdir(parents=True)
    (ds / "SDFVD2.0_fake").mkdir(parents=True)

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
    extract_all_frames_sdfvd(
        str(ds), output_root=str(out), frame_interval=2, limit_per_video=2, use_face_detection=True
    )

    items = load_image_paths_with_groups(base_dir=str(out))
    assert len(items) > 0

    labels = {lbl for _, lbl, _ in items}
    assert labels == {0, 1}

    train, val, test = stratified_group_split(items, seed=7, train_ratio=0.8, val_ratio=0.1)

    p2g = {p: g for p, _, g in items}
    g_train = {p2g[p] for p, _ in train}
    g_val = {p2g[p] for p, _ in val}
    g_test = {p2g[p] for p, _ in test}

    assert g_train.isdisjoint(g_val)
    assert g_train.isdisjoint(g_test)
    assert g_val.isdisjoint(g_test)

def test_loaders_shapes(tmp_path):
    """Test dataloader creation and tensor shapes."""
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
    """Sanity check helpers integration."""
    assert parse_original_from_video_stem("real_someOriginal_aug_4_frame_001") == ("someOriginal", 0)
    assert parse_original_from_video_stem("fake_foo_aug_0") == ("foo", 1)
    assert parse_original_from_video_stem("unknown_stem") == ("unknown_stem", -1)

    assert label_from_name("SDFVD2.0_real") == "real"
    assert label_from_name("fake_videos") == "fake"
    assert label_from_name("random_dir") is None
