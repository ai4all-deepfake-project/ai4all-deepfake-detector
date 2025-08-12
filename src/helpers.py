"""
Lightweight utilities for training, evaluation, and reproducibility.

Provided:
- set_seed(seed, deterministic=True)
- get_device(prefer_cuda=True, prefer_mps=True) -> torch.device
- preds_from_logits(y_pred, threshold=0.5)
- calculate_metrics(y_true, y_pred, average='weighted')
- print_metrics(metrics: dict, header='Evaluation Metrics')
- after_subplot(fig)  
"""

from __future__ import annotations
from typing import Dict, Iterable, Optional, Tuple, Union, List

import os
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from pathlib import Path
import re

from sklearn.model_selection import train_test_split


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds across Python, NumPy, and PyTorch.
    deterministic=True may reduce speed but improves reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuDNN settings for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For newer PyTorch versions, this enforces determinism further:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def get_device(prefer_cuda: bool = True, prefer_mps: bool = True) -> torch.device:
    """
    Return the best available device as a torch.device.
    - CUDA if available (and prefer_cuda)
    - Apple MPS if available (and prefer_mps)
    - else CPU
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon
    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore
        return torch.device("mps")
    return torch.device("cpu")


def _to_1d_numpy(x: Union[torch.Tensor, np.ndarray, Iterable]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().reshape(-1)
    if isinstance(x, np.ndarray):
        return x.reshape(-1)
    return np.array(list(x)).reshape(-1)


def preds_from_logits(
    y_pred: Union[torch.Tensor, np.ndarray, Iterable],
    threshold: float = 0.5
) -> np.ndarray:
    """
    Convert logits/probabilities to hard class predictions.
    Supports:
      - shape (N, 2): argmax over classes
      - shape (N,): binary logit/prob — threshold at 0.5 after sigmoid
      - shape (N, 1): binary logit/prob — squeeze and threshold
    Returns int array of shape (N,) with values in {0, 1} (or {0..C-1}).
    """
    if isinstance(y_pred, torch.Tensor):
        arr = y_pred.detach().cpu().numpy()
    elif isinstance(y_pred, np.ndarray):
        arr = y_pred
    else:
        arr = np.array(list(y_pred))

    arr = np.asarray(arr)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        # Multiclass logits/probs
        return arr.argmax(axis=1).astype(int)
    elif arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]

    # Now arr is (N,)
    # If looks like logits, apply sigmoid; if already prob-like, threshold still works.
    # We’ll treat values as logits if they’re unbounded.
    # Simple heuristic: apply sigmoid unconditionally—harmless for probs in [0,1].
    probs = 1 / (1 + np.exp(-arr))
    return (probs >= threshold).astype(int)


def calculate_metrics(
    y_true: Union[torch.Tensor, np.ndarray, Iterable],
    y_pred: Union[torch.Tensor, np.ndarray, Iterable],
    average: str = "weighted",
) -> Dict[str, float]:
    """
    Compute accuracy/precision/recall/F1.
    y_pred can be hard labels or raw logits/probabilities; we’ll coerce to labels.
    """
    yt = _to_1d_numpy(y_true)
    yp = y_pred

    # If yp looks like probabilities/logits or has 2D shape, convert to labels
    if isinstance(yp, (torch.Tensor, np.ndarray)) and (
        (isinstance(yp, torch.Tensor) and yp.ndim >= 1) or
        (isinstance(yp, np.ndarray) and yp.ndim >= 1)
    ):
        yp_labels = preds_from_logits(yp)
    else:
        yp_labels = _to_1d_numpy(yp)

    acc = float(accuracy_score(yt, yp_labels))
    prec = float(precision_score(yt, yp_labels, average=average, zero_division=0))
    rec = float(recall_score(yt, yp_labels, average=average, zero_division=0))
    f1 = float(f1_score(yt, yp_labels, average=average, zero_division=0))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}


def print_metrics(metrics: Dict[str, float], header: str = "Evaluation Metrics") -> None:
    print(header + ":")
    for k in ("accuracy", "precision", "recall", "f1_score"):
        if k in metrics and metrics[k] is not None:
            print(f"{k}: {metrics[k]:.4f}")
    print("")


def after_subplot(fig) -> None:  # pragma: no cover
    # Intentionally light; customize if you want gridlines/titles
    try:
        fig.tight_layout()
    except Exception:
        pass

# ----------------------------- Data helpers ----------------------------- #


# filenames like "..._frame_000.jpg" → strip the trailing frame tag
_FRAME_SUFFIX_RE = re.compile(r"^(?P<stem>.*)_frame_\d{3}$")

def strip_frame_suffix(stem: str) -> str:
    m = _FRAME_SUFFIX_RE.match(stem)
    return m.group("stem") if m else stem

def parse_original_from_video_stem(video_stem: str) -> Tuple[str, int]:
    """
    'real_<original>_aug_<id>' → (original, 0)
    'fake_<original>_aug_<id>' → (original, 1)
    If not found, label = -1 and original = best-effort stem.
    """
    base = strip_frame_suffix(video_stem)
    left = base.split("_aug_")[0]  # "real_<original>" or "fake_<original>" (ideally)

    if left.startswith("real_"):
        return left[len("real_"):], 0
    if left.startswith("fake_"):
        return left[len("fake_"):], 1
    if "real_" in left:
        return left.split("real_", 1)[-1], 0
    if "fake_" in left:
        return left.split("fake_", 1)[-1], 1
    return left, -1

def label_from_name(name: str) -> Optional[str]:
    low = name.lower()
    if low == "real" or low.endswith("_real") or low.startswith("real"):
        return "real"
    if low == "fake" or low.endswith("_fake") or low.startswith("fake"):
        return "fake"
    return None

def dir_has_videos(p: str, video_exts: Tuple[str, ...]) -> bool:
    try:
        for fname in os.listdir(p):
            if any(fname.lower().endswith(e) for e in video_exts):
                return True
        return False
    except Exception:
        return False

def find_label_dirs(root: str,
                    video_exts: Tuple[str, ...] = (".mp4", ".mov", ".mkv", ".avi"),
                    max_depth: int = 2) -> Dict[str, List[str]]:
    """
    Walk up to max_depth under root; collect dirs that look like real/fake and have videos.
    """
    root_parts = Path(root).resolve().parts
    found = {"real": [], "fake": []}

    for dp, dn, _ in os.walk(root):
        depth = len(Path(dp).resolve().parts) - len(root_parts)
        if depth > max_depth:
            dn[:] = []  # stop descending deeper
            continue
        lbl = label_from_name(os.path.basename(dp))
        if lbl and dir_has_videos(dp, video_exts):
            found[lbl].append(dp)
    return found

def groups_to_indices(data_with_groups: List[Tuple[str, int, str]],
                      keep_groups: Iterable[str]) -> List[int]:
    keep = set(keep_groups)
    return [i for i, (_, _, g) in enumerate(data_with_groups) if g in keep]

def safe_group_stratified_split(groups: List[str],
                                labels: List[int],
                                train_ratio: float,
                                val_ratio: float,
                                seed: int) -> Tuple[List[str], List[str], List[str]]:
    """
    Split group IDs into train / val / test with stratification if feasible.
    Falls back cleanly for tiny sets. Returns lists of group IDs.
    """
    # 1) train vs temp
    stratify1 = labels if len(set(labels)) > 1 else None
    try:
        tr, temp, y_tr, y_temp = train_test_split(
            groups, labels, test_size=(1 - train_ratio),
            stratify=stratify1, random_state=seed
        )
    except ValueError:
        tr, temp, y_tr, y_temp = train_test_split(
            groups, labels, test_size=(1 - train_ratio),
            stratify=None, random_state=seed
        )

    # 2) temp → val/test
    if len(temp) < 2:
        return tr, [], temp

    rel_val = val_ratio / max(1e-12, (1 - train_ratio))
    stratify2 = y_temp if len(set(y_temp)) > 1 else None
    try:
        va, te, _, _ = train_test_split(
            temp, y_temp, test_size=(1 - rel_val),
            stratify=stratify2, random_state=seed
        )
    except ValueError:
        va, te, _, _ = train_test_split(
            temp, y_temp, test_size=(1 - rel_val),
            stratify=None, random_state=seed
        )
    return tr, va, te
