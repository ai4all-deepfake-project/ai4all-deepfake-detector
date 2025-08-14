# src/predictor.py
"""
Deepfake prediction pipeline for video input. Extracts frames, preprocesses,
runs inference, aggregates predictions, and can generate Grad-CAM maps.
Designed to be called from app.ipynb for user-facing inference.

Tests live at the bottom of this same file. In notebook, run:
!pytest -q src/predictor.py
"""

import os
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


# ----------------------------- Core Inference ----------------------------- #

def load_model(model_path: str = "best_model.pth") -> Tuple[torch.nn.Module, torch.device]:
    from model import get_model
    from helpers import get_device

    device = get_device()
    model = get_model("efficientnet_b0").to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device



def extract_frames(video_path: str,
                   frame_rate: int = 1,
                   max_frames: int = 10) -> List[Image.Image]:
    """
    Extract up to `max_frames` frames, sampled roughly evenly across the video.
    If `frame_rate` > 1, it will stride within the evenly-sampled indices.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError("Video has no frames")

    # Even sampling; oversample then stride by frame_rate
    n = min(max_frames * max(1, frame_rate), total)
    sample_idxs = np.linspace(0, total - 1, num=n, dtype=int)
    sample_idxs = sample_idxs[::max(1, frame_rate)]

    frames: List[Image.Image] = []
    for idx in sample_idxs[:max_frames]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()

    if not frames:
        raise ValueError("No frames could be extracted from video")

    return frames


def preprocess_frame(frame: Image.Image) -> torch.Tensor:
    """
    Preprocess a single frame to 224x224 ImageNet-normalized tensor.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(frame)


def predict_frame(model: torch.nn.Module,
                  frame: Image.Image,
                  device: torch.device) -> Dict[str, float]:
    """
    Predict class probabilities for a single frame.
    Returns {"Real": p0, "Fake": p1}
    """
    x = preprocess_frame(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        if logits.ndim != 2 or logits.size(1) != 2:
            raise ValueError("Model must return [N, 2] logits for (Real, Fake).")
        probs = torch.softmax(logits, dim=1)

    return {"Real": float(probs[0, 0].item()),
            "Fake": float(probs[0, 1].item())}


def aggregate_predictions(frame_predictions: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Average frame-level probabilities to produce a video-level decision.
    """
    if len(frame_predictions) == 0:
        raise ValueError("No frame predictions to aggregate.")

    avg_real = sum(p["Real"] for p in frame_predictions) / len(frame_predictions)
    avg_fake = sum(p["Fake"] for p in frame_predictions) / len(frame_predictions)

    if avg_fake >= avg_real:
        prediction, confidence = "Fake", avg_fake
    else:
        prediction, confidence = "Real", avg_real

    return {
        "prediction": prediction,
        "confidence": float(confidence),
        "fake_probability": float(avg_fake),
        "real_probability": float(avg_real),
    }


# ----------------------------- Grad-CAM ---------------------------------- #

class GradCAM:
    """Minimal Grad-CAM for a single conv feature layer."""
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.hook_handles = []
        self.hook_handles.append(
            target_layer.register_forward_hook(self._save_activation)
        )
        # `register_full_backward_hook` handles more autograd cases
        self.hook_handles.append(
            target_layer.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_inputs, grad_outputs):
        self.gradients = grad_outputs[0]

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int = 1) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for `class_idx` (default 1='Fake').
        """
        if not input_tensor.requires_grad:
            input_tensor.requires_grad_(True)

        logits = self.model(input_tensor)  # [1, 2]
        self.model.zero_grad(set_to_none=True)
        target = logits[:, class_idx].sum()
        target.backward(retain_graph=True)

        grads = self.gradients  # [B, C, H, W]
        acts = self.activations  # [B, C, H, W]
        if grads is None or acts is None:
            raise RuntimeError("Gradients/activations were not captured. Check target_layer.")

        weights = grads.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = (weights * acts).sum(dim=1, keepdim=False)  # [B, H, W]
        cam = F.relu(cam)
        cam = cam[0]  # [H, W]

        maxv = cam.max()
        if maxv > 0:
            cam = cam / maxv
        return cam.detach().cpu().numpy()

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()


def generate_gradcam(model: torch.nn.Module,
                     frame: Image.Image,
                     target_layer: Optional[torch.nn.Module] = None,
                     model_name: str = "efficientnet_b0",
                     fake_class_idx: int = 1) -> np.ndarray:
    """
    Generate an overlayed RGB Grad-CAM image for a PIL frame.
    """
    if target_layer is None:
        from model import get_target_layer
        target_layer = get_target_layer(model)

    device = next(model.parameters()).device

    gradcam = GradCAM(model, target_layer)
    x = preprocess_frame(frame).unsqueeze(0).to(device)
    x.requires_grad_(True)

    heat = gradcam.generate_heatmap(x, class_idx=fake_class_idx)
    gradcam.remove_hooks()

    heat_resized = cv2.resize(heat, (frame.size[0], frame.size[1]))
    heat_uint8 = np.uint8(255 * heat_resized)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

    orig = np.array(frame)
    overlay = cv2.addWeighted(orig, 0.6, heat_color, 0.4, 0.0)
    return overlay


def predict_video(video_path: str,
                  use_gradcam: bool = False,
                  model_path: str = "best_model.pth",
                  max_frames: int = 10) -> Dict:
    model, device = load_model(model_path=model_path)
    frames = extract_frames(video_path, frame_rate=1, max_frames=max_frames)

    frame_predictions = [predict_frame(model, f, device) for f in frames]
    result = aggregate_predictions(frame_predictions)
    result["frame_predictions"] = [p["Fake"] for p in frame_predictions]

    if use_gradcam:
        from model import get_target_layer
        tl = get_target_layer(model)
        grad_frames = [Image.fromarray(generate_gradcam(model, f)) for f in frames[:3]]
        result["gradcam_frames"] = grad_frames

    return result

# =============================================================================
#                                   TESTS
# =============================================================================
import sys
import types
import tempfile
import pytest


class _TinyCNN(nn.Module):
    """
    A tiny CNN with one conv block + GAP + linear head -> 2 logits.
    Fast and deterministic; supports Grad-CAM on a conv layer.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(8, 2)

        # Bias the logits slightly toward Fake for deterministic aggregation
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias[:] = torch.tensor([1.5, 2.0])  # Real ~0.27, Fake ~0.73

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


def _install_fake_helpers_and_model(tiny_model: nn.Module):
    """
    Monkey-install `helpers` and `model` modules so load_model() resolves to fakes.
    """
    helpers_mod = types.ModuleType("helpers")
    def get_device():
        return torch.device("cpu")
    helpers_mod.get_device = get_device

    model_mod = types.ModuleType("model")
    def get_model(model_name: str = "efficientnet_b0"):
        return tiny_model
    def get_target_layer(model, model_name: str = "efficientnet_b0"):
        # Use the second Conv2d for Grad-CAM
        return model.features[-2]
    model_mod.get_model = get_model
    model_mod.get_target_layer = get_target_layer

    sys.modules["helpers"] = helpers_mod
    sys.modules["model"] = model_mod


def _make_dummy_video(path: str, frames: int = 12, w: int = 128, h: int = 96):
    """Create a small synthetic .mp4 video for tests."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, 12.0, (w, h))
    assert vw.isOpened()
    for i in range(frames):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # moving square + gradient
        cv2.rectangle(img, (5 + i, 5), (25 + i, 25), (0, 255, 0), -1)
        img[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)
        vw.write(img)
    vw.release()
    assert os.path.exists(path)


def test_load_model_and_predict_frame():
    tiny = _TinyCNN().eval()
    _install_fake_helpers_and_model(tiny)

    # Save and load a state_dict to exercise the load path
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        torch.save(tiny.state_dict(), f.name)
        model, device = load_model(model_path=f.name, model_name="whatever")
    os.unlink(f.name)

    assert isinstance(model, nn.Module)
    assert isinstance(device, torch.device)

    frame = Image.new("RGB", (320, 200), color=(10, 40, 200))
    out = predict_frame(model, frame, device)
    assert set(out.keys()) == {"Real", "Fake"}
    assert 0.0 <= out["Real"] <= 1.0
    assert 0.0 <= out["Fake"] <= 1.0
    assert abs(out["Real"] + out["Fake"] - 1.0) < 1e-6


def test_extract_frames_and_aggregate(tmp_path):
    tiny = _TinyCNN().eval()
    _install_fake_helpers_and_model(tiny)

    vid = tmp_path / "toy.mp4"
    _make_dummy_video(str(vid), frames=15)

    frames = extract_frames(str(vid), frame_rate=2, max_frames=5)
    assert 1 <= len(frames) <= 5
    assert all(isinstance(f, Image.Image) for f in frames)

    model, device = load_model(model_path="__nope__.pth", model_name="n/a")
    preds = [predict_frame(model, f, device) for f in frames]
    agg = aggregate_predictions(preds)

    assert set(agg.keys()) == {"prediction", "confidence", "fake_probability", "real_probability"}
    assert 0.0 <= agg["fake_probability"] <= 1.0
    assert 0.0 <= agg["real_probability"] <= 1.0
    assert abs(agg["fake_probability"] + agg["real_probability"] - 1.0) < 1e-6


def test_gradcam_end_to_end():
    tiny = _TinyCNN().eval()
    _install_fake_helpers_and_model(tiny)

    frame = Image.new("RGB", (200, 120), color=(30, 120, 60))
    model, device = load_model(model_path="__nope__.pth", model_name="n/a")

    # Use our fake get_target_layer
    from model import get_target_layer
    tl = get_target_layer(model, "n/a")

    overlay = generate_gradcam(model, frame, target_layer=tl, model_name="n/a")
    assert isinstance(overlay, np.ndarray)
    assert overlay.ndim == 3 and overlay.shape[2] == 3
    assert overlay.shape[0] == frame.size[1] and overlay.shape[1] == frame.size[0]


def test_predict_video_integration(tmp_path):
    tiny = _TinyCNN().eval()
    _install_fake_helpers_and_model(tiny)

    vid = tmp_path / "tiny.mp4"
    _make_dummy_video(str(vid), frames=9)

    result = predict_video(str(vid), use_gradcam=True, model_path="__nope__.pth",
                           model_name="n/a", max_frames=6)

    assert "prediction" in result and "confidence" in result
    assert "frame_predictions" in result and isinstance(result["frame_predictions"], list)
    assert len(result["frame_predictions"]) >= 1

    assert "gradcam_frames" in result
    gfs = result["gradcam_frames"]
    assert 1 <= len(gfs) <= 3
    # PIL Images
    assert all(hasattr(img, "size") for img in gfs)
