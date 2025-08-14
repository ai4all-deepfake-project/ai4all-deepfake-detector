# src/transfer_learning.py
"""
Transfer-learning helpers for ResNet (ResNet18 only).
Supports binary (1-logit) and multiclass heads, with proper freeze/unfreeze logic.
"""

from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


# --------- Internal helpers ---------

def _replace_head_resnet(model: nn.Module, out_features: int) -> nn.Module:
    """Replace ResNet final fully-connected layer with out_features outputs."""
    if not hasattr(model, "fc") or not isinstance(model.fc, nn.Linear):
        raise RuntimeError("Expected a ResNet-like model with an 'fc' Linear head.")
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, out_features)
    return model


def _freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def _unfreeze_head(model: nn.Module) -> None:
    for p in model.fc.parameters():
        p.requires_grad = True


def _resnet_backbone_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    Ordered backbone stages for staged unfreezing:
    [conv1, bn1, layer1, layer2, layer3, layer4]
    """
    stages: List[Tuple[str, nn.Module]] = []
    if hasattr(model, "conv1"):  stages.append(("conv1", model.conv1))
    if hasattr(model, "bn1"):    stages.append(("bn1", model.bn1))
    if hasattr(model, "layer1"): stages.append(("layer1", model.layer1))
    if hasattr(model, "layer2"): stages.append(("layer2", model.layer2))
    if hasattr(model, "layer3"): stages.append(("layer3", model.layer3))
    if hasattr(model, "layer4"): stages.append(("layer4", model.layer4))
    if not stages:
        raise RuntimeError("Could not locate ResNet backbone stages.")
    return stages


# --------- Public API ---------

def get_model_transfer_learning(
    model_name: str = "resnet18",
    task: str = "binary",
    n_classes: int = 2,
    freeze_backbone: bool = True,
    unfreeze_from: Optional[int] = None,
) -> nn.Module:
    """
    Build a transfer-learning ResNet18 with fixed freeze/unfreeze logic.

    Args:
        model_name: must be "resnet18"
        task: "binary" (1 logit) or "multiclass"
        n_classes: number of classes for multiclass (ignored for binary)
        freeze_backbone: if True, freeze backbone initially
        unfreeze_from: staged unfreeze start index into
                       [conv1, bn1, layer1, layer2, layer3, layer4]
                       If None and freeze_backbone=True, only head is trainable.
                       Ignored if freeze_backbone=False.

    Returns:
        model with requested head and proper freeze state
    """
    model_name = model_name.lower()
    if model_name != "resnet18":
        raise ValueError("Model 'resnet18' is the only supported model.")

    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    backbone_stages = _resnet_backbone_layers(model)

    # Head
    if task.lower() == "binary":
        model = _replace_head_resnet(model, out_features=1)
    elif task.lower() == "multiclass":
        model = _replace_head_resnet(model, out_features=n_classes)
    else:
        raise ValueError("task must be 'binary' or 'multiclass'")

    # Freeze policy
    if freeze_backbone:
        _freeze_all(model)
        _unfreeze_head(model)

        if unfreeze_from is not None:
            if unfreeze_from >= len(backbone_stages):
                print(f"Warning: unfreeze_from={unfreeze_from} >= total stages={len(backbone_stages)}")
            else:
                unfrozen_params = 0
                stage_name = backbone_stages[unfreeze_from][0]
                for _, stage in backbone_stages[unfreeze_from:]:
                    for p in stage.parameters():
                        if not p.requires_grad:
                            p.requires_grad = True
                            unfrozen_params += p.numel()
                print(f"Unfroze backbone stages from index {unfreeze_from} "
                      f"({stage_name}) → end = {unfrozen_params:,} params")
    else:
        for p in model.parameters():
            p.requires_grad = True

    # Report
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\n=== Transfer Learning Model Created ===")
    print(f"Architecture: {model_name}")
    print(f"Task: {task}")
    if task.lower() == "multiclass":
        print(f"Classes: {n_classes}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"Frozen: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
    if freeze_backbone:
        print("Backbone frozen: Yes")
        if unfreeze_from is not None:
            print(f"Unfrozen from stage index: {unfreeze_from}")
    else:
        print("Backbone frozen: No (fully trainable)")
    print("=" * 40)

    return model


def diagnose_model_freeze_state(model: nn.Module, verbose: bool = True):
    """
    Diagnose and report the freeze state of a model.
    """
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    layer_info = []

    for name, p in model.named_parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
            status = "TRAINABLE"
        else:
            frozen_params += p.numel()
            status = "FROZEN"
        if verbose:
            layer_info.append(f"{name}: {p.numel():,} params - {status}")

    print(f"\n=== Model Freeze Diagnosis ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"Frozen: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
    if verbose:
        print("\nDetailed layer breakdown:")
        for info in layer_info:
            print(f"  {info}")
    print("=" * 30)

    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params,
        "trainable_ratio": trainable_params / total_params if total_params else 0.0,
    }


# =========================
#           TESTS
# =========================
import pytest


def test_binary_head_shape():
    m = get_model_transfer_learning(task="binary", freeze_backbone=True)
    x = torch.randn(2, 3, 224, 224)
    m.eval()
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, 1), f"Expected (2, 1), got {y.shape}"


def test_multiclass_head_shape():
    m = get_model_transfer_learning(task="multiclass", n_classes=5, freeze_backbone=True)
    x = torch.randn(2, 3, 224, 224)
    m.eval()
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, 5), f"Expected (2, 5), got {y.shape}"


def test_freezing_logic_head_only():
    m = get_model_transfer_learning(task="binary", freeze_backbone=True, unfreeze_from=None)
    head_trainable = all(p.requires_grad for p in m.fc.parameters())
    backbone_trainable = any(
        p.requires_grad for n, p in m.named_parameters() if not n.startswith("fc")
    )
    assert head_trainable, "Head should be trainable"
    assert not backbone_trainable, "Backbone should be frozen for head-only training"


def test_partial_unfreezing():
    # Unfreeze from layer2 (index 3) onward: [conv1=0, bn1=1, layer1=2, layer2=3, layer3=4, layer4=5]
    m = get_model_transfer_learning(task="binary", freeze_backbone=True, unfreeze_from=3)
    head_trainable = all(p.requires_grad for p in m.fc.parameters())

    # Some backbone params should now be trainable
    some_backbone_trainable = any(
        p.requires_grad for n, p in m.named_parameters()
        if not n.startswith("fc") and (
            n.startswith("layer2") or n.startswith("layer3") or n.startswith("layer4")
        )
    )
    assert head_trainable, "Head should be trainable"
    assert some_backbone_trainable, "Expected some backbone layers to be trainable with unfreeze_from=3"


def test_fully_trainable():
    m = get_model_transfer_learning(task="binary", freeze_backbone=False)
    assert all(p.requires_grad for p in m.parameters()), \
        "All parameters should be trainable when freeze_backbone=False"


def test_diagnose_function():
    m = get_model_transfer_learning(task="binary", freeze_backbone=True, unfreeze_from=2)
    stats = diagnose_model_freeze_state(m, verbose=False)
    assert "total" in stats and "trainable" in stats and "frozen" in stats and "trainable_ratio" in stats
    assert stats["total"] == stats["trainable"] + stats["frozen"]
    assert 0 < stats["trainable_ratio"] < 1  # some frozen, some trainable


def test_unsupported_model_raises():
    with pytest.raises(ValueError):
        get_model_transfer_learning(model_name="efficientnet_b0")  # now unsupported
