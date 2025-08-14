# src/optimization.py
"""
Simplified optimization components for transfer learning.
Defaults to binary classification with BCEWithLogitsLoss.
Supports AdamW, SGD+Nesterov, and RAdam.
"""

import torch
from torch import nn
from torch.optim import AdamW, SGD, RAdam


# =========================
# Loss (BINARY by default)
# =========================
def get_loss_function(pos_weight: torch.Tensor | None = None) -> nn.Module:
    """
    Return BCEWithLogitsLoss for binary classification.

    Args:
        pos_weight: Optional tensor([neg/pos]) to handle class imbalance.
    """
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# =========================
# Optimizers
# =========================
def get_optimizer(
    model: nn.Module,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    optimizer_name: str = "adamw",
) -> torch.optim.Optimizer:
    """
    Create optimizer for transfer learning.
    Only optimizes trainable parameters (frozen backbone is ignored).

    optimizer_name: 'adamw' (default), 'sgd', or 'radam'
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    opt = optimizer_name.lower()
    if opt == "adamw":
        return AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    elif opt == "sgd":
        return SGD(
            trainable_params, lr=learning_rate, weight_decay=weight_decay,
            momentum=0.9, nesterov=True
        )
    elif opt == "radam":
        return RAdam(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError("optimizer_name must be one of: 'adamw', 'sgd', 'radam'")


# =========================
# Mixed precision
# =========================
def get_mixed_precision_scaler():
    """Return GradScaler if CUDA available, None otherwise."""
    if torch.cuda.is_available():
        try:
            # New PyTorch 2.0+ API
            from torch.amp import GradScaler
            return GradScaler('cuda')
        except ImportError:
            # Fallback for older PyTorch versions
            from torch.cuda.amp import GradScaler
            return GradScaler()
    return None


# =============================================================================
#                                    TESTS
# =============================================================================

import pytest


def test_loss_function_bce():
    """Test BCEWithLogitsLoss creation and forward."""
    loss_fn = get_loss_function()
    assert isinstance(loss_fn, nn.BCEWithLogitsLoss)

    logits = torch.randn(4, 1)                 # (B, 1) logits
    targets = torch.randint(0, 2, (4,)).float()  # (B,) in {0,1}
    loss = loss_fn(logits, targets.unsqueeze(1))
    assert isinstance(loss.item(), float)
    assert loss.item() > 0


def test_optimizer_only_trainable_params():
    """Test optimizer only gets trainable parameters."""
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(10, 5)
            self.classifier = nn.Linear(5, 1)  # 1-logit head for binary

        def forward(self, x):
            return self.classifier(self.backbone(x))

    model = TestModel()

    # Freeze backbone (like transfer learning)
    for p in model.backbone.parameters():
        p.requires_grad = False

    optimizer = get_optimizer(model, learning_rate=0.001, weight_decay=0.01)

    # Should only have classifier parameters
    optimizer_params = {id(p) for g in optimizer.param_groups for p in g['params']}
    model_trainable_params = {id(p) for p in model.parameters() if p.requires_grad}

    assert optimizer_params == model_trainable_params

    # Should have 2 tensors (weight + bias of classifier)
    total_params = sum(len(g['params']) for g in optimizer.param_groups)
    assert total_params == 2


def test_optimizer_variants_build_and_step():
    """Ensure all optimizers build and can do a training step."""
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
        def forward(self, x): return self.net(x)

    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,)).float()

    for name in ["adamw", "sgd", "radam"]:
        model = Tiny()
        opt = get_optimizer(model, optimizer_name=name, learning_rate=1e-3)
        loss_fn = get_loss_function()
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y.unsqueeze(1))
        loss.backward()
        opt.step()
        assert isinstance(loss.item(), float)


def test_mixed_precision_scaler():
    """Test mixed precision scaler creation."""
    scaler = get_mixed_precision_scaler()
    if torch.cuda.is_available():
        assert scaler is not None
        assert hasattr(scaler, 'scale')
    else:
        assert scaler is None


def test_integration_with_transfer_learning():
    """Integration with a transfer model; force 1-logit head."""
    try:
        from model import get_model_transfer_learning
        model = get_model_transfer_learning(n_classes=1)  # if your helper supports it
    except Exception:
        # Fallback: EfficientNet head → 1 logit
        import torchvision.models as models
        model = models.efficientnet_b0(weights=None)
        for p in model.parameters():
            p.requires_grad = False
        in_f = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_f, 1)

    optimizer = get_optimizer(model, learning_rate=0.001, optimizer_name="sgd")
    loss_fn = get_loss_function()

    x = torch.randn(2, 3, 224, 224)
    y = torch.randint(0, 2, (2,)).float()

    model.train()
    optimizer.zero_grad()
    logits = model(x)                   # (B,1)
    loss = loss_fn(logits, y.unsqueeze(1))
    loss.backward()
    optimizer.step()

    assert logits.shape == (2, 1)
    assert isinstance(loss.item(), float)
