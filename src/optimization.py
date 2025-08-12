# src/optimization.py
"""
Simplified optimization components for transfer learning.
"""

import torch
from torch import nn
from torch.optim import AdamW


def get_loss_function():
    """Return CrossEntropyLoss for binary classification."""
    return nn.CrossEntropyLoss()


def get_optimizer(model, learning_rate=0.001, weight_decay=0.01):
    """
    Create AdamW optimizer for transfer learning.
    Only optimizes trainable parameters (frozen backbone is ignored).
    """
    # Only get parameters that require gradients (unfrozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    return AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)


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


def test_loss_function():
    """Test loss function creation."""
    loss_fn = get_loss_function()
    assert isinstance(loss_fn, nn.CrossEntropyLoss)
    
    # Test it works
    logits = torch.randn(4, 2)
    targets = torch.randint(0, 2, (4,))
    loss = loss_fn(logits, targets)
    assert isinstance(loss.item(), float)
    assert loss.item() > 0


def test_optimizer_only_trainable_params():
    """Test optimizer only gets trainable parameters."""
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(10, 5)
            self.classifier = nn.Linear(5, 2)

        def forward(self, x):
            return self.classifier(self.backbone(x))

    model = TestModel()
    
    # Freeze backbone (like transfer learning)
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    optimizer = get_optimizer(model, learning_rate=0.001, weight_decay=0.01)
    
    # Should only have classifier parameters
    optimizer_params = set()
    for group in optimizer.param_groups:
        for param in group['params']:
            optimizer_params.add(id(param))
    
    # Check only trainable params are in optimizer
    model_trainable_params = set()
    for param in model.parameters():
        if param.requires_grad:
            model_trainable_params.add(id(param))
    
    assert optimizer_params == model_trainable_params
    
    # Should have 2 parameters (weight and bias of classifier)
    total_params = sum(len(group['params']) for group in optimizer.param_groups)
    assert total_params == 2


def test_optimizer_works_with_training():
    """Test optimizer works in a training step."""
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    
    optimizer = get_optimizer(model, learning_rate=0.001)
    loss_fn = get_loss_function()
    
    # Training step
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))
    
    optimizer.zero_grad()
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
    
    # Should complete without errors
    assert True


def test_mixed_precision_scaler():
    """Test mixed precision scaler creation."""
    scaler = get_mixed_precision_scaler()
    if torch.cuda.is_available():
        assert scaler is not None
        # Works with both old and new PyTorch APIs
        assert hasattr(scaler, 'scale')
    else:
        assert scaler is None


def test_integration_with_transfer_learning():
    """Test integration with transfer learning model."""
    try:
        from model import get_model_transfer_learning
        model = get_model_transfer_learning(n_classes=2)
    except ImportError:
        # Fallback model
        import torchvision.models as models
        model = models.efficientnet_b0(weights=None)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Linear(num_ftrs, 2)
    
    # Get optimizer
    optimizer = get_optimizer(model, learning_rate=0.001)
    loss_fn = get_loss_function()
    
    # Should work with actual model
    x = torch.randn(2, 3, 224, 224)
    y = torch.randint(0, 2, (2,))
    
    model.train()
    optimizer.zero_grad()
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
    
    assert output.shape == (2, 2)
    assert isinstance(loss.item(), float)