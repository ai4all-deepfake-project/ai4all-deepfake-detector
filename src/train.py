# src/train.py
"""
Flexible training for transfer learning.
- Binary (1-logit + BCEWithLogits) or Multiclass (C logits + CE)
- Optimizers and loss from src.optimization
- FIXED freeze enforcement with proper layer identification
- Better binary metrics: AUC (optional), best-F1 threshold, balanced accuracy
"""

from typing import Literal, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.optimization import get_loss_function, get_optimizer

# -------- optional sklearn for ROC-AUC --------
try:
    from sklearn.metrics import roc_auc_score
    _HAS_SK = True
except Exception:
    _HAS_SK = False


# -------- FIXED freeze helpers --------
def _freeze_all(model: nn.Module):
    """Freeze all parameters in the model."""
    for p in model.parameters():
        p.requires_grad = False

def _unfreeze_head(model: nn.Module):
    """Unfreeze the classifier/head parameters only."""
    # Try multiple common head names
    head_found = False
    
    # EfficientNet style classifier
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True
        head_found = True
        print(f"Unfroze classifier with {sum(p.numel() for p in model.classifier.parameters()):,} params")
    
    # ResNet style fc layer
    elif hasattr(model, "fc"):
        for p in model.fc.parameters():
            p.requires_grad = True
        head_found = True
        print(f"Unfroze fc layer with {sum(p.numel() for p in model.fc.parameters()):,} params")
    
    # Custom head
    elif hasattr(model, "head"):
        for p in model.head.parameters():
            p.requires_grad = True
        head_found = True
        print(f"Unfroze head with {sum(p.numel() for p in model.head.parameters()):,} params")
    
    if not head_found:
        # Fallback: find the last Linear layer in the model
        last_linear = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                last_linear = (name, module)
        
        if last_linear:
            name, module = last_linear
            for p in module.parameters():
                p.requires_grad = True
            print(f"Unfroze last Linear layer '{name}' with {sum(p.numel() for p in module.parameters()):,} params")
        else:
            raise RuntimeError("Could not find any classifier/fc/head layer to unfreeze.")

def _unfreeze_from_block(model: nn.Module, start_idx: int):
    """
    Unfreeze layers starting from a specific block index.
    Works for EfficientNet-style models with .features sequential blocks.
    """
    if start_idx is None:
        return
        
    unfrozen_params = 0
    
    # EfficientNet style: model.features is a Sequential
    if hasattr(model, "features") and isinstance(model.features, nn.Sequential):
        total_blocks = len(model.features)
        if start_idx >= total_blocks:
            print(f"Warning: start_idx {start_idx} >= total blocks {total_blocks}, no additional layers unfrozen")
            return
            
        for i in range(start_idx, total_blocks):
            for p in model.features[i].parameters():
                p.requires_grad = True
                unfrozen_params += p.numel()
        print(f"Unfroze features[{start_idx}:] with {unfrozen_params:,} params")
    
    # ResNet style: try to find layer blocks
    elif hasattr(model, "layer1"):  # ResNet has layer1, layer2, layer3, layer4
        layers = []
        for i in range(1, 5):  # layer1 through layer4
            if hasattr(model, f"layer{i}"):
                layers.append(getattr(model, f"layer{i}"))
        
        if start_idx < len(layers):
            for i in range(start_idx, len(layers)):
                for p in layers[i].parameters():
                    p.requires_grad = True
                    unfrozen_params += p.numel()
            print(f"Unfroze layer{start_idx+1}+ with {unfrozen_params:,} params")
    
    else:
        print(f"Warning: Could not find features or layer blocks to unfreeze from index {start_idx}")

def enforce_freeze(
    model: nn.Module,
    mode: Literal["keep", "head_only", "partial"] = "keep",
    unfreeze_from: Optional[int] = None,
):
    """
    FIXED freeze enforcement with better layer detection.
    
    - keep: use model's current requires_grad flags
    - head_only: freeze all, unfreeze head only
    - partial: freeze all, unfreeze head + some backbone layers from unfreeze_from
    """
    if mode == "keep":
        return
    
    print(f"Applying freeze mode: {mode}")
    
    # First freeze everything
    _freeze_all(model)
    
    # Then unfreeze the head
    _unfreeze_head(model)
    
    # For partial mode, also unfreeze some backbone layers
    if mode == "partial":
        if unfreeze_from is not None:
            _unfreeze_from_block(model, unfreeze_from)
        else:
            print("Warning: partial mode requested but unfreeze_from is None")


# -------- metrics for binary classification --------
def _binary_metrics_from_probs(probs: torch.Tensor, y_true: torch.Tensor):
    """
    probs: (N,) in [0,1]; y_true: (N,) in {0,1}
    Returns dict: acc@0.5, bal_acc@0.5, best_f1, best_thresh, (optional) auc
    """
    y_true = y_true.long()
    preds05 = (probs >= 0.5).long()
    tp = ((preds05 == 1) & (y_true == 1)).sum().item()
    tn = ((preds05 == 0) & (y_true == 0)).sum().item()
    fp = ((preds05 == 1) & (y_true == 0)).sum().item()
    fn = ((preds05 == 0) & (y_true == 1)).sum().item()

    acc05 = (tp + tn) / max(1, tp + tn + fp + fn)
    tpr = tp / max(1, tp + fn)  # recall+
    tnr = tn / max(1, tn + fp)  # recall-
    bal_acc05 = 0.5 * (tpr + tnr)

    best_f1, best_th = 0.0, 0.5
    for th in torch.linspace(0, 1, 101, device=probs.device):
        p = (probs >= th).long()
        tp = ((p == 1) & (y_true == 1)).sum().item()
        fp = ((p == 1) & (y_true == 0)).sum().item()
        fn = ((p == 0) & (y_true == 1)).sum().item()
        prec = tp / max(1, tp + fp)
        rec  = tp / max(1, tp + fn)
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        if f1 > best_f1:
            best_f1, best_th = f1, float(th)

    out = {"acc@0.5": acc05, "bal_acc@0.5": bal_acc05, "best_f1": best_f1, "best_thresh": best_th}
    if _HAS_SK:
        try:
            out["auc"] = roc_auc_score(y_true.cpu().numpy(), probs.detach().cpu().numpy())
        except Exception:
            pass
    return out


def simple_train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 10,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    optimizer_name: Literal["adamw", "sgd", "radam"] = "sgd",
    task: Literal["binary", "multiclass"] = "binary",
    ckpt_path: str = "best_transfer_model.pth",
    freeze_mode: Literal["keep", "head_only", "partial"] = "head_only",  # Changed default
    unfreeze_from: Optional[int] = None,
    early_stopping_patience: int = 5,  # Added early stopping
):
    """
    IMPROVED training with better defaults and early stopping to prevent overfitting.
    
    - Binary: outputs (B,1), BCEWithLogitsLoss, targets float {0,1}
    - Multiclass: outputs (B,C), CrossEntropyLoss, targets long {0..C-1}
    - Freeze control: keep / head_only / partial(+unfreeze_from)
    - Early stopping to prevent overfitting
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Enforce desired freeze policy
    enforce_freeze(model, mode=freeze_mode, unfreeze_from=unfreeze_from)

    # Count and report parameter status
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = n_trainable + n_frozen
    
    print(f"\n=== Model Parameter Summary ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {n_trainable:,} ({100*n_trainable/total_params:.1f}%)")
    print(f"Frozen parameters: {n_frozen:,} ({100*n_frozen/total_params:.1f}%)")
    print(f"Freeze mode: {freeze_mode}")
    if unfreeze_from is not None:
        print(f"Unfreezing from block: {unfreeze_from}")
    print("=" * 30)

    # Loss & optimizer
    loss_fn = get_loss_function() if task == "binary" else nn.CrossEntropyLoss()
    optimizer = get_optimizer(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer_name=optimizer_name,
    )

    # Early stopping variables
    best_val = float("inf")
    patience_counter = 0
    
    for epoch in range(1, n_epochs + 1):
        # ---- Train ----
        model.train()
        total_train = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)

            if task == "binary":
                loss = loss_fn(logits, yb.float().unsqueeze(1))
            else:
                loss = loss_fn(logits, yb.long())

            loss.backward()
            optimizer.step()
            total_train += loss.item()

        # ---- Validate ----
        model.eval()
        total_val = 0.0
        correct = 0
        n_samples = 0

        all_probs, all_targets = [], []  # for binary metrics

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)

                if task == "binary":
                    loss = loss_fn(logits, yb.float().unsqueeze(1))
                    probs = torch.sigmoid(logits).squeeze(1)
                    preds = (probs >= 0.5).long()
                    all_probs.append(probs.detach()); all_targets.append(yb.detach())
                else:
                    loss = loss_fn(logits, yb.long())
                    preds = logits.argmax(dim=1)

                total_val += loss.item()
                correct += (preds == yb.long()).sum().item()
                n_samples += yb.numel()

        train_loss = total_train / max(1, len(train_loader))
        val_loss   = total_val   / max(1, len(val_loader))
        acc = 100.0 * correct / max(1, n_samples)

        if task == "binary":
            all_probs = torch.cat(all_probs) if all_probs else torch.zeros(0, device=device)
            all_targets = torch.cat(all_targets) if all_targets else torch.zeros(0, device=device, dtype=torch.long)
            m = _binary_metrics_from_probs(all_probs, all_targets) if all_probs.numel() else {"acc@0.5": 0.0, "bal_acc@0.5": 0.0, "best_f1": 0.0, "best_thresh": 0.5}
            auc_str = f"{m.get('auc', float('nan')):.4f}" if "auc" in m else "nan"
            print(f"Epoch {epoch}/{n_epochs} | Train {train_loss:.4f} | Val {val_loss:.4f} | Acc {acc:.2f}%")
            print(f"    AUC: {auc_str} | F1*: {m['best_f1']:.4f} @thr={m['best_thresh']:.2f} | BalAcc@0.5: {m['bal_acc@0.5']:.4f}")
        else:
            print(f"Epoch {epoch}/{n_epochs} | Train {train_loss:.4f} | Val {val_loss:.4f} | Acc {acc:.2f}%")

        # Early stopping and model saving
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            print("  ✓ New best model saved")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{early_stopping_patience})")
            
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

    return model


# =========================
#            TESTS
# =========================
import pytest


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, n: int, task: str = "binary", n_classes: int = 2):
        self.x = torch.randn(n, 3, 224, 224)
        if task == "binary":
            self.y = torch.randint(0, 2, (n,))
        else:
            self.y = torch.randint(0, n_classes, (n,))
        self.task = task
        self.n_classes = n_classes

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def _quick_dataloaders(task="binary", n_classes=2):
    train = ToyDataset(12, task=task, n_classes=n_classes)
    val = ToyDataset(8, task=task, n_classes=n_classes)
    return (
        DataLoader(train, batch_size=4, shuffle=True),
        DataLoader(val, batch_size=4, shuffle=False),
    )


def test_train_binary_with_sgd(tmp_path):
    from src.transfer_learning import get_model_transfer_learning
    model = get_model_transfer_learning(task="binary", freeze_backbone=True)

    train_loader, val_loader = _quick_dataloaders(task="binary")
    ckpt = tmp_path / "best_binary.pth"

    trained = simple_train(
        model,
        train_loader,
        val_loader,
        n_epochs=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        optimizer_name="sgd",
        task="binary",
        ckpt_path=str(ckpt),
        freeze_mode="head_only"
    )

    assert isinstance(trained, nn.Module)
    assert ckpt.exists()
    # shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained.eval()
    x = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        y = trained(x)
    assert y.shape == (1, 1)


def test_train_multiclass_with_radam(tmp_path):
    from src.transfer_learning import get_model_transfer_learning
    model = get_model_transfer_learning(task="multiclass", n_classes=2, freeze_backbone=True)

    train_loader, val_loader = _quick_dataloaders(task="multiclass", n_classes=2)
    ckpt = tmp_path / "best_multiclass.pth"

    trained = simple_train(
        model,
        train_loader,
        val_loader,
        n_epochs=1,
        learning_rate=1e-3,
        weight_decay=1e-4,  # Fixed typo: was weight_delay
        optimizer_name="radam",
        task="multiclass",
        ckpt_path=str(ckpt),
        freeze_mode="head_only"
    )

    assert isinstance(trained, nn.Module)
    assert ckpt.exists()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained.eval()
    x = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        y = trained(x)
    assert y.shape == (1, 2)


def test_partial_unfreeze_counts():
    from src.transfer_learning import get_model_transfer_learning
    model = get_model_transfer_learning(task="binary", freeze_backbone=True)
    # Enforce partial unfreeze of tail blocks (features[5:])
    enforce_freeze(model, mode="partial", unfreeze_from=7)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_froz  = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    assert n_train > 0 and n_froz > 0 and n_froz > n_train