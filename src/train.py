# src/train.py
"""
Simplified training - only trains the classifier head, much faster!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def simple_train(model, train_loader, val_loader, n_epochs=10, lr=0.001):
    """
    Simple training function for transfer learning.
    Only the classifier head gets trained - much faster!
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Only optimize the classifier parameters (backbone is frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training only {sum(p.numel() for p in trainable_params):,} parameters")
    print(f"Frozen: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,} parameters")
    
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        accuracy = 100. * correct / len(val_loader.dataset)
        
        print(f'Epoch {epoch+1}/{n_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Val Accuracy: {accuracy:.2f}%')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_transfer_model.pth')
            print(f'  ✓ New best model saved!')
        print()
    
    print("Training completed!")
    return model


# =============================================================================
#                                    TESTS
# =============================================================================

import pytest
import tempfile
import os


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, n: int):
        self.x = torch.randn(n, 3, 224, 224)
        self.y = torch.randint(0, 2, (n,))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def test_simple_train_functionality():
    """Test that simple_train function works without errors"""
    try:
        from model import get_model_transfer_learning
        model = get_model_transfer_learning(n_classes=2)
    except ImportError:
        import torchvision.models as models
        model = models.efficientnet_b0(weights=None)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Linear(num_ftrs, 2)
    
    train_dataset = ToyDataset(16)
    val_dataset = ToyDataset(8)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        original_cwd = os.getcwd()
        os.chdir(tmp_dir)
        
        try:
            trained_model = simple_train(model, train_loader, val_loader, n_epochs=2, lr=0.01)
            
            assert trained_model is not None
            assert isinstance(trained_model, nn.Module)
            assert os.path.exists('best_transfer_model.pth')
            
            trained_model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            test_input = torch.randn(1, 3, 224, 224).to(device)
            with torch.no_grad():
                output = trained_model(test_input)
            
            assert output.shape == (1, 2)
            
        finally:
            os.chdir(original_cwd)


def test_parameter_freezing():
    """Test that only classifier parameters are trainable"""
    try:
        from model import get_model_transfer_learning
        model = get_model_transfer_learning(n_classes=2)
    except ImportError:
        import torchvision.models as models
        model = models.efficientnet_b0(weights=None)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Linear(num_ftrs, 2)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    frozen_params = [p for p in model.parameters() if not p.requires_grad]
    
    assert len(trainable_params) > 0
    assert len(frozen_params) > 0
    
    total_trainable = sum(p.numel() for p in trainable_params)
    total_frozen = sum(p.numel() for p in frozen_params)
    
    assert total_trainable < total_frozen


def test_training_step():
    """Test individual training components work"""
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 224 * 224, 10),
        nn.ReLU(),
        nn.Linear(10, 2)
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    test_input = torch.randn(2, 3, 224, 224).to(device)
    output = model(test_input)
    assert output.shape == (2, 2)
    
    criterion = nn.CrossEntropyLoss()
    target = torch.randint(0, 2, (2,)).to(device)
    loss = criterion(output, target)
    assert isinstance(loss.item(), float)
    assert loss.item() > 0
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()