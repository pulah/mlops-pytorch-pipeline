import sys
from pathlib import Path

# Add the repository root directory to sys.path to enable imports of the src package
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest
import torch
import torch.nn as nn
from src.model import get_model
from src.dataset import get_transforms

def test_get_model_resnet18():
    model = get_model(architecture="resnet18", num_classes=10)
    assert isinstance(model, nn.Module)
    
    # Test output shape with random tensor mimicking CIFAR-10 shape
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10)

def test_get_model_unsupported():
    with pytest.raises(ValueError):
        get_model(architecture="unsupported_arch", num_classes=10)

def test_get_transforms():
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)
    
    assert train_transform is not None
    assert val_transform is not None
