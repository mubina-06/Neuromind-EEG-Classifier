"""
Unit tests for EEG classification models.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from models.model import build_model, build_ensemble
except ImportError:
    pytest.skip("Model modules not available", allow_module_level=True)


class TestModels:
    """Test neural network models."""
    
    def test_resnet18_creation(self):
        """Test ResNet18 model creation."""
        model = build_model("resnet18", num_classes=3, pretrained=False)
        assert model is not None
        
        # Test forward pass
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        assert output.shape == (1, 3)
    
    def test_efficientnet_creation(self):
        """Test EfficientNet-B0 model creation."""
        model = build_model("efficientnet_b0", num_classes=3, pretrained=False)
        assert model is not None
        
        # Test forward pass
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        assert output.shape == (1, 3)
    
    def test_ensemble_model(self):
        """Test ensemble model creation."""
        model1 = build_model("resnet18", num_classes=3, pretrained=False)
        model2 = build_model("efficientnet_b0", num_classes=3, pretrained=False)
        
        ensemble = build_ensemble([model1, model2], weights=[0.6, 0.4])
        assert ensemble is not None
        
        # Test forward pass
        x = torch.randn(1, 3, 224, 224)
        output = ensemble(x)
        assert output.shape == (1, 3)
    
    def test_model_parameters(self):
        """Test model parameter counts."""
        resnet = build_model("resnet18", num_classes=3, pretrained=False)
        efficientnet = build_model("efficientnet_b0", num_classes=3, pretrained=False)
        
        resnet_params = sum(p.numel() for p in resnet.parameters())
        efficientnet_params = sum(p.numel() for p in efficientnet.parameters())
        
        # ResNet18 should have more parameters than EfficientNet-B0
        assert resnet_params > efficientnet_params
        assert resnet_params > 10_000_000  # At least 10M parameters
        assert efficientnet_params > 1_000_000  # At least 1M parameters


if __name__ == "__main__":
    pytest.main([__file__])