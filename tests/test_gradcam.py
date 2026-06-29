"""
Unit tests for Grad-CAM explainability module.

Tests the gradient-weighted class activation mapping implementation
for EEG spectrogram visualization and interpretability.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from utils.gradcam import GradCAM, overlay_heatmap
    from models.model import build_model
except ImportError:
    pytest.skip("GradCAM modules not available", allow_module_level=True)


class MockModel(nn.Module):
    """Simple mock model for testing GradCAM functionality."""
    
    def __init__(self):
        super(MockModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        features = self.relu(self.conv3(x))  # Target layer for GradCAM
        x = self.pool(features)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TestGradCAM:
    """Test Grad-CAM visualization functionality."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a simple mock model for testing."""
        model = MockModel()
        model.eval()
        return model
    
    @pytest.fixture
    def sample_input(self):
        """Create sample EEG spectrogram input."""
        # Simulate 224x224 RGB spectrogram
        batch_size = 1
        channels = 3
        height, width = 224, 224
        return torch.randn(batch_size, channels, height, width)
    
    @pytest.fixture
    def gradcam_instance(self, mock_model):
        """Create GradCAM instance with mock model."""
        target_layer = mock_model.conv3  # Use last conv layer
        return GradCAM(mock_model, target_layer)
    
    def test_gradcam_initialization(self, mock_model):
        """Test GradCAM object initialization."""
        target_layer = mock_model.conv3
        gradcam = GradCAM(mock_model, target_layer)
        
        assert gradcam.model == mock_model
        assert gradcam.target_layer == target_layer
        assert gradcam.gradients is None
        assert gradcam.activations is None
    
    def test_gradcam_heatmap_generation(self, gradcam_instance, sample_input):
        """Test heatmap generation with valid inputs."""
        class_idx = 0  # Test with first class (Focused)
        
        # Generate heatmap
        heatmap = gradcam_instance.generate_heatmap(sample_input, class_idx)
        
        # Validate heatmap properties
        assert isinstance(heatmap, torch.Tensor)
        assert heatmap.ndim == 3  # Should be 2D heatmap with batch dimension
        assert heatmap.shape[0] == 1  # Batch size
        assert heatmap.min() >= 0  # ReLU applied, should be non-negative
        assert not torch.isnan(heatmap).any()  # No NaN values
        assert not torch.isinf(heatmap).any()  # No infinite values
    
    def test_gradcam_different_classes(self, gradcam_instance, sample_input):
        """Test heatmap generation for all mental state classes."""
        classes = [0, 1, 2]  # Focused, Relaxed, Stressed
        heatmaps = []
        
        for class_idx in classes:
            heatmap = gradcam_instance.generate_heatmap(sample_input, class_idx)
            heatmaps.append(heatmap)
            
            # Each heatmap should be valid
            assert isinstance(heatmap, torch.Tensor)
            assert heatmap.shape[0] == 1
        
        # Heatmaps for different classes should be different
        # (with high probability for random model)
        assert not torch.equal(heatmaps[0], heatmaps[1])
        assert not torch.equal(heatmaps[1], heatmaps[2])
    
    def test_gradcam_batch_processing(self, gradcam_instance):
        """Test GradCAM with batch input."""
        batch_size = 4
        batch_input = torch.randn(batch_size, 3, 224, 224)
        
        # Should handle batch input gracefully
        heatmap = gradcam_instance.generate_heatmap(batch_input, class_idx=0)
        
        assert isinstance(heatmap, torch.Tensor)
        assert heatmap.shape[0] == batch_size
    
    def test_gradcam_invalid_class_index(self, gradcam_instance, sample_input):
        """Test GradCAM behavior with invalid class indices."""
        # Test with out-of-bounds class index
        with pytest.raises((IndexError, RuntimeError)):
            gradcam_instance.generate_heatmap(sample_input, class_idx=10)
        
        # Test with negative class index
        with pytest.raises((IndexError, RuntimeError)):
            gradcam_instance.generate_heatmap(sample_input, class_idx=-1)
    
    def test_gradcam_empty_input(self, gradcam_instance):
        """Test GradCAM behavior with empty or malformed input."""
        # Test with wrong input dimensions
        with pytest.raises((RuntimeError, ValueError)):
            empty_input = torch.tensor([])
            gradcam_instance.generate_heatmap(empty_input, class_idx=0)
        
        # Test with wrong number of channels
        with pytest.raises(RuntimeError):
            wrong_channels = torch.randn(1, 5, 224, 224)  # 5 channels instead of 3
            gradcam_instance.generate_heatmap(wrong_channels, class_idx=0)
    
    def test_heatmap_overlay_function(self, sample_input):
        """Test heatmap overlay functionality."""
        # Create mock heatmap
        heatmap = torch.rand(1, 224, 224)
        
        # Convert input tensor to PIL Image
        sample_image = sample_input.squeeze(0).permute(1, 2, 0).numpy()
        sample_image = (sample_image - sample_image.min()) / (sample_image.max() - sample_image.min())
        sample_image = Image.fromarray((sample_image * 255).astype(np.uint8))
        
        # Test overlay function
        overlaid_image = overlay_heatmap(sample_image, heatmap, alpha=0.4)
        
        assert isinstance(overlaid_image, Image.Image)
        assert overlaid_image.size == sample_image.size
        assert overlaid_image.mode in ['RGB', 'RGBA']
    
    def test_heatmap_overlay_parameters(self, sample_input):
        """Test heatmap overlay with different alpha values."""
        heatmap = torch.rand(1, 224, 224)
        
        # Convert to PIL Image
        sample_image = sample_input.squeeze(0).permute(1, 2, 0).numpy()
        sample_image = (sample_image - sample_image.min()) / (sample_image.max() - sample_image.min())
        sample_image = Image.fromarray((sample_image * 255).astype(np.uint8))
        
        # Test different alpha values
        alpha_values = [0.0, 0.3, 0.5, 0.7, 1.0]
        
        for alpha in alpha_values:
            overlaid = overlay_heatmap(sample_image, heatmap, alpha=alpha)
            assert isinstance(overlaid, Image.Image)
            assert overlaid.size == sample_image.size
    
    def test_gradcam_numerical_stability(self, gradcam_instance):
        """Test GradCAM numerical stability with extreme inputs."""
        # Test with very small values
        small_input = torch.full((1, 3, 224, 224), 1e-8)
        heatmap_small = gradcam_instance.generate_heatmap(small_input, class_idx=0)
        assert torch.isfinite(heatmap_small).all()
        
        # Test with large values
        large_input = torch.full((1, 3, 224, 224), 1e3)
        heatmap_large = gradcam_instance.generate_heatmap(large_input, class_idx=0)
        assert torch.isfinite(heatmap_large).all()
    
    def test_gradcam_reproducibility(self, gradcam_instance, sample_input):
        """Test that GradCAM produces consistent results."""
        class_idx = 1
        
        # Generate heatmap twice with same input
        torch.manual_seed(42)  # Set seed for reproducibility
        heatmap1 = gradcam_instance.generate_heatmap(sample_input.clone(), class_idx)
        
        torch.manual_seed(42)  # Reset seed
        heatmap2 = gradcam_instance.generate_heatmap(sample_input.clone(), class_idx)
        
        # Results should be identical (or very close due to floating point)
        assert torch.allclose(heatmap1, heatmap2, atol=1e-6)


class TestGradCAMIntegration:
    """Integration tests for GradCAM with actual models."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_gradcam_with_gpu(self, mock_model, sample_input):
        """Test GradCAM functionality on GPU if available."""
        device = torch.device('cuda')
        
        # Move model and input to GPU
        model_gpu = mock_model.to(device)
        input_gpu = sample_input.to(device)
        
        # Create GradCAM instance
        gradcam = GradCAM(model_gpu, model_gpu.conv3)
        
        # Generate heatmap on GPU
        heatmap = gradcam.generate_heatmap(input_gpu, class_idx=0)
        
        assert heatmap.device == device
        assert isinstance(heatmap, torch.Tensor)
    
    def test_gradcam_memory_usage(self, gradcam_instance, sample_input):
        """Test GradCAM memory efficiency."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss
        
        # Generate multiple heatmaps
        for i in range(10):
            heatmap = gradcam_instance.generate_heatmap(sample_input, class_idx=i % 3)
            del heatmap  # Explicitly delete to help garbage collection
        
        memory_after = process.memory_info().rss
        memory_increase = (memory_after - memory_before) / 1024 / 1024  # MB
        
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100, f"Memory usage increased by {memory_increase:.1f}MB"


class TestGradCAMVisualization:
    """Test visualization and export functionality."""
    
    def test_heatmap_normalization(self):
        """Test heatmap normalization for visualization."""
        # Create sample heatmap with known values
        raw_heatmap = torch.tensor([[[1.0, 2.0, 3.0],
                                    [4.0, 5.0, 6.0],
                                    [7.0, 8.0, 9.0]]])
        
        # Normalize to 0-1 range
        normalized = (raw_heatmap - raw_heatmap.min()) / (raw_heatmap.max() - raw_heatmap.min())
        
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        assert normalized.shape == raw_heatmap.shape
    
    def test_colormap_application(self):
        """Test colormap application to heatmaps."""
        import matplotlib.cm as cm
        
        # Create normalized heatmap
        heatmap = torch.rand(1, 64, 64)
        heatmap_np = heatmap.squeeze().numpy()
        
        # Apply colormap
        colored_heatmap = cm.hot(heatmap_np)
        
        assert colored_heatmap.shape == (64, 64, 4)  # RGBA
        assert colored_heatmap.min() >= 0.0
        assert colored_heatmap.max() <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])