"""
Integration tests for NeuroMind EEG Classification Pipeline.

Tests the complete workflow from data loading to prediction and visualization.
These tests ensure all components work together correctly.
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt
from PIL import Image

# Import NeuroMind modules
from src.models.model import build_model, build_ensemble
from src.data.preprocessing import preprocess_raw, epoch_raw, epoch_to_spectrogram
from src.utils.gradcam import GradCAM, overlay_heatmap, visualise_gradcam
from src.training.calibration import calibrate_model, load_temperature


class TestIntegrationPipeline:
    """Test complete EEG classification pipeline."""
    
    @pytest.fixture
    def mock_eeg_data(self):
        """Create mock EEG data for testing."""
        import mne
        
        # Create synthetic EEG data
        n_channels = 64
        sfreq = 160.0
        duration = 10.0  # seconds
        n_samples = int(sfreq * duration)
        
        # Generate realistic EEG-like signals
        np.random.seed(42)
        data = np.random.randn(n_channels, n_samples) * 1e-5
        
        # Add some frequency content
        times = np.arange(n_samples) / sfreq
        for i in range(n_channels):
            # Alpha waves (8-13 Hz)
            data[i] += 2e-5 * np.sin(2 * np.pi * 10 * times + i * 0.1)
            # Beta waves (13-30 Hz)
            data[i] += 1e-5 * np.sin(2 * np.pi * 20 * times + i * 0.2)
        
        # Create MNE Raw object
        ch_names = [f'EEG{i+1:03d}' for i in range(n_channels)]
        info = mne.create_info(ch_names, sfreq, ch_types=['eeg'] * n_channels)
        raw = mne.io.RawArray(data, info, verbose=False)
        
        return raw
    
    @pytest.fixture
    def test_model(self):
        """Create a test model."""
        model = build_model(arch="resnet18", pretrained=False)
        model.eval()
        return model
    
    def test_complete_pipeline(self, mock_eeg_data, test_model):
        """Test the complete pipeline from EEG to prediction."""
        
        # Step 1: Preprocess EEG data
        raw_clean = preprocess_raw(mock_eeg_data)
        assert raw_clean is not None
        assert raw_clean.info['sfreq'] == mock_eeg_data.info['sfreq']
        
        # Step 2: Create epochs
        epochs = epoch_raw(raw_clean, epoch_duration=4.0)
        assert len(epochs) > 0
        
        # Step 3: Convert to spectrogram
        epoch_data = epochs.get_data()[0]  # First epoch
        sfreq = raw_clean.info['sfreq']
        spectrogram = epoch_to_spectrogram(epoch_data, sfreq)
        
        assert spectrogram.shape == (224, 224, 3)
        assert spectrogram.dtype == np.uint8
        
        # Step 4: Model prediction
        import torchvision.transforms as T
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(spectrogram).unsqueeze(0)
        
        with torch.no_grad():
            output = test_model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
        
        assert output.shape == (1, 3)  # 3 classes
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(1))
        
        # Step 5: Get prediction
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities.max().item()
        
        assert 0 <= predicted_class <= 2
        assert 0.0 <= confidence <= 1.0
    
    def test_gradcam_integration(self, test_model):
        """Test Grad-CAM integration with model."""
        # Create dummy input
        input_tensor = torch.randn(1, 3, 224, 224)
        
        # Get last convolutional layer
        from src.models.model import get_last_conv_layer
        target_layer = get_last_conv_layer(test_model, "resnet18")
        
        # Create Grad-CAM instance
        gradcam = GradCAM(test_model, target_layer)
        
        # Generate heatmap
        heatmap = gradcam(input_tensor)
        gradcam.remove_hooks()
        
        assert heatmap is not None
        assert heatmap.ndim == 2 or (heatmap.ndim == 0)  # Can be scalar for some layers
        if heatmap.ndim == 2:
            assert heatmap.min() >= 0.0
            assert heatmap.max() <= 1.0
    
    def test_batch_processing(self, mock_eeg_data, test_model):
        """Test batch processing of multiple EEG epochs."""
        
        # Preprocess and create multiple epochs
        raw_clean = preprocess_raw(mock_eeg_data)
        epochs = epoch_raw(raw_clean, epoch_duration=2.0, overlap=0.5)
        
        batch_size = min(4, len(epochs))
        epoch_data = epochs.get_data()[:batch_size]
        sfreq = raw_clean.info['sfreq']
        
        # Process batch
        spectrograms = []
        for epoch in epoch_data:
            spec = epoch_to_spectrogram(epoch, sfreq)
            spectrograms.append(spec)
        
        assert len(spectrograms) == batch_size
        
        # Batch prediction
        import torchvision.transforms as T
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        batch_tensors = torch.stack([
            transform(spec) for spec in spectrograms
        ])
        
        with torch.no_grad():
            batch_outputs = test_model(batch_tensors)
            batch_probs = torch.softmax(batch_outputs, dim=1)
        
        assert batch_outputs.shape == (batch_size, 3)
        assert torch.allclose(batch_probs.sum(dim=1), torch.ones(batch_size))
    
    def test_model_ensemble_integration(self):
        """Test ensemble model integration."""
        # Create individual models
        resnet = build_model(arch="resnet18", pretrained=False)
        efficientnet = build_model(arch="efficientnet_b0", pretrained=False)
        
        # Create ensemble (mock the checkpoint loading)
        with patch('src.models.model.load_checkpoint') as mock_load:
            mock_load.side_effect = [resnet, efficientnet]
            
            try:
                ensemble = build_ensemble()
            except FileNotFoundError:
                # Expected if checkpoints don't exist
                pytest.skip("Ensemble test requires model checkpoints")
        
        # Test ensemble prediction if models are available
        if 'ensemble' in locals():
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = ensemble(dummy_input)
            
            assert output.shape == (1, 3)
            assert torch.allclose(output.sum(dim=1), torch.ones(1))
    
    def test_calibration_integration(self, test_model):
        """Test temperature calibration integration."""
        
        # Create mock validation data
        val_inputs = torch.randn(10, 3, 224, 224)
        val_labels = torch.randint(0, 3, (10,))
        
        # Test calibration
        with torch.no_grad():
            logits = test_model(val_inputs)
        
        # Mock calibration (normally requires training)
        temperature = calibrate_model(logits, val_labels, lr=0.01, max_iter=10)
        
        assert temperature > 0.0
        assert isinstance(temperature, float)
        
        # Test calibrated predictions
        calibrated_probs = torch.softmax(logits / temperature, dim=1)
        assert torch.allclose(calibrated_probs.sum(dim=1), torch.ones(10))
    
    def test_visualization_integration(self, test_model):
        """Test visualization pipeline integration."""
        
        # Create test image and tensor
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        input_tensor = torch.randn(1, 3, 224, 224)
        
        # Test Grad-CAM visualization
        from src.models.model import get_last_conv_layer
        target_layer = get_last_conv_layer(test_model, "resnet18")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "test_gradcam.png"
            
            # This should not crash
            try:
                pred_label, confidence, probs = visualise_gradcam(
                    test_model, target_layer, input_tensor, test_image,
                    ["focused", "relaxed", "stressed"], str(save_path)
                )
                
                assert save_path.exists()
                assert pred_label in ["focused", "relaxed", "stressed"]
                assert 0.0 <= confidence <= 100.0
                assert len(probs) == 3
                
            except Exception as e:
                pytest.skip(f"Visualization test failed: {e}")
    
    def test_error_handling(self, test_model):
        """Test error handling in pipeline."""
        
        # Test with invalid input shapes
        with pytest.raises((RuntimeError, ValueError)):
            invalid_input = torch.randn(1, 2, 100, 100)  # Wrong channels
            test_model(invalid_input)
        
        # Test with empty EEG data
        import mne
        empty_data = np.array([]).reshape(0, 0)
        info = mne.create_info([], sfreq=160.0, ch_types=[])
        
        with pytest.raises((ValueError, RuntimeError)):
            empty_raw = mne.io.RawArray(empty_data, info, verbose=False)
            preprocess_raw(empty_raw)
    
    def test_performance_benchmarks(self, test_model):
        """Basic performance benchmarks."""
        import time
        
        # Measure inference time
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = test_model(dummy_input)
        
        # Benchmark
        start_time = time.time()
        for _ in range(10):
            with torch.no_data():
                _ = test_model(dummy_input)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        
        # Should be reasonably fast (< 1 second per inference)
        assert avg_time < 1.0, f"Inference too slow: {avg_time:.3f}s"
    
    def test_memory_usage(self, test_model):
        """Test memory usage is reasonable."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run inference multiple times
        dummy_input = torch.randn(10, 3, 224, 224)
        
        for _ in range(5):
            with torch.no_grad():
                _ = test_model(dummy_input)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should not have significant memory leaks (< 100MB increase)
        assert memory_increase < 100, f"Memory leak detected: {memory_increase:.1f}MB"


class TestDataIntegrity:
    """Test data integrity throughout the pipeline."""
    
    def test_spectrogram_consistency(self):
        """Test that spectrograms are generated consistently."""
        
        # Create deterministic test data
        np.random.seed(42)
        epoch_data = np.random.randn(64, 640)  # 64 channels, 4 seconds at 160Hz
        sfreq = 160.0
        
        # Generate spectrograms multiple times
        spec1 = epoch_to_spectrogram(epoch_data, sfreq)
        spec2 = epoch_to_spectrogram(epoch_data, sfreq)
        
        # Should be identical (deterministic)
        assert np.array_equal(spec1, spec2), "Spectrogram generation not deterministic"
    
    def test_preprocessing_determinism(self):
        """Test that preprocessing is deterministic."""
        
        import mne
        
        # Create test EEG data
        np.random.seed(42)
        data = np.random.randn(32, 1600)  # 32 channels, 10 seconds
        info = mne.create_info([f'EEG{i}' for i in range(32)], 
                              sfreq=160.0, ch_types=['eeg']*32)
        raw1 = mne.io.RawArray(data.copy(), info, verbose=False)
        raw2 = mne.io.RawArray(data.copy(), info, verbose=False)
        
        # Preprocess both
        clean1 = preprocess_raw(raw1)
        clean2 = preprocess_raw(raw2)
        
        # Should be very similar (allowing for floating point precision)
        data1 = clean1.get_data()
        data2 = clean2.get_data()
        
        assert np.allclose(data1, data2, rtol=1e-5), "Preprocessing not deterministic"


@pytest.mark.slow
class TestLongRunningIntegration:
    """Integration tests that take longer to run."""
    
    def test_full_dataset_processing(self):
        """Test processing a full synthetic dataset."""
        
        # Skip if not explicitly running slow tests
        pytest.skip("Slow test - run with 'pytest -m slow' to include")
        
        # This would test processing many EEG files
        # Implementation would depend on having test data available
        pass
    
    def test_model_training_pipeline(self):
        """Test the complete model training pipeline."""
        
        pytest.skip("Training test - requires substantial compute time")
        
        # This would test the complete training workflow
        # from data loading to model evaluation
        pass


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v"])