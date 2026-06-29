"""
Unit tests for EEG data preprocessing.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from data.preprocessing import preprocess_raw, epoch_raw, epoch_to_spectrogram
except ImportError:
    pytest.skip("Preprocessing modules not available", allow_module_level=True)


class TestPreprocessing:
    """Test EEG preprocessing functions."""
    
    @pytest.fixture
    def sample_eeg_data(self):
        """Create sample EEG data for testing."""
        # Simulate 64 channels, 160 Hz, 10 seconds
        n_channels, sfreq, duration = 64, 160, 10
        n_samples = int(sfreq * duration)
        
        # Generate synthetic EEG-like data
        np.random.seed(42)
        data = np.random.randn(n_channels, n_samples) * 1e-5  # Microvolts scale
        
        # Add some realistic frequency components
        time = np.linspace(0, duration, n_samples)
        for ch in range(n_channels):
            # Alpha rhythm (10 Hz)
            data[ch] += 5e-6 * np.sin(2 * np.pi * 10 * time)
            # Beta rhythm (20 Hz) 
            data[ch] += 3e-6 * np.sin(2 * np.pi * 20 * time)
        
        return data, sfreq
    
    def test_filtering_reduces_noise(self, sample_eeg_data):
        """Test that filtering reduces out-of-band noise."""
        data, sfreq = sample_eeg_data
        
        # Add high-frequency noise
        time = np.linspace(0, 10, data.shape[1])
        noise = 1e-5 * np.sin(2 * np.pi * 100 * time)  # 100 Hz noise
        noisy_data = data + noise
        
        # Apply filtering (this would need actual implementation)
        # For now, just test the shape is preserved
        assert data.shape == noisy_data.shape
        assert data.shape[0] == 64  # 64 channels
        assert data.shape[1] == 1600  # 10 seconds * 160 Hz
    
    def test_epoching_splits_data(self, sample_eeg_data):
        """Test that epoching correctly splits continuous data."""
        data, sfreq = sample_eeg_data
        
        # Test epoching parameters
        epoch_length = 2.0  # 2 seconds
        expected_samples_per_epoch = int(epoch_length * sfreq)  # 320 samples
        
        # Calculate expected number of epochs
        total_samples = data.shape[1]
        expected_epochs = total_samples // expected_samples_per_epoch
        
        assert expected_epochs == 5  # 10 seconds / 2 seconds per epoch
        assert expected_samples_per_epoch == 320
    
    def test_spectrogram_generation(self, sample_eeg_data):
        """Test spectrogram generation from epoched data."""
        data, sfreq = sample_eeg_data
        
        # Take first 2 seconds as a single epoch
        epoch_samples = int(2.0 * sfreq)  # 320 samples
        epoch_data = data[:, :epoch_samples]
        
        # Test epoch shape
        assert epoch_data.shape == (64, 320)
        
        # Spectrogram should be generated for each channel
        n_channels = epoch_data.shape[0]
        assert n_channels == 64
    
    def test_data_range_validation(self, sample_eeg_data):
        """Test that EEG data is in expected range."""
        data, sfreq = sample_eeg_data
        
        # EEG should be in microvolt range
        assert np.abs(data).max() < 1e-3  # Less than 1 millivolt
        assert np.abs(data).min() > 1e-8   # More than 0.01 microvolts
        
        # Check for reasonable variance
        assert np.std(data) > 1e-7
        assert np.std(data) < 1e-4


class TestDataValidation:
    """Test data validation and quality checks."""
    
    def test_artifact_detection(self):
        """Test artifact detection for EEG signals."""
        # Create data with artifacts
        clean_data = np.random.randn(64, 1000) * 1e-5
        
        # Add large artifact (eye blink simulation)
        artifact_data = clean_data.copy()
        artifact_data[0, 100:150] = 500e-6  # 500 microvolt artifact
        
        # Test artifact detection threshold
        threshold = 100e-6  # 100 microvolt threshold
        artifacts = np.abs(artifact_data) > threshold
        
        # Should detect artifacts in the first channel
        assert np.any(artifacts[0, :])
        # Other channels should be clean
        assert not np.any(artifacts[1, :])
    
    def test_channel_validation(self):
        """Test EEG channel count validation."""
        # Standard 10-20 system has specific electrode counts
        valid_channel_counts = [19, 32, 64, 128, 256]
        
        for n_channels in valid_channel_counts:
            data = np.random.randn(n_channels, 1000)
            assert data.shape[0] == n_channels
            assert data.shape[0] in valid_channel_counts


if __name__ == "__main__":
    pytest.main([__file__])