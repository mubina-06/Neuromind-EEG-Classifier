"""
preprocess.py
-------------
EEG signal processing pipeline using PhysioNet EEGBCI dataset.
Handles data download, filtering, epoching, and spectrogram generation.
"""

import numpy as np
import mne
from mne.datasets import eegbci
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# Label definitions
CLASSES = ["focused", "relaxed", "stressed"]
LABEL_MAP = {0: "focused", 1: "relaxed", 2: "stressed"}

# PhysioNet EEGBCI run mapping → mental state
RUN_LABEL = {
    1: 1,   # relaxed (eyes open)
    2: 1,   # relaxed (eyes closed)
    3: 2,   # stressed (real movement)
    4: 0,   # focused (motor imagery)
    5: 2,   # stressed (real movement)
    6: 0,   # focused (motor imagery)
}


def load_physionet_data(n_subjects=10, runs=None):
    """
    Downloads and loads real EEG data from PhysioNet EEGBCI dataset.
    
    Parameters
    ----------
    n_subjects : int
        Number of subjects to load (1-109)
    runs : list
        Run numbers to use (default: [1,3,4])
    
    Returns
    -------
    raw_list : list of mne.io.Raw
    labels : list of int (0=focused, 1=relaxed, 2=stressed)
    """
    if runs is None:
        runs = [1, 3, 4]  # rest, real movement, imagined movement

    raw_list, labels = [], []
    
    print(f"[INFO] Loading PhysioNet EEGBCI — {n_subjects} subjects, runs {runs}")
    print("[INFO] Auto-downloading if not cached (~50MB first time)...")

    for subj in range(1, n_subjects + 1):
        for run in runs:
            try:
                # MNE auto-downloads from PhysioNet
                fnames = eegbci.load_data(subj, [run], verbose=False)
                raw = mne.io.read_raw_edf(fnames[0], preload=True, verbose=False)
                
                # Standardize channel names
                eegbci.standardize(raw)
                mne.datasets.eegbci.standardize(raw)
                
                # Set montage
                montage = mne.channels.make_standard_montage("standard_1005")
                raw.set_montage(montage, on_missing="ignore", verbose=False)
                
                raw_list.append(raw)
                labels.append(RUN_LABEL[run])
                
            except Exception as e:
                print(f"  [WARN] Subject {subj} run {run} failed: {e}")
                continue

    if not raw_list:
        raise RuntimeError("Failed to load any PhysioNet data. Check internet connection.")

    print(f"[INFO] Loaded {len(raw_list)} recordings.")
    _print_distribution(labels)
    return raw_list, labels


def preprocess_raw(raw):
    """
    Apply EEG preprocessing: bandpass filter, notch filter, average reference.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    
    Returns
    -------
    raw : mne.io.Raw
        Preprocessed EEG data
    """
    raw = raw.copy()
    
    # Bandpass filter: 4-45 Hz (all relevant brain waves)
    raw.filter(l_freq=4.0, h_freq=45.0, method="fir", verbose=False)
    
    # Notch filter: 60 Hz (US power line interference)
    raw.notch_filter(freqs=60.0, verbose=False)
    
    # Average reference: standardize across channels
    raw.set_eeg_reference("average", projection=False, verbose=False)
    
    return raw


def epoch_raw(raw, epoch_duration=4.0, overlap=0.5):
    """
    Segment EEG data into fixed-length epochs.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed EEG data
    epoch_duration : float
        Length of each epoch in seconds
    overlap : float
        Overlap between epochs (0.0 to 1.0)
    
    Returns
    -------
    epochs : mne.Epochs
        Segmented EEG data
    """
    events = mne.make_fixed_length_events(
        raw, duration=epoch_duration, overlap=overlap
    )
    epochs = mne.Epochs(
        raw, events, tmin=0, tmax=epoch_duration,
        baseline=None, reject={"eeg": 500e-6},  # 500µV threshold
        preload=True, verbose=False,
    )
    return epochs


def epoch_to_spectrogram(epoch_data, sfreq, fmin=4, fmax=45, img_size=(224, 224)):
    """
    Convert EEG epoch to STFT spectrogram image.
    
    Parameters
    ----------
    epoch_data : np.ndarray
        EEG epoch data (channels × time)
    sfreq : float
        Sampling frequency
    fmin, fmax : float
        Frequency range for spectrogram
    img_size : tuple
        Output image size
    
    Returns
    -------
    np.ndarray
        Spectrogram image (RGB)
    """
    from scipy.signal import stft
    
    n_channels, n_times = epoch_data.shape
    nperseg = min(256, n_times // 4)
    noverlap = nperseg // 2
    all_specs = []
    
    # Compute STFT for each channel
    for ch in range(n_channels):
        f, t, Zxx = stft(epoch_data[ch], fs=sfreq,
                         nperseg=nperseg, noverlap=noverlap)
        power = np.abs(Zxx) ** 2
        mask = (f >= fmin) & (f <= fmax)
        all_specs.append(power[mask])
    
    # Average across channels
    mean_spec = np.mean(all_specs, axis=0)
    
    # Log transform and normalize
    mean_spec = np.log1p(mean_spec)
    mean_spec -= mean_spec.min()
    if mean_spec.max() > 0:
        mean_spec /= mean_spec.max()
    
    # Convert to uint8
    mean_spec = (mean_spec * 255).astype(np.uint8)
    
    # Render as image
    fig, ax = plt.subplots(figsize=(img_size[1]/100, img_size[0]/100), dpi=100)
    ax.imshow(mean_spec, aspect="auto", origin="lower", cmap="viridis")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    
    # Convert to numpy array
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)
    
    return np.array(Image.fromarray(buf).resize(img_size, Image.LANCZOS))


def save_spectrograms(raw_list, labels, output_dir="data/spectrograms", epoch_duration=4.0):
    """
    Complete pipeline: preprocess → epoch → STFT → save PNG images.
    
    Parameters
    ----------
    raw_list : list of mne.io.Raw
        List of raw EEG recordings
    labels : list of int
        Corresponding labels for each recording
    output_dir : str
        Output directory for spectrograms
    epoch_duration : float
        Duration of each epoch in seconds
    
    Returns
    -------
    str
        Path to output directory
    """
    output_dir = Path(output_dir)
    for cls in CLASSES:
        (output_dir / cls).mkdir(parents=True, exist_ok=True)
    
    total = 0
    for i, (raw, label) in enumerate(zip(raw_list, labels)):
        cls_name = LABEL_MAP[label]
        print(f"[INFO] Recording {i+1}/{len(raw_list)} → {cls_name}")
        
        try:
            # Preprocess
            raw_processed = preprocess_raw(raw)
            epochs = epoch_raw(raw_processed, epoch_duration=epoch_duration)
            
            if len(epochs) == 0:
                continue
                
            # Generate spectrograms
            data = epochs.get_data()
            sfreq = raw.info["sfreq"]
            
            for j, epoch in enumerate(data):
                img = epoch_to_spectrogram(epoch, sfreq)
                fname = output_dir / cls_name / f"s{i:03d}_e{j:03d}.png"
                Image.fromarray(img).save(fname)
                total += 1
                
        except Exception as e:
            print(f"  [WARN] Skipped: {e}")
    
    print(f"[INFO] Saved {total} spectrogram images → '{output_dir}'")
    return str(output_dir)


def _print_distribution(labels):
    """Print label distribution."""
    from collections import Counter
    c = Counter(labels)
    print("[INFO] Label distribution:")
    for k, v in sorted(c.items()):
        print(f"  {LABEL_MAP[k]:<10}: {v}")


if __name__ == "__main__":
    # Example usage
    raw_list, labels = load_physionet_data(n_subjects=5)
    save_spectrograms(raw_list, labels)