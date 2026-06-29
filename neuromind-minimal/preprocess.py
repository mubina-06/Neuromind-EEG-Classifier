"""EEG preprocessing and spectrogram generation"""
import numpy as np
import mne
from mne.datasets import eegbci
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

CLASSES = ["focused", "relaxed", "stressed"]
LABEL_MAP = {0: "focused", 1: "relaxed", 2: "stressed"}
RUN_LABEL = {1: 1, 3: 2, 4: 0}  # rest->relaxed, real->stressed, imagery->focused

def load_eeg_data(n_subjects=5):
    """Load PhysioNet EEG data"""
    raw_list, labels = [], []
    for subj in range(1, n_subjects + 1):
        for run in [1, 3, 4]:
            try:
                fnames = eegbci.load_data(subj, [run], verbose=False)
                raw = mne.io.read_raw_edf(fnames[0], preload=True, verbose=False)
                eegbci.standardize(raw)
                raw_list.append(raw)
                labels.append(RUN_LABEL[run])
            except:
                continue
    return raw_list, labels

def preprocess_eeg(raw):
    """Apply filters and referencing"""
    raw = raw.copy()
    raw.filter(4.0, 45.0, method="fir", verbose=False)
    raw.notch_filter(60.0, verbose=False)
    raw.set_eeg_reference("average", verbose=False)
    return raw

def create_spectrograms(raw_list, labels, output_dir="data"):
    """Generate spectrograms from EEG data"""
    from scipy.signal import stft
    
    output_dir = Path(output_dir)
    for cls in CLASSES:
        (output_dir / cls).mkdir(parents=True, exist_ok=True)
    
    total = 0
    for i, (raw, label) in enumerate(zip(raw_list, labels)):
        raw_proc = preprocess_eeg(raw)
        events = mne.make_fixed_length_events(raw_proc, duration=4.0)
        epochs = mne.Epochs(raw_proc, events, tmin=0, tmax=4.0, 
                           baseline=None, preload=True, verbose=False)
        
        for j, epoch in enumerate(epochs.get_data()):
            # Average across channels and compute STFT
            avg_signal = np.mean(epoch, axis=0)
            f, t, Zxx = stft(avg_signal, fs=raw.info["sfreq"], nperseg=256)
            power = np.abs(Zxx) ** 2
            power = np.log1p(power)
            
            # Normalize and save as image
            power = (power - power.min()) / (power.max() - power.min())
            power = (power * 255).astype(np.uint8)
            
            fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
            ax.imshow(power, cmap="viridis", aspect="auto")
            ax.axis("off")
            plt.tight_layout(pad=0)
            
            fname = output_dir / LABEL_MAP[label] / f"s{i:03d}_e{j:03d}.png"
            plt.savefig(fname, bbox_inches="tight", pad_inches=0)
            plt.close()
            total += 1
    
    print(f"Generated {total} spectrograms")
    return str(output_dir)