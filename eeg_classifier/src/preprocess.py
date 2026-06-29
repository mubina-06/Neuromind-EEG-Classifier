"""
preprocess.py
-------------
Real EEG data pipeline using PhysioNet EEGBCI dataset.
Downloads automatically via MNE — no registration needed.

Dataset: EEG Motor Movement/Imagery Dataset (PhysioNet)
- 109 subjects, 64 EEG channels, 160 Hz sampling rate
- Tasks mapped to: Focused / Relaxed / Stressed

Label Mapping (task-based):
  Rest (eyes open)     → Relaxed   (1)
  Motor imagery task   → Focused   (0)
  Real movement task   → Stressed  (2)

Reference: https://physionet.org/content/eegmmidb/1.0.0/
"""

import numpy as np
import mne
from mne.datasets import eegbci
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# ── Label definitions ──────────────────────────────────────────────────────────
CLASSES   = ["focused", "relaxed", "stressed"]
LABEL_MAP = {0: "focused", 1: "relaxed", 2: "stressed"}

# PhysioNet EEGBCI run mapping → mental state
# Run 1  = baseline eyes open  → relaxed
# Run 2  = baseline eyes closed → relaxed
# Run 3  = real left/right fist → stressed (active motor)
# Run 4  = imagined left/right  → focused  (mental effort)
# Run 5  = real fists/feet      → stressed
# Run 6  = imagined fists/feet  → focused
RUN_LABEL = {
    1: 1,   # relaxed
    2: 1,   # relaxed
    3: 2,   # stressed
    4: 0,   # focused
    5: 2,   # stressed
    6: 0,   # focused
}


# ── Download & load real PhysioNet EEG data ────────────────────────────────────
def load_physionet_data(n_subjects=10, runs=None):
    """
    Downloads and loads real EEG data from PhysioNet EEGBCI dataset.
    Auto-downloads to ~/mne_data/ on first run (~50MB for 10 subjects).

    Parameters
    ----------
    n_subjects : int   number of subjects to load (1-109)
    runs       : list  run numbers to use (default: [1,3,4])

    Returns
    -------
    raw_list : list of mne.io.Raw
    labels   : list of int  (0=focused, 1=relaxed, 2=stressed)
    """
    if runs is None:
        runs = [1, 3, 4]   # rest, real movement, imagined movement

    raw_list, labels = [], []

    print(f"[INFO] Loading PhysioNet EEGBCI — {n_subjects} subjects, runs {runs}")
    print("[INFO] Auto-downloading if not cached (~50MB first time)...")

    for subj in range(1, n_subjects + 1):
        for run in runs:
            try:
                # MNE auto-downloads from PhysioNet
                fnames = eegbci.load_data(subj, [run], verbose=False)
                raw    = mne.io.read_raw_edf(fnames[0], preload=True, verbose=False)

                # Standardise channel names
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
    _print_dist(labels)
    return raw_list, labels


def _print_dist(labels):
    from collections import Counter
    c = Counter(labels)
    print("[INFO] Label distribution:")
    for k, v in sorted(c.items()):
        print(f"  {LABEL_MAP[k]:<10}: {v}")


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess_raw(raw):
    """
    Bandpass 4–45 Hz + notch 60 Hz (PhysioNet uses 60 Hz power line) + avg ref.
    """
    raw = raw.copy()
    raw.filter(l_freq=4.0, h_freq=45.0, method="fir", verbose=False)
    raw.notch_filter(freqs=60.0, verbose=False)   # US power line = 60 Hz
    raw.set_eeg_reference("average", projection=False, verbose=False)
    return raw


def epoch_raw(raw, epoch_duration=4.0, overlap=0.5):
    """Segment into fixed-length epochs."""
    events = mne.make_fixed_length_events(
        raw, duration=epoch_duration, overlap=overlap
    )
    epochs = mne.Epochs(
        raw, events, tmin=0, tmax=epoch_duration,
        baseline=None, reject={"eeg": 500e-6},  # relaxed threshold for PhysioNet
        preload=True, verbose=False,
    )
    return epochs


# ── STFT Spectrogram ───────────────────────────────────────────────────────────
def epoch_to_spectrogram(epoch_data, sfreq, fmin=4, fmax=45,
                          img_size=(224, 224)):
    """
    Convert EEG epoch → STFT spectrogram image.
    Averages power across all channels, log-scales, renders as viridis image.
    """
    from scipy.signal import stft

    n_channels, n_times = epoch_data.shape
    nperseg  = min(256, n_times // 4)
    noverlap = nperseg // 2
    all_specs = []

    for ch in range(n_channels):
        f, t, Zxx = stft(epoch_data[ch], fs=sfreq,
                         nperseg=nperseg, noverlap=noverlap)
        power     = np.abs(Zxx) ** 2
        mask      = (f >= fmin) & (f <= fmax)
        all_specs.append(power[mask])

    mean_spec = np.mean(all_specs, axis=0)
    mean_spec = np.log1p(mean_spec)
    mean_spec -= mean_spec.min()
    if mean_spec.max() > 0:
        mean_spec /= mean_spec.max()
    mean_spec = (mean_spec * 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(img_size[1]/100, img_size[0]/100), dpi=100)
    ax.imshow(mean_spec, aspect="auto", origin="lower", cmap="viridis")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)

    return np.array(Image.fromarray(buf).resize(img_size, Image.LANCZOS))


# ── Save all spectrograms ──────────────────────────────────────────────────────
def save_spectrograms(raw_list, labels, output_dir="data/spectrograms",
                      epoch_duration=4.0):
    """Full pipeline: preprocess → epoch → STFT → save PNG."""
    output_dir = Path(output_dir)
    for cls in CLASSES:
        (output_dir / cls).mkdir(parents=True, exist_ok=True)

    total = 0
    for i, (raw, label) in enumerate(zip(raw_list, labels)):
        cls_name = LABEL_MAP[label]
        print(f"[INFO] Recording {i+1}/{len(raw_list)} → {cls_name}")
        try:
            raw_c  = preprocess_raw(raw)
            epochs = epoch_raw(raw_c, epoch_duration=epoch_duration)
            if len(epochs) == 0:
                continue
            data  = epochs.get_data()
            sfreq = raw.info["sfreq"]
            for j, ep in enumerate(data):
                img   = epoch_to_spectrogram(ep, sfreq)
                fname = output_dir / cls_name / f"s{i:03d}_e{j:03d}.png"
                Image.fromarray(img).save(fname)
                total += 1
        except Exception as e:
            print(f"  [WARN] Skipped: {e}")

    print(f"[INFO] Saved {total} spectrogram images → '{output_dir}'")
    return str(output_dir)


# ── Auto-detect data source ────────────────────────────────────────────────────
def load_data(deap_dir="data/raw/deap", n_subjects=10):
    """
    Priority:
    1. DEAP .dat files (if present in deap_dir)
    2. PhysioNet EEGBCI (auto-download, always works)
    """
    deap_path = Path(deap_dir)
    if deap_path.exists() and list(deap_path.glob("s*.dat")):
        print("[INFO] DEAP dataset found — loading real DEAP data...")
        from src.preprocess import _load_deap
        return _load_deap(deap_dir)
    else:
        print("[INFO] Using PhysioNet EEGBCI (real EEG, free, auto-download)...")
        return load_physionet_data(n_subjects=n_subjects)


def _load_deap(data_dir, max_subjects=5):
    """Load DEAP .dat files with arousal-based labelling."""
    import pickle
    data_dir = Path(data_dir)
    files    = sorted(data_dir.glob("s*.dat"))[:max_subjects]
    sfreq    = 128.0
    ch_names = [f"EEG{i+1:02d}" for i in range(32)]
    info     = mne.create_info(ch_names, sfreq=sfreq, ch_types=["eeg"]*32)
    raws, labels = [], []
    for f in files:
        with open(f, 'rb') as fp:
            d = pickle.load(fp, encoding='latin1')
        for trial in range(d['data'].shape[0]):
            raw   = mne.io.RawArray(d['data'][trial, :32] * 1e-6, info, verbose=False)
            ar    = d['labels'][trial, 1]
            label = 2 if ar >= 6 else (1 if ar <= 3 else 0)
            raws.append(raw); labels.append(label)
    return raws, labels
