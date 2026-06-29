"""
deap_loader.py
--------------
Loads real DEAP dataset (.dat files) and maps valence/arousal scores
to 3 mental state classes: focused, relaxed, stressed.

DEAP Dataset:
  - 32 participants, 40 trials each
  - 32 EEG channels + 8 peripheral channels
  - Sampling rate: 128 Hz
  - Labels: valence (1-9), arousal (1-9), dominance, liking

Download from: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/

Place .dat files in: data/raw/deap/
"""

import os
import pickle
import numpy as np
import mne
from pathlib import Path


# ── Label mapping from valence/arousal to mental state ────────────────────────
def valence_arousal_to_class(valence: float, arousal: float) -> int:
    """
    Maps DEAP valence/arousal scores to 3 mental states.

    Mapping logic (based on circumplex model of affect):
      - Focused  : high arousal + high valence  (alert, engaged)
      - Relaxed  : low arousal  + high valence  (calm, content)
      - Stressed : high arousal + low valence   (tense, anxious)

    Parameters
    ----------
    valence : float  1-9 scale
    arousal : float  1-9 scale

    Returns
    -------
    int  0=focused, 1=relaxed, 2=stressed
    """
    mid = 5.0  # midpoint of 1-9 scale
    if arousal >= mid and valence >= mid:
        return 0  # focused
    elif arousal < mid and valence >= mid:
        return 1  # relaxed
    else:
        return 2  # stressed


def load_deap_participant(dat_path: str, sfreq: float = 128.0):
    """
    Load one DEAP participant .dat file.

    Returns
    -------
    raw_list : list of mne.io.RawArray  (one per trial)
    labels   : list of int
    """
    with open(dat_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    eeg_data = data['data'][:, :32, :]   # (40 trials, 32 EEG channels, 8064 samples)
    labels_va = data['labels'][:, :2]    # (40 trials, [valence, arousal])

    ch_names = [f"EEG{i+1:03d}" for i in range(32)]
    ch_types = ["eeg"] * 32
    info     = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    raw_list, labels = [], []
    for trial_idx in range(eeg_data.shape[0]):
        trial_data = eeg_data[trial_idx]  # (32, 8064)
        raw = mne.io.RawArray(trial_data * 1e-6, info, verbose=False)
        label = valence_arousal_to_class(labels_va[trial_idx, 0],
                                         labels_va[trial_idx, 1])
        raw_list.append(raw)
        labels.append(label)

    return raw_list, labels


def load_deap_dataset(data_dir: str = "data/raw/deap", max_participants: int = 5):
    """
    Load multiple DEAP participants.

    Parameters
    ----------
    data_dir         : folder containing s01.dat, s02.dat, ...
    max_participants : limit for quick testing

    Returns
    -------
    all_raws   : list of mne.io.RawArray
    all_labels : list of int
    """
    data_dir = Path(data_dir)
    dat_files = sorted(data_dir.glob("s*.dat"))[:max_participants]

    if not dat_files:
        raise FileNotFoundError(
            f"No DEAP .dat files found in '{data_dir}'.\n"
            "Download from https://www.eecs.qmul.ac.uk/mmv/datasets/deap/ "
            "and place files in data/raw/deap/"
        )

    all_raws, all_labels = [], []
    for f in dat_files:
        print(f"[INFO] Loading {f.name}...")
        raws, lbls = load_deap_participant(str(f))
        all_raws.extend(raws)
        all_labels.extend(lbls)

    print(f"[INFO] Loaded {len(all_raws)} trials from {len(dat_files)} participants.")
    return all_raws, all_labels
