# EEG Brain Signal Classifier using CNN and Spectrograms

## Overview
Classifies mental states (Focused / Relaxed / Stressed) from real EEG signals using:
- ResNet18 + EfficientNet-B0 with transfer learning
- STFT spectrogram images as CNN input
- Grad-CAM for explainability
- Temperature scaling for confidence calibration
- Streamlit demo app

## Dataset
PhysioNet EEG Motor Movement/Imagery Dataset
- 109 subjects, 64 EEG channels, 160 Hz
- Free, no registration, auto-downloads via MNE-Python
- Link: https://physionet.org/content/eegmmidb/1.0.0/

## Label Mapping
| Task | Mental State |
|------|-------------|
| Rest (eyes open) | Relaxed |
| Motor imagery | Focused |
| Real movement | Stressed |

## Installation
```bash
pip install torch torchvision mne scipy matplotlib seaborn scikit-learn tqdm streamlit
```

## How to Run

### Step 1 — Train both models (downloads data automatically)
```bash
python download_data.py
```

### Step 2 — Run Streamlit demo app
```bash
streamlit run app/streamlit_app.py
```

### Step 3 — Run evaluation only
```bash
python main.py --skip_preprocess
```

## Project Structure
```
eeg_classifier/
├── src/
│   ├── preprocess.py     # MNE loading, 4-45Hz filter, STFT spectrogram
│   ├── dataset.py        # PyTorch Dataset + augmentation (SpecAugment)
│   ├── model.py          # ResNet18, EfficientNet-B0, Ensemble
│   ├── train.py          # Training loop, early stopping
│   ├── evaluate.py       # Accuracy, F1, ROC, calibration curves
│   ├── gradcam.py        # Grad-CAM for both models
│   └── calibration.py    # Temperature scaling
├── app/
│   └── streamlit_app.py  # Professional demo app
├── models/checkpoints/   # Saved model weights
├── results/              # Evaluation plots
│   └── gradcam/          # Grad-CAM outputs
├── data/spectrograms/    # Generated spectrogram images
├── download_data.py      # Full pipeline entry point
└── requirements.txt
```

## Results

| Model | Accuracy | F1 Macro |
|-------|----------|----------|
| ResNet18 | ~55-65% | ~0.50 |
| EfficientNet-B0 | ~55-65% | ~0.50 |
| Ensemble | ~60-70% | ~0.55 |

*Results improve significantly with more subjects (increase N_SUBJECTS in download_data.py)*

## Augmentation (Task 1)
- Random horizontal flip
- Random rotation ±10 degrees
- Color jitter (brightness/contrast ±20%)
- SpecAugment: time-frequency masking
- Gaussian noise

## Known Limitations
- Label mapping is task-based (not emotion-based like DEAP)
- Small dataset (10 subjects) limits accuracy
- CPU training is slow (~1.5s/batch) — use GPU for faster training
- Relaxed class is underrepresented (only rest recordings)
- Increase N_SUBJECTS to 30+ for better accuracy
