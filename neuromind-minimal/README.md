# 🧠 NeuroMind EEG Classifier

AI-powered EEG brain signal classification system for mental state detection.

## Features
- **67.3% accuracy** on medical EEG data
- **3 mental states**: Focused, Relaxed, Stressed  
- **Multiple models**: ResNet18, EfficientNet-B0, Ensemble
- **Web interface**: Streamlit app for predictions

## Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd neuromind-minimal
pip install -r requirements.txt

# Download data and train models
python download_data.py

# Run web app
streamlit run app.py
```

## Dataset
- **Source**: PhysioNet EEG Motor Movement/Imagery Database
- **Subjects**: 109 healthy volunteers
- **Samples**: 687 spectrograms (297 focused, 119 relaxed, 271 stressed)

## Performance
| Model | Accuracy | Speed |
|-------|----------|-------|
| ResNet18 | 65.2% | 850ms |
| EfficientNet-B0 | 63.8% | 1200ms |
| Ensemble | 67.3% | 1350ms |

## Team
- Gurram Durga Anuhya (AP23110010664)
- Patan Mubina (AP23110010657)
- Supervisor: Dr. T. Anitha Kumari, SRM University-AP