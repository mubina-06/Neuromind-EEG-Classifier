# 🧠 NeuroMind EEG Brain Signal Classifier

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-green.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An advanced AI-powered system for classifying mental states (Focused, Relaxed, Stressed) from EEG brain signals using deep learning and explainable AI.

## 🎯 Key Features

- **Real-time EEG Classification**: 67.3% accuracy on medical-grade data
- **Multiple CNN Architectures**: ResNet18, EfficientNet-B0, and Ensemble models
- **Explainable AI**: Grad-CAM visualization for medical interpretability
- **Professional Web Interface**: Modern Streamlit application
- **Complete ML Pipeline**: From raw EEG signals to deployment

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 8GB RAM (16GB recommended)
- NVIDIA GPU (optional, for training)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/neuromind-eeg-classifier.git
cd neuromind-eeg-classifier
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download data and train models**
```bash
python download_data.py
```

5. **Run the application**
```bash
# Modern interface (recommended)
streamlit run app/modern_app.py

# Or basic interface
streamlit run app/streamlit_app.py
```

## 📊 Dataset

- **Source**: PhysioNet EEG Motor Movement/Imagery Database
- **Subjects**: 109 healthy volunteers
- **Channels**: 64 EEG electrodes
- **Classes**: 3 mental states (Focused, Relaxed, Stressed)
- **Total Samples**: 687 spectrograms

## 🏗️ Architecture

### Signal Processing Pipeline
```
Raw EEG → Bandpass Filter (4-45Hz) → Notch Filter (60Hz) → 
Average Reference → Epoching → STFT → Spectrograms (224×224)
```

### Deep Learning Models
- **ResNet18**: Speed-optimized (850ms inference)
- **EfficientNet-B0**: Memory-efficient (21.4MB)
- **Ensemble**: Best accuracy (67.3%)

## 📈 Performance

| Model | Accuracy | F1-Score | Inference Time |
|-------|----------|----------|----------------|
| ResNet18 | 65.2% | 0.62 | 850ms |
| EfficientNet-B0 | 63.8% | 0.60 | 1200ms |
| **Ensemble** | **67.3%** | **0.64** | 1350ms |

### Per-Class Results
- 🎯 **Focused**: 69% precision, 71% recall
- 😌 **Relaxed**: 58% precision, 55% recall
- 😰 **Stressed**: 71% precision, 73% recall

## 🔬 Technical Details

### Filtering Techniques
- **Bandpass Filter**: 4-45 Hz (preserves all brain waves)
- **Notch Filter**: 60 Hz (removes power line interference)
- **Average Reference**: Standardizes signals across channels
- **Artifact Rejection**: 500µV threshold for clean signals

### Data Augmentation
- **SpecAugment**: Time-frequency masking for EEG
- **Geometric**: Rotation, flip, affine transforms
- **Color**: Brightness, contrast, saturation jitter
- **Noise**: Gaussian noise for robustness

### Explainable AI
- **Grad-CAM**: Shows where AI focuses during classification
- **Temperature Scaling**: Calibrated confidence scores
- **Medical Interpretability**: Transparent decision-making

## 🖥️ Web Application

### Features
- **Live Analysis**: Single file processing with explanations
- **Batch Processing**: Multiple files with progress tracking
- **Performance Dashboard**: Model metrics and comparisons
- **Modern UI**: Professional black theme with animations

### Screenshots
*Add screenshots of your application here*

## 📁 Project Structure

```
neuromind-eeg-classifier/
├── src/                    # Core implementation
│   ├── preprocess.py      # EEG signal processing
│   ├── model.py           # CNN architectures
│   ├── train.py           # Training pipeline
│   ├── evaluate.py        # Performance evaluation
│   ├── gradcam.py         # Explainable AI
│   └── calibration.py     # Confidence calibration
├── app/                   # Web interfaces
│   ├── modern_app.py      # Professional interface
│   └── streamlit_app.py   # Basic interface
├── notebooks/             # Jupyter analysis
├── docs/                  # Documentation
├── requirements.txt       # Dependencies
└── download_data.py       # Complete pipeline
```

## 🧪 Testing

Run the complete test suite:
```bash
python -m pytest tests/
```

Manual testing checklist:
- [ ] Data download and preprocessing
- [ ] Model training and validation
- [ ] Web interface functionality
- [ ] Grad-CAM visualization
- [ ] Batch processing

## 📚 Documentation

- [Technical Report](docs/NeuroMind_EEG_Project_Report.md)
- [API Documentation](docs/api.md)
- [User Guide](docs/user_guide.md)
- [Development Guide](docs/development.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PhysioNet** for providing medical-grade EEG data
- **MNE-Python** community for excellent EEG processing tools
- **PyTorch** team for the deep learning framework
- **Streamlit** for the web application platform

## 📞 Contact

- **Gurram Durga Anuhya** - AP23110010664
- **Patan Mubina** - AP23110010657
- **Supervisor**: Dr. T. Anitha Kumari, SRM University-AP

## 🔗 Links

- [PhysioNet Database](https://physionet.org/content/eegmmidb/1.0.0/)
- [Live Demo](http://localhost:8501) (when running locally)
- [Project Report](docs/NeuroMind_EEG_Project_Report.md)

---

**Made with ❤️ for advancing AI in healthcare**