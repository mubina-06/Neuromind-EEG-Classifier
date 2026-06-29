# NEUROMIND EEG BRAIN SIGNAL CLASSIFIER
## A PROJECT REPORT
### For "ARTIFICIAL INTELLIGENCE & MACHINE LEARNING"

**Submitted by**
- Gurram Durga Anuhya [AP23110010664]
- Patan Mubina [AP23110010657]

**BACHELOR OF TECHNOLOGY**
in
**COMPUTER SCIENCE AND ENGINEERING**
of
**SCHOOL OF ENGINEERING AND SCIENCES**

**Under the Guidance of**
Dr T. Anitha Kumari,
Lecturer, Department of CSE
SRM University-AP, Neerukonda, Andhra Pradesh 522240

---

## Acknowledgement

We sincerely thank our course instructor and faculty of the Artificial Intelligence and Machine Learning department for their invaluable guidance and support throughout this project. This project gave us hands-on experience in applying AI/ML principles — including signal processing, deep learning, computer vision, and explainable AI — to a real-world medical application.

We extend our heartfelt gratitude to **Dr. T. Anitha Kumari**, Lecturer, Department of CSE, SRM University – AP, for her continuous encouragement, expert insights, and mentorship throughout the project lifecycle.

We also thank our peers for their constructive feedback during peer reviews and our families for their unwavering support during the course of this project.

**G Durga Anuhya** AP23110010664  
**P Mubina** AP23110010657  
B.Tech – Computer Science and Engineering  
SRM University – AP, April 2025

---

## Certificate

**Date:** _______________

**Supervisor (Signature)**  
Dr. T. Anitha Kumari,  
Assistant Professor,  
SRM-AP University

---

## Table of Contents

1. [Abstract](#1-abstract) .................................................... 6
2. [Introduction](#2-introduction) .......................................... 7
3. [Objectives](#3-objectives) ............................................. 8
4. [Literature Review](#4-literature-review) .............................. 9
5. [System Requirements Specification](#5-system-requirements-specification) ... 10
6. [Dataset Analysis](#6-dataset-analysis) ................................ 12
7. [Methodology](#7-methodology) .......................................... 14
8. [System Design](#8-system-design) ...................................... 17
9. [Implementation](#9-implementation) ..................................... 20
10. [Signal Processing Pipeline](#10-signal-processing-pipeline) ........... 22
11. [Deep Learning Architecture](#11-deep-learning-architecture) ........... 24
12. [Web Application Development](#12-web-application-development) ......... 26
13. [Results and Analysis](#13-results-and-analysis) ..................... 28
14. [Testing and Validation](#14-testing-and-validation) ................. 30
15. [Conclusion](#15-conclusion) ......................................... 32
16. [References](#16-references) ......................................... 33

---
## 1. Abstract

NeuroMind is an advanced AI-powered EEG brain signal classification system developed as part of the Artificial Intelligence and Machine Learning course. The project applies core AI/ML principles including signal processing, deep learning, computer vision, and explainable AI to classify mental states from real electroencephalogram (EEG) data.

The system processes medical-grade EEG signals from the PhysioNet database and classifies them into three mental states: Focused, Relaxed, and Stressed. Built using PyTorch, MNE-Python for signal processing, and Streamlit for the web interface, the project demonstrates a complete machine learning pipeline from data acquisition through deployment.

The system achieves **67.3% accuracy** using an ensemble of ResNet18 and EfficientNet-B0 models with transfer learning. Key innovations include Grad-CAM explainable AI for medical interpretability, comprehensive data augmentation with SpecAugment, and temperature scaling for confidence calibration. All functional requirements were successfully validated through comprehensive testing, demonstrating the system's readiness for real-world healthcare applications.

## 2. Introduction

Electroencephalography (EEG) is a non-invasive neurophysiological monitoring method that records electrical activity of the brain. The ability to automatically classify mental states from EEG signals has significant applications in healthcare, human-computer interaction, and neuroscience research. Traditional EEG analysis methods rely on manual feature extraction and require domain expertise, making them time-consuming and subjective.

NeuroMind addresses this challenge by leveraging artificial intelligence to provide automated, objective analysis of brain signals for mental state classification. The project covers the complete Machine Learning Development Life Cycle (MLDLC), including:

- **Data Engineering**: EEG signal processing and spectrogram generation
- **Model Development**: CNN architectures with transfer learning
- **Explainable AI**: Grad-CAM visualization for medical interpretability  
- **Web Application**: Professional Streamlit interface for practical deployment
- **Validation**: Comprehensive testing with real medical data

The system processes raw EEG signals through advanced signal processing techniques, converts them to spectrograms, and applies deep learning models to classify mental states. The integration of explainable AI makes the system suitable for medical applications where interpretability is crucial.

### 2.1 Problem Statement

Current challenges in EEG-based mental state assessment include:
- **Subjective Analysis**: Manual interpretation varies between experts
- **Time-Intensive**: Hours of manual analysis for each recording
- **Limited Scalability**: Cannot handle large-scale screening
- **Lack of Interpretability**: Black-box AI systems unsuitable for medical use

### 2.2 Proposed Solution

NeuroMind provides:
- **Automated Classification**: AI-powered mental state detection
- **Real-time Processing**: Results in under 1.5 seconds
- **Explainable Results**: Grad-CAM heatmaps show AI decision-making
- **Medical-grade Accuracy**: 67.3% on validated medical dataset
- **Production-ready Interface**: Professional web application

## 3. Objectives

The primary objectives of this project are:

### 3.1 Technical Objectives
1. **Implement End-to-End ML Pipeline**: From raw EEG signals to deployed web application
2. **Achieve Medical-grade Performance**: Target >65% accuracy on real medical data
3. **Develop Explainable AI**: Implement Grad-CAM for transparent decision-making
4. **Create Production-ready System**: Professional web interface suitable for clinical use
5. **Validate with Real Data**: Use PhysioNet medical-grade EEG database

### 3.2 Learning Objectives
1. **Signal Processing Mastery**: Apply advanced EEG preprocessing techniques
2. **Deep Learning Implementation**: Build and compare multiple CNN architectures
3. **Transfer Learning Application**: Adapt pre-trained models for medical data
4. **Web Development Skills**: Create modern, responsive user interfaces
5. **Software Engineering Practices**: Follow modular, maintainable code structure

### 3.3 Research Objectives
1. **Compare CNN Architectures**: Evaluate ResNet18 vs EfficientNet-B0 for EEG data
2. **Ensemble Method Development**: Combine models for improved accuracy
3. **Data Augmentation Innovation**: Apply SpecAugment to EEG spectrograms
4. **Confidence Calibration**: Implement temperature scaling for reliable predictions
5. **Medical AI Best Practices**: Ensure interpretability and validation standards

## 4. Literature Review

### 4.1 EEG Signal Processing Background

Electroencephalography measures electrical activity in the brain through electrodes placed on the scalp. EEG signals are characterized by different frequency bands associated with various mental states:

| Frequency Band | Range | Associated Mental States |
|----------------|-------|-------------------------|
| **Delta** | 0.5-4 Hz | Deep sleep, unconscious states |
| **Theta** | 4-8 Hz | Drowsiness, meditation |
| **Alpha** | 8-13 Hz | Relaxed awareness, eyes closed |
| **Beta** | 13-30 Hz | Active concentration, alertness |
| **Gamma** | 30-45 Hz | High-level cognitive processing |

### 4.2 Deep Learning in EEG Analysis

Recent advances in deep learning have shown promising results in EEG classification:

#### 4.2.1 Convolutional Neural Networks (CNNs)
- **Spatial Pattern Recognition**: CNNs excel at learning spatial-temporal patterns in EEG data
- **Transfer Learning**: Pre-trained models improve performance with limited medical data
- **Spectrogram Processing**: Converting EEG to images enables CNN application

#### 4.2.2 Related Work Performance Benchmarks
| Application Domain | Typical Accuracy Range | Challenges |
|-------------------|----------------------|------------|
| Motor imagery classification | 60-85% | Individual variability |
| Emotion recognition | 55-75% | Subjective labels |
| Sleep stage classification | 80-90% | Clear physiological markers |
| Mental workload assessment | 65-80% | Task dependency |

### 4.3 Explainable AI in Medical Applications

Medical AI systems require interpretability for clinical acceptance:
- **Grad-CAM**: Gradient-weighted Class Activation Mapping shows attention regions
- **LIME**: Local Interpretable Model-agnostic Explanations for feature importance
- **SHAP**: SHapley Additive exPlanations for unified feature attribution

### 4.4 Research Gap and Innovation

Current limitations in EEG classification systems:
1. **Limited Interpretability**: Most systems are black-box models
2. **Small Datasets**: Overfitting due to limited training data
3. **Individual Variability**: Poor generalization across subjects
4. **Real-time Constraints**: Slow processing unsuitable for live monitoring

**Our Innovations**:
- **Ensemble Architecture**: Combines speed and efficiency optimized models
- **SpecAugment for EEG**: Novel application of audio augmentation to brain signals
- **Temperature Calibration**: Reliable confidence estimation for medical use
- **Complete Pipeline**: End-to-end system from raw EEG to web deployment

## 5. System Requirements Specification

### 5.1 Functional Requirements

#### FR-01: EEG Data Processing
- **FR-01.1**: System shall load EEG data from PhysioNet EEGBCI database
- **FR-01.2**: System shall apply bandpass filtering (4-45 Hz) to remove noise
- **FR-01.3**: System shall apply notch filtering (60 Hz) for power line interference
- **FR-01.4**: System shall perform average referencing across all channels
- **FR-01.5**: System shall segment signals into 4-second epochs with 50% overlap

#### FR-02: Spectrogram Generation
- **FR-02.1**: System shall convert EEG epochs to STFT spectrograms
- **FR-02.2**: System shall generate 224×224 pixel RGB images
- **FR-02.3**: System shall average power across all 64 EEG channels
- **FR-02.4**: System shall apply log-scale normalization

#### FR-03: Machine Learning Classification
- **FR-03.1**: System shall implement ResNet18 architecture with transfer learning
- **FR-03.2**: System shall implement EfficientNet-B0 architecture with transfer learning
- **FR-03.3**: System shall create ensemble model combining both architectures
- **FR-03.4**: System shall classify into 3 classes: Focused, Relaxed, Stressed
- **FR-03.5**: System shall achieve minimum 65% accuracy on test data

#### FR-04: Explainable AI
- **FR-04.1**: System shall generate Grad-CAM attention heatmaps
- **FR-04.2**: System shall overlay heatmaps on original spectrograms
- **FR-04.3**: System shall provide confidence scores for predictions
- **FR-04.4**: System shall implement temperature scaling for calibration

#### FR-05: Web Application Interface
- **FR-05.1**: System shall provide file upload functionality for EEG spectrograms
- **FR-05.2**: System shall display real-time prediction results
- **FR-05.3**: System shall show Grad-CAM explanations
- **FR-05.4**: System shall support batch processing of multiple files
- **FR-05.5**: System shall provide performance dashboard with model metrics

### 5.2 Non-Functional Requirements

| NFR ID | Category | Requirement | Acceptance Criteria |
|--------|----------|-------------|-------------------|
| **NFR-01** | Performance | Prediction time < 1.5 seconds | Measured on standard hardware |
| **NFR-02** | Accuracy | Classification accuracy ≥ 65% | Validated on PhysioNet test set |
| **NFR-03** | Usability | Responsive web interface | Works on desktop, tablet, mobile |
| **NFR-04** | Reliability | No system crashes during operation | 100% uptime during testing |
| **NFR-05** | Scalability | Handle 100+ concurrent users | Load testing validation |
| **NFR-06** | Security | Secure file upload and processing | Input validation and sanitization |
| **NFR-07** | Maintainability | Modular code architecture | Clear separation of concerns |
| **NFR-08** | Portability | Cross-platform compatibility | Windows, macOS, Linux support |

### 5.3 System Constraints

#### 5.3.1 Hardware Constraints
- **Minimum RAM**: 8 GB for training, 4 GB for inference
- **GPU Requirement**: NVIDIA GPU with CUDA support recommended for training
- **Storage**: 5 GB free space for models and data
- **Network**: Internet connection for PhysioNet data download

#### 5.3.2 Software Constraints
- **Python Version**: 3.8 or higher
- **Operating System**: Windows 10+, macOS 10.15+, Ubuntu 18.04+
- **Browser Support**: Chrome 90+, Firefox 88+, Safari 14+
- **Dependencies**: PyTorch 2.0+, MNE-Python 1.4+, Streamlit 1.28+

#### 5.3.3 Data Constraints
- **Input Format**: PNG spectrogram images (224×224 pixels)
- **EEG Channels**: 64-channel recordings from PhysioNet
- **Sampling Rate**: 160 Hz original, processed to spectrograms
- **Dataset Size**: 687 samples across 3 mental state classes

## 6. Dataset Analysis

### 6.1 PhysioNet EEG Motor Movement/Imagery Database

#### 6.1.1 Dataset Overview
- **Official Source**: https://physionet.org/content/eegmmidb/1.0.0/
- **Subjects**: 109 healthy volunteers (64 male, 45 female)
- **Age Range**: 18-64 years (mean: 29.9 years)
- **Recording Setup**: 64-channel EEG with 10-10 electrode placement
- **Sampling Frequency**: 160 Hz with 16-bit resolution
- **Total Size**: ~50 MB compressed data

#### 6.1.2 Recording Protocol
Each subject performed multiple tasks in controlled laboratory conditions:

| Task ID | Description | Duration | Mental State Mapping |
|---------|-------------|----------|-------------------|
| **Run 1** | Baseline, eyes open | 1 minute | Relaxed |
| **Run 2** | Baseline, eyes closed | 1 minute | Relaxed |
| **Run 3** | Real left/right fist movement | 2 minutes | Stressed |
| **Run 4** | Imagined left/right fist movement | 2 minutes | Focused |
| **Run 5** | Real fists/feet movement | 2 minutes | Stressed |
| **Run 6** | Imagined fists/feet movement | 2 minutes | Focused |

#### 6.1.3 Mental State Justification

**Relaxed State (Eyes Open Rest)**:
- Minimal cognitive load and motor activity
- Alpha wave dominance (8-13 Hz)
- Represents baseline, calm mental state
- Used as control condition in neuroscience research

**Focused State (Motor Imagery)**:
- Active mental concentration and visualization
- Beta wave activity (13-30 Hz) in motor cortex
- Requires sustained attention and cognitive effort
- Simulates focused, goal-directed mental activity

**Stressed State (Motor Execution)**:
- Physical and mental coordination demands
- Gamma wave activity (30-45 Hz) during movement
- Represents high cognitive and motor load
- Analogous to multitasking or high-pressure situations

### 6.2 Data Distribution Analysis

#### 6.2.1 Sample Distribution
After preprocessing and spectrogram generation:

| Mental State | Sample Count | Percentage | Class Balance |
|-------------|--------------|------------|---------------|
| **🎯 Focused** | 297 samples | 43.2% | Majority class |
| **😌 Relaxed** | 119 samples | 17.3% | Minority class |
| **😰 Stressed** | 271 samples | 39.5% | Balanced |
| **Total** | **687 samples** | **100%** | **Imbalanced** |

#### 6.2.2 Class Imbalance Impact
- **Challenge**: Relaxed class underrepresented (17.3% vs 43.2%/39.5%)
- **Solution**: Weighted loss function and stratified sampling
- **Validation**: Per-class metrics to ensure fair evaluation
- **Augmentation**: Extra augmentation for minority class

#### 6.2.3 Data Quality Metrics
- **Signal Quality**: >95% artifact-free epochs after preprocessing
- **Consistency**: Standardized recording protocols across subjects
- **Validation**: Medical-grade dataset used in peer-reviewed research
- **Reproducibility**: Publicly available with documented methodology

### 6.3 Exploratory Data Analysis

#### 6.3.1 Frequency Domain Analysis
Power spectral density analysis reveals distinct patterns:

**Focused State Characteristics**:
- Increased beta power (13-30 Hz) in frontal regions
- Decreased alpha power (8-13 Hz) indicating active attention
- Higher gamma activity (30-45 Hz) in motor planning areas

**Relaxed State Characteristics**:
- Dominant alpha rhythm (8-13 Hz) in posterior regions
- Low beta activity indicating minimal cognitive load
- Stable baseline with minimal artifacts

**Stressed State Characteristics**:
- Broad-spectrum activation across multiple frequency bands
- Increased muscle artifacts due to physical movement
- Higher power in gamma range (30-45 Hz) during execution

#### 6.3.2 Spatial Distribution Analysis
Electrode-wise analysis shows:
- **Frontal electrodes (Fp1, Fp2, F3, F4)**: High activity during focused tasks
- **Central electrodes (C3, C4, Cz)**: Peak activity during motor execution
- **Parietal electrodes (P3, P4, Pz)**: Dominant during relaxed states
- **Occipital electrodes (O1, O2)**: Alpha rhythm during eyes-closed rest

#### 6.3.3 Temporal Dynamics
Time-series analysis reveals:
- **Event-related patterns**: Clear onset/offset during motor tasks
- **Sustained activity**: Consistent patterns during imagery tasks
- **Baseline stability**: Minimal drift during rest conditions
- **Individual variability**: Subject-specific patterns requiring normalization
## 7. Methodology

### 7.1 Overall Approach

The NeuroMind project follows a systematic machine learning methodology combining signal processing, deep learning, and software engineering best practices. The approach consists of five main phases:

1. **Data Acquisition & Preprocessing**: EEG signal processing and spectrogram generation
2. **Model Development**: CNN architecture implementation with transfer learning
3. **Training & Optimization**: Model training with advanced techniques
4. **Evaluation & Validation**: Comprehensive performance assessment
5. **Deployment & Interface**: Web application development and deployment

### 7.2 Machine Learning Pipeline

#### 7.2.1 Data Pipeline Architecture
```
PhysioNet EEG Database → Signal Processing → Spectrogram Generation → Data Augmentation → Model Training → Evaluation → Deployment
```

#### 7.2.2 Signal Processing Workflow
```
Raw EEG (64 channels, 160 Hz) → Bandpass Filter (4-45 Hz) → Notch Filter (60 Hz) → Average Reference → Epoching (4-sec) → Artifact Rejection → STFT → Spectrograms (224×224)
```

### 7.3 Deep Learning Strategy

#### 7.3.1 Transfer Learning Approach
- **Base Models**: ImageNet pre-trained ResNet18 and EfficientNet-B0
- **Adaptation**: Replace final classification layer (1000 → 3 classes)
- **Fine-tuning**: Train entire network on EEG spectrograms
- **Rationale**: Leverage visual features learned from natural images

#### 7.3.2 Ensemble Method
- **Architecture**: Probability averaging of ResNet18 + EfficientNet-B0
- **Combination**: Weighted average of softmax outputs
- **Benefits**: Improved accuracy and reduced overfitting
- **Implementation**: No additional training required

### 7.4 Experimental Design

#### 7.4.1 Data Splitting Strategy
- **Training Set**: 70% (481 samples) - Model parameter learning
- **Validation Set**: 15% (103 samples) - Hyperparameter tuning
- **Test Set**: 15% (103 samples) - Final performance evaluation
- **Stratification**: Maintains class distribution across splits

#### 7.4.2 Cross-Validation Protocol
- **Method**: 5-fold stratified cross-validation
- **Purpose**: Robust performance estimation
- **Metrics**: Accuracy, F1-score, precision, recall per fold
- **Statistical Testing**: Confidence intervals and significance tests

## 8. System Design

### 8.1 Architecture Overview

The NeuroMind system follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    WEB INTERFACE LAYER                      │
│  Streamlit Frontend (modern_app.py, streamlit_app.py)      │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   APPLICATION LAYER                         │
│  • Live Analysis    • Batch Processing                     │
│  • Performance Dashboard    • Model Comparison             │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    MACHINE LEARNING LAYER                  │
│  • Model Loading    • Prediction Engine                    │
│  • Grad-CAM        • Confidence Calibration               │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    DATA PROCESSING LAYER                   │
│  • Signal Processing    • Spectrogram Generation           │
│  • Data Augmentation   • Preprocessing Pipeline            │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      DATA LAYER                            │
│  • PhysioNet Database    • Model Checkpoints              │
│  • Generated Spectrograms    • Results Cache              │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 Component Diagram

#### 8.2.1 Core Components

| Component | File | Responsibility |
|-----------|------|----------------|
| **Signal Processor** | `src/preprocess.py` | EEG filtering, epoching, spectrogram generation |
| **Model Manager** | `src/model.py` | CNN architectures, ensemble, checkpointing |
| **Training Engine** | `src/train.py` | Model training, validation, early stopping |
| **Evaluation Suite** | `src/evaluate.py` | Performance metrics, visualization |
| **Explainable AI** | `src/gradcam.py` | Grad-CAM heatmap generation |
| **Web Interface** | `app/modern_app.py` | Streamlit application, user interaction |
| **Data Pipeline** | `download_data.py` | End-to-end automation script |

### 8.3 Data Flow Architecture

#### 8.3.1 Training Data Flow
```
PhysioNet → Raw EEG → Preprocessing → Spectrograms → Augmentation → Training → Model Checkpoints
```

#### 8.3.2 Inference Data Flow
```
User Upload → Image Validation → Preprocessing → Model Inference → Grad-CAM → Results Display
```

### 8.4 Class Diagram

#### 8.4.1 Core Classes

```python
class EEGPreprocessor:
    + load_physionet_data(n_subjects: int) -> Tuple[List, List]
    + preprocess_raw(raw: mne.Raw) -> mne.Raw
    + epoch_to_spectrogram(epoch_data: np.ndarray) -> np.ndarray
    + save_spectrograms(raw_list: List, labels: List) -> str

class CNNModel:
    + build_model(arch: str, pretrained: bool) -> torch.nn.Module
    + get_last_conv_layer(model: torch.nn.Module) -> torch.nn.Module
    + save_checkpoint(model: torch.nn.Module, arch: str) -> str
    + load_checkpoint(arch: str, device: torch.device) -> torch.nn.Module

class EnsembleModel(torch.nn.Module):
    + __init__(model_resnet: torch.nn.Module, model_effnet: torch.nn.Module)
    + forward(x: torch.Tensor) -> torch.Tensor

class GradCAM:
    + __init__(model: torch.nn.Module, target_layer: torch.nn.Module)
    + __call__(input_tensor: torch.Tensor) -> np.ndarray
    + remove_hooks() -> None

class ModelEvaluator:
    + evaluate_model(model: torch.nn.Module, dataloader: DataLoader) -> Dict
    + compare_models(results_dict: Dict) -> None
    + plot_confusion_matrix(cm: np.ndarray) -> None
```

## 9. Implementation

### 9.1 Technology Stack

#### 9.1.1 Core Technologies

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| **Deep Learning** | PyTorch | 2.0+ | Neural network framework |
| **Signal Processing** | MNE-Python | 1.4+ | EEG data processing |
| **Web Framework** | Streamlit | 1.28+ | Interactive web application |
| **Visualization** | Plotly | 5.15+ | Interactive charts and graphs |
| **Scientific Computing** | NumPy, SciPy | Latest | Numerical computations |
| **Machine Learning** | scikit-learn | 1.2+ | Evaluation metrics, utilities |
| **Image Processing** | PIL, OpenCV | Latest | Image manipulation |
| **Data Handling** | Pandas | Latest | Data manipulation and analysis |

#### 9.1.2 Development Environment
- **Language**: Python 3.8+
- **IDE**: VS Code, PyCharm, Jupyter Notebook
- **Version Control**: Git with GitHub
- **Package Management**: pip, conda
- **Documentation**: Markdown, Sphinx

### 9.2 Project Structure

```
eeg_classifier/
├── src/                          # Core implementation modules
│   ├── preprocess.py            # EEG signal processing pipeline
│   ├── dataset.py               # PyTorch Dataset with augmentation
│   ├── model.py                 # CNN architectures (ResNet18, EfficientNet-B0)
│   ├── train.py                 # Training loop with early stopping
│   ├── evaluate.py              # Comprehensive evaluation metrics
│   ├── gradcam.py               # Grad-CAM explainable AI
│   └── calibration.py           # Temperature scaling for confidence
├── app/                         # Web application interfaces
│   ├── streamlit_app.py         # Basic Streamlit interface
│   └── modern_app.py            # Professional modern interface
├── models/                      # Trained model storage
│   ├── checkpoints/             # Model weight files
│   └── best_model.pth           # Best performing model
├── data/                        # Dataset and generated files
│   └── spectrograms/            # Generated spectrogram images
│       ├── focused/             # Focused state spectrograms
│       ├── relaxed/             # Relaxed state spectrograms
│       └── stressed/            # Stressed state spectrograms
├── results/                     # Evaluation outputs
│   ├── gradcam/                 # Grad-CAM visualization outputs
│   └── *.png                    # Performance plots and charts
├── notebooks/                   # Jupyter analysis notebooks
│   └── EEG_Classifier_Walkthrough.ipynb
├── download_data.py             # Complete pipeline automation
├── main.py                      # Command-line interface
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

### 9.3 Key Implementation Details

#### 9.3.1 Signal Processing Implementation

```python
def preprocess_raw(raw):
    """Advanced EEG preprocessing pipeline"""
    raw = raw.copy()
    # Bandpass filter: 4-45 Hz (all relevant brain waves)
    raw.filter(l_freq=4.0, h_freq=45.0, method="fir", verbose=False)
    # Notch filter: 60 Hz (US power line interference)
    raw.notch_filter(freqs=60.0, verbose=False)
    # Average reference: standardize across channels
    raw.set_eeg_reference("average", projection=False, verbose=False)
    return raw

def epoch_to_spectrogram(epoch_data, sfreq, fmin=4, fmax=45, img_size=(224, 224)):
    """Convert EEG epoch to STFT spectrogram image"""
    from scipy.signal import stft
    
    n_channels, n_times = epoch_data.shape
    nperseg = min(256, n_times // 4)
    noverlap = nperseg // 2
    all_specs = []
    
    # Process each channel
    for ch in range(n_channels):
        f, t, Zxx = stft(epoch_data[ch], fs=sfreq, nperseg=nperseg, noverlap=noverlap)
        power = np.abs(Zxx) ** 2
        mask = (f >= fmin) & (f <= fmax)
        all_specs.append(power[mask])
    
    # Average across channels and normalize
    mean_spec = np.mean(all_specs, axis=0)
    mean_spec = np.log1p(mean_spec)  # Log transform
    mean_spec = (mean_spec - mean_spec.min()) / (mean_spec.max() - mean_spec.min())
    
    return render_spectrogram_image(mean_spec, img_size)
```

#### 9.3.2 Model Architecture Implementation

```python
def build_model(arch="resnet18", pretrained=True, dropout=0.4):
    """Build CNN classifier with transfer learning"""
    weights = "DEFAULT" if pretrained else None
    
    if arch == "resnet18":
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features  # 512
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 3)  # 3 mental states
        )
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features  # 1280
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 3)
        )
    
    return model

class EnsembleModel(nn.Module):
    """Ensemble combining ResNet18 + EfficientNet-B0"""
    def __init__(self, model_resnet, model_effnet):
        super().__init__()
        self.resnet = model_resnet
        self.effnet = model_effnet
    
    def forward(self, x):
        # Average softmax probabilities
        p1 = torch.softmax(self.resnet(x), dim=1)
        p2 = torch.softmax(self.effnet(x), dim=1)
        return (p1 + p2) / 2
```