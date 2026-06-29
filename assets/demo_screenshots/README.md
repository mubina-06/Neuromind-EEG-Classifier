# Demo Screenshots

This directory contains screenshots demonstrating the NeuroMind EEG Classifier web application features.

## Screenshot Overview

### 1. **Model Performance Dashboard** (`model_performance_dashboard.png`)
- **Key Metrics Display**: 67.3% accuracy, 0.64 F1-score, 1.2s inference time
- **Model Comparison**: Visual comparison between ResNet18, EfficientNet-B0, and Ensemble
- **Professional UI**: Clean metrics cards with glassmorphism design
- **Real-time Stats**: 687 total samples processed

### 2. **Training Progress Analysis** (`training_progress.png`)
- **Training Curves**: Real-time training and validation accuracy progression
- **Confusion Matrix**: Detailed classification performance breakdown
- **Per-Class Metrics**: Precision, recall, and F1-scores for each mental state
- **Performance Visualization**: Clear charts showing model learning progression

### 3. **System Information Panel** (`system_information.png`)
- **Hardware Detection**: Automatic CPU/GPU detection and optimization
- **Model Architecture**: ResNet18 details with parameter count (~11M)
- **Input Specifications**: 224x224x3 RGB spectrogram format
- **Class Information**: 3 target classes (Focused, Relaxed, Stressed)

### 4. **Advanced Model Analysis** (`advanced_model_analysis.png`)
- **Architecture Comparison Table**: Detailed specs for all three models
- **Performance vs Speed Trade-off**: Interactive scatter plot analysis
- **Resource Utilization**: Memory usage and computational requirements
- **Model Selection Guide**: Data-driven recommendations for different use cases

### 5. **Feature Importance Analysis** (`feature_importance_analysis.png`)
- **Frequency Band Analysis**: EEG frequency importance (Alpha, Beta, Gamma, etc.)
- **Channel Importance**: Top 10 EEG electrodes contributing to classification
- **Model Interpretability**: Radar chart comparing interpretability aspects
- **Scientific Validation**: Links model decisions to neuroscientific knowledge

### 6. **Model Selection Recommendations** (`model_selection_recommendations.png`)
- **Smart Recommendations**: AI-powered model selection based on requirements
- **Use Case Optimization**: ResNet18 for speed, EfficientNet for efficiency, Ensemble for accuracy
- **Professional Cards**: Clean UI cards with pros/cons for each architecture
- **Decision Support**: Helps users choose optimal model for their needs

### 7. **Main Application Interface** (`main_application_interface.png`)
- **Professional Navigation**: Clean sidebar with model configuration
- **Real EEG Data**: Integration with PhysioNet EEG database
- **Live Signal Display**: Multi-channel EEG signal visualization
- **Modern Dark Theme**: Professional glassmorphism design with excellent UX

### 8. **AI Brain State Analysis** (`ai_brain_state_analysis.png`)
- **Spectrogram Generation**: STFT visualization of EEG signals (4-45 Hz)
- **Real-time Classification**: "STRESSED" state detection with confidence
- **Calibrated Confidence**: Temperature-scaled probability scores
- **Professional Results**: Large, clear prediction display with emoji indicators

### 9. **Grad-CAM Explainable AI** (`gradcam_explainable_ai.png`)
- **Medical Interpretability**: Shows which brain regions AI focuses on
- **Triple Visualization**: Original spectrogram, attention heatmap, combined overlay
- **Scientific Explanation**: Detailed interpretation of AI decision-making
- **Research Quality**: Suitable for medical/academic applications

## Technical Specifications

### UI/UX Features Demonstrated
- **🎨 Modern Design**: Glassmorphism effects, gradient backgrounds
- **📱 Responsive Layout**: Works on desktop and mobile devices  
- **🌙 Dark Theme**: Professional dark mode throughout
- **📊 Interactive Charts**: Plotly-powered visualizations
- **⚡ Real-time Updates**: Live data processing and display
- **🔍 Detailed Analytics**: Comprehensive performance metrics

### AI/ML Features Showcased
- **🧠 Multi-Model Architecture**: ResNet18, EfficientNet-B0, Ensemble
- **📈 Performance Metrics**: Accuracy, F1-score, confusion matrices
- **🔬 Explainable AI**: Grad-CAM attention visualization
- **⚙️ Model Comparison**: Side-by-side architecture analysis
- **🎯 Confidence Calibration**: Temperature-scaled probability scores
- **📊 Feature Analysis**: EEG frequency and channel importance

### Data Science Capabilities
- **📡 Real EEG Data**: PhysioNet medical-grade dataset integration
- **🔄 Signal Processing**: Complete preprocessing pipeline visualization
- **📊 Comprehensive Evaluation**: ROC curves, calibration plots, performance matrices
- **🎨 Publication Quality**: Research-grade visualizations and metrics
- **⚡ Real-time Inference**: Fast prediction with detailed explanations

## Usage in Documentation

These screenshots are used throughout the project documentation:

- **README.md**: Main interface and results screenshots
- **User Guide**: Step-by-step application walkthrough
- **Technical Report**: Performance analysis and model comparison
- **Portfolio**: Demonstrating professional AI/ML application development

## Screenshot Guidelines

When updating screenshots:
1. **Consistency**: Use the same browser and window size (1920x1080 recommended)
2. **Quality**: High-resolution PNG format
3. **Realistic Data**: Show actual model predictions and real EEG data
4. **Professional Appearance**: Clean, uncluttered interface
5. **Dark Theme**: Consistent dark mode for professional appearance

## File Naming Convention

- Use descriptive names reflecting the main feature shown
- Use underscores for spaces: `model_performance_dashboard.png`
- Include version numbers if multiple iterations: `main_interface_v2.png`
- Keep names concise but descriptive: `gradcam_visualization.png`