# 📋 How to Push NeuroMind EEG Project to GitHub

## Step 1: Install Git
Since Git is not installed on your system, you need to install it first:

### Option A: Download Git for Windows
1. Go to https://git-scm.com/download/win
2. Download and install Git for Windows
3. During installation, keep all default settings

### Option B: Using Package Manager (if available)
```bash
# If you have Chocolatey installed
choco install git

# If you have winget installed  
winget install Git.Git
```

## Step 2: Configure Git (First time only)
After installing Git, configure your username and email:
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Step 3: Handle Large Files
Your project has some large files that might cause issues. Here's what to do:

### Files to Remove/Ignore (Already added to .gitignore):
- ✅ Model files (*.pth) - 45MB+ files
- ✅ Data files (data/ folder) - Contains large EEG datasets  
- ✅ Virtual environment (.venv/) - Not needed in repository
- ✅ Cache files (__pycache__/) - Python bytecode
- ✅ Results/outputs - Generated files

### Create a Models Download Script
Instead of including large model files, create a script to download them:

```python
# download_models.py
import requests
import os
from pathlib import Path

def download_models():
    """Download pre-trained models from external source"""
    # You can upload models to Google Drive, Hugging Face, or similar
    # and download them with this script
    pass
```

## Step 4: Push to GitHub

### 4.1 Initialize Repository
```bash
cd "C:\Users\mubin\Downloads\Aritificial Intelligence sem-6"
git init
```

### 4.2 Add Remote Repository
First create a new repository on GitHub, then:
```bash
git remote add origin https://github.com/mubina-06/Neuromind-EEG-Classifier.git
```

### 4.3 Add Files (excluding large files via .gitignore)
```bash
git add .
git status  # Check what will be committed
```

### 4.4 Commit Changes
```bash
git commit -m "Initial commit: NeuroMind EEG Brain Signal Classifier

Features:
- Real-time EEG signal classification
- ResNet18 + EfficientNet ensemble models
- Streamlit web interface
- Grad-CAM explainability
- PhysioNet dataset integration
- Modern dark theme UI"
```

### 4.5 Push to GitHub
```bash
git branch -M main
git push -u origin main
```

## Step 5: Handle Large Files with Git LFS (Alternative)

If you want to include model files, use Git Large File Storage:

### 5.1 Install Git LFS
```bash
git lfs install
```

### 5.2 Track Large Files
```bash
git lfs track "*.pth"
git lfs track "*.h5"
git lfs track "*.pkl"
git add .gitattributes
```

### 5.3 Then Add and Push Normally
```bash
git add .
git commit -m "Add models with Git LFS"
git push origin main
```

## Step 6: Create a Professional README

Your repository should have a professional README.md:

```markdown
# 🧠 NeuroMind EEG Brain Signal Classifier

Advanced real-time EEG brain signal classification using deep learning.

## 🚀 Features
- **Real-time Classification**: Focused, Relaxed, Stressed states
- **Deep Learning Models**: ResNet18 + EfficientNet ensemble
- **Interactive Interface**: Modern Streamlit web app
- **Explainable AI**: Grad-CAM visualizations
- **Professional UI**: Dark theme with scientific aesthetics

## 📦 Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download models: `python download_data.py`
4. Run app: `streamlit run eeg_classifier/app/streamlit_app.py`

## 🎯 Usage
Open `http://localhost:8501` in your browser for the EEG classification interface.

## 📊 Performance
- **Accuracy**: 67.3% on PhysioNet EEGBCI dataset
- **Speed**: <1s inference time
- **Models**: ResNet18, EfficientNet-B0, Ensemble

## 🔬 Technical Details
- **Dataset**: PhysioNet EEGBCI (109 subjects, 64 channels)
- **Preprocessing**: 4-45 Hz bandpass, STFT spectrograms
- **Architecture**: CNN with transfer learning
- **Framework**: PyTorch, Streamlit, MNE-Python
```

## ⚠️ Important Notes

1. **Large Files**: Model files (45MB+) may hit GitHub's 100MB limit
2. **Data Folder**: Contains ~687 spectrogram images - consider hosting externally
3. **Virtual Environment**: Never commit .venv/ folder to Git
4. **Sensitive Info**: Make sure no API keys or passwords are in code

## 🔧 Troubleshooting

### Error: "File too large"
- Use .gitignore to exclude large files
- Consider Git LFS for essential large files
- Host models on external services (Google Drive, Hugging Face)

### Error: "Git not recognized"  
- Install Git from https://git-scm.com/download/win
- Restart terminal after installation

### Error: "Permission denied"
- Check if repository exists and you have write access
- Use personal access token for authentication

## 📞 Next Steps

1. **Install Git** using the link above
2. **Create GitHub repository** at github.com/mubina-06/Neuromind-EEG-Classifier
3. **Follow Step 4** to push your code
4. **Add professional README** with project description
5. **Set up GitHub Pages** for project website (optional)

---

Your NeuroMind EEG project will be ready to share with the world! 🌟