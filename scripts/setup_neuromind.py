#!/usr/bin/env python3
"""
NeuroMind Setup and Launch Script
================================
Automated setup and launch for the NeuroMind EEG Classification System
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'torch', 'numpy', 'plotly', 
        'matplotlib', 'mne', 'scikit-learn', 'pandas'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} - Missing")
    
    return missing

def install_dependencies(missing_packages):
    """Install missing dependencies"""
    if not missing_packages:
        return True
    
    print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
    try:
        cmd = [sys.executable, "-m", "pip", "install"] + missing_packages
        subprocess.run(cmd, check=True)
        print("✅ All dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_model_files():
    """Check if model files exist"""
    model_paths = [
        "models/best_model.pth",
        "models/checkpoints/best_resnet18.pth",
        "models/checkpoints/best_efficientnet_b0.pth"
    ]
    
    found_models = []
    for path in model_paths:
        if Path(path).exists():
            found_models.append(path)
            print(f"✅ Model found: {path}")
    
    if not found_models:
        print("⚠️  No trained models found. You can:")
        print("   1. Run 'python download_data.py' to train models")
        print("   2. Or use the demo mode with simulated predictions")
        return False
    
    return True

def launch_interface():
    """Launch the NeuroMind interface"""
    interfaces = {
        "1": ("NeuroMind Live (Futuristic UI)", "eeg_classifier/app/neuromind_live_app.py"),
        "2": ("Modern Interface", "eeg_classifier/app/modern_app.py"),
        "3": ("Classic Interface", "eeg_classifier/app/streamlit_app.py")
    }
    
    print("\n🚀 Choose interface to launch:")
    for key, (name, _) in interfaces.items():
        print(f"   {key}. {name}")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        if choice in interfaces:
            name, app_path = interfaces[choice]
            break
        print("❌ Invalid choice. Please enter 1, 2, or 3.")
    
    print(f"\n🧠 Launching {name}...")
    
    if not Path(app_path).exists():
        print(f"❌ Error: {app_path} not found!")
        return False
    
    try:
        cmd = [sys.executable, "-m", "streamlit", "run", app_path]
        print("🌐 Opening in your browser...")
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch interface: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Interface stopped by user.")
        return True

def main():
    """Main setup and launch function"""
    print("🧠 NeuroMind EEG Classification Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check dependencies
    print("\n📋 Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        install_choice = input(f"\n❓ Install missing packages? (y/n): ").lower().strip()
        if install_choice == 'y':
            if not install_dependencies(missing):
                return
        else:
            print("⚠️  Cannot continue without required packages.")
            return
    
    # Check model files
    print("\n🤖 Checking AI models...")
    model_available = check_model_files()
    
    if not model_available:
        train_choice = input("\n❓ Download and train models now? (y/n): ").lower().strip()
        if train_choice == 'y':
            print("🔄 Starting model training...")
            try:
                subprocess.run([sys.executable, "download_data.py"], check=True)
                print("✅ Model training completed!")
            except subprocess.CalledProcessError as e:
                print(f"❌ Model training failed: {e}")
                print("📝 Continuing with demo mode...")
    
    # Launch interface
    print("\n" + "=" * 40)
    launch_interface()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Setup cancelled by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("📧 Please report this issue on GitHub.")