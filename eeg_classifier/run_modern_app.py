#!/usr/bin/env python3
"""
NeuroMind - Modern EEG Classifier Launcher
==========================================
Simple launcher script for the modern Streamlit app
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("🧠 NeuroMind - EEG Brain Signal Classifier")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("app/modern_app.py").exists():
        print("❌ Error: Please run this script from the eeg_classifier directory")
        print("   Current directory:", os.getcwd())
        return
    
    # Install requirements if needed
    try:
        import plotly
        import pandas
    except ImportError:
        print("📦 Installing additional requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements_modern.txt"])
    
    print("🚀 Starting NeuroMind application...")
    print("📱 The app will open in your browser automatically")
    print("🔗 URL: http://localhost:8501")
    print("\n💡 Press Ctrl+C to stop the application")
    print("=" * 50)
    
    # Run the Streamlit app
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "app/modern_app.py",
            "--server.headless", "false",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 NeuroMind application stopped")

if __name__ == "__main__":
    main()