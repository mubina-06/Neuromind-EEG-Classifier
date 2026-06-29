#!/usr/bin/env python3
"""
NeuroMind Live EEG Detection Launcher
=====================================
Launch the futuristic EEG brain signal classification interface
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the NeuroMind Live interface"""
    print("🧠 Starting NeuroMind Live EEG Detection Interface...")
    print("🚀 Launching futuristic brain signal analyzer...")
    
    # Path to the app
    app_path = Path("eeg_classifier/app/neuromind_live_app.py")
    
    if not app_path.exists():
        print(f"❌ Error: {app_path} not found!")
        sys.exit(1)
    
    try:
        # Launch Streamlit app
        cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
        print(f"📡 Running: {' '.join(cmd)}")
        print("🌐 Opening in your default browser...")
        print("💡 Use Ctrl+C to stop the server")
        
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 NeuroMind Live interface stopped.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()