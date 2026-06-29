#!/usr/bin/env python3
"""
Complete training pipeline for NeuroMind EEG Classifier.

This script trains all models (ResNet18, EfficientNet-B0, Ensemble) 
and saves the best performing versions.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

def main():
    parser = argparse.ArgumentParser(description="Train EEG classification models")
    parser.add_argument("--models", nargs="+", 
                       choices=["resnet18", "efficientnet", "ensemble", "all"],
                       default=["all"],
                       help="Models to train")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Training batch size")
    parser.add_argument("--data-path", type=str, default="data/spectrograms",
                       help="Path to spectrogram data")
    parser.add_argument("--output-dir", type=str, default="models",
                       help="Directory to save trained models")
    
    args = parser.parse_args()
    
    print("🧠 NeuroMind EEG Classifier - Training Pipeline")
    print("=" * 50)
    
    # Import training modules
    try:
        from training.trainer import train_model
        from models.model import get_model
        print("✅ Successfully imported training modules")
    except ImportError as e:
        print(f"❌ Failed to import modules: {e}")
        print("Make sure you're running from the project root directory")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    models_to_train = args.models if "all" not in args.models else ["resnet18", "efficientnet", "ensemble"]
    
    for model_name in models_to_train:
        print(f"\n🚀 Training {model_name.upper()} model...")
        try:
            model = get_model(model_name)
            best_accuracy = train_model(
                model=model,
                model_name=model_name,
                data_path=args.data_path,
                epochs=args.epochs,
                batch_size=args.batch_size,
                output_dir=args.output_dir
            )
            print(f"✅ {model_name} training completed. Best accuracy: {best_accuracy:.3f}")
        except Exception as e:
            print(f"❌ Training failed for {model_name}: {e}")
    
    print("\n🎉 Training pipeline completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())