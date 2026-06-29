#!/usr/bin/env python3
"""
Model evaluation script for NeuroMind EEG Classifier.

Evaluates trained models and generates performance reports.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

def main():
    parser = argparse.ArgumentParser(description="Evaluate EEG classification models")
    parser.add_argument("--models-dir", type=str, default="models",
                       help="Directory containing trained models")
    parser.add_argument("--data-path", type=str, default="data/spectrograms",
                       help="Path to test data")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Directory to save evaluation results")
    parser.add_argument("--generate-gradcam", action="store_true",
                       help="Generate Grad-CAM visualizations")
    
    args = parser.parse_args()
    
    print("🔬 NeuroMind EEG Classifier - Model Evaluation")
    print("=" * 50)
    
    # Import evaluation modules
    try:
        from training.evaluate import evaluate_model
        from utils.gradcam import generate_gradcam_visualizations
        print("✅ Successfully imported evaluation modules")
    except ImportError as e:
        print(f"❌ Failed to import modules: {e}")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find model files
    model_files = list(Path(args.models_dir).glob("*.pth"))
    
    if not model_files:
        print(f"❌ No model files found in {args.models_dir}")
        return 1
    
    print(f"📋 Found {len(model_files)} models to evaluate")
    
    results = {}
    
    for model_file in model_files:
        model_name = model_file.stem
        print(f"\n🔍 Evaluating {model_name}...")
        
        try:
            accuracy, f1_score, report = evaluate_model(
                model_path=str(model_file),
                data_path=args.data_path,
                output_dir=args.output_dir
            )
            
            results[model_name] = {
                "accuracy": accuracy,
                "f1_score": f1_score,
                "report": report
            }
            
            print(f"✅ {model_name}: Accuracy={accuracy:.3f}, F1={f1_score:.3f}")
            
            # Generate Grad-CAM if requested
            if args.generate_gradcam:
                print(f"🎨 Generating Grad-CAM for {model_name}...")
                gradcam_dir = Path(args.output_dir) / "gradcam" / model_name
                gradcam_dir.mkdir(parents=True, exist_ok=True)
                generate_gradcam_visualizations(
                    model_path=str(model_file),
                    data_path=args.data_path,
                    output_dir=str(gradcam_dir)
                )
                
        except Exception as e:
            print(f"❌ Evaluation failed for {model_name}: {e}")
    
    # Print summary
    print("\n📊 Evaluation Summary")
    print("-" * 30)
    for model_name, result in results.items():
        print(f"{model_name:20}: {result['accuracy']:.3f} acc, {result['f1_score']:.3f} f1")
    
    print(f"\n💾 Results saved to: {args.output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())