"""
download_data.py
----------------
Complete pipeline: Download PhysioNet EEG data → Preprocess → Train models → Evaluate.
This is the main entry point for the entire NeuroMind system.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Configure MNE data path
MNE_DATA_PATH = os.path.join(os.path.expanduser("~"), "mne_data")
os.makedirs(MNE_DATA_PATH, exist_ok=True)
os.environ["MNE_DATA"] = MNE_DATA_PATH

import mne
mne.set_config("MNE_DATA", MNE_DATA_PATH)
mne.set_config("MNE_DATASETS_EEGBCI_PATH", MNE_DATA_PATH)

import torch
from pathlib import Path

# Import our modules
sys.path.insert(0, os.path.dirname(__file__))
from src.preprocess import load_physionet_data, save_spectrograms
from src.model import build_model, save_checkpoint, build_ensemble

# Configuration
N_SUBJECTS = 10  # Number of subjects to download (max 109)
RUNS = [1, 3, 4]  # EEG tasks: rest, real movement, motor imagery
DATA_DIR = "data/spectrograms"
RESULTS_DIR = "results"
EPOCHS = 15
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    """Main pipeline execution."""
    print("=" * 60)
    print("  🧠 NeuroMind EEG Brain Signal Classifier")
    print("  Complete ML Pipeline: Data → Models → Evaluation")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Subjects: {N_SUBJECTS}")
    print(f"  Data will be saved to: {DATA_DIR}")
    print("=" * 60)
    
    # Step 1: Download and preprocess EEG data
    print("\n[STEP 1] 📥 Downloading PhysioNet EEG data...")
    try:
        raw_list, labels = load_physionet_data(n_subjects=N_SUBJECTS, runs=RUNS)
        print(f"✅ Successfully loaded {len(raw_list)} EEG recordings")
    except Exception as e:
        print(f"❌ Failed to download data: {e}")
        print("💡 Check your internet connection and try again")
        return
    
    # Step 2: Generate spectrograms
    print(f"\n[STEP 2] 🎨 Generating spectrograms...")
    existing_spectrograms = list(Path(DATA_DIR).rglob("*.png")) if Path(DATA_DIR).exists() else []
    
    if len(existing_spectrograms) < 100:
        try:
            output_dir = save_spectrograms(raw_list, labels, DATA_DIR)
            print(f"✅ Spectrograms saved to: {output_dir}")
        except Exception as e:
            print(f"❌ Failed to generate spectrograms: {e}")
            return
    else:
        print(f"✅ Using existing {len(existing_spectrograms)} spectrograms")
    
    # Step 3: Prepare dataset
    print(f"\n[STEP 3] 📊 Preparing dataset...")
    try:
        from src.dataset import get_dataloaders
        loaders, total_samples = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)
        print(f"✅ Dataset ready: {total_samples} samples")
        print(f"   Train: {len(loaders['train'].dataset)} samples")
        print(f"   Val:   {len(loaders['val'].dataset)} samples") 
        print(f"   Test:  {len(loaders['test'].dataset)} samples")
    except Exception as e:
        print(f"❌ Failed to prepare dataset: {e}")
        return
    
    # Step 4: Train ResNet18
    print(f"\n[STEP 4] 🚀 Training ResNet18...")
    try:
        model_resnet = build_model("resnet18", pretrained=True)
        from src.train import train_model
        model_resnet, history_resnet = train_model(
            model_resnet, loaders, num_epochs=EPOCHS, device=DEVICE, arch="resnet18"
        )
        save_checkpoint(model_resnet, "resnet18", "best")
        print("✅ ResNet18 training completed")
    except Exception as e:
        print(f"❌ ResNet18 training failed: {e}")
        return
    
    # Step 5: Train EfficientNet-B0
    print(f"\n[STEP 5] ⚡ Training EfficientNet-B0...")
    try:
        model_effnet = build_model("efficientnet_b0", pretrained=True)
        model_effnet, history_effnet = train_model(
            model_effnet, loaders, num_epochs=EPOCHS, device=DEVICE, arch="efficientnet_b0"
        )
        save_checkpoint(model_effnet, "efficientnet_b0", "best")
        print("✅ EfficientNet-B0 training completed")
    except Exception as e:
        print(f"❌ EfficientNet-B0 training failed: {e}")
        return
    
    # Step 6: Evaluate models
    print(f"\n[STEP 6] 📈 Evaluating models...")
    try:
        from src.evaluate import evaluate_model, compare_models
        Path(RESULTS_DIR).mkdir(exist_ok=True)
        
        # Evaluate individual models
        results_resnet = evaluate_model(
            model_resnet, loaders["test"], device=DEVICE, 
            save_dir=RESULTS_DIR, arch="resnet18"
        )
        
        results_effnet = evaluate_model(
            model_effnet, loaders["test"], device=DEVICE,
            save_dir=RESULTS_DIR, arch="efficientnet_b0"
        )
        
        # Build and evaluate ensemble
        try:
            ensemble = build_ensemble(DEVICE)
            results_ensemble = evaluate_model(
                ensemble, loaders["test"], device=DEVICE,
                save_dir=RESULTS_DIR, arch="ensemble"
            )
            
            all_results = {
                "ResNet18": results_resnet,
                "EfficientNet-B0": results_effnet,
                "Ensemble": results_ensemble
            }
        except Exception as e:
            print(f"⚠️  Ensemble evaluation failed: {e}")
            all_results = {
                "ResNet18": results_resnet,
                "EfficientNet-B0": results_effnet
            }
        
        # Compare all models
        compare_models(all_results, save_dir=RESULTS_DIR)
        print("✅ Model evaluation completed")
        
    except Exception as e:
        print(f"❌ Model evaluation failed: {e}")
        return
    
    # Step 7: Generate Grad-CAM visualizations
    print(f"\n[STEP 7] 🔍 Generating Grad-CAM visualizations...")
    try:
        from src.gradcam import save_gradcam_all_classes
        gradcam_dir = Path(RESULTS_DIR) / "gradcam"
        gradcam_dir.mkdir(exist_ok=True)
        
        save_gradcam_all_classes(
            model_resnet, "resnet18", DATA_DIR, str(gradcam_dir), DEVICE
        )
        save_gradcam_all_classes(
            model_effnet, "efficientnet_b0", DATA_DIR, str(gradcam_dir), DEVICE
        )
        print("✅ Grad-CAM visualizations generated")
        
    except Exception as e:
        print(f"⚠️  Grad-CAM generation failed: {e}")
    
    # Final summary
    best_model = max(all_results.items(), key=lambda x: x[1]["accuracy"])
    print(f"\n{'='*60}")
    print("  🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"  📊 Best Model: {best_model[0]} ({best_model[1]['accuracy']*100:.1f}% accuracy)")
    print(f"  💾 Models saved to: models/checkpoints/")
    print(f"  📈 Results saved to: {RESULTS_DIR}/")
    print(f"  🎨 Spectrograms: {DATA_DIR}/")
    print(f"\n  🚀 Ready to run the web application:")
    print(f"     streamlit run app/modern_app.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error: {e}")
        print("💡 Check the error message above and try again")