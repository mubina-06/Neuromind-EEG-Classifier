"""
download_data.py
----------------
Full pipeline: download real EEG → preprocess → train both models → evaluate → Grad-CAM.
Trains ResNet18 + EfficientNet-B0, then builds ensemble.

Run: python download_data.py
"""

import os, sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

MNE_DATA_PATH = os.path.join(os.path.expanduser("~"), "mne_data")
os.makedirs(MNE_DATA_PATH, exist_ok=True)
os.environ["MNE_DATA"] = MNE_DATA_PATH

import mne
mne.set_config("MNE_DATA", MNE_DATA_PATH)
mne.set_config("MNE_DATASETS_EEGBCI_PATH", MNE_DATA_PATH)

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from pathlib import Path
from mne.datasets import eegbci

from src.preprocess  import preprocess_raw, epoch_raw, epoch_to_spectrogram, CLASSES, LABEL_MAP
from src.dataset     import get_dataloaders
from src.model       import build_model, save_checkpoint, build_ensemble, get_last_conv_layer
from src.train       import train_model
from src.evaluate    import evaluate_model, compare_models
from src.calibration import calibrate_model
from src.gradcam     import save_gradcam_all_classes

import torchvision.transforms as T
from PIL import Image

# ── Config ─────────────────────────────────────────────────────────────────────
N_SUBJECTS  = 10
RUNS        = [1, 3, 4]
DATA_DIR    = "data/spectrograms"
RESULTS_DIR = "results"
EPOCHS      = 15
BATCH_SIZE  = 16
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_LABEL   = {1:1, 2:1, 3:2, 4:0, 5:2, 6:0}


def download_and_load():
    print(f"\n[STEP 1] Downloading PhysioNet EEGBCI ({N_SUBJECTS} subjects)...")
    raw_list, labels = [], []
    for subj in range(1, N_SUBJECTS + 1):
        for run in RUNS:
            try:
                fnames = eegbci.load_data(subj, [run], path=MNE_DATA_PATH, verbose=False)
                raw    = mne.io.read_raw_edf(fnames[0], preload=True, verbose=False)
                eegbci.standardize(raw)
                montage = mne.channels.make_standard_montage("standard_1005")
                raw.set_montage(montage, on_missing="ignore", verbose=False)
                raw_list.append(raw)
                labels.append(RUN_LABEL[run])
                print(f"  [OK] Subject {subj:03d} Run {run} -> {LABEL_MAP[RUN_LABEL[run]]}")
            except Exception as e:
                print(f"  [FAIL] Subject {subj} Run {run}: {e}")
    print(f"\n[INFO] Loaded {len(raw_list)} recordings.")
    return raw_list, labels


def generate_spectrograms(raw_list, labels):
    print(f"\n[STEP 2] Generating STFT spectrograms...")
    output_dir = Path(DATA_DIR)
    for cls in CLASSES:
        (output_dir / cls).mkdir(parents=True, exist_ok=True)
    total = 0
    for i, (raw, label) in enumerate(zip(raw_list, labels)):
        cls_name = LABEL_MAP[label]
        print(f"  {i+1}/{len(raw_list)} -> {cls_name}")
        try:
            raw_c  = preprocess_raw(raw)
            epochs = epoch_raw(raw_c)
            if len(epochs) == 0: continue
            data  = epochs.get_data()
            sfreq = raw.info["sfreq"]
            for j, ep in enumerate(data):
                img   = epoch_to_spectrogram(ep, sfreq)
                fname = output_dir / cls_name / f"s{i:03d}_e{j:03d}.png"
                Image.fromarray(img).save(fname)
                total += 1
        except Exception as e:
            print(f"  [WARN] {e}")
    print(f"[INFO] Saved {total} spectrograms -> '{DATA_DIR}'")
    return total


def train_arch(arch, loaders):
    print(f"\n[TRAIN] {arch.upper()}...")
    model = build_model(arch=arch, pretrained=True)
    model, history = train_model(
        model, loaders,
        num_epochs=EPOCHS,
        save_dir="models",
        device=DEVICE,
        arch=arch,
    )
    save_checkpoint(model, arch, tag="best")
    return model, history


def run_pipeline():
    print("=" * 56)
    print("  EEG Brain Signal Classifier — Full Pipeline")
    print(f"  Device  : {DEVICE}")
    print(f"  Subjects: {N_SUBJECTS}")
    print("=" * 56)

    # Step 1: Data
    raw_list, labels = download_and_load()

    # Step 2: Spectrograms (skip if already exist)
    existing = list(Path(DATA_DIR).rglob("*.png")) if Path(DATA_DIR).exists() else []
    if len(existing) < 100:
        generate_spectrograms(raw_list, labels)
    else:
        print(f"\n[STEP 2] Using existing {len(existing)} spectrograms.")

    # Step 3: DataLoaders
    print(f"\n[STEP 3] Preparing dataset...")
    loaders, n = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)

    # Step 4: Train ResNet18
    model_resnet, _ = train_arch("resnet18", loaders)

    # Step 5: Train EfficientNet-B0
    model_effnet, _ = train_arch("efficientnet_b0", loaders)

    # Step 6: Calibration
    print(f"\n[STEP 6] Calibrating confidence (Temperature Scaling)...")
    scaler_r, T_r = calibrate_model(model_resnet, loaders["val"],
                                     save_path="models/checkpoints/temperature.pt",
                                     device=DEVICE)

    # Step 7: Evaluate all models
    print(f"\n[STEP 7] Evaluating all models...")
    Path(RESULTS_DIR).mkdir(exist_ok=True)

    res_resnet = evaluate_model(model_resnet, loaders["test"],
                                device=DEVICE, save_dir=RESULTS_DIR,
                                arch="resnet18", temperature=T_r)

    res_effnet = evaluate_model(model_effnet, loaders["test"],
                                device=DEVICE, save_dir=RESULTS_DIR,
                                arch="efficientnet_b0")

    # Ensemble
    try:
        ensemble   = build_ensemble(DEVICE)
        res_ens    = evaluate_model(ensemble, loaders["test"],
                                    device=DEVICE, save_dir=RESULTS_DIR,
                                    arch="ensemble")
        all_results = {
            "ResNet18":        res_resnet,
            "EfficientNet-B0": res_effnet,
            "Ensemble":        res_ens,
        }
    except Exception as e:
        print(f"[WARN] Ensemble failed: {e}")
        all_results = {
            "ResNet18":        res_resnet,
            "EfficientNet-B0": res_effnet,
        }

    compare_models(all_results, save_dir=RESULTS_DIR)

    # Step 8: Grad-CAM for all classes
    print(f"\n[STEP 8] Generating Grad-CAM for all classes...")
    save_gradcam_all_classes(model_resnet, "resnet18",
                              DATA_DIR, "results/gradcam", DEVICE)
    save_gradcam_all_classes(model_effnet, "efficientnet_b0",
                              DATA_DIR, "results/gradcam", DEVICE)

    # Final summary
    best = max(all_results.items(), key=lambda x: x[1]["accuracy"])
    print(f"\n{'='*56}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Best model : {best[0]} ({best[1]['accuracy']*100:.2f}%)")
    print(f"  Checkpoints -> models/checkpoints/")
    print(f"  Results     -> results/")
    print(f"  Grad-CAM    -> results/gradcam/")
    print(f"\n  Run demo:")
    print(f"  streamlit run app/streamlit_app.py")
    print(f"{'='*56}")


if __name__ == "__main__":
    run_pipeline()
