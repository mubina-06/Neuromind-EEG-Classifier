"""
main.py
-------
Full pipeline entry point.

Usage:
  python main.py                          # mock data, resnet18, 20 epochs
  python main.py --deap_dir data/raw/deap # real DEAP data
  python main.py --arch efficientnet_b0 --epochs 30
"""

import argparse, sys, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.preprocess import load_data, save_spectrograms, load_physionet_data
from src.dataset    import get_dataloaders
from src.model      import build_model, get_last_conv_layer
from src.train      import train_model
from src.evaluate   import evaluate_model
from src.gradcam    import visualise_gradcam

import numpy as np
import torchvision.transforms as T
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch",       default="resnet18",
                   choices=["resnet18","resnet50","efficientnet_b0"])
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--batch_size", type=int,   default=16)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--deap_dir",   default="data/raw/deap")
    p.add_argument("--data_dir",   default="data/spectrograms")
    p.add_argument("--save_dir",   default="models")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--n_subjects",  type=int,   default=10,
                   help="Number of PhysioNet subjects to download (1-109)")
    p.add_argument("--skip_preprocess", action="store_true")
    p.add_argument("--freeze_backbone", action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*52}")
    print("  EEG Brain Signal Classifier")
    print(f"  Device  : {device}")
    print(f"  Arch    : {args.arch}")
    print(f"  Epochs  : {args.epochs}")
    print(f"{'='*52}\n")

    # ── Step 1: Load & preprocess ──────────────────────────────────────────────
    if not args.skip_preprocess:
        print("STEP 1: Loading EEG data and generating spectrograms...")
        raw_list, labels = load_data(
            deap_dir=args.deap_dir,
            n_subjects=args.n_subjects
        )
        save_spectrograms(raw_list, labels, output_dir=args.data_dir)
    else:
        print("STEP 1: Skipping preprocessing (using existing spectrograms).")

    # ── Step 2: Dataset ────────────────────────────────────────────────────────
    print("\nSTEP 2: Preparing dataset (70/15/15 split)...")
    loaders, n_total = get_dataloaders(
        data_dir   = args.data_dir,
        batch_size = args.batch_size,
    )

    # ── Step 3: Model ──────────────────────────────────────────────────────────
    print("\nSTEP 3: Building ResNet18 model...")
    model = build_model(
        arch            = args.arch,
        pretrained      = True,
        freeze_backbone = args.freeze_backbone,
    )

    # ── Step 4: Train ──────────────────────────────────────────────────────────
    print("\nSTEP 4: Training...")
    model, history = train_model(
        model       = model,
        dataloaders = loaders,
        num_epochs  = args.epochs,
        lr          = args.lr,
        save_dir    = args.save_dir,
        device      = device,
    )

    # ── Step 5: Evaluate ───────────────────────────────────────────────────────
    print("\nSTEP 5: Evaluating on test set...")
    results = evaluate_model(
        model      = model,
        dataloader = loaders["test"],
        device     = device,
        save_dir   = args.output_dir,
    )

    # ── Step 6: Grad-CAM ───────────────────────────────────────────────────────
    print("\nSTEP 6: Generating Grad-CAM...")
    _run_gradcam(model, args, device)

    # ── Step 7: Sample prediction ──────────────────────────────────────────────
    print("\nSTEP 7: Sample prediction...")
    _sample_prediction(model, args, device)

    print(f"\n{'='*52}")
    print("  PIPELINE COMPLETE")
    print(f"  Models  → {args.save_dir}/")
    print(f"  Outputs → {args.output_dir}/")
    print(f"  Run demo: streamlit run app.py")
    print(f"{'='*52}")


def _run_gradcam(model, args, device):
    data_dir = Path(args.data_dir)
    for cls in ["focused","relaxed","stressed"]:
        imgs = list((data_dir / cls).glob("*.png"))
        if not imgs: continue
        transform = T.Compose([
            T.Resize((224,224)), T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        img_pil = Image.open(imgs[0]).convert("RGB")
        img_np  = np.array(img_pil.resize((224,224)))
        tensor  = transform(img_pil).unsqueeze(0)
        target_layer = get_last_conv_layer(model, args.arch)
        visualise_gradcam(
            model, target_layer, tensor, img_np,
            ["focused","relaxed","stressed"],
            save_path=f"{args.output_dir}/gradcam_sample.png",
            device=device,
        )
        break


def _sample_prediction(model, args, device):
    from src.preprocess import epoch_to_spectrogram
    CLASSES = ["focused","relaxed","stressed"]
    data_dir = Path(args.data_dir)
    transform = T.Compose([
        T.Resize((224,224)), T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    for cls in CLASSES:
        imgs = list((data_dir / cls).glob("*.png"))
        if not imgs: continue
        img_pil = Image.open(imgs[0]).convert("RGB")
        tensor  = transform(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy()
        pred  = CLASSES[probs.argmax()]
        conf  = probs.max() * 100
        print(f"\n{'='*42}")
        print("  EEG MENTAL STATE PREDICTION")
        print(f"{'='*42}")
        print(f"  Predicted : {pred.upper()}")
        print(f"  Confidence: {conf:.2f}%")
        print("\n  All Probabilities:")
        for c, p in zip(CLASSES, probs):
            bar = "█" * int(p * 100 / 5)
            print(f"    {c:<10} {p*100:>6.2f}%  {bar}")
        print(f"{'='*42}")
        break


if __name__ == "__main__":
    main()
