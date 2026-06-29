"""
predict.py
----------
Inference module: takes a raw EEG signal (or a spectrogram image path)
and returns the predicted mental state with confidence scores.
"""

from pathlib import Path
from typing import Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from src.preprocess import preprocess_raw, epoch_raw, epoch_to_spectrogram
from src.model import build_model, get_last_conv_layer
from src.gradcam import visualise_gradcam

CLASSES = ["focused", "relaxed", "stressed"]

# Standard ImageNet normalisation (same as training)
INFER_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


def load_trained_model(
    checkpoint_path: str,
    arch: str = "resnet18",
    device: torch.device = None,
) -> torch.nn.Module:
    """Load a saved model checkpoint for inference."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(arch=arch, pretrained=False)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"[INFO] Model loaded from '{checkpoint_path}' on {device}")
    return model


def predict_from_image(
    img_path: str,
    model: torch.nn.Module,
    device: torch.device = None,
    run_gradcam: bool = True,
    arch: str = "resnet18",
    save_dir: str = "outputs",
) -> dict:
    """
    Predict mental state from a spectrogram image file.

    Returns
    -------
    dict with keys: predicted_class, confidence, all_probabilities
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess image
    img_pil  = Image.open(img_path).convert("RGB")
    img_np   = np.array(img_pil.resize((224, 224)))
    tensor   = INFER_TRANSFORM(img_pil).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    # Inference
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx   = probs.argmax()
    pred_class = CLASSES[pred_idx]
    confidence = float(probs[pred_idx]) * 100

    # Grad-CAM
    if run_gradcam:
        target_layer = get_last_conv_layer(model, arch)
        save_path    = str(Path(save_dir) / f"gradcam_{Path(img_path).stem}.png")
        visualise_gradcam(
            model, target_layer, tensor, img_np,
            CLASSES, save_path=save_path, device=device,
        )

    result = {
        "predicted_class":    pred_class,
        "confidence":         round(confidence, 2),
        "all_probabilities":  {cls: round(float(p) * 100, 2)
                               for cls, p in zip(CLASSES, probs)},
    }

    _print_result(result)
    return result


def predict_from_raw_eeg(
    raw,   # mne.io.BaseRaw
    model: torch.nn.Module,
    arch: str = "resnet18",
    device: torch.device = None,
    save_dir: str = "outputs",
) -> list:
    """
    Full pipeline: raw EEG → preprocess → epoch → spectrogram → predict.
    Returns a list of prediction dicts (one per epoch).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from src.preprocess import preprocess_raw, epoch_raw, epoch_to_spectrogram

    raw_clean  = preprocess_raw(raw)
    epochs     = epoch_raw(raw_clean)
    epoch_data = epochs.get_data()
    sfreq      = raw.info["sfreq"]

    results = []
    for i, ep in enumerate(epoch_data):
        img_np = epoch_to_spectrogram(ep, sfreq)
        img_pil = Image.fromarray(img_np)
        tensor  = INFER_TRANSFORM(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        pred_idx   = probs.argmax()
        pred_class = CLASSES[pred_idx]
        confidence = float(probs[pred_idx]) * 100

        result = {
            "epoch":           i,
            "predicted_class": pred_class,
            "confidence":      round(confidence, 2),
            "all_probabilities": {cls: round(float(p) * 100, 2)
                                  for cls, p in zip(CLASSES, probs)},
        }
        results.append(result)
        print(f"  Epoch {i:3d} → {pred_class:10s}  ({confidence:.1f}%)")

    return results


def _print_result(result: dict):
    """Pretty-print a single prediction result."""
    print("\n" + "=" * 40)
    print("  EEG MENTAL STATE PREDICTION")
    print("=" * 40)
    print(f"  Predicted State : {result['predicted_class'].upper()}")
    print(f"  Confidence      : {result['confidence']:.2f}%")
    print("\n  All Probabilities:")
    for cls, prob in result["all_probabilities"].items():
        bar = "█" * int(prob / 5)
        print(f"    {cls:<10} {prob:>6.2f}%  {bar}")
    print("=" * 40)
