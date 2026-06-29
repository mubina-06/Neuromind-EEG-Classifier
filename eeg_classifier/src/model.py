"""
model.py
--------
Task 2: ResNet18 + EfficientNet-B0 with transfer learning.
Task 3: Ensemble model combining both.
Checkpoints saved to models/checkpoints/
"""

import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path

NUM_CLASSES   = 3
CKPT_DIR      = Path("models/checkpoints")


def build_model(arch="resnet18", pretrained=True,
                freeze_backbone=False, dropout=0.4):
    """
    Build CNN classifier.
    arch: 'resnet18' | 'resnet50' | 'efficientnet_b0'
    """
    weights = "DEFAULT" if pretrained else None

    if arch == "resnet18":
        model = models.resnet18(weights=weights)
        in_f  = model.fc.in_features          # 512
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_f, NUM_CLASSES)
        )

    elif arch == "resnet50":
        model = models.resnet50(weights=weights)
        in_f  = model.fc.in_features          # 2048
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_f, NUM_CLASSES)
        )

    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        in_f  = model.classifier[1].in_features  # 1280
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_f, NUM_CLASSES)          # Linear(1280, 3)
        )

    else:
        raise ValueError(f"Unknown arch: {arch}")

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "fc" not in name and "classifier" not in name:
                param.requires_grad = False
        print(f"[INFO] Backbone frozen.")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] {arch} | Params: {total:,} | Trainable: {trainable:,}")
    return model


def get_last_conv_layer(model, arch):
    """Returns last conv layer for Grad-CAM (Task 7)."""
    if arch in ("resnet18", "resnet50"):
        return model.layer4[-1]
    elif arch == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(f"Grad-CAM not defined for: {arch}")


def save_checkpoint(model, arch, tag="best"):
    """Save model checkpoint to models/checkpoints/"""
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    path = CKPT_DIR / f"{tag}_{arch}.pth"
    torch.save(model.state_dict(), path)
    print(f"[INFO] Checkpoint saved -> '{path}'")
    return str(path)


def load_checkpoint(arch, tag="best", device=None):
    """Load a saved checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path  = CKPT_DIR / f"{tag}_{arch}.pth"
    model = build_model(arch=arch, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    print(f"[INFO] Loaded checkpoint: '{path}'")
    return model


# ── Task 3: Ensemble ───────────────────────────────────────────────────────────
class EnsembleModel(nn.Module):
    """
    Averages softmax probabilities from ResNet18 + EfficientNet-B0.
    No additional training needed — uses pretrained checkpoints.
    """
    def __init__(self, model_resnet, model_effnet):
        super().__init__()
        self.resnet = model_resnet
        self.effnet = model_effnet

    def forward(self, x):
        p1 = torch.softmax(self.resnet(x), dim=1)
        p2 = torch.softmax(self.effnet(x),  dim=1)
        return (p1 + p2) / 2   # average probabilities


def build_ensemble(device=None):
    """Load both models and return ensemble."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = load_checkpoint("resnet18",       device=device)
    effnet = load_checkpoint("efficientnet_b0", device=device)
    ensemble = EnsembleModel(resnet, effnet).to(device)
    ensemble.eval()
    print("[INFO] Ensemble model ready (ResNet18 + EfficientNet-B0)")
    return ensemble
