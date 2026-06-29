"""
model.py
--------
CNN architectures for EEG classification: ResNet18, EfficientNet-B0, and Ensemble.
Implements transfer learning from ImageNet pre-trained models.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path

NUM_CLASSES = 3  # focused, relaxed, stressed
CKPT_DIR = Path("models/checkpoints")


def build_model(arch="resnet18", pretrained=True, freeze_backbone=False, dropout=0.4):
    """
    Build CNN classifier with transfer learning.
    
    Parameters
    ----------
    arch : str
        Architecture name: 'resnet18', 'resnet50', 'efficientnet_b0'
    pretrained : bool
        Use ImageNet pre-trained weights
    freeze_backbone : bool
        Freeze backbone parameters (only train classifier)
    dropout : float
        Dropout rate for classifier
    
    Returns
    -------
    torch.nn.Module
        CNN model ready for training
    """
    weights = "DEFAULT" if pretrained else None
    
    if arch == "resnet18":
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features  # 512
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, NUM_CLASSES)
        )
        
    elif arch == "resnet50":
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features  # 2048
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, NUM_CLASSES)
        )
        
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features  # 1280
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, NUM_CLASSES)
        )
        
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    
    # Optionally freeze backbone
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "fc" not in name and "classifier" not in name:
                param.requires_grad = False
        print(f"[INFO] Backbone frozen for {arch}")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] {arch} | Total: {total_params:,} | Trainable: {trainable_params:,}")
    
    return model


def get_last_conv_layer(model, arch):
    """
    Get the last convolutional layer for Grad-CAM.
    
    Parameters
    ----------
    model : torch.nn.Module
        CNN model
    arch : str
        Architecture name
    
    Returns
    -------
    torch.nn.Module
        Last convolutional layer
    """
    if arch in ("resnet18", "resnet50"):
        return model.layer4[-1]
    elif arch == "efficientnet_b0":
        return model.features[-1]
    else:
        raise ValueError(f"Grad-CAM not implemented for: {arch}")


def save_checkpoint(model, arch, tag="best"):
    """
    Save model checkpoint.
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained model
    arch : str
        Architecture name
    tag : str
        Checkpoint tag (e.g., 'best', 'epoch_10')
    
    Returns
    -------
    str
        Path to saved checkpoint
    """
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    path = CKPT_DIR / f"{tag}_{arch}.pth"
    torch.save(model.state_dict(), path)
    print(f"[INFO] Checkpoint saved → '{path}'")
    return str(path)


def load_checkpoint(arch, tag="best", device=None):
    """
    Load a saved checkpoint.
    
    Parameters
    ----------
    arch : str
        Architecture name
    tag : str
        Checkpoint tag
    device : torch.device
        Device to load model on
    
    Returns
    -------
    torch.nn.Module
        Loaded model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    path = CKPT_DIR / f"{tag}_{arch}.pth"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    model = build_model(arch=arch, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    print(f"[INFO] Loaded checkpoint: '{path}'")
    return model


class EnsembleModel(nn.Module):
    """
    Ensemble model combining ResNet18 and EfficientNet-B0.
    Averages softmax probabilities from both models.
    """
    
    def __init__(self, model_resnet, model_effnet):
        """
        Initialize ensemble model.
        
        Parameters
        ----------
        model_resnet : torch.nn.Module
            ResNet18 model
        model_effnet : torch.nn.Module
            EfficientNet-B0 model
        """
        super().__init__()
        self.resnet = model_resnet
        self.effnet = model_effnet
    
    def forward(self, x):
        """
        Forward pass through ensemble.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        
        Returns
        -------
        torch.Tensor
            Averaged predictions
        """
        # Get softmax probabilities from both models
        p1 = torch.softmax(self.resnet(x), dim=1)
        p2 = torch.softmax(self.effnet(x), dim=1)
        
        # Average probabilities
        return (p1 + p2) / 2


def build_ensemble(device=None):
    """
    Build ensemble model from saved checkpoints.
    
    Parameters
    ----------
    device : torch.device
        Device to load models on
    
    Returns
    -------
    EnsembleModel
        Ensemble model ready for inference
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load individual models
    resnet = load_checkpoint("resnet18", device=device)
    effnet = load_checkpoint("efficientnet_b0", device=device)
    
    # Create ensemble
    ensemble = EnsembleModel(resnet, effnet).to(device)
    ensemble.eval()
    print("[INFO] Ensemble model ready (ResNet18 + EfficientNet-B0)")
    return ensemble


def get_model_info(arch):
    """
    Get model architecture information.
    
    Parameters
    ----------
    arch : str
        Architecture name
    
    Returns
    -------
    dict
        Model information
    """
    model = build_model(arch, pretrained=False)
    
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / (1024 ** 2)  # Assuming float32
    
    return {
        "architecture": arch,
        "parameters": total_params,
        "size_mb": round(model_size_mb, 2),
        "input_size": (3, 224, 224),
        "num_classes": NUM_CLASSES
    }


if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build models
    resnet = build_model("resnet18", pretrained=True)
    effnet = build_model("efficientnet_b0", pretrained=True)
    
    print("\nModel Information:")
    for arch in ["resnet18", "efficientnet_b0"]:
        info = get_model_info(arch)
        print(f"{arch}: {info['parameters']:,} params, {info['size_mb']} MB")