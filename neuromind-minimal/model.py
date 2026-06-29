"""CNN models for EEG classification"""
import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path

def build_model(arch="resnet18"):
    """Build CNN model with transfer learning"""
    if arch == "resnet18":
        model = models.resnet18(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, 3)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights="DEFAULT")
        model.classifier = nn.Linear(model.classifier[1].in_features, 3)
    return model

def save_model(model, arch):
    """Save trained model"""
    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), f"models/{arch}.pth")

def load_model(arch):
    """Load trained model"""
    model = build_model(arch)
    model.load_state_dict(torch.load(f"models/{arch}.pth", map_location="cpu"))
    return model.eval()

class EnsembleModel(nn.Module):
    """Ensemble of ResNet18 + EfficientNet-B0"""
    def __init__(self):
        super().__init__()
        self.resnet = load_model("resnet18")
        self.effnet = load_model("efficientnet_b0")
    
    def forward(self, x):
        p1 = torch.softmax(self.resnet(x), dim=1)
        p2 = torch.softmax(self.effnet(x), dim=1)
        return (p1 + p2) / 2