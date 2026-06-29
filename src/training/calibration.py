"""
calibration.py — Task 4
-----------------------
Temperature Scaling for confidence calibration.
Finds optimal temperature T on validation set that minimises NLL.
Calibrated softmax: softmax(logits / T)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class TemperatureScaler(nn.Module):
    """
    Wraps a model and applies temperature scaling to its logits.
    T > 1 makes predictions less confident (softens probabilities).
    T < 1 makes predictions more confident.
    """
    def __init__(self, model):
        super().__init__()
        self.model       = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature

    def calibrate(self, val_loader, device=None, lr=0.01, max_iter=50):
        """
        Find optimal temperature T using NLL loss on validation set.
        Only the temperature parameter is optimised — model weights frozen.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.eval()
        self.to(device)

        # Collect all logits and labels from val set
        all_logits, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                logits = self.model(inputs)
                all_logits.append(logits.cpu())
                all_labels.append(labels)

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        # Optimise temperature
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr,
                                       max_iter=max_iter)

        def eval_step():
            optimizer.zero_grad()
            loss = criterion(all_logits / self.temperature, all_labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)

        T = self.temperature.item()
        print(f"[INFO] Optimal temperature T = {T:.4f}")
        return T

    def get_calibrated_probs(self, logits):
        """Apply temperature scaling and return calibrated probabilities."""
        return torch.softmax(logits / self.temperature, dim=1)


def calibrate_model(model, val_loader, save_path="models/checkpoints/temperature.pt",
                    device=None):
    """
    Full calibration pipeline.
    Returns TemperatureScaler wrapping the model.
    """
    scaler = TemperatureScaler(model)
    T      = scaler.calibrate(val_loader, device=device)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"temperature": T}, save_path)
    print(f"[INFO] Temperature saved -> '{save_path}'")
    return scaler, T


def load_temperature(path="models/checkpoints/temperature.pt"):
    """Load saved temperature value."""
    if Path(path).exists():
        data = torch.load(path, map_location="cpu")
        return data["temperature"]
    return 1.0   # default: no calibration
