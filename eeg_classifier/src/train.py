"""
train.py
--------
Training loop with Adam optimizer, LR scheduler, early stopping.
"""

import copy, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


def train_model(model, dataloaders, num_epochs=30, lr=1e-4,
                weight_decay=1e-4, patience=7,
                save_dir="models", device=None, arch="resnet18"):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on: {device}")

    model = model.to(device)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    ckpt = Path(save_dir) / f"best_{arch}.pth"

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    history       = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
    best_val_loss = float("inf")
    best_weights  = copy.deepcopy(model.state_dict())
    no_improve    = 0

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}  " + "-"*30)

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss = running_correct = total = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=phase, leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss    = criterion(outputs, labels)
                    preds   = outputs.argmax(dim=1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss    += loss.item() * inputs.size(0)
                running_correct += (preds == labels).sum().item()
                total           += inputs.size(0)

            e_loss = running_loss / total
            e_acc  = running_correct / total
            history[f"{phase}_loss"].append(e_loss)
            history[f"{phase}_acc"].append(e_acc)
            print(f"  {phase.upper():5s} → Loss: {e_loss:.4f}  Acc: {e_acc:.4f}")

            if phase == "val":
                scheduler.step(e_loss)
                if e_loss < best_val_loss:
                    best_val_loss = e_loss
                    best_weights  = copy.deepcopy(model.state_dict())
                    torch.save(best_weights, ckpt)
                    ckpt_dir = Path("models/checkpoints")
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(best_weights, ckpt_dir / f"best_{arch}.pth")
                    print(f"  [OK] Best model saved (val_loss={best_val_loss:.4f})")
                    no_improve = 0
                else:
                    no_improve += 1

        if no_improve >= patience:
            print(f"\n[INFO] Early stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_weights)
    _plot_history(history, save_dir)
    print(f"\n[INFO] Training complete. Best val loss: {best_val_loss:.4f}")
    return model, history


def _plot_history(history, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("Loss Curve"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy Curve"); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    out = Path(save_dir) / "training_curves.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"[INFO] Training curves saved → '{out}'")
