"""
evaluate.py — Task 5
--------------------
Full evaluation suite:
- Accuracy, Precision, Recall, F1 (per class + macro)
- Confusion matrix
- ROC curves + AUC (one-vs-rest)
- Calibration curve (reliability diagram)
- Per-class accuracy breakdown
All plots saved to /results/
"""

from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve

CLASSES     = ["focused", "relaxed", "stressed"]
RESULTS_DIR = Path("results")


def evaluate_model(model, dataloader, device=None,
                   save_dir="results", arch="model",
                   temperature=1.0):
    """
    Full evaluation with all metrics and plots.

    Parameters
    ----------
    temperature : float  calibration temperature (1.0 = uncalibrated)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs  = inputs.to(device)
            logits  = model(inputs)
            probs   = torch.softmax(logits / temperature, dim=1)
            preds   = probs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    # ── Core metrics ───────────────────────────────────────────────────────────
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=[0,1,2],
        zero_division=0
    )
    mp, mr, mf1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    # ── Per-class accuracy ─────────────────────────────────────────────────────
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)

    # ── Print report ───────────────────────────────────────────────────────────
    print(f"\n{'='*56}")
    print(f"  EVALUATION — {arch.upper()}")
    print(f"{'='*56}")
    print(f"  Overall Accuracy  : {acc*100:.2f}%")
    print(f"  Macro Precision   : {mp:.4f}")
    print(f"  Macro Recall      : {mr:.4f}")
    print(f"  Macro F1-Score    : {mf1:.4f}")
    print(f"\n  Per-Class Results:")
    print(f"  {'Class':<12} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>5}")
    print("  " + "-"*44)
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:<12} {per_class_acc[i]:>6.3f} {precision[i]:>6.3f} "
              f"{recall[i]:>6.3f} {f1[i]:>6.3f} {int(support[i]):>5}")
    print(f"{'='*56}")
    print(classification_report(all_labels, all_preds,
                                 target_names=CLASSES, zero_division=0))

    # ── Plots ──────────────────────────────────────────────────────────────────
    _plot_confusion_matrix(cm, save_dir, arch)
    _plot_roc_curves(all_labels, all_probs, save_dir, arch)
    _plot_calibration(all_labels, all_probs, save_dir, arch)

    return {
        "accuracy":          acc,
        "macro_precision":   mp,
        "macro_recall":      mr,
        "macro_f1":          mf1,
        "per_class_acc":     per_class_acc.tolist(),
        "per_class_f1":      f1.tolist(),
        "confusion_matrix":  cm.tolist(),
        "all_probs":         all_probs,
        "all_preds":         all_preds,
        "all_labels":        all_labels,
    }


def _plot_confusion_matrix(cm, save_dir, arch):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {arch}")
    plt.tight_layout()
    out = Path(save_dir) / f"confusion_matrix_{arch}.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"[INFO] Confusion matrix -> '{out}'")


def _plot_roc_curves(labels, probs, save_dir, arch):
    """ROC curves for all 3 classes (one-vs-rest) + AUC scores."""
    y_bin   = label_binarize(labels, classes=[0, 1, 2])
    colors  = ["#4CAF50", "#2196F3", "#F44336"]
    fig, ax = plt.subplots(figsize=(7, 5))

    for i, (cls, color) in enumerate(zip(CLASSES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{cls} (AUC = {roc_auc:.3f})")

    ax.plot([0,1],[0,1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {arch}")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = Path(save_dir) / f"roc_curves_{arch}.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"[INFO] ROC curves -> '{out}'")


def _plot_calibration(labels, probs, save_dir, arch):
    """Reliability diagram (calibration curve)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    colors  = ["#4CAF50", "#2196F3", "#F44336"]
    y_bin   = label_binarize(labels, classes=[0, 1, 2])

    for i, (cls, color) in enumerate(zip(CLASSES, colors)):
        try:
            fraction_pos, mean_pred = calibration_curve(
                y_bin[:, i], probs[:, i], n_bins=10
            )
            ax.plot(mean_pred, fraction_pos, "s-",
                    color=color, label=cls, lw=2)
        except Exception:
            pass

    ax.plot([0,1],[0,1], "k--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration Curve — {arch}")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = Path(save_dir) / f"calibration_{arch}.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"[INFO] Calibration curve -> '{out}'")


def compare_models(results_dict, save_dir="results"):
    """
    Print and plot side-by-side comparison of multiple models.
    results_dict: {"resnet18": {...}, "efficientnet_b0": {...}, "ensemble": {...}}
    """
    Path(save_dir).mkdir(exist_ok=True)
    print(f"\n{'='*60}")
    print("  MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Model':<20} {'Accuracy':>10} {'F1 Macro':>10}")
    print("  " + "-"*42)
    for name, res in results_dict.items():
        print(f"  {name:<20} {res['accuracy']*100:>9.2f}% "
              f"{res['macro_f1']:>10.4f}")
    print(f"{'='*60}")

    # Bar chart comparison
    names  = list(results_dict.keys())
    accs   = [r["accuracy"]*100 for r in results_dict.values()]
    f1s    = [r["macro_f1"]*100 for r in results_dict.values()]
    x      = np.arange(len(names))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, accs, width, label="Accuracy %", color="#4CAF50", alpha=0.85)
    ax.bar(x + width/2, f1s,  width, label="F1 Macro %", color="#2196F3", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Score (%)"); ax.set_title("Model Comparison")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 100)
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i - width/2, a + 1, f"{a:.1f}%", ha="center", fontsize=9)
        ax.text(i + width/2, f + 1, f"{f:.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    out = Path(save_dir) / "model_comparison.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"[INFO] Comparison chart -> '{out}'")
