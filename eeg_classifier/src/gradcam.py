"""
gradcam.py — Task 7
-------------------
Grad-CAM for ResNet18 and EfficientNet-B0.
Saves sample outputs for all 3 classes to results/gradcam/
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm_module
from PIL import Image

CLASSES = ["focused", "relaxed", "stressed"]


class GradCAM:
    """
    Grad-CAM implementation.
    Works with any CNN — hooks on the specified target layer.
    """
    def __init__(self, model, target_layer):
        self.model        = model
        self._gradients   = None
        self._activations = None
        self._fwd = target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, '_activations', o.detach())
        )
        self._bwd = target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, '_gradients', go[0].detach())
        )

    def __call__(self, input_tensor, class_idx=None):
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        # Global average pool gradients → importance weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self._activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam).squeeze().cpu().numpy()

        # Handle case where cam is scalar (single spatial location)
        if cam.ndim == 0:
            cam = np.array([[cam.item()]])

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def remove_hooks(self):
        self._fwd.remove()
        self._bwd.remove()


def overlay_heatmap(original_img, heatmap, alpha=0.4):
    """Blend Grad-CAM heatmap onto original spectrogram."""
    H, W = original_img.shape[:2]
    hm   = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)
    ) / 255.0
    rgb     = (cm_module.get_cmap("jet")(hm)[:, :, :3] * 255).astype(np.uint8)
    blended = (alpha * rgb + (1 - alpha) * original_img).astype(np.uint8)
    return blended


def visualise_gradcam(model, target_layer, input_tensor, original_img,
                      class_names, save_path="results/gradcam/gradcam.png",
                      device=None):
    """
    Full Grad-CAM pipeline: compute → overlay → save 3-panel figure.
    alpha=0.4 as specified in Task 7.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    input_tensor = input_tensor.to(device)

    gcam    = GradCAM(model, target_layer)
    heatmap = gcam(input_tensor)
    gcam.remove_hooks()

    with torch.no_grad():
        out   = model(input_tensor)
        probs = torch.softmax(out, dim=1).squeeze().cpu().numpy()

    pred_idx   = probs.argmax()
    pred_label = class_names[pred_idx]
    confidence = probs[pred_idx] * 100
    blended    = overlay_heatmap(original_img, heatmap, alpha=0.4)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor('#0F1117')

    axes[0].imshow(original_img)
    axes[0].set_title("STFT Spectrogram", color='#CCC', fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet", aspect='auto')
    axes[1].set_title("Grad-CAM Heatmap", color='#CCC', fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(blended)
    axes[2].set_title(f"Overlay — {pred_label} ({confidence:.1f}%)",
                      color='#CCC', fontsize=10)
    axes[2].axis("off")

    plt.suptitle("EEG Mental State — Grad-CAM Explainability",
                 color='white', fontsize=12, y=1.01)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor='#0F1117')
    plt.close()
    print(f"[INFO] Grad-CAM saved -> '{save_path}'")
    return pred_label, confidence, probs


def save_gradcam_all_classes(model, arch, data_dir="data/spectrograms",
                              save_dir="results/gradcam", device=None):
    """
    Task 7: Save Grad-CAM outputs for all 3 classes.
    """
    import torchvision.transforms as T
    from src.model import get_last_conv_layer

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    target_layer = get_last_conv_layer(model, arch)

    for cls in CLASSES:
        imgs = list(Path(data_dir, cls).glob("*.png"))
        if not imgs:
            print(f"[WARN] No images for class '{cls}'")
            continue
        img_pil = Image.open(imgs[0]).convert("RGB")
        img_np  = np.array(img_pil.resize((224, 224)))
        tensor  = transform(img_pil).unsqueeze(0)

        visualise_gradcam(
            model, target_layer, tensor, img_np,
            CLASSES,
            save_path=f"{save_dir}/gradcam_{arch}_{cls}.png",
            device=device,
        )
