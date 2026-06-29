"""
dataset.py
----------
PyTorch Dataset with heavy augmentation (Task 1).
Supports ResNet18 and EfficientNet-B0 (Task 2).
Train 70% / Val 15% / Test 15% split.
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import torchvision.transforms.functional as TF

CLASSES   = ["focused", "relaxed", "stressed"]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}


# ── SpecAugment-style time-frequency masking ───────────────────────────────────
class SpecAugment:
    """
    Randomly masks rectangular blocks in the spectrogram image.
    Simulates missing frequency bands or time segments.
    """
    def __init__(self, num_masks=2, mask_size=30):
        self.num_masks = num_masks
        self.mask_size = mask_size

    def __call__(self, img):
        img = np.array(img).copy()
        H, W = img.shape[:2]
        for _ in range(self.num_masks):
            # Frequency mask (horizontal band)
            f0 = random.randint(0, max(0, H - self.mask_size))
            img[f0:f0 + self.mask_size, :] = 0
            # Time mask (vertical band)
            t0 = random.randint(0, max(0, W - self.mask_size))
            img[:, t0:t0 + self.mask_size] = 0
        return Image.fromarray(img)


class GaussianNoise:
    """Adds random Gaussian noise to the image."""
    def __init__(self, std=0.02):
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std


# ── Transforms ─────────────────────────────────────────────────────────────────
def get_transforms(split="train", img_size=224):
    """
    Training: heavy augmentation to expand effective dataset size.
    Val/Test: only resize + normalize.
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return T.Compose([
            T.Resize((img_size, img_size)),
            # Geometric augmentations
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            # Color augmentations
            T.ColorJitter(brightness=0.2, contrast=0.2,
                          saturation=0.1, hue=0.05),
            # SpecAugment masking
            SpecAugment(num_masks=2, mask_size=25),
            T.ToTensor(),
            T.Normalize(mean, std),
            # Gaussian noise
            GaussianNoise(std=0.02),
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])


# ── Dataset ────────────────────────────────────────────────────────────────────
class EEGSpectrogramDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root      = Path(root)
        self.transform = transform
        self.samples   = []

        for cls in CLASSES:
            cls_dir = self.root / cls
            if not cls_dir.exists():
                continue
            for img_path in sorted(cls_dir.glob("*.png")):
                self.samples.append((str(img_path), CLASS2IDX[cls]))

        if not self.samples:
            raise RuntimeError(
                f"No images found in '{root}'. Run preprocessing first."
            )
        print(f"[INFO] Dataset: {len(self.samples)} images | {self._counts()}")

    def _counts(self):
        from collections import Counter
        c = Counter(lbl for _, lbl in self.samples)
        return " | ".join(f"{CLASSES[k]}:{v}" for k, v in sorted(c.items()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class _TransformWrapper(Dataset):
    def __init__(self, base, transform):
        self.base = base
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img_path, label = self.base.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_dataloaders(data_dir, batch_size=32, img_size=224,
                    val_split=0.15, test_split=0.15,
                    num_workers=0, seed=42):
    """
    Returns train/val/test DataLoaders.
    Training set uses heavy augmentation (Task 1).
    """
    full    = EEGSpectrogramDataset(data_dir)
    n       = len(full)
    n_test  = max(1, int(n * test_split))
    n_val   = max(1, int(n * val_split))
    n_train = n - n_val - n_test

    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full, [n_train, n_val, n_test], generator=g
    )

    train_ds.dataset = _TransformWrapper(full, get_transforms("train", img_size))
    val_ds.dataset   = _TransformWrapper(full, get_transforms("val",   img_size))
    test_ds.dataset  = _TransformWrapper(full, get_transforms("test",  img_size))

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size,
                            shuffle=True,  num_workers=num_workers, pin_memory=False),
        "val":   DataLoader(val_ds,   batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=False),
        "test":  DataLoader(test_ds,  batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=False),
    }
    print(f"[INFO] Split -> train:{n_train} | val:{n_val} | test:{n_test}")
    return loaders, n
