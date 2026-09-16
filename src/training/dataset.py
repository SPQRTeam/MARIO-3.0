"""Dataset for robot color classification from labeled crops."""

from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


_IMAGENET_NORM = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Val / inference: deterministic resize (matches previous behavior).
TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(**_IMAGENET_NORM),
])

# Train only: mild geometry + photometric jitter. Hue is fixed to 0 so we do not
# turn red into orange/blue — still helps with lighting/crop variation vs overfitting.
TRANSFORM_TRAIN = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224, scale=(0.82, 1.0), ratio=(0.92, 1.08)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.0),
    transforms.ToTensor(),
    transforms.Normalize(**_IMAGENET_NORM),
])


class RobotCropDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, int]], *, augment: bool = False):
        """
        Args:
            samples: list of (image_path, class_index) tuples
            augment: if True, apply ``TRANSFORM_TRAIN`` (training only).
        """
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tf = TRANSFORM_TRAIN if self.augment else TRANSFORM
        tensor = tf(img)
        return tensor, label


def training_color_list(
    colors: List[str],
    *,
    skip: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Classi usate per il training del CNN.

    Di default esclude ``unknown``: resta nel tool di labeling ma non ha logits
    dedicati (evita dati rumorosi); in inferenza il modello può comunque
    segnalare incertezza via soglia di confidenza → ``unknown``.
    """
    if skip is None:
        skip = frozenset({"unknown"})
    sk = {str(s).strip().lower() for s in skip}
    return [c for c in colors if c.strip().lower() not in sk]


def load_splits(images_dir: Path, labels_file: Path, colors: List[str]):
    """
    Load labels.json and return train/val/test splits.

    Split strategy (insertion order, stable):
    - ~85% train, ~8% val, ~7% test — most labeled images are used for training
      (older 70/10/20 wasted too much data for small datasets).

    Returns:
        train, val, test: lists of (Path, int) tuples
        idx2color: dict mapping index to color name
    """
    color2idx = {c: i for i, c in enumerate(colors)}
    idx2color = {i: c for i, c in enumerate(colors)}

    with open(labels_file) as f:
        labels: dict = json.load(f)

    # Preserve insertion order (Python 3.7+) so test set is stable
    items = [(images_dir / fname, color2idx[color])
             for fname, color in labels.items()
             if color in color2idx and (images_dir / fname).exists()]

    if not items:
        return [], [], [], idx2color

    n = len(items)
    if n <= 2:
        return items, [], [], idx2color

    i1 = int(round(n * 0.85))
    i1 = max(1, min(i1, n - 2))
    i2 = int(round(n * 0.93))
    i2 = max(i1 + 1, min(i2, n))

    train = items[:i1]
    val = items[i1:i2]
    test = items[i2:]

    return train, val, test, idx2color
