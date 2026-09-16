"""Training, evaluation, and inference for the color classifier CNN."""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..ml.models.color_classifier import ColorClassifierCNNResnet
from .dataset import RobotCropDataset, TRANSFORM


def _make_loader(samples, batch_size=128, shuffle=True, augment: bool = False):
    if not samples:
        return None
    return DataLoader(
        RobotCropDataset(samples, augment=augment),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
    )


def train(
    train_samples: List[Tuple[Path, int]],
    val_samples: List[Tuple[Path, int]],
    num_classes: int,
    idx2color: Optional[dict] = None,
    save_path: Optional[Path] = None,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    augment_train: bool = True,
    early_stop_patience: int = 12,
    device: Optional[torch.device] = None,
) -> ColorClassifierCNNResnet:
    """Train the CNN and return the best model (by val accuracy).

    Uses train-time augmentation, Adam weight decay, and optional early stopping
    on val accuracy to reduce overfit to small labeled sets.

    If save_path and idx2color are provided, the best model is saved to disk
    each time val accuracy improves.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ColorClassifierCNNResnet(in_channels=3, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_loader = _make_loader(train_samples, augment=augment_train)
    val_loader = _make_loader(val_samples, shuffle=False, augment=False)

    best_val_acc = -1.0
    best_state = None
    stagnant = 0
    stop_after = None if early_stop_patience <= 0 else early_stop_patience

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc = evaluate(model, val_loader, device) if val_loader else 0.0
        marker = " *" if val_acc > best_val_acc else ""
        es_note = ""
        if stop_after is not None and stagnant > 0:
            es_note = f"  (no val improv. {stagnant}/{stop_after})"
        print(
            f"  epoch {epoch+1:3d}/{epochs}  loss={total_loss/len(train_loader):.4f}  "
            f"val={val_acc*100:.1f}%{marker}{es_note}",
            flush=True,
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            stagnant = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if save_path is not None and idx2color is not None:
                save_model(model, save_path, idx2color)
        else:
            stagnant += 1
            if stop_after is not None and stagnant >= stop_after:
                print(f"  early stop: val accuracy plateaued for {stop_after} epochs.", flush=True)
                break

    if best_state is None:
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def evaluate(model: ColorClassifierCNNResnet, loader: DataLoader, device: torch.device) -> float:
    """Return accuracy (0-1) on the given loader."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / total if total > 0 else 0.0


UNKNOWN_CONFIDENCE_THRESHOLD = 0.80


def resolve_hint_color(
    idx_pred: int,
    conf: float,
    idx2color: dict,
    ui_colors: List[str],
    *,
    unknown_threshold: float = UNKNOWN_CONFIDENCE_THRESHOLD,
) -> str:
    """
    Mappa (classe argmax, confidenza) al nome colore per la UI.

    Se ``unknown`` **non** è stato addestrato come classe (assente da ``idx2color``),
    con confidenza **sotto** la soglia (default 80 %) si restituisce ``unknown``.

    Se ``unknown`` **è** una classe del checkpoint, si usa solo l'argmax (il modello
    può imparare a classificare l'ambiguità e ridurre falsi positivi sui colori).
    """
    unk = next((c for c in ui_colors if c.strip().lower() == "unknown"), None)
    trained_unknown = unk is not None and any(
        str(v).strip().lower() == "unknown" for v in idx2color.values()
    )
    if trained_unknown:
        return idx2color[int(idx_pred)]
    if unk is not None and conf < float(unknown_threshold):
        return unk
    return idx2color[int(idx_pred)]


def predict(model: ColorClassifierCNNResnet, img: "Path | np.ndarray", device: torch.device) -> Tuple[int, float]:
    """Return (class_index, confidence) for a single image (path or BGR ndarray)."""
    import cv2
    import numpy as np
    if isinstance(img, np.ndarray):
        arr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        arr = cv2.cvtColor(cv2.imread(str(img)), cv2.COLOR_BGR2RGB)
    tensor = TRANSFORM(arr).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, idx = probs.max(dim=1)
    return idx.item(), conf.item()


def predict_max_confidence_batch(
    paths: List[Path],
    model: ColorClassifierCNNResnet,
    device: torch.device,
    batch_size: int = 128,
    show_progress: bool = False,
) -> List[float]:
    """
    Max softmax confidence for each image path, same order as ``paths``.
    Batched inference (much faster than calling ``predict`` per file for large folders).
    """
    if not paths:
        return []
    n = len(paths)
    samples = [(p, 0) for p in paths]
    loader = DataLoader(
        RobotCropDataset(samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    out: List[float] = []
    model.eval()
    seen = 0
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device, non_blocking=device.type == "cuda")
            probs = torch.softmax(model(imgs), dim=1)
            conf, _ = probs.max(dim=1)
            out.extend(float(c) for c in conf.cpu().tolist())
            seen = len(out)
            if show_progress:
                print(f"\r  scoring for sort order: {seen}/{n} ", end="", flush=True)
    if show_progress:
        print()
    return out


def predict_class_indices_batch(
    paths: List[Path],
    model: ColorClassifierCNNResnet,
    device: torch.device,
    batch_size: int = 128,
    show_progress: bool = False,
) -> List[int]:
    """
    Argmax class index per image path, same order as ``paths`` (batched).
    """
    if not paths:
        return []
    n = len(paths)
    samples = [(p, 0) for p in paths]
    loader = DataLoader(
        RobotCropDataset(samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    out: List[int] = []
    model.eval()
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device, non_blocking=device.type == "cuda")
            logits = model(imgs)
            pred = logits.argmax(dim=1)
            out.extend(int(x) for x in pred.cpu().tolist())
            if show_progress:
                print(f"\r  auto-label: {len(out)}/{n} ", end="", flush=True)
    if show_progress:
        print()
    return out


def save_model(model: ColorClassifierCNNResnet, path: Path, idx2color: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "idx2color": idx2color}, path)


def load_model(
    path: Path,
    device: torch.device,
    num_classes: Optional[int] = None,
) -> Tuple[ColorClassifierCNNResnet, dict]:
    """
    Load a saved color classifier. Class count comes from the checkpoint unless
    ``num_classes`` is passed and must match (useful for sanity-checking scripts).
    """
    checkpoint = torch.load(path, map_location=device)
    idx2color = {int(k): v for k, v in checkpoint["idx2color"].items()}
    nc = len(idx2color)
    if num_classes is not None and num_classes != nc:
        raise ValueError(
            f"Checkpoint has {nc} classes {list(idx2color.values())}, "
            f"expected num_classes={num_classes}."
        )
    model = ColorClassifierCNNResnet(in_channels=3, num_classes=nc).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, idx2color
