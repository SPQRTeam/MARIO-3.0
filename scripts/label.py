"""
Label robot jersey crops and train the color CNN from the keyboard.

Workflow
    - You see each unlabeled crop; pick a color with the mouse **or** the first
      letter of the color name (``r``=red, …). **Enter** / **Space** confirms the
      hint bar when it is shown.
    - **t** — train the model on all labels in ``labels.json`` (same split logic as
      ``train_color.py``). Do this whenever you want an updated checkpoint.
    - **l** — auto-label su tutte le immagini della cartella tranne quelle con
      etichetta **umana** (``human_label_keys.json``), poi train sul merged split.
    - **s** — re-order **remaining** images (from the current one onward) by
      **ascending confidence** (hardest first) so you can fix bad predictions;
      press **t** again to retrain on the corrected dataset.
    - **n** — next image **without** saving a label (skip).
    - **z** / left arrow — go back one image.
    - **q** — quit (saves ``labels.json``).

Only images **not** already in ``labels.json`` are shown. By default **no checkpoint
is loaded**: you start in full manual mode (no hint bar) even if
``color_classifier_active.pth`` exists; press **t** after some labels to train,
then hints work in the same session. Pass ``--hints`` to load an existing model
for the confidence bar from the first screen. Queue order: file name by default;
``--uncertain-first`` needs a model (loads checkpoint once for sorting). ``--shuffle`` = random.

Offline batch training::

    python scripts/train_color.py -i <robot_pics_dir> -c <same classes as here>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

# Make sure the repo root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.dataset import load_splits, RobotCropDataset, training_color_list
from src.training.train import (
    train,
    evaluate,
    predict,
    predict_max_confidence_batch,
    predict_class_indices_batch,
    resolve_hint_color,
    save_model,
    load_model,
)
from src.utils.drawing import TEAM_COLORS
from torch.utils.data import DataLoader

# ── UI constants ──────────────────────────────────────────────────────────────
WINDOW      = "Label Tool"
IMG_SIZE    = 300          # square display size for the crop
BTN_H       = 60
BAR_H       = 28           # hint confidence bar height
MARGIN      = 8
FONT        = cv2.FONT_HERSHEY_SIMPLEX

DEFAULT_AUTO_RETRAIN_EVERY = 0  # 0 = no periodic retrain; use key ``t`` or --auto-retrain-every
TRAIN_EPOCHS = 10  # CNN epochs when pressing t / l in label.py or label_server
AUTOLABEL_THRESHOLD = 0.90  # test accuracy needed to offer auto-labeling
MODEL_PATH          = Path("data/models/color_classifier_active.pth")

# ── color → BGR for button rendering ─────────────────────────────────────────
def color_bgr(name: str) -> tuple:
    return TEAM_COLORS.get(name.lower(), (100, 100, 100))

def text_color(bgr: tuple) -> tuple:
    return (0, 0, 0) if sum(bgr) > 360 else (255, 255, 255)


def label_disagrees(hint: str, saved: str) -> bool:
    """True se classe modello (hint) e etichetta salvata differiscono (case-insensitive)."""
    return (hint or "").strip().lower() != (saved or "").strip().lower()


# ── persistence ───────────────────────────────────────────────────────────────
def load_labels(output_dir: Path) -> dict:
    p = output_dir / "labels.json"
    return json.load(open(p)) if p.exists() else {}

def save_labels(output_dir: Path, labels: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "labels.json", "w") as f:
        json.dump(labels, f, indent=2)


HUMAN_KEYS_NAME = "human_label_keys.json"


def load_human_keys(output_dir: Path, labels: dict) -> set:
    """Nomi file etichettati da umano (non sovrascrivibili da autolabel replace_all).

    Se ``human_label_keys.json`` non esiste ma ``labels`` ha voci, si inizializza
    con tutte le chiavi presenti (migrazione: il lavoro già salvato si considera GT).
    """
    p = output_dir / HUMAN_KEYS_NAME
    if p.exists():
        data = json.load(open(p))
        return set(data) if isinstance(data, list) else set(data)
    if labels:
        s = set(labels.keys())
        save_human_keys(output_dir, s)
        return s
    return set()


def save_human_keys(output_dir: Path, keys: set) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / HUMAN_KEYS_NAME, "w") as f:
        json.dump(sorted(keys), f, indent=2)


def mark_human(output_dir: Path, labels: dict, fname: str) -> None:
    """Segna un'etichetta come confermata da umano (click, scorciatoia o hint+Enter)."""
    keys = load_human_keys(output_dir, labels)
    keys.add(fname)
    save_human_keys(output_dir, keys)


# ── training helpers ──────────────────────────────────────────────────────────
def _merge_splits_almost_all(train_s, val_s, test_s, val_frac: float = 0.06):
    """
    Combine train/val/test and reserve a small tail for validation only
    (early stopping). Almost every labeled image is used for training.
    """
    all_s = list(train_s) + list(val_s) + list(test_s)
    n = len(all_s)
    if n < 4:
        return all_s, [], []
    nv = max(1, round(n * val_frac))
    nv = min(nv, n - 2)
    return all_s[:-nv], all_s[-nv:], []


def do_retrain(
    labels: dict,
    images_dir: Path,
    colors: list,
    model_state: dict,
    device: torch.device,
    *,
    merged: bool = False,
    include_unknown: bool = False,
):
    """
    Run one training pass from current labels.json under images_dir.
    ``merged=True`` (after **l**) trains on nearly all labeled pixels; ``False`` (**t**)
    uses the normal 85/8/7 split from ``load_splits``.

    ``include_unknown``: se True, ``unknown`` è una classe CNN (utile per imparare
    a sopprimere falsi positivi sui colori); se False (default), ``unknown`` è
    solo UI / abstain via soglia di confidenza.
    Returns (updated model_state or same, test_acc or None).
    """
    skip = frozenset() if include_unknown else None
    tcolors = training_color_list(colors, skip=skip)
    if len(tcolors) < 2:
        print(
            "[retrain] servono almeno 2 classi di training (es. red, blue). "
            "Senza --train-include-unknown, 'unknown' non conta come classe."
        )
        return model_state, None

    n_unknown = sum(1 for v in labels.values() if str(v).strip().lower() == "unknown")
    if include_unknown and n_unknown:
        print(f"[retrain] inclusi {n_unknown} campioni con etichetta 'unknown' (classe CNN).")
    elif not include_unknown and n_unknown:
        print(f"[retrain] esclusi dal training: {n_unknown} campioni con etichetta 'unknown'.")

    train_s, val_s, test_s, idx2color = load_splits(
        images_dir, images_dir / "labels.json", tcolors
    )

    if merged:
        train_s, val_s, test_s = _merge_splits_almost_all(train_s, val_s, test_s)

    if len(train_s) < 2:
        print("[retrain] not enough labeled samples in train split (need ≥2), skipping.")
        return model_state, None

    num_classes = len(tcolors)
    model = train(
        train_s,
        val_s,
        num_classes=num_classes,
        idx2color=idx2color,
        save_path=MODEL_PATH,
        epochs=TRAIN_EPOCHS,
        device=device,
        augment_train=True,
        weight_decay=1e-4,
        early_stop_patience=8,
    )
    test_loader = DataLoader(RobotCropDataset(test_s), batch_size=32, shuffle=False) if test_s else None
    test_acc = evaluate(model, test_loader, device) if test_loader else 0.0
    tag = " [merged train]" if merged else ""
    print(f"[retrain] test accuracy: {test_acc*100:.1f}%{tag}")

    save_model(model, MODEL_PATH, idx2color)
    return {"model": model, "idx2color": idx2color}, test_acc


def maybe_auto_retrain(
    labels: dict,
    images_dir: Path,
    colors: list,
    model_state: dict,
    device: torch.device,
    auto_retrain_every: int,
    *,
    include_unknown: bool = False,
):
    """If auto_retrain_every > 0 and len(labels) is a multiple, run do_retrain."""
    n = len(labels)
    if auto_retrain_every <= 0 or n == 0 or n % auto_retrain_every != 0:
        return model_state, None

    print(f"\n[active learning] {n} labels reached — auto retraining…")
    return do_retrain(
        labels, images_dir, colors, model_state, device, include_unknown=include_unknown
    )


def load_or_none(colors, device):
    if MODEL_PATH.exists():
        try:
            model, idx2color = load_model(MODEL_PATH, device)
            return {"model": model, "idx2color": idx2color}
        except Exception:
            pass
    return None


def build_unlabeled_queue(
    images: list,
    labels: dict,
    model_state,
    device: torch.device,
    *,
    shuffle: bool,
    uncertain_first: bool,
):
    """
    Filter to unlabeled paths and apply ordering (CLI / web shared logic).
    Returns a new list of Path objects.
    """
    import random

    unlabeled = [img for img in images if img.name not in labels]
    if uncertain_first:
        if model_state is None or not unlabeled:
            pass  # keep sorted-by-glob order
        else:
            print(
                f"Ordine per confidenza (solo ordinamento): {len(unlabeled)} immagini…",
                flush=True,
            )
            try:
                confs = predict_max_confidence_batch(
                    unlabeled, model_state["model"], device, show_progress=True
                )
                scored = list(zip(confs, unlabeled))
            except Exception:
                scored = []
                for p in unlabeled:
                    try:
                        _, conf = predict(model_state["model"], p, device)
                        scored.append((conf, p))
                    except Exception:
                        scored.append((0.0, p))
            scored.sort(key=lambda x: x[0])
            unlabeled = [p for _, p in scored]
    elif shuffle:
        unlabeled = list(unlabeled)
        random.shuffle(unlabeled)
    return unlabeled


# ── canvas rendering ──────────────────────────────────────────────────────────
def render(crop, colors, img_path, idx, total, n_labeled,
           hint_color=None, hint_conf=None, test_acc=None, saved_label=None):
    n       = len(colors)
    btn_w   = IMG_SIZE // n
    extra   = BAR_H + MARGIN if hint_color else 0
    status_h = 22
    review_h = 18 if (saved_label is not None and hint_color is not None) else 0
    canvas_h = IMG_SIZE + status_h + review_h + MARGIN + extra + MARGIN + BTN_H + MARGIN

    canvas = np.zeros((canvas_h, IMG_SIZE, 3), dtype=np.uint8)

    # crop
    canvas[:IMG_SIZE] = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

    # status line
    acc_str = f"  test:{test_acc*100:.0f}%" if test_acc is not None else ""
    status  = f"{idx+1}/{total}  labeled:{n_labeled}{acc_str}  {img_path.name}"
    cv2.putText(canvas, status, (4, IMG_SIZE + 16), FONT, 0.38, (180, 180, 180), 1)

    y0 = IMG_SIZE + status_h
    if saved_label is not None and hint_color is not None:
        mismatch = label_disagrees(hint_color, saved_label)
        line2 = (
            f"salvato: {saved_label}  |  hint: {hint_color} {hint_conf*100:.0f}%"
            + ("  [MISMATCH]" if mismatch else "  [ok]")
        )
        col = (60, 120, 255) if mismatch else (120, 200, 120)
        cv2.putText(canvas, line2, (4, y0 + 2), FONT, 0.34, col, 1)
        y0 += review_h

    y = y0 + MARGIN

    # hint bar
    if hint_color is not None:
        bgr  = color_bgr(hint_color)
        bar_w = int(IMG_SIZE * hint_conf)
        cv2.rectangle(canvas, (0, y), (bar_w, y + BAR_H), bgr, -1)
        cv2.rectangle(canvas, (0, y), (IMG_SIZE, y + BAR_H), (200, 200, 200), 1)
        label = f"hint: {hint_color}  {hint_conf*100:.0f}%  (Enter=confirm)"
        cv2.putText(canvas, label, (6, y + BAR_H - 8), FONT, 0.42, text_color(bgr), 1)
        y += BAR_H + MARGIN

    # color buttons
    btn_y = y
    for i, cname in enumerate(colors):
        x1 = i * btn_w + 2
        x2 = (i + 1) * btn_w - 2
        bgr = color_bgr(cname)
        cv2.rectangle(canvas, (x1, btn_y), (x2, btn_y + BTN_H), bgr, -1)
        cv2.rectangle(canvas, (x1, btn_y), (x2, btn_y + BTN_H), (255, 255, 255), 1)
        label = f"[{cname[0]}] {cname}"
        tx = x1 + max(0, (btn_w - len(label) * 9) // 2)
        cv2.putText(canvas, label, (tx, btn_y + BTN_H // 2 + 6), FONT, 0.5, text_color(bgr), 1)

    return canvas, btn_y, btn_w, BTN_H


# ── uncertain review ──────────────────────────────────────────────────────────
def _review_uncertain(images, output_dir, colors, device, top_n):
    """Immagini già etichettate dove hint (modello) ≠ etichetta salvata — tipici errori autolabel."""
    model_state = load_or_none(colors, device)
    if model_state is None:
        print("No model found. Run labeling first to train a model.")
        return

    labels = load_labels(output_dir)
    if not labels:
        print("No labels found.")
        return

    model = model_state["model"]
    i2c = model_state["idx2color"]
    labeled_paths = [p for p in images if p.name in labels]
    if not labeled_paths:
        print("No labeled images in this folder.")
        return

    print(f"Confronto hint vs etichetta su {len(labeled_paths)} immagini…")
    try:
        idxs = predict_class_indices_batch(
            labeled_paths, model, device, show_progress=True
        )
        confs = predict_max_confidence_batch(
            labeled_paths, model, device, show_progress=False
        )
    except Exception as e:
        print(f"Batch fallito ({e}), fallback per-file…")
        idxs, confs = [], []
        for p in labeled_paths:
            try:
                ix, cf = predict(model, p, device)
                idxs.append(ix)
                confs.append(cf)
            except Exception:
                idxs.append(-1)
                confs.append(0.0)

    mismatches: List[Tuple[float, Path]] = []
    for p, ix, conf in zip(labeled_paths, idxs, confs):
        if ix < 0:
            continue
        hint = resolve_hint_color(ix, conf, i2c, colors)
        if label_disagrees(hint, labels[p.name]):
            mismatches.append((conf, p))

    mismatches.sort(key=lambda x: x[0])
    uncertain = [p for _, p in mismatches[:top_n]]
    if not uncertain:
        print("Nessun mismatch: tutte le etichette coincidono con il modello (sui file controllati).")
        return

    print(
        f"Revisione: {len(uncertain)} immagini con hint ≠ etichetta "
        f"(confidenza modello crescente tra i mismatch)."
    )

    _label_loop(
        uncertain,
        labels,
        output_dir,
        colors,
        device,
        model_state,
        test_acc=None,
        autolabel_threshold=1.1,
        review_labeled=True,
        review_show_saved=True,
    )


def _label_loop(
    images,
    labels,
    output_dir,
    colors,
    device,
    model_state,
    test_acc,
    autolabel_threshold,
    auto_retrain_every: int = 0,
    *,
    review_labeled: bool = False,
    review_show_saved: bool = False,
    train_include_unknown: bool = False,
):
    """Core labeling loop, extracted so it can be reused by review mode.

    ``review_labeled``: la coda contiene immagini già in ``labels`` (revisione).
    ``review_show_saved``: mostra confronto etichetta salvata vs hint nel canvas.
    ``train_include_unknown``: passa a ``do_retrain`` / auto-retrain (vedi CLI).
    """
    _ = autolabel_threshold  # reserved for CLI compatibility
    clicked = [None]
    images_dir = images[0].parent if images else output_dir
    i = 0  # before nested defs so ``nonlocal i`` is always initialized

    def _note_acc(new_acc):
        nonlocal test_acc
        if new_acc is not None:
            test_acc = new_acc

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        btn_y, btn_w, btn_h, n = param
        if btn_y <= y <= btn_y + btn_h:
            col_idx = x // btn_w
            if 0 <= col_idx < n:
                clicked[0] = colors[col_idx]

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    while 0 <= i < len(images):
        if not review_labeled:
            while i < len(images) and images[i].name in labels:
                i += 1
        if i >= len(images):
            break
        img_path = images[i]
        crop = cv2.imread(str(img_path))
        if crop is None:
            i += 1
            continue

        hint_color = hint_conf = None
        if model_state is not None:
            try:
                idx_pred, conf = predict(model_state["model"], img_path, device)
                hint_color = resolve_hint_color(
                    idx_pred, conf, model_state["idx2color"], colors
                )
                hint_conf = conf
            except Exception:
                pass

        canvas, btn_y, btn_w, btn_h = render(
            crop, colors, img_path, i, len(images), len(labels),
            hint_color,
            hint_conf,
            test_acc,
            saved_label=(labels.get(img_path.name) if review_show_saved else None),
        )
        cv2.setMouseCallback(WINDOW, on_mouse, (btn_y, btn_w, btn_h, len(colors)))
        cv2.imshow(WINDOW, canvas)

        clicked[0] = None
        chosen = None
        while chosen is None:
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                save_labels(output_dir, labels)
                print(f"Quit. {len(labels)} labels saved.")
                cv2.destroyAllWindows()
                return
            elif key == 81 or key == ord('z'):  # left arrow or z = back
                i = max(0, i - 1)
                break
            elif key == ord("n"):  # next without saving
                i += 1
                break
            elif key == ord("s"):  # sort remaining by ascending confidence
                remaining = images[i:]
                if not remaining:
                    break
                if model_state is None:
                    print("[label.py] Ordinamento: serve un modello (premi t dopo alcune etichette).")
                    break
                try:
                    confs = predict_max_confidence_batch(
                        remaining, model_state["model"], device, show_progress=True
                    )
                    scored = list(zip(confs, remaining))
                except Exception:
                    scored = []
                    for p in remaining:
                        try:
                            _, conf = predict(model_state["model"], p, device)
                            scored.append((conf, p))
                        except Exception:
                            scored.append((0.0, p))
                scored.sort(key=lambda x: x[0])
                images[:] = images[:i] + [p for _, p in scored]
                print(f"[label.py] Rimanenti ordinati per confidenza ↑ ({len(remaining)} immagini).")
                break
            elif key == ord("l"):  # rilabel tutta la cartella + training su tutte le etichette
                if model_state is None:
                    print("[label.py] Auto-label: serve un modello (premi t dopo alcune etichette).")
                    break
                _autolabel(
                    images,
                    labels,
                    model_state,
                    output_dir,
                    colors,
                    device,
                    replace_all=True,
                    images_dir=images_dir,
                )
                model_state, new_acc = do_retrain(
                    labels,
                    images_dir,
                    colors,
                    model_state,
                    device,
                    merged=True,
                    include_unknown=train_include_unknown,
                )
                _note_acc(new_acc)
                print("[label.py] Auto-label completato e modello addestrato su tutte le etichette.")
                break
            elif key in (13, 32) and hint_color is not None:  # Enter/Space = confirm hint
                chosen = hint_color
            elif key == ord("t"):
                model_state, new_acc = do_retrain(
                    labels,
                    images_dir,
                    colors,
                    model_state,
                    device,
                    include_unknown=train_include_unknown,
                )
                _note_acc(new_acc)
                break  # redraw same image with updated model hints
            else:
                # first-char shortcut: e.g. 'r' → red, 'b' → blue, 'y' → yellow
                for color_name in colors:
                    if key == ord(color_name[0]):
                        chosen = color_name
                        break
            if clicked[0] is not None:
                chosen = clicked[0]
                clicked[0] = None

        if chosen is not None:
            labels[img_path.name] = chosen
            save_labels(output_dir, labels)
            mark_human(output_dir, labels, img_path.name)
            model_state, new_acc = maybe_auto_retrain(
                labels,
                images_dir,
                colors,
                model_state,
                device,
                auto_retrain_every,
                include_unknown=train_include_unknown,
            )
            _note_acc(new_acc)
            i += 1

    cv2.destroyAllWindows()
    print(f"Done. {len(labels)} labels saved.")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  "-i", type=Path, required=True,
                        help="Directory with robot crop images")
    parser.add_argument("--colors", "-c", nargs="+", required=True,
                        help="Color labels, e.g. --colors red blue yellow")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output dir for labels.json (default: same as --input)")
    parser.add_argument("--autolabel-threshold", type=float, default=AUTOLABEL_THRESHOLD)
    parser.add_argument(
        "--review-uncertain",
        action="store_true",
        help="Solo immagini dove hint del modello ≠ etichetta salvata (errori tipici autolabel), conf ↑",
    )
    parser.add_argument("--review-top", type=int, default=500,
                        help="How many uncertain images to review (default: 500)")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomize unlabeled order (instant).",
    )
    parser.add_argument(
        "--hints",
        action="store_true",
        help="Load data/models/color_classifier_active.pth at startup for the hint bar (optional). Default: no model until you press t.",
    )
    parser.add_argument(
        "--uncertain-first",
        action="store_true",
        help="Score every unlabeled image with the CNN and show lowest confidence first (loads checkpoint; slow).",
    )
    parser.add_argument(
        "--auto-retrain-every",
        type=int,
        default=DEFAULT_AUTO_RETRAIN_EVERY,
        metavar="N",
        help="Retrain automatically every N new labels saved (0 = off; use t in the GUI for manual retrain).",
    )
    parser.add_argument(
        "--train-include-unknown",
        action="store_true",
        help="Addestra anche la classe 'unknown' (campioni ambigui → meno falsi positivi sui colori; default: unknown solo UI + soglia conf.).",
    )
    args = parser.parse_args()

    output_dir = args.output or args.input
    colors     = args.colors
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images = sorted(args.input.glob("*.png")) + sorted(args.input.glob("*.jpg"))
    if not images:
        print(f"No images found in {args.input}")
        return

    if args.review_uncertain:
        _review_uncertain(images, output_dir, colors, device, args.review_top)
        return

    labels = load_labels(output_dir)
    need_ckpt = args.hints or args.uncertain_first
    model_state = load_or_none(colors, device) if need_ckpt else None

    print(f"Images: {len(images)}  Already labeled: {len(labels)}  Colors: {colors}")
    if args.train_include_unknown:
        print(
            "Modalità training: 'unknown' è classe CNN (come train_color.py --include-unknown)."
        )
    if not need_ckpt:
        print(
            "Avvio senza modello: solo etichette manuali; dopo t le hint funzionano nella sessione. "
            "Usa --hints per caricare subito un .pth."
        )
    print(
        "t=retrain | l=rilabel cartella + train | s=sort rest by conf | n=skip | "
        "z=back | Enter/Space=hint | q=quit"
        + (f" | auto-retrain every {args.auto_retrain_every} labels" if args.auto_retrain_every > 0 else "")
    )

    unlabeled = build_unlabeled_queue(
        images,
        labels,
        model_state,
        device,
        shuffle=args.shuffle,
        uncertain_first=args.uncertain_first,
    )
    if args.uncertain_first and (model_state is None or not unlabeled):
        print(
            "[label.py] --uncertain-first: nessun modello o nessuna immagine; "
            "ordine per nome file."
        )
    elif args.uncertain_first and model_state is not None:
        print(
            "Ordine: confidenza crescente (più incerte prima). "
            "Senza flag = ordine per nome file."
        )
    elif args.shuffle:
        print(f"Ordine casuale ({len(unlabeled)} immagini).")
    else:
        print(
            f"Ordine per nome file ({len(unlabeled)} immagini). "
            f"Opzioni: --uncertain-first (lento) | --shuffle"
        )

    _label_loop(
        unlabeled,
        labels,
        output_dir,
        colors,
        device,
        model_state,
        test_acc=None,
        autolabel_threshold=args.autolabel_threshold,
        auto_retrain_every=max(0, args.auto_retrain_every),
        train_include_unknown=args.train_include_unknown,
    )


def _autolabel(
    images,
    labels,
    model_state,
    output_dir,
    colors,
    device,
    *,
    replace_all: bool = False,
    images_dir: Optional[Path] = None,
):
    """Auto-label with the current model (batched + progress).

    If ``replace_all`` is False, only images in ``images`` that are missing from
    ``labels`` are predicted. If True, every ``*.png`` / ``*.jpg`` under
    ``images_dir`` is considered, ma le voci in ``human_label_keys.json`` (etichette
    manuali) non vengono sovrascritte.
    """
    model = model_state["model"]
    i2c = model_state["idx2color"]
    human_keys = load_human_keys(output_dir, labels)
    if replace_all:
        root = images_dir if images_dir is not None else (
            images[0].parent if images else None
        )
        if root is None or not root.is_dir():
            print("[auto-label] replace_all: directory immagini non valida.")
            return
        all_paths = sorted(root.glob("*.png")) + sorted(root.glob("*.jpg"))
        if not all_paths:
            print(f"[auto-label] nessuna immagine in {root}.")
            return
        to_label = [p for p in all_paths if p.name not in human_keys]
        n_skip = len(all_paths) - len(to_label)
        if n_skip:
            print(
                f"[auto-label] salto {n_skip} immagini con etichetta umana (GT); "
                f"predizione su {len(to_label)}.",
                flush=True,
            )
        if not to_label:
            print("[auto-label] niente da predire (tutto etichettato a mano o vuoto).")
            return
        print(f"[auto-label] rilabello {len(to_label)} immagini in {root}…", flush=True)
    else:
        to_label = [p for p in images if p.name not in labels]
        if not to_label:
            print("[auto-label] nessuna immagine senza etichetta nella coda.")
            return
        print(f"[auto-label] {len(to_label)} immagini…", flush=True)
    try:
        indices = predict_class_indices_batch(
            to_label, model, device, show_progress=True
        )
        confs = predict_max_confidence_batch(
            to_label, model, device, show_progress=False
        )
    except Exception as e:
        print(f"[auto-label] batch fallito ({e}), fallback per-file…", flush=True)
        indices, confs = [], []
        for p in to_label:
            try:
                idx_pred, cf = predict(model, p, device)
                indices.append(idx_pred)
                confs.append(cf)
            except Exception:
                indices.append(None)
                confs.append(0.0)
    count = 0
    for p, ix, cf in zip(to_label, indices, confs):
        if ix is None:
            continue
        labels[p.name] = resolve_hint_color(ix, float(cf), i2c, colors)
        count += 1
    save_labels(output_dir, labels)
    print(f"[auto-label] completato: {count} immagini. Totale etichette: {len(labels)}", flush=True)


if __name__ == "__main__":
    main()
