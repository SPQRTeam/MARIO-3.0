"""Train the color classifier CNN on the full labeled dataset."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.dataset import load_splits, RobotCropDataset, training_color_list
from src.training.train import train, evaluate, save_model
from torch.utils.data import DataLoader
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  "-i", type=Path, required=True,
                        help="Directory with robot crop images")
    parser.add_argument("--colors", "-c", nargs="+", required=True,
                        help="Color labels UI (order = class index). 'unknown' è escluso dal training salvo --include-unknown.")
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help=(
            "Addestra anche la classe 'unknown': il modello impara pattern ambigui e può "
            "ridurre falsi positivi sui colori (default: unknown solo come abstain via soglia)."
        ),
    )
    parser.add_argument("--output", "-o", type=Path,
                        default=Path("data/models/color_classifier_active.pth"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable train-time augmentation (stronger fit to crops, more overfit risk).",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam L2 penalty.")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=50,
        metavar="N",
        help="Stop after N epochs without val improvement (0 = train all epochs).",
    )
    args = parser.parse_args()

    labels_file = args.input / "labels.json"
    if not labels_file.exists():
        print(f"No labels.json found in {args.input}")
        return

    skip = frozenset() if args.include_unknown else frozenset({"unknown"})
    tcolors = training_color_list(args.colors, skip=skip)
    if len(tcolors) < 2:
        print(
            "Servono almeno 2 classi per il training (dopo esclusione di 'unknown'). "
            "Usa --include-unknown se vuoi addestrare anche unknown."
        )
        return

    with open(labels_file) as f:
        raw_labels = json.load(f)
    n_skip = sum(
        1
        for v in raw_labels.values()
        if str(v).strip().lower() == "unknown" and "unknown" not in {c.lower() for c in tcolors}
    )
    if n_skip:
        print(f"Esclusi dal training: {n_skip} campioni con etichetta 'unknown'.")

    train_s, val_s, test_s, idx2color = load_splits(args.input, labels_file, tcolors)
    print(f"Classi CNN: {tcolors}")
    print(f"Train: {len(train_s)}  Val: {len(val_s)}  Test: {len(test_s)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train(
        train_s,
        val_s,
        num_classes=len(tcolors),
        idx2color=idx2color,
        save_path=args.output,
        epochs=args.epochs,
        device=device,
        augment_train=not args.no_augment,
        weight_decay=args.weight_decay,
        early_stop_patience=args.early_stop_patience,
    )

    test_loader = DataLoader(RobotCropDataset(test_s), batch_size=32, shuffle=False)
    test_acc = evaluate(model, test_loader, device)
    print(f"Test accuracy: {test_acc*100:.1f}%")
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
