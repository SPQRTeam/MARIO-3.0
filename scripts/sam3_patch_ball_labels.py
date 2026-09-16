#!/usr/bin/env python3
"""
Per immagini dove manca la palla nelle label: aggiunge solo la classe palla con SAM3 (testo),
unendo alle righe esistenti (robot, ecc.) senza duplicare la classe ball.

Usage:
  python scripts/sam3_patch_ball_labels.py \\
    --images-dir frames_Final_FO \\
    --labels-dir mydataset/labels \\
    --ball-class 1 \\
    --ball-prompt "soccer ball"

  # Solo stem elencati:
  python scripts/sam3_patch_ball_labels.py --from-list missing_ball.txt ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import collect_dataset_sam3 as c3


def parse_args():
    p = argparse.ArgumentParser(description="SAM3: append ball-only YOLO lines where missing")
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--labels-dir", type=Path, required=True)
    p.add_argument("--ball-class", type=int, default=1)
    p.add_argument("--ball-prompt", type=str, default="soccer ball",
                   help="SAM3 text prompt for the ball")
    p.add_argument("--from-list", type=Path, default=None,
                   help="Optional file: one image path or stem per line (else scan all labels missing ball)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--threshold", type=float, default=0.35,
                   help="SAM3 instance threshold (ball is small; try 0.25–0.4)")
    p.add_argument("--max-balls", type=int, default=1,
                   help="Keep only this many ball boxes, highest SAM3 score first (default 1)")
    p.add_argument("--dry-run", action="store_true", help="Print only, do not write")
    return p.parse_args()


def _read_non_ball_lines(path: Path, ball_class: int) -> list[str]:
    if not path.exists():
        return []
    lines = []
    for line in path.read_text().strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if int(line.split()[0]) == ball_class:
            continue
        lines.append(line)
    return lines


def _dets_to_ball_lines(detections: list, ball_class: int, max_balls: int = 1) -> list[str]:
    """SAM3 can return multiple ball masks; keep top ``max_balls`` by score."""
    candidates: list[tuple[float, str]] = []
    for cls_id, xc, yc, bw, bh, score, _mask in detections:
        if cls_id != ball_class:
            continue
        sc = float(score) if score is not None else 0.0
        line = f"{ball_class} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
        candidates.append((sc, line))
    candidates.sort(key=lambda t: -t[0])
    return [c[1] for c in candidates[: max(0, max_balls)]]


def main() -> int:
    args = parse_args()
    token = c3.load_hf_token()
    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    model, processor = c3.load_sam3(token, device)

    ball_prompts = {args.ball_prompt: args.ball_class}

    image_paths: list[Path] = []

    if args.from_list:
        for line in args.from_list.read_text().strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            p = Path(line).expanduser()
            if p.is_file():
                image_paths.append(p.resolve())
                continue
            stem = p.stem if p.suffix else line
            found = None
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                cand = args.images_dir / f"{stem}{ext}"
                if cand.is_file():
                    found = cand.resolve()
                    break
            if found:
                image_paths.append(found)
            else:
                print(f"  SKIP (not found): {line}")
    else:
        for lab in sorted(args.labels_dir.glob("*.txt")):
            text = lab.read_text().strip()
            has_ball = False
            if text:
                for ln in text.split("\n"):
                    ln = ln.strip()
                    if not ln:
                        continue
                    if int(ln.split()[0]) == args.ball_class:
                        has_ball = True
                        break
            if not has_ball:
                stem = lab.stem
                found = None
                for ext in (".jpg", ".jpeg", ".png", ".webp"):
                    cand = args.images_dir / f"{stem}{ext}"
                    if cand.is_file():
                        found = cand.resolve()
                        break
                if found:
                    image_paths.append(found)

    if not image_paths:
        print("Nothing to patch (all labels already have ball class or no candidates).")
        return 0

    print(f"Patching {len(image_paths)} image(s)… (max {args.max_balls} ball / frame by score)")

    for img_path in image_paths:
        stem = img_path.stem
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  SKIP {img_path.name}: cannot read image")
            continue
        detections = c3.detect_frame(
            frame, model, processor, device, ball_prompts, threshold=args.threshold
        )
        ball_lines = _dets_to_ball_lines(
            detections, args.ball_class, max_balls=args.max_balls
        )
        label_path = args.labels_dir / f"{stem}.txt"
        keep = _read_non_ball_lines(label_path, args.ball_class)
        merged = keep + ball_lines
        body = "\n".join(merged) + ("\n" if merged else "")

        if args.dry_run:
            print(f"  [dry-run] {stem}: +{len(ball_lines)} ball line(s)")
            continue

        label_path.write_text(body)
        print(f"  {stem}: wrote {len(keep)} kept + {len(ball_lines)} ball → {label_path.name}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
