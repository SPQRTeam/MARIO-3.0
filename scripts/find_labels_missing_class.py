#!/usr/bin/env python3
"""
Elenca le immagini (per stem) le cui label YOLO non contengono una classe data (es. palla = 1).

Usage:
  python scripts/find_labels_missing_class.py --labels-dir data/ds/labels --class-id 1 \\
      --images-dir data/ds/images --list-out missing_ball.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Find label files missing a YOLO class id")
    p.add_argument("--labels-dir", type=Path, required=True, help="Directory with .txt labels")
    p.add_argument("--class-id", type=int, default=1, help="Class index to require (default: 1 = ball)")
    p.add_argument("--images-dir", type=Path, default=None,
                   help="If set, print matching image paths; else print stems only")
    p.add_argument("--list-out", type=Path, default=None, help="Write one path or stem per line")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.labels_dir.is_dir():
        print(f"ERROR: not a directory: {args.labels_dir}")
        return 1

    missing = []
    for lab in sorted(args.labels_dir.glob("*.txt")):
        text = lab.read_text().strip()
        has_cls = False
        if text:
            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if int(parts[0]) == args.class_id:
                    has_cls = True
                    break
        if not has_cls:
            stem = lab.stem
            if args.images_dir:
                for ext in (".jpg", ".jpeg", ".png", ".webp"):
                    img = args.images_dir / f"{stem}{ext}"
                    if img.exists():
                        missing.append(str(img.resolve()))
                        break
                else:
                    missing.append(stem)
            else:
                missing.append(stem)

    print(f"Missing class {args.class_id}: {len(missing)} file(s)")
    for m in missing:
        print(m)

    if args.list_out:
        args.list_out.write_text("\n".join(missing) + "\n")
        print(f"Wrote {args.list_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
