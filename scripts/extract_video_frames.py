#!/usr/bin/env python3
"""
Estrae frame da un video con OpenCV (solo stream video → niente errori Opus).

Esempi:
  # Un frame ogni 5 secondi (come ffmpeg fps=1/5)
  python scripts/extract_video_frames.py -i video.mp4 -o frames/ --every-seconds 5

  # Un frame ogni 30 frame
  python scripts/extract_video_frames.py -i video.mp4 -o frames/ --every-n-frames 30

  # Con filtro scene (stesso criterio di collect_dataset_sam3)
  python scripts/extract_video_frames.py -i video.mp4 -o frames/ --every-seconds 1 --scene-diff 0.018
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import collect_dataset_sam3 as sam3
except ImportError:
    sam3 = None


def parse_args():
    p = argparse.ArgumentParser(description="Extract JPEG frames from video (OpenCV, video only)")
    p.add_argument("--input", "-i", type=Path, required=True, help="Video file")
    p.add_argument("--output", "-o", type=Path, required=True, help="Output directory for JPEGs")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--every-n-frames",
        type=int,
        metavar="N",
        help="Save 1 frame every N frames",
    )
    g.add_argument(
        "--every-seconds",
        type=float,
        metavar="S",
        help="Save ~1 frame every S seconds (uses video FPS)",
    )
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=None, help="Stop after saving this many images")
    p.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality 0-100 (default 95)")
    p.add_argument("--scene-diff", type=float, default=None,
                   help="Skip frame if too similar to last saved (requires scripts/collect_dataset_sam3.py)")
    p.add_argument("--force-every-frames", type=int, default=None,
                   help="With --scene-diff: force save at least every N frames")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        print(f"ERROR: not found: {args.input}")
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        print(f"ERROR: cannot open video: {args.input}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.every_n_frames is not None:
        stride = max(1, args.every_n_frames)
    else:
        stride = max(1, int(round(fps * args.every_seconds)))

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    scene_diff = args.scene_diff
    force_every = args.force_every_frames or 0
    if scene_diff is not None and sam3 is None:
        print("ERROR: --scene-diff needs collect_dataset_sam3.py in scripts/")
        return 1

    last_gray = None
    last_saved_idx = -10**9
    skipped = 0
    saved = 0
    frame_idx = args.start_frame

    pbar = tqdm.tqdm(total=max(0, total - args.start_frame), desc="frames", unit="f")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pbar.update(1)

        if (frame_idx - args.start_frame) % stride != 0:
            frame_idx += 1
            continue

        if scene_diff is not None:
            g = sam3._downscale_gray(frame)
            if not sam3._should_save_keyframe(
                g,
                last_gray,
                frame_idx,
                last_saved_idx if last_saved_idx >= 0 else -10**9,
                scene_diff,
                force_every,
            ):
                skipped += 1
                frame_idx += 1
                continue
            last_gray = g

        name = f"frame_{frame_idx:07d}.jpg"
        out_path = args.output / name
        cv2.imwrite(
            str(out_path),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, int(max(0, min(100, args.jpeg_quality)))],
        )
        saved += 1
        last_saved_idx = frame_idx

        if args.max_frames and saved >= args.max_frames:
            break

        frame_idx += 1

    pbar.close()
    cap.release()

    print(f"Saved {saved} images → {args.output.resolve()}")
    if scene_diff and skipped:
        print(f"Skipped {skipped} similar frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
