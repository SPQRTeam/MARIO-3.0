#!/usr/bin/env python3
"""
Pre-label with YOLO only (fast) → YOLO-format images + labels + dataset.yaml.

Video:
  python scripts/prelabel_yolo_dataset.py -w data/models/yolo12_sam3.pt \\
    -v match.mp4 -o data/mygame/yolo_prelabel --every 5 --scene-diff 0.018

Immagini già in cartella (es. dataset_sam3/images): solo inferenza, label con stesso stem:
  python scripts/prelabel_yolo_dataset.py -w data/models/yolo12_sam3.pt \\
    --images-dir data/Final_GO/dataset_sam3/images \\
    -o data/Final_GO/dataset_sam3 --every 1 --conf 0.25 --imgsz 960
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import collect_dataset_sam3 as sam3  # noqa: E402 — reuse scene sampling helpers


def parse_args():
    p = argparse.ArgumentParser(description="Fast YOLO-only pre-labeling → YOLO dataset")
    p.add_argument("--weights", "-w", type=Path, required=True, help="YOLO weights (.pt)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", "-v", type=Path, help="Input video")
    src.add_argument("--images-dir", type=Path,
                     help="Folder of existing .jpg/.png — writes labels/*.txt with same stem as each image")
    p.add_argument("--output", "-o", type=Path, required=True, help="Dataset root (images/, labels/, dataset.yaml)")
    p.add_argument("--every", "-e", type=int, default=5,
                   help="Video: every N frames. Images: every N-th file (sorted). Default 5 (use 1 for all images)")
    p.add_argument("--scene-diff", type=float, default=None, metavar="MIN",
                   help="Skip near-duplicate frames vs last saved (same as SAM3 collector)")
    p.add_argument("--force-every-frames", type=int, default=None,
                   help="Force a save at least every N frames if scene-diff is on")
    p.add_argument("--start-frame", type=int, default=0, help="Video only: start at this frame index")
    p.add_argument("--max-frames", type=int, default=None, help="Video: max saved frames. Images: max processed")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Minimum confidence for YOLO candidates before top-k per class")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--robot-class", type=int, default=0, help="Class id for robot (default 0)")
    p.add_argument("--ball-class", type=int, default=1, help="Class id for ball (default 1)")
    p.add_argument("--max-robots", type=int, default=10,
                   help="Keep at most this many robot boxes, highest conf first (default 10)")
    p.add_argument("--max-balls", type=int, default=1,
                   help="Keep at most this many ball boxes, highest conf first (default 1)")
    return p.parse_args()


def _write_dataset_yaml(out_dir: Path, names_list: list) -> None:
    nc = len(names_list)
    data = {
        "path": str(out_dir.resolve()),
        "train": "images",
        "val": "images",
        "nc": nc,
        "names": names_list,
    }
    with open(out_dir / "dataset.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {out_dir / 'dataset.yaml'}")


def _boxes_to_yolo_lines(
    result,
    robot_cls: int = 0,
    ball_cls: int = 1,
    max_robots: int = 10,
    max_balls: int = 1,
) -> list[str]:
    """
    Ultralytics Results → YOLO lines: top ``max_robots`` robots + top ``max_balls`` balls by confidence.
    Other classes are dropped. At most max_robots + max_balls lines (default 11).
    """
    r = result
    if r.boxes is None or len(r.boxes) == 0:
        return []

    robots: list[tuple[float, list]] = []
    balls: list[tuple[float, list]] = []

    boxes = r.boxes
    for i in range(len(boxes)):
        cls = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        xywhn = boxes.xywhn[i].tolist()
        if cls == robot_cls:
            robots.append((conf, xywhn))
        elif cls == ball_cls:
            balls.append((conf, xywhn))

    robots.sort(key=lambda t: -t[0])
    balls.sort(key=lambda t: -t[0])
    picked_robots = robots[:max_robots]
    picked_balls = balls[:max_balls]

    lines: list[str] = []
    for conf, xywhn in picked_robots:
        lines.append(
            f"{robot_cls} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}\n"
        )
    for conf, xywhn in picked_balls:
        lines.append(
            f"{ball_cls} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}\n"
        )
    return lines


def _lines_from_result(args, result) -> list[str]:
    return _boxes_to_yolo_lines(
        result,
        robot_cls=args.robot_class,
        ball_cls=args.ball_class,
        max_robots=args.max_robots,
        max_balls=args.max_balls,
    )


def _load_model(weights: Path):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: pip install ultralytics")
        raise SystemExit(1)
    model = YOLO(str(weights))
    mdn = model.names
    if isinstance(mdn, dict):
        names_list = [mdn[i] for i in range(len(mdn))]
    else:
        names_list = list(mdn)
    return model, names_list


def main():
    args = parse_args()
    if not args.weights.exists():
        print(f"ERROR: weights not found: {args.weights}")
        return 1

    if args.video is not None:
        if not args.video.exists():
            print(f"ERROR: video not found: {args.video}")
            return 1
        return _run_video(args)

    if args.images_dir is not None:
        if not args.images_dir.is_dir():
            print(f"ERROR: not a directory: {args.images_dir}")
            return 1
        return _run_images_dir(args)

    return 1


def _run_video(args) -> int:
    model, names_list = _load_model(args.weights)
    print(f"Classes: {names_list}")
    print(
        f"Per frame: top {args.max_robots} class-{args.robot_class} (robot) + "
        f"top {args.max_balls} class-{args.ball_class} (ball) by conf "
        f"(max {args.max_robots + args.max_balls} boxes; other classes ignored)"
    )

    out = args.output
    images_dir = out / "images"
    labels_dir = out / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.video}")
        return 1

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = args.start_frame
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    saved = 0
    skipped_similar = 0
    last_key_gray = None
    last_saved_idx = None
    scene_diff_min = args.scene_diff
    force_every = args.force_every_frames or 0

    print(f"Video {args.video.name}: {total} frames, every {args.every}, conf={args.conf}, imgsz={args.imgsz}")
    if scene_diff_min is not None:
        print(f"  Scene filter min diff={scene_diff_min}, force every {force_every or 'off'}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.every == 0:
            if scene_diff_min is not None:
                g = sam3._downscale_gray(frame)
                if not sam3._should_save_keyframe(
                    g,
                    last_key_gray,
                    frame_idx,
                    last_saved_idx if last_saved_idx is not None else -10**9,
                    scene_diff_min,
                    force_every,
                ):
                    skipped_similar += 1
                    frame_idx += 1
                    continue

            results = model.predict(
                frame,
                conf=args.conf,
                imgsz=args.imgsz,
                verbose=False,
                device=args.device,
            )
            r = results[0]
            lines = _lines_from_result(args, r)

            name = f"frame_{frame_idx:07d}"
            cv2.imwrite(str(images_dir / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            label_path = labels_dir / f"{name}.txt"
            label_path.write_text("".join(lines))

            saved += 1
            last_saved_idx = frame_idx
            if scene_diff_min is not None:
                last_key_gray = sam3._downscale_gray(frame)

            nbox = len(lines)
            print(f"  [{frame_idx}/{total}] saved ({nbox} boxes)")

            if args.max_frames and saved >= args.max_frames:
                break

        frame_idx += 1

    cap.release()

    _write_dataset_yaml(args.output, names_list)
    print(f"Done. {saved} frames → {args.output} (skipped_similar={skipped_similar})")
    return 0


def _run_images_dir(args) -> int:
    model, names_list = _load_model(args.weights)
    print(f"Classes: {names_list}")
    print(
        f"Per frame: top {args.max_robots} class-{args.robot_class} + top {args.max_balls} class-{args.ball_class} "
        f"(max {args.max_robots + args.max_balls} boxes; other classes ignored)"
    )

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = sorted(
        p for p in args.images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )
    if not paths:
        print(f"No images in {args.images_dir}")
        return 1

    out = args.output
    labels_dir = out / "labels"
    out_images = out / "images"
    labels_dir.mkdir(parents=True, exist_ok=True)

    in_resolved = args.images_dir.resolve()
    out_resolved = out_images.resolve()
    same_folder = in_resolved == out_resolved
    if not same_folder:
        out_images.mkdir(parents=True, exist_ok=True)

    print(f"Images: {len(paths)} files in {args.images_dir}, every {args.every}, conf={args.conf}, imgsz={args.imgsz}")
    if same_folder:
        print("  (output images/ == --images-dir: only writing labels/*.txt, no image copy)")

    scene_diff_min = args.scene_diff
    force_every = args.force_every_frames or 0
    if scene_diff_min is not None:
        print(f"  Scene filter min diff={scene_diff_min}, force every {force_every or 'off'}")

    saved = 0
    skipped_similar = 0
    last_key_gray = None
    last_saved_idx = None

    for i, img_path in enumerate(paths):
        if i % args.every != 0:
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  SKIP unreadable: {img_path.name}")
            continue

        if scene_diff_min is not None:
            g = sam3._downscale_gray(frame)
            if not sam3._should_save_keyframe(
                g,
                last_key_gray,
                i,
                last_saved_idx if last_saved_idx is not None else -10**9,
                scene_diff_min,
                force_every,
            ):
                skipped_similar += 1
                continue

        results = model.predict(
            frame,
            conf=args.conf,
            imgsz=args.imgsz,
            verbose=False,
            device=args.device,
        )
        lines = _lines_from_result(args, results[0])
        stem = img_path.stem
        label_path = labels_dir / f"{stem}.txt"
        label_path.write_text("".join(lines))

        if not same_folder:
            dst = out_images / img_path.name
            if not dst.exists() or dst.resolve() != img_path.resolve():
                shutil.copy2(img_path, dst)

        saved += 1
        last_saved_idx = i
        if scene_diff_min is not None:
            last_key_gray = sam3._downscale_gray(frame)

        print(f"  [{saved}] {img_path.name} ({len(lines)} boxes)")

        if args.max_frames and saved >= args.max_frames:
            break

    _write_dataset_yaml(args.output, names_list)
    print(f"Done. {saved} labels → {labels_dir} (skipped_similar={skipped_similar})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
