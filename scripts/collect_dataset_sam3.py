#!/usr/bin/env python3
"""
Collect YOLO training dataset using SAM 3 (facebook/sam3) from Hugging Face.

SAM 3 supports text-prompted segmentation — we prompt with custom text
to get instance masks, then extract bounding boxes for YOLO labels.

Requires:
  pip install "transformers>=5.0.0" torch pillow
  HF token in hf_token.txt at the project root (needed to download gated model).

Usage:
  python scripts/collect_dataset_sam3.py --image path/to/image.jpg --show
  python scripts/collect_dataset_sam3.py --video path/to/video.mp4 --every 30 --show
  python scripts/collect_dataset_sam3.py --video video.mp4 --prompts "robot:0" "ball:1" "referee:2" --show

Campione “intelligente” (meno frame quasi uguali, meno chiamate SAM3):
  python scripts/collect_dataset_sam3.py -v video.mp4 --every 5 --scene-diff 0.018 \\
      --force-every-frames 120 --prompts "soccer player:0" "soccer ball:1" --yolo-guided data/models/yolo12_sam3.pt
  --scene-diff: salta se la scena (grigio ridimensionato) è troppo simile al frame salvato prima.
  --force-every-frames: dopo N frame senza salvataggi, salva comunque (pan lenti / poca differenza).
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
import src.utils as utils

ROOT_DIR = Path(__file__).parent.parent

MODEL_ID = "facebook/sam3"

# Smart sampling: compare downscaled grayscale to avoid near-duplicate keyframes
_SCENE_SMALL = (160, 90)


def _downscale_gray(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.resize(frame_bgr, _SCENE_SMALL), cv2.COLOR_BGR2GRAY)


def _mean_abs_diff_norm(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0)


def _should_save_keyframe(
    gray: np.ndarray,
    last_gray: np.ndarray,
    frame_idx: int,
    last_saved_idx: int,
    scene_diff_min: float,
    force_every_frames: int,
) -> bool:
    """Return False if this frame is too similar to last saved (and no force)."""
    if last_gray is None:
        return True
    d = _mean_abs_diff_norm(gray, last_gray)
    if d >= scene_diff_min:
        return True
    if force_every_frames and (frame_idx - last_saved_idx) >= force_every_frames:
        return True
    return False

EXTENDED_PROMPTS = {
    "soccer player with red shirt": 0,
    "soccer player with blue shirt": 1,
    "soccer player with white shirt": 2,
    "soccer player with yellow shirt": 3,
    "soccer player": 4,
    "goalkeeper": 5,
    "ball": 6,
    "human referee": 7,
}

DEFAULT_PROMPTS = {
    "soccer player": 0,
    "soccer ball": 1,
    "human referee": 2,
}

# YOLO class index → (prompt_cls_id, label)
# Adjust if your YOLO model has different class ordering
YOLO_TO_PROMPT_CLS = {
    0: 0,   # robot/player → class 0
    1: 1,   # ball         → class 1
}


def load_hf_token() -> str:
    token_path = ROOT_DIR / "hf_token.txt"
    if not token_path.exists():
        print(f"ERROR: HF token not found at {token_path}")
        print("Create the file with your Hugging Face token.")
        sys.exit(1)
    return token_path.read_text().strip()


def parse_prompts(prompt_args: list) -> dict:
    """
    Parse prompt arguments like "robot:0" "ball:1" into {text: class_id} dict.
    If no class id is given, auto-assign incrementally.
    """
    prompts = {}
    for i, p in enumerate(prompt_args):
        if ":" in p:
            text, cls_id = p.rsplit(":", 1)
            prompts[text.strip()] = int(cls_id.strip())
        else:
            prompts[p.strip()] = i
    return prompts


def load_yolo(model_path: Path):
    """Load a YOLO model (ultralytics)."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)
    print(f"Loading YOLO from {model_path}...")
    model = YOLO(str(model_path))
    print("YOLO loaded.")
    return model


def detect_frame_yolo_guided(frame_bgr: np.ndarray, yolo_model, sam_model,
                              sam_processor, device: str,
                              yolo_to_cls: dict = None,
                              conf_threshold: float = 0.25) -> list:
    """
    Use YOLO to locate objects, then use SAM with point prompts for clean masks.

    This is much more reliable for small objects like the ball because YOLO
    provides precise center-point hints, and SAM then segments the exact object.

    Returns list of (cls_id, xc_norm, yc_norm, w_norm, h_norm, score, mask).
    """
    if yolo_to_cls is None:
        yolo_to_cls = YOLO_TO_PROMPT_CLS

    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # Run YOLO detection
    results = yolo_model(frame_bgr, conf=conf_threshold, verbose=False)
    if not results or results[0].boxes is None:
        return []

    boxes = results[0].boxes
    detections = []

    for i in range(len(boxes)):
        yolo_cls = int(boxes.cls[i].item())
        score = float(boxes.conf[i].item())
        cls_id = yolo_to_cls.get(yolo_cls)
        if cls_id is None:
            continue

        # YOLO box in xyxy pixel coords
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # Use SAM with a point prompt at the YOLO detection center
        try:
            input_points = [[[cx, cy]]]
            input_labels = [[1]]  # 1 = foreground point
            inputs = sam_processor(
                images=pil_image,
                input_points=input_points,
                input_labels=input_labels,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = sam_model(**inputs)

            masks_out = sam_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )[0]  # shape: (num_masks, H, W)

            # Pick the mask with highest IOU score, or fallback to YOLO box
            iou_scores = outputs.iou_scores[0] if hasattr(outputs, "iou_scores") else None
            if iou_scores is not None and len(iou_scores[0]) > 0:
                best_idx = iou_scores[0].argmax().item()
                mask_np = masks_out[0][best_idx].numpy().astype(bool)
            else:
                mask_np = masks_out[0][0].numpy().astype(bool)

            # Derive bbox from mask for tighter fit
            coords = np.where(mask_np)
            if len(coords[0]) > 0:
                my1, my2 = int(coords[0].min()), int(coords[0].max())
                mx1, mx2 = int(coords[1].min()), int(coords[1].max())
            else:
                mx1, my1, mx2, my2 = int(x1), int(y1), int(x2), int(y2)
                mask_np = None

        except Exception as e:
            # SAM failed: fall back to YOLO box, no mask
            mx1, my1, mx2, my2 = int(x1), int(y1), int(x2), int(y2)
            mask_np = None

        xc_n = (mx1 + mx2) / 2.0 / w
        yc_n = (my1 + my2) / 2.0 / h
        bw_n = (mx2 - mx1) / w
        bh_n = (my2 - my1) / h

        if bw_n > 0.002 and bh_n > 0.002:
            detections.append((cls_id, xc_n, yc_n, bw_n, bh_n, score, mask_np))

    return detections


def load_sam3(token: str, device: str):
    """Load SAM 3 model and processor from HF (gated, needs token)."""
    try:
        from transformers import Sam3Processor, Sam3Model
    except ImportError:
        print("ERROR: Sam3 requires transformers >= 5.0.0")
        print("  Run: pip install \"transformers>=5.0.0\"")
        sys.exit(1)

    print(f"Loading {MODEL_ID} on {device}...")
    processor = Sam3Processor.from_pretrained(MODEL_ID, token=token)
    model = Sam3Model.from_pretrained(MODEL_ID, token=token).to(device)
    model.eval()
    print("SAM 3 loaded.")
    return model, processor


def detect_frame(frame_bgr: np.ndarray, model, processor, device: str,
                 prompts: dict = None, threshold: float = 0.5):
    """
    Run SAM 3 text-prompted segmentation on a single frame.

    Returns list of (cls_id, xc_norm, yc_norm, w_norm, h_norm, score, mask).
    Coordinates are normalized [0, 1].
    """
    if prompts is None:
        prompts = DEFAULT_PROMPTS

    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    detections = []

    for text_prompt, cls_id in prompts.items():
        inputs = processor(
            images=pil_image,
            text=text_prompt,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        masks = results.get("masks", [])
        boxes = results.get("boxes", [])
        scores = results.get("scores", [])

        # Per-prompt thresholds
        prompt_lower = text_prompt.lower()
        if "player" in prompt_lower or ("soccer" in prompt_lower and "ball" not in prompt_lower):
            min_score = 0.53
        elif "ball" in prompt_lower:
            min_score = 0.3
        else:
            min_score = threshold

        for i in range(len(masks)):
            score = scores[i].item() if torch.is_tensor(scores[i]) else float(scores[i])
            if score < min_score:
                continue

            # Get bbox: prefer the model's box output, fall back to mask
            if len(boxes) > i:
                box = boxes[i]
                if torch.is_tensor(box):
                    box = box.tolist()
                x1, y1, x2, y2 = box
            else:
                mask_np = masks[i].cpu().numpy() if torch.is_tensor(masks[i]) else np.array(masks[i])
                coords = np.where(mask_np > 0)
                if len(coords[0]) == 0:
                    continue
                y1, y2 = coords[0].min(), coords[0].max()
                x1, x2 = coords[1].min(), coords[1].max()

            xc = (x1 + x2) / 2.0 / w
            yc = (y1 + y2) / 2.0 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            if bw > 0.005 and bh > 0.005:
                mask_np = masks[i].cpu().numpy() if torch.is_tensor(masks[i]) else np.array(masks[i])
                detections.append((cls_id, xc, yc, bw, bh, score, mask_np))

    # Cross-class NMS: if overlap with referee, keep referee;
    # otherwise keep the lower cls_id.
    detections = _cross_class_nms(detections, prompts)

    return detections


def _iou(a, b):
    """IoU between two boxes (xc, yc, w, h) all normalized."""
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def _cross_class_nms(detections, prompts=None, iou_threshold=0.5):
    """
    Cross-class NMS: when two classes overlap, keep "human referee".
    For all other overlaps, keep the lower cls_id (higher priority).
    """
    # Find the cls_id for "human referee"
    referee_cls = None
    if prompts:
        for text, cls_id in prompts.items():
            if "referee" in text.lower():
                referee_cls = cls_id
                break

    detections.sort(key=lambda d: d[0])
    keep = []
    for det in detections:
        box = (det[1], det[2], det[3], det[4])
        overlap_idx = None
        for i, kept in enumerate(keep):
            if kept[0] != det[0]:
                kept_box = (kept[1], kept[2], kept[3], kept[4])
                if _iou(box, kept_box) > iou_threshold:
                    overlap_idx = i
                    break
        if overlap_idx is None:
            keep.append(det)
        elif referee_cls is not None and det[0] == referee_cls:
            # New detection is referee: replace the overlapping one
            keep[overlap_idx] = det
        # else: keep the existing one (lower cls_id wins)
    return keep


PALETTE = [
    (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 255), (255, 128, 0), (0, 128, 255),
]


def draw_detections(image_bgr: np.ndarray, detections: list,
                    prompts: dict = None,
                    draw_masks: bool = True) -> np.ndarray:
    """Draw bounding boxes and semi-transparent masks on the image."""
    if prompts is None:
        prompts = DEFAULT_PROMPTS
    names = {v: k for k, v in prompts.items()}

    vis = image_bgr.copy()
    img_h, img_w = vis.shape[:2]

    for cls_id, xc, yc, bw, bh, score, mask_np in detections:
        color = PALETTE[cls_id % len(PALETTE)]

        # Draw mask overlay
        if draw_masks and mask_np is not None:
            mask_bool = mask_np.astype(bool)
            if mask_bool.shape[:2] == (img_h, img_w):
                overlay = vis.copy()
                overlay[mask_bool] = color
                cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)

        # Draw bbox
        x1 = int((xc - bw / 2) * img_w)
        y1 = int((yc - bh / 2) * img_h)
        x2 = int((xc + bw / 2) * img_w)
        y2 = int((yc + bh / 2) * img_h)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{names.get(cls_id, 'obj')} {score:.2f}"
        cv2.putText(vis, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return vis


def _mask_to_polygon(mask_np, img_h, img_w, max_points=100):
    """Convert a binary mask to a normalized YOLO segmentation polygon."""
    mask_uint8 = (mask_np > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Take the largest contour
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None
    # Simplify if too many points
    if len(contour) > max_points:
        epsilon = 0.5
        while len(contour) > max_points:
            contour = cv2.approxPolyDP(contour, epsilon, True)
            epsilon += 0.5
    # Normalize coordinates
    points = contour.reshape(-1, 2)
    normalized = []
    for x, y in points:
        normalized.append(f"{x / img_w:.6f}")
        normalized.append(f"{y / img_h:.6f}")
    return " ".join(normalized)


def save_yolo(labels_dir: Path, frame_id: int, detections: list, img_h: int, img_w: int,
              stem: str = None):
    """Save detections in YOLO format: bbox labels + segmentation labels."""
    name = stem if stem is not None else f"frame_{frame_id:07d}"

    # Bounding box labels (YOLO detect format)
    with open(labels_dir / f"{name}.txt", "w") as f:
        for cls_id, xc, yc, w, h, _score, _mask in detections:
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    # Segmentation labels (YOLO segment format: cls x1 y1 x2 y2 ... xn yn)
    seg_dir = labels_dir.parent / "labels_seg"
    seg_dir.mkdir(parents=True, exist_ok=True)
    with open(seg_dir / f"{name}.txt", "w") as f:
        for cls_id, _xc, _yc, _w, _h, _score, mask_np in detections:
            if mask_np is None:
                continue
            polygon = _mask_to_polygon(mask_np, img_h, img_w)
            if polygon:
                f.write(f"{cls_id} {polygon}\n")


def write_dataset_yaml(output_dir: Path, prompts: dict):
    names = {v: k for k, v in prompts.items()}
    nc = max(names.keys()) + 1
    names_list = [names.get(i, f"class_{i}") for i in range(nc)]
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {output_dir.resolve()}\n")
        f.write("train: images\nval: images\n\n")
        f.write(f"nc: {nc}\nnames: {names_list}\n")
    print(f"dataset.yaml written to {yaml_path}")


def process_single_image(image_path: Path, model, processor, device: str,
                         prompts: dict, show: bool = False, yolo_model=None,
                         save_to: Path = None):
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"ERROR: cannot read image {image_path}")
        return

    names = {v: k for k, v in prompts.items()}
    h, w = frame.shape[:2]
    print(f"Running SAM 3 on {image_path.name} ({w}x{h})...")

    if yolo_model is not None:
        print("  Mode: YOLO-guided (YOLO → SAM point prompts)")
        detections = detect_frame_yolo_guided(frame, yolo_model, model, processor, device)
    else:
        print(f"  Prompts: {prompts}")
        detections = detect_frame(frame, model, processor, device, prompts)

    print(f"  Found {len(detections)} detections")
    for cls_id, xc, yc, bw, bh, score, _ in detections:
        name = names.get(cls_id, f"class_{cls_id}")
        print(f"    {name}: center=({xc:.3f}, {yc:.3f}) "
              f"size=({bw:.3f}x{bh:.3f}) score={score:.2f}")

    if save_to is not None:
        images_dir = save_to / "images"
        labels_dir = save_to / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem
        out_img = images_dir / f"{stem}.jpg"
        if not out_img.exists() or out_img.resolve() != image_path.resolve():
            cv2.imwrite(str(out_img), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        save_yolo(labels_dir, 0, detections, h, w, stem=stem)
        print(f"  Saved → {out_img} + labels/{stem}.txt")

    if show:
        vis = draw_detections(frame, detections, prompts)
        cv2.imshow("SAM 3 detections", vis)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(video_path: Path, output_dir: Path, model, processor,
                  device: str, prompts: dict, every_n_frames: int = 1,
                  max_frames: int = None, show: bool = False,
                  start_frame: int = 0, yolo_model=None,
                  scene_diff_min: float = None,
                  force_every_frames: int = None):
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: cannot open video {video_path}")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved = 0
    frame_idx = start_frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"  Starting from frame {start_frame}")

    print(f"Processing {video_path.name} ({total} frames, every {every_n_frames})")
    print(f"  Prompts: {prompts}")
    if scene_diff_min is not None:
        print(f"  Scene filter: min diff={scene_diff_min} (skip near-duplicates)"
              + (f", force every {force_every_frames} frames" if force_every_frames else ""))
    skipped_similar = 0
    last_key_gray = None
    last_saved_idx = None
    if show:
        print("  Controls: RIGHT/D=skip 10 | LEFT/A=back 10 | SPACE=pause | Q=quit")

    skip_count = 0
    paused = False
    quit_requested = False

    while True:
        if skip_count > 0:
            cap.grab()
            frame_idx += 1
            skip_count -= 1
            continue

        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n_frames == 0:
            if scene_diff_min is not None:
                g = _downscale_gray(frame)
                if not _should_save_keyframe(
                    g,
                    last_key_gray,
                    frame_idx,
                    last_saved_idx if last_saved_idx is not None else -10**9,
                    scene_diff_min,
                    force_every_frames or 0,
                ):
                    skipped_similar += 1
                    frame_idx += 1
                    continue

            try:
                if yolo_model is not None:
                    detections = detect_frame_yolo_guided(
                        frame, yolo_model, model, processor, device
                    )
                else:
                    detections = detect_frame(frame, model, processor, device, prompts)

                name = f"frame_{frame_idx:07d}"
                cv2.imwrite(str(images_dir / f"{name}.jpg"), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                fh, fw = frame.shape[:2]
                save_yolo(labels_dir, frame_idx, detections, fh, fw)

                saved += 1
                last_saved_idx = frame_idx
                if scene_diff_min is not None:
                    last_key_gray = g
                print(f"  [{frame_idx}/{total}] saved ({len(detections)} detections)")

            except Exception as e:
                detections = []
                print(f"  [{frame_idx}/{total}] ERROR: {e}")

            if show:
                vis = draw_detections(frame, detections, prompts)
                cv2.putText(vis, f"frame {frame_idx}/{total}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("SAM 3 detections", vis)

                while True:
                    key = cv2.waitKey(0 if paused else 1) & 0xFF
                    if key == ord('q'):
                        quit_requested = True
                        break
                    elif key == 83 or key == ord('d'):  # RIGHT arrow
                        skip_count = 10 * every_n_frames
                        print(f"  >> skip {skip_count} frames")
                        break
                    elif key == 81 or key == ord('a'):  # LEFT arrow
                        new_pos = max(0, frame_idx - 10 * every_n_frames)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                        frame_idx = new_pos
                        print(f"  << rewind to frame {new_pos}")
                        break
                    elif key == ord(' '):
                        paused = not paused
                        print(f"  {'PAUSED' if paused else 'RESUMED'}")
                        if not paused:
                            break
                    elif not paused:
                        break

                if quit_requested:
                    print("Interrupted by user")
                    break

            if max_frames and saved >= max_frames:
                break

        frame_idx += 1

    cap.release()
    if show:
        cv2.destroyAllWindows()

    write_dataset_yaml(output_dir, prompts)
    print(f"Done. Saved {saved} frames to {output_dir}")
    if scene_diff_min is not None and skipped_similar:
        print(f"  Skipped {skipped_similar} similar frames (scene filter).")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect YOLO dataset using SAM 3 (facebook/sam3)"
    )
    parser.add_argument("--image", "-i", type=Path, default=None,
                        help="Single image to process (for testing)")
    parser.add_argument("--images-dir", type=Path, default=None,
                        help="Process all .jpg/.png in this folder (use with --output)")
    parser.add_argument("--save-to", type=Path, default=None,
                        help="With --image: write images/ + labels/ under this directory")
    parser.add_argument("--video", "-v", type=Path, default=None,
                        help="Video file to process")
    parser.add_argument("--section", "-s", type=str, default=None,
                        help="Section letter (uses config.yaml)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory")
    parser.add_argument("--every", "-e", type=int, default=1,
                        help="Process 1 frame every N (default: 1)")
    parser.add_argument("--max-frames", "-m", type=int, default=None,
                        help="Max frames to save")
    parser.add_argument("--show", action="store_true",
                        help="Show detections with masks in a window")
    parser.add_argument("--prompts", "-p", nargs="+", type=str, default=None,
                        help='Text prompts as "name:class_id" (default: "robot:0" "ball:1"). '
                             'Example: --prompts "robot:0" "ball:1" "referee:2"')
    parser.add_argument("--config", "-c", type=Path, default=None,
                        help="Path to config.yaml")
    parser.add_argument("--device", "-d", type=str, default=None,
                        help="Device (default: cuda if available, else cpu)")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                        help="Detection confidence threshold (default: 0.5)")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Frame to start from (default: 0)")
    parser.add_argument("--yolo-guided", type=Path, default=None,
                        metavar="YOLO_MODEL",
                        help="Use a YOLO model to locate objects first, then refine with SAM "
                             "point prompts. Much more reliable for small objects like the ball. "
                             "Example: --yolo-guided data/models/yolo12_sam3.pt")
    parser.add_argument("--scene-diff", type=float, default=None,
                        metavar="MIN",
                        help="Skip candidate frames whose downscaled grayscale is too similar "
                             "to the last saved frame (mean abs diff 0..1, try 0.012–0.03). "
                             "Reduces duplicate labels and SAM3 cost.")
    parser.add_argument("--force-every-frames", type=int, default=None,
                        metavar="N",
                        help="If no frame was saved for N frames, save the next candidate anyway "
                             "(e.g. 90–150 at 30 fps) so slow pans are still sampled.")
    return parser.parse_args()


def main():
    args = parse_args()
    token = load_hf_token()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    prompts = parse_prompts(args.prompts) if args.prompts else DEFAULT_PROMPTS

    model, processor = load_sam3(token, device)
    yolo_model = load_yolo(args.yolo_guided) if args.yolo_guided else None

    # Single image or directory of images
    if args.image:
        process_single_image(
            args.image, model, processor, device,
            prompts, show=args.show, yolo_model=yolo_model,
            save_to=args.save_to,
        )
        if args.save_to:
            write_dataset_yaml(args.save_to, prompts)
        return

    if args.images_dir:
        if not args.output:
            print("ERROR: --images-dir requires --output DATASET_DIR")
            sys.exit(1)
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        paths = sorted(
            p for p in args.images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        )
        if not paths:
            print(f"No images in {args.images_dir}")
            sys.exit(1)
        print(f"Processing {len(paths)} images → {args.output}")
        for p in paths:
            process_single_image(
                p, model, processor, device,
                prompts, show=False, yolo_model=yolo_model,
                save_to=args.output,
            )
        write_dataset_yaml(args.output, prompts)
        print("Done.")
        return

    # Video mode
    if args.video:
        video_path = args.video
        output_dir = args.output or video_path.parent / "sam3_dataset"
    elif args.section:
        config = utils.load_config(args.config)
        section_name = f"mario_{args.section}"
        video_path = config.video_path(section_name)
        output_dir = args.output or config.game_dir / "sam3_dataset"
    else:
        print("ERROR: provide --image, --video, or --section")
        sys.exit(1)

    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}")
        sys.exit(1)

    process_video(
        video_path=video_path,
        output_dir=output_dir,
        model=model,
        processor=processor,
        device=device,
        prompts=prompts,
        every_n_frames=args.every,
        max_frames=args.max_frames,
        show=args.show,
        start_frame=args.start_frame,
        yolo_model=yolo_model,
        scene_diff_min=args.scene_diff,
        force_every_frames=args.force_every_frames,
    )


if __name__ == "__main__":
    main()
