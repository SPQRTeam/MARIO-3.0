#!/usr/bin/env python3
"""
Run YOLOv12 inference on images or video using trained weights.

Usage:
  python scripts/inference_yolo.py --weights runs/detect/sam3_yolov12/weights/best.pt --video path/to/video.mp4
  python scripts/inference_yolo.py --weights best.pt --image path/to/image.jpg --show
  python scripts/inference_yolo.py --weights best.pt --video video.mp4 --show --save
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv12 inference")
    parser.add_argument("--weights", "-w", type=Path, required=True,
                        help="Path to best.pt weights")
    parser.add_argument("--image", "-i", type=Path, default=None,
                        help="Single image")
    parser.add_argument("--video", "-v", type=Path, default=None,
                        help="Video file")
    parser.add_argument("--show", action="store_true",
                        help="Show results in window")
    parser.add_argument("--save", action="store_true",
                        help="Save annotated output")
    parser.add_argument("--conf", type=float, default=0.2,
                        help="Confidence threshold (default: 0.5)")
    parser.add_argument("--imgsz", type=int, default=480,
                        help="Image size (default: 480)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: auto)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.weights.exists():
        print(f"ERROR: weights not found: {args.weights}")
        return

    source = args.image or args.video
    if source is None:
        print("ERROR: provide --image or --video")
        return
    if not source.exists():
        print(f"ERROR: {source} not found")
        return

    model = YOLO(str(args.weights))

    results = model.predict(
        source=str(source),
        conf=args.conf,
        imgsz=args.imgsz,
        show=args.show,
        save=args.save,
        device=args.device,
        stream=True,
    )

    for r in results:
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            print(f"Frame: {len(boxes)} detections")
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = r.names[cls]
                xyxy = box.xyxy[0].tolist()
                print(f"  {name} ({conf:.2f}): [{xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f}]")

    print(f"\nDone. Results saved in runs/detect/predict/")


if __name__ == "__main__":
    main()
