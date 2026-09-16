#!/usr/bin/env python3
"""
Crop robot da video: OpenCV legge solo 1 frame ogni N, YOLO ``predict`` sul frame,
salva ``{frame_idx}-{i}.png`` (i = indice del robot in quel frame).

Niente tracking completo, niente homography — solo campionamento + inferenza.

Esempi::

    python scripts/prepare_robot_pics.py -i partita.mp4 -o training/robot_pics --every-n-frames 100

    # Pesi custom (default: config.yaml → models.yolo_v12)
    python scripts/prepare_robot_pics.py -i video.mp4 -o out/ -m data/models/yolo12_sam3.pt

Modalità ``--filter``: sottocampiona PNG già esistenti (stesso schema nome file).
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_FRAME_RE = re.compile(r"^(\d+)-(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)


def _sort_key(p: Path) -> Tuple[int, int]:
    m = _FRAME_RE.match(p.name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 0, 0


def list_robot_crops(d: Path) -> List[Path]:
    out: List[Path] = []
    for pat in ("*.png", "*.jpg", "*.jpeg"):
        out.extend(d.glob(pat))
    return sorted(out, key=_sort_key)


def run_filter_folder(inp: Path, out: Path, every_n: int, offset: int, dry_run: bool) -> int:
    n = max(1, every_n)
    off = offset % n
    inp = inp.resolve()
    out = out.resolve()

    if not inp.is_dir():
        print(f"ERROR: cartella non trovata: {inp}", file=sys.stderr)
        return 1

    paths = list_robot_crops(inp)
    if not paths:
        print(f"ERROR: nessuna immagine in {inp}", file=sys.stderr)
        return 1

    keep: List[Path] = []
    skipped_unparsed = 0
    for p in paths:
        m = _FRAME_RE.match(p.name)
        if not m:
            skipped_unparsed += 1
            continue
        fid = int(m.group(1))
        if fid % n == off:
            keep.append(p)

    print(
        f"Trovate {len(paths)} immagini → ne tengo {len(keep)} "
        f"(ogni {n} frame, offset {off})"
    )
    if skipped_unparsed:
        print(f"  (ignorate {skipped_unparsed} con nome non standard)")

    if dry_run:
        print("Dry-run: nessun file copiato.")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    for p in keep:
        shutil.copy2(p, out / p.name)

    print(f"Copiate {len(keep)} file → {out}")
    return 0


def _load_yaml_config(config_path: Path) -> Dict[str, Any]:
    import yaml

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_sparse_yolo_video(
    video_path: Path,
    out_dir: Path,
    *,
    stride: int,
    config_path: Path,
    weights_path: Path | None,
    start_frame: int,
    max_frames: int | None,
) -> int:
    import cv2
    from ultralytics import YOLO
    import tqdm

    from src.core.detection import Detection, suppress_nested_robot_detections

    video_path = video_path.resolve()
    if not video_path.is_file():
        print(f"ERROR: video non trovato: {video_path}", file=sys.stderr)
        return 1

    cfg = _load_yaml_config(config_path)
    wpath = (
        Path(weights_path)
        if weights_path is not None
        else (ROOT / cfg["models"]["yolo_v12"])
    )
    if not wpath.is_file():
        print(f"ERROR: pesi YOLO non trovati: {wpath}", file=sys.stderr)
        return 1

    rns = cfg.get("robot_nested_suppression") or {}

    model = YOLO(str(wpath))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: impossibile aprire il video: {video_path}", file=sys.stderr)
        return 1

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = list(range(start_frame, total, stride))
    if max_frames is not None:
        indices = indices[: max(0, max_frames)]

    saved = 0
    for fi in tqdm.tqdm(indices, desc="frame", unit="f"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        results = model.predict(frame, verbose=False)
        dets = Detection.from_results(results)
        if rns.get("enabled"):
            dets = suppress_nested_robot_detections(
                dets,
                min_overlap_of_smaller=float(rns.get("min_overlap_of_smaller", 0.5)),
                max_area_ratio=float(rns.get("max_area_ratio", 0.85)),
            )
        robots = [d for d in dets if d.cls_name == "robot"]
        for j, det in enumerate(robots):
            crop = det.box.as_int().cut(frame)
            if crop.size == 0:
                continue
            fn = f"{fi}-{j}.png"
            cv2.imwrite(str(out_dir / fn), crop)
            saved += 1

    cap.release()
    print(f"Salvati {saved} crop in {out_dir} ({len(indices)} frame campionati).")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Crop robot: YOLO ogni N frame (OpenCV), oppure --filter su cartella PNG."
    )
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="File video .mp4 / .mkv … oppure cartella PNG con --filter",
    )
    p.add_argument("-o", "--output", type=Path, required=True, help="Cartella output crop")
    p.add_argument(
        "--every-n-frames",
        type=int,
        default=100,
        metavar="N",
        help="Leggi e inferisci 1 frame ogni N (default 100)",
    )
    p.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Primo indice frame da campionare (default 0)",
    )
    p.add_argument(
        "--max-sampled-frames",
        type=int,
        default=None,
        metavar="K",
        help="Ferma dopo K frame campionati (opzionale)",
    )
    p.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="config.yaml per pesi e robot_nested_suppression (default: repo root)",
    )
    p.add_argument(
        "-m",
        "--model",
        type=Path,
        default=None,
        help="Override percorso pesi YOLOv12 (robot)",
    )
    p.add_argument(
        "--filter",
        action="store_true",
        help="Input è una cartella: copia solo PNG con frame_id mod N",
    )
    p.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Solo --filter: congruenza mod N",
    )
    p.add_argument("--dry-run", action="store_true", help="Solo --filter")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.filter:
        return run_filter_folder(
            args.input,
            args.output,
            args.every_n_frames,
            args.offset,
            args.dry_run,
        )

    cfg_path = args.config or (ROOT / "config.yaml")
    if not cfg_path.is_file():
        print(f"ERROR: config non trovato: {cfg_path}", file=sys.stderr)
        return 1

    return run_sparse_yolo_video(
        args.input,
        args.output,
        stride=max(1, args.every_n_frames),
        config_path=cfg_path,
        weights_path=args.model,
        start_frame=max(0, args.start_frame),
        max_frames=args.max_sampled_frames,
    )


if __name__ == "__main__":
    raise SystemExit(main())
