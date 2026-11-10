"""Moondream 3.1 as an object detector: text labels in, tracked boxes out.

Runs on ``moondream_engine``: dense weights in VRAM, MoE experts split between VRAM and
pinned RAM. Weights (10.5 GB) download from Hugging Face on first use.
"""

import logging
from typing import List, NamedTuple, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import psutil
from PIL import Image
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

ROBOT_CLS = 0
UNTRACKED_ID = -1
# Overlap between neighbouring tiles, as a fraction of the tile size.
TILE_OVERLAP = 0.2
# Boxes from different views overlapping more than this are the same object.
MERGE_IOU = 0.5


class MoondreamObject(NamedTuple):
    box_xyxy: Tuple[float, float, float, float]
    confidence: float
    cls: int
    track_id: int


def tile_boxes(width: int, height: int, grid: int) -> List[Tuple[int, int, int, int]]:
    """``grid`` x ``grid`` overlapping crops covering the frame, as (x1, y1, x2, y2)."""
    tile_w = width / (grid - (grid - 1) * TILE_OVERLAP)
    tile_h = height / (grid - (grid - 1) * TILE_OVERLAP)
    step_w, step_h = tile_w * (1 - TILE_OVERLAP), tile_h * (1 - TILE_OVERLAP)
    return [
        (
            int(col * step_w),
            int(row * step_h),
            min(width, int(col * step_w + tile_w)),
            min(height, int(row * step_h + tile_h)),
        )
        for row in range(grid)
        for col in range(grid)
    ]


def to_pixels(obj: dict, view: Tuple[int, int, int, int]) -> Tuple[List[int], float]:
    """Normalized box inside ``view`` -> (frame pixel box, confidence)."""
    vx1, vy1, vx2, vy2 = view
    vw, vh = vx2 - vx1, vy2 - vy1
    box = [
        int(round(vx1 + obj["x_min"] * vw)),
        int(round(vy1 + obj["y_min"] * vh)),
        int(round(vx1 + obj["x_max"] * vw)),
        int(round(vy1 + obj["y_max"] * vh)),
    ]
    return box, float(obj.get("confidence", 1.0))


def _iou(a: List[int], b: List[int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def merge_boxes(boxes: List[Tuple[List[int], float]]) -> List[Tuple[List[int], float]]:
    """Greedy NMS across views: keep the most confident of overlapping boxes."""
    kept: List[Tuple[List[int], float]] = []
    for box, conf in sorted(boxes, key=lambda b: -b[1]):
        if all(_iou(box, other) <= MERGE_IOU for other, _ in kept):
            kept.append((box, conf))
    return kept


class RobotTracker:
    """
    Keeps robot IDs across model runs by matching feet positions.

    Runs are about a second apart: robots can move further than their own box, so
    IoU-based trackers such as ByteTrack hand them a new ID. Distances are measured in
    box heights, so the gate means the same near and far from the camera.
    """

    def __init__(self, gate: float, max_lost: int):
        self.gate = gate
        self.max_lost = max_lost
        self._next_id = 1
        self._tracks: dict = {}  # id -> (box, consecutive runs without a match)

    @staticmethod
    def _distance(a: List[int], b: List[int]) -> float:
        dx = (a[0] + a[2] - b[0] - b[2]) / 2
        dy = a[3] - b[3]
        return float(np.hypot(dx, dy)) / max(a[3] - a[1], b[3] - b[1], 1)

    def update(self, boxes: List[List[int]]) -> List[Tuple[int, List[int]]]:
        """Return (track_id, box) for every box of this run, in input order."""
        ids = list(self._tracks)
        id_of_box = {}
        if ids and boxes:
            costs = np.array(
                [[self._distance(self._tracks[i][0], box) for box in boxes] for i in ids]
            )
            for row, col in zip(*linear_sum_assignment(costs)):
                if costs[row, col] <= self.gate:
                    id_of_box[col] = ids[row]

        tracks = {}
        result = []
        for col, box in enumerate(boxes):
            track_id = id_of_box.get(col)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
            tracks[track_id] = (box, 0)
            result.append((track_id, box))
        for track_id in ids:
            box, lost = self._tracks[track_id]
            if track_id not in tracks and lost < self.max_lost:
                tracks[track_id] = (box, lost + 1)
        self._tracks = tracks
        return result


class MoondreamDetector:
    """Detects ``moondream.labels`` in a frame and assigns track IDs to robots."""

    def __init__(self, config):
        cfg = config.moondream
        self.stride = int(cfg.stride)
        self.labels = dict(cfg.labels)  # text prompt -> class id
        self.grids = list(cfg.grids)
        self.max_objects = int(cfg.max_objects)
        self.ball_max_size_px = int(cfg.ball_max_size_px)
        self.ball_max_aspect = float(cfg.ball_max_aspect)

        # Offloaded experts are pinned, so they can't go to swap: too little RAM takes the desktop down.
        available_gb = psutil.virtual_memory().available / 1e9
        if available_gb < float(cfg.min_available_ram_gb):
            raise MemoryError(
                f"Moondream needs {cfg.min_available_ram_gb} GB of available RAM, "
                f"only {available_gb:.1f} GB left. Close something and retry."
            )

        from .moondream_engine import Moondream31

        self.model = Moondream31(
            vram_reserve_gb=float(cfg.vram_reserve_gb),
            max_resident_layers=cfg.get("max_resident_layers"),
            max_crops=int(cfg.max_crops),
        )

        self.tracker = RobotTracker(float(cfg.track_gate), int(cfg.track_max_lost))

    def _boxes_by_label(self, frame_bgr: npt.NDArray) -> dict:
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        width, height = image.size
        views = [(0, 0, width, height)]
        for grid in self.grids:
            views += tile_boxes(width, height, grid)

        found = {label: [] for label in self.labels}
        for view in views:
            self.model.encode_image(image.crop(view))  # once for all labels
            for label in self.labels:
                objects = self.model.detect(label, max_objects=self.max_objects)
                found[label] += [to_pixels(obj, view) for obj in objects]
        return {label: merge_boxes(boxes) for label, boxes in found.items()}

    def _is_ball_shaped(self, box: List[int]) -> bool:
        """With no ball in view the model may return lines, marks or robots instead."""
        width, height = box[2] - box[0], box[3] - box[1]
        return (
            max(width, height) <= self.ball_max_size_px
            and max(width, height) <= self.ball_max_aspect * max(min(width, height), 1)
        )

    def detect(self, frame_bgr: npt.NDArray) -> List[MoondreamObject]:
        """Run the model on one frame. Call it on consecutive strided frames only."""
        robots: List[Tuple[List[int], float]] = []
        objects: List[MoondreamObject] = []
        for label, boxes in self._boxes_by_label(frame_bgr).items():
            cls = self.labels[label]
            if cls == ROBOT_CLS:
                robots += boxes
            else:
                # Balls skip the tracker: they can move too far between runs to match reliably.
                objects += [
                    MoondreamObject(tuple(box), conf, cls, UNTRACKED_ID)
                    for box, conf in boxes
                    if self._is_ball_shaped(box)
                ]
        tracked = self.tracker.update([box for box, _ in robots])
        robot_objects = [
            MoondreamObject(tuple(box), conf, ROBOT_CLS, track_id)
            for (track_id, box), (_, conf) in zip(tracked, robots)
        ]
        return robot_objects + objects
