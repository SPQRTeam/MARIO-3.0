"""Detection dataclass and frame iterator for video processing."""

import dataclasses
import queue
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Iterator, Tuple, Optional
import cv2

from ..utils.opencv_qt_fonts import ensure_opencv_qt_fonts

ensure_opencv_qt_fonts()

import numpy.typing as npt
from pathlib import Path

from ..vision.bounding_box import BoundingBox
from ..streaming import cap_from_youtube, parse_deltas
from ..commentator.commentary_labels import L_GOAL
from ..commentator.events_commentary import (
    spoken_line_for_ocr_goal,
)

import logging
logger = logging.getLogger(__name__)


def _box_area(box: BoundingBox) -> float:
    return float(box.w * box.h)


def _intersection_area(a: BoundingBox, b: BoundingBox) -> float:
    xi1 = max(a.x1, b.x1)
    yi1 = max(a.y1, b.y1)
    xi2 = min(a.x2, b.x2)
    yi2 = min(a.y2, b.y2)
    iw = max(0.0, float(xi2 - xi1))
    ih = max(0.0, float(yi2 - yi1))
    return iw * ih


def suppress_nested_robot_detections(
    detections: List["Detection"],
    min_overlap_of_smaller: float = 0.5,
    max_area_ratio: float = 0.85,
) -> List["Detection"]:
    """
    Drop smaller robot boxes that largely overlap a larger one (same physical robot:
    e.g. torso + full body). Keeps the largest box per cluster.

    A candidate *small* is removed only if some already-kept *large* has strictly
    greater area, intersection(small,large)/area(small) >= min_overlap_of_smaller,
    and area(small)/area(large) <= max_area_ratio. This avoids merging two nearby
    robots of similar size.
    """
    robots = [d for d in detections if d.cls_name == "robot"]
    rest = [d for d in detections if d.cls_name != "robot"]
    if len(robots) <= 1:
        return detections

    robots.sort(key=lambda d: _box_area(d.box), reverse=True)
    kept: List[Detection] = []
    for r in robots:
        ar = _box_area(r.box)
        drop = False
        for k in kept:
            ak = _box_area(k.box)
            if ak <= ar:
                continue
            inter = _intersection_area(r.box, k.box)
            if ar <= 0:
                continue
            overlap_frac = inter / ar
            size_ratio = ar / ak
            if (
                overlap_frac >= float(min_overlap_of_smaller)
                and size_ratio <= float(max_area_ratio)
            ):
                drop = True
                break
        if not drop:
            kept.append(r)

    return rest + kept

# measured in frames
OCR_PERIOD = 30

@dataclass
class Detection:
    """
    Single object detection with tracking information.

    Attributes:
        box: Bounding box of the detection.
        confidence: Detection confidence score (0-1).
        cls: Class ID (0=robot, 1=ball).
        cls_name: Class name string.
        id: Track ID from ByteTrack.
        color: Predicted color (for robots), None or "white" for ball.
        position: Field coordinates (x, y) after homography transform.
    """
    box: BoundingBox
    confidence: float
    cls: int
    cls_name: str
    id: int
    color: Optional[str] = None
    position: Optional[Tuple[float, float]] = None

    @staticmethod
    def from_results(results) -> List["Detection"]:
        """
        Convert YOLO results to Detection objects.

        Args:
            results: YOLO detection results.

        Returns:
            List of Detection objects.
        """
        result = results[0]
        boxes_xywh = result.boxes.xywh.cpu().numpy()
        boxes = [
            BoundingBox.from_xcycwh(b[0], b[1], b[2], b[3])
            for b in boxes_xywh
        ]
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        class_names = ["robot" if c == 0 else "ball" for c in classes]

        # YOLOv12 with integrated tracker has IDs, YOLOv8 does not
        if result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy().astype(int)
        else:
            ids = [-1 for _ in boxes]

        return [
            Detection(*t)
            for t in zip(boxes, confidences, classes, class_names, ids)
        ]

    @staticmethod
    def from_moondream(objects) -> List["Detection"]:
        """Convert ``MoondreamDetector.detect`` output to Detection objects."""
        detections = []
        for (x1, y1, x2, y2), confidence, cls, track_id in objects:
            box = BoundingBox.from_xcycwh((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
            cls_name = "robot" if cls == 0 else "ball"
            detections.append(Detection(box, confidence, cls, cls_name, track_id))
        return detections

    def __repr__(self) -> str:
        """String representation of the detection."""
        return (
            f"Detection(id={self.id}, cls={self.cls_name}, "
            f"conf={self.confidence:.2f}, color={self.color})"
        )


def _interpolated(start: Detection, end: Detection, t: float) -> Detection:
    """Copy of ``end`` with its box at fraction ``t`` of the way from ``start``'s box."""
    x1, y1, x2, y2 = (
        a + (b - a) * t
        for a, b in (
            (start.box.x1, end.box.x1),
            (start.box.y1, end.box.y1),
            (start.box.x2, end.box.x2),
            (start.box.y2, end.box.y2),
        )
    )
    box = BoundingBox.from_xcycwh((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
    return dataclasses.replace(end, box=box)


def interpolate_detections(start: List[Detection], end: List[Detection], t: float) -> List[Detection]:
    """
    Detections at fraction ``t`` (0-1] of the way between two model runs.

    Robots pair up by track ID and balls by distance. Objects seen only at the start
    stay put until the end run, and objects seen only at the end appear at ``t == 1``.
    """
    if t >= 1:
        return [dataclasses.replace(det) for det in end]

    start_robots = {det.id: det for det in start if det.cls_name == "robot"}
    start_balls = [det for det in start if det.cls_name == "ball"]
    end_robots = [det for det in end if det.cls_name == "robot"]
    end_balls = [det for det in end if det.cls_name == "ball"]

    result = [
        _interpolated(start_robots[det.id], det, t) for det in end_robots if det.id in start_robots
    ]
    end_ids = {det.id for det in end_robots}
    result += [dataclasses.replace(det) for tid, det in start_robots.items() if tid not in end_ids]

    if start_balls:
        for ball in end_balls:
            nearest = min(
                start_balls,
                key=lambda b: (b.box.xc - ball.box.xc) ** 2 + (b.box.y2 - ball.box.y2) ** 2,
            )
            result.append(_interpolated(nearest, ball, t))
        if not end_balls:
            result += [dataclasses.replace(det) for det in start_balls]
    return result


@dataclass
class OverlayReading:
    score_left: int
    score_right: int

    @classmethod
    def null_value(cls):
        return cls(-1, -1)


class FrameIteratorBase:
    """
    Iterator for processing video frames with YOLO detection models.

    Uses YOLOv12 with ByteTrack for robot detection and tracking,
    and YOLOv8 for ball detection.
    """

    _moondream_thread: Optional[threading.Thread] = None

    def __init__(
        self,
        models: any,
        calibration_frame_offset: int,
        bytetrack_config: Path = None,
        use_v12_only: bool = False,
        robot_nested_suppression: Optional[dict] = None,
        mario_config: Any = None,
        team_map: Any = None,
    ):
        """
        Initialize frame iterator with YOLO models.

        Args:
            video_file: Path to input video file.
            models: object with yolo_v8 (ball) and yolo_v12 (robots) attributes.
            bytetrack_config: Path to ByteTrack configuration YAML.
            robot_nested_suppression: Optional dict with ``enabled``, ``min_overlap_of_smaller``,
                ``max_area_ratio`` (see config.yaml).
            mario_config: Full MARIO config (team names + ``commentating`` OCR mapping for goals).
        """
        self.mario_config = mario_config
        self.team_map = team_map
        self._w = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self._h = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.yolo_v8 = models.yolo_v8
        self.yolo_v12 = models.yolo_v12
        self.moondream = models.moondream
        self._moondream_anchor: Optional[List[Detection]] = None  # latest model run
        self._moondream_queue: Optional[queue.Queue] = None
        self._moondream_stop = threading.Event()
        self.ocr_vlm = models.ocr_vlm
        self.llm = models.llm
        self.bytetrack_config = str(bytetrack_config) if bytetrack_config else None
        self.use_v12_only = use_v12_only
        self._robot_nested_suppression = robot_nested_suppression or {}
        self.match_tts = models.match_tts

        # Filled each ``__next__`` for optional per-frame profiling (see tracker ``debug_frame_timings``).
        self.last_iter_timings: Dict[str, float] = {}

        self.last_reading = OverlayReading.null_value()
        self._current_frame = 0
        self._ocr_lock = threading.Lock()

        # OCR crop boxes: read from config so they can be calibrated per video.
        # Format: [y1, y2, x1, x2] in pixels at the video's native resolution.
        self._ocr_crop_left = mario_config.commentating.ocr_crop_left
        self._ocr_crop_right = mario_config.commentating.ocr_crop_right
        self._show_debug_ocr = mario_config.commentating.show_debug_ocr
        logger.info("OCR debug windows enabled: %s", self._show_debug_ocr)

        self.enable_ocr = mario_config.features.enable_ocr_score
        self._ocr_epoch = 0
        self._ocr_in_flight = False  # True while a VLM pair is pending; skip new cycles

        # Get calibration frame
        initial_frame = self.video.get(cv2.CAP_PROP_POS_FRAMES)
        self.video.set(cv2.CAP_PROP_POS_FRAMES, initial_frame + calibration_frame_offset)
        ret, self.calibration_frame = self.video.read()
        self.video.set(cv2.CAP_PROP_POS_FRAMES, initial_frame)

        if not ret:
            raise ValueError(
                f"Cannot read from video.\n"
                f"The video file may be corrupted or in an unsupported format, or it may be a streaming issue if in streaming mode."
            )

    @property
    def frame_width(self):
        return self._w

    @property
    def frame_height(self):
        return self._h

    def seek(self, frame_idx: int):
        """Seek to a specific frame."""
        self._stop_moondream_prefetch()
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self._current_frame = frame_idx
        self._moondream_anchor = None

    def _read_moondream_window(self) -> List[Tuple[npt.NDArray, List[Detection]]]:
        """
        Read up to ``stride`` frames, run Moondream on the last one, and return them all
        with boxes interpolated from the previous run.
        """
        # The very first frame has no previous run to interpolate from.
        window_size = self.moondream.stride if self._moondream_anchor is not None else 1
        frames = []
        for _ in range(window_size):
            ret, frame = self.video.read()
            if not ret:
                break
            frames.append(frame)
        if not frames:
            return []

        end = Detection.from_moondream(self.moondream.detect(frames[-1]))
        start = self._moondream_anchor if self._moondream_anchor is not None else end
        self._moondream_anchor = end
        return [
            (frame, interpolate_detections(start, end, i / len(frames)))
            for i, frame in enumerate(frames, start=1)
        ]

    def _put_moondream_item(self, item: Any) -> None:
        while not self._moondream_stop.is_set():
            try:
                self._moondream_queue.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    def _prefetch_moondream(self) -> None:
        """Background loop: keeps the model busy while the main thread handles frames."""
        try:
            while not self._moondream_stop.is_set():
                window = self._read_moondream_window()
                if not window:
                    break
                for item in window:
                    self._put_moondream_item(item)
            self._put_moondream_item(None)
        except Exception as e:
            self._put_moondream_item(e)

    def _stop_moondream_prefetch(self) -> None:
        if self._moondream_thread is not None:
            self._moondream_stop.set()
            self._moondream_thread.join()
            self._moondream_thread = None

    def _next_moondream_frame(self) -> Tuple[bool, Optional[npt.NDArray], List[Detection]]:
        if self._moondream_thread is None:
            self._moondream_queue = queue.Queue(maxsize=2 * self.moondream.stride)
            self._moondream_stop.clear()
            self._moondream_thread = threading.Thread(target=self._prefetch_moondream, daemon=True)
            self._moondream_thread.start()

        item = self._moondream_queue.get()
        if item is None:
            return False, None, []
        if isinstance(item, Exception):
            raise item
        frame, detections = item
        return True, frame, detections

    def __iter__(self) -> Iterator[Tuple[npt.NDArray, List[Detection]]]:
        """Return iterator."""
        return self

    def __len__(self) -> int:
        """Total number of frames in the video."""
        return self._len

    def __next__(self) -> Tuple[npt.NDArray, List[Detection]]:
        """
        Process next frame and return detections.

        Returns:
            Tuple of (frame, detections).

        Raises:
            StopIteration: When video ends.
        """
        t0 = time.perf_counter()
        if self.moondream is not None:
            # The model runs on a background thread: waiting for it shows up in video_read_ms.
            ret, frame, detections = self._next_moondream_frame()
        else:
            ret, frame = self.video.read()
        t1 = time.perf_counter()
        video_read_ms = (t1 - t0) * 1000.0

        if not ret:
            logger.info(f"Finished processing {self._current_frame} frames")
            raise StopIteration
        
        if self._current_frame == self._len:
            logger.info(f"Reached expected length at {self._current_frame} frames")
            raise StopIteration

        t2 = time.perf_counter()
        yolo_v12_ms = 0.0
        yolo_v8_ms = 0.0
        if self.moondream is None:
            # YOLOv12 for robot detection with ByteTrack
            results_v12 = self.yolo_v12.track(
                frame,
                verbose=False,
                tracker=self.bytetrack_config,
                persist=True
            ) if self.bytetrack_config else self.yolo_v12.predict(frame, verbose=False)
            detections_v12 = Detection.from_results(results_v12)
            t3 = time.perf_counter()
            yolo_v12_ms = (t3 - t2) * 1000.0

            if self.use_v12_only:
                detections = detections_v12
            else:
                # YOLOv8 for ball detection, fallback to v12 if missed
                t4 = time.perf_counter()
                results_v8 = self.yolo_v8(frame, verbose=False)
                detections_v8 = Detection.from_results(results_v8)
                balls_v8 = [det for det in detections_v8 if det.cls_name == "ball"]
                balls = balls_v8 if balls_v8 else [det for det in detections_v12 if det.cls_name == "ball"]
                detections = [det for det in detections_v12 if det.cls_name == "robot"] + balls
                yolo_v8_ms = (time.perf_counter() - t4) * 1000.0

        t5 = time.perf_counter()
        rns = self._robot_nested_suppression
        if rns.get("enabled"):
            detections = suppress_nested_robot_detections(
                detections,
                min_overlap_of_smaller=float(rns.get("min_overlap_of_smaller", 0.5)),
                max_area_ratio=float(rns.get("max_area_ratio", 0.85)),
            )
        nested_suppress_ms = (time.perf_counter() - t5) * 1000.0

        ocr_schedule_ms = 0.0
        if self.enable_ocr and self._current_frame % OCR_PERIOD == 0 and not self._ocr_in_flight:
            t_ocr = time.perf_counter()
            try:
                ly1, ly2, lx1, lx2 = self._ocr_crop_left
                ry1, ry2, rx1, rx2 = self._ocr_crop_right
                crop_left = frame[ly1:ly2, lx1:lx2].copy()
                crop_right = frame[ry1:ry2, rx1:rx2].copy()
                if self._show_debug_ocr:
                    logger.info(
                        f"[OCR] schedule frame={self._current_frame} "
                        f"crops left=y{ly1}:{ly2},x{lx1}:{lx2} "
                        f"right=y{ry1}:{ry2},x{rx1}:{rx2}",
                    )
                if crop_left.size == 0 or crop_right.size == 0:
                    logger.warning(
                        "OCR crop is empty — check ocr_crop_left/right in commentating.yaml "
                        "(frame size: %dx%d)", int(self._h), int(self._w)
                    )
                with self._ocr_lock:
                    sl = self.last_reading.score_left
                    sr = self.last_reading.score_right
                prompt_l = (
                    f"Is the number in the image {sl} or {sl+1}? Return ONLY the digit. No words, no punctuation."
                    if sl >= 0
                    else "Extract the single number from this image. Return ONLY the digit. No words, no punctuation."
                )
                prompt_r = (
                    f"Is the number in the image {sr} or {sr+1}? Return ONLY the digit. No words, no punctuation."
                    if sr >= 0
                    else "Extract the single number from this image. Return ONLY the digit. No words, no punctuation."
                )

                def _parse_score(val: str):
                    m = re.search(r"\d+", val)
                    return int(m.group()) if m else None

                if self._show_debug_ocr:
                    try:
                        cv2.imshow("OCR left", crop_left)
                        cv2.imshow("OCR right", crop_right)
                        cv2.waitKey(1)
                    except cv2.error:
                        # Wayland / headless: fallback to disk
                        cv2.imwrite("/tmp/mario_ocr_left.png", crop_left)
                        cv2.imwrite("/tmp/mario_ocr_right.png", crop_right)
                        logger.info("OCR debug (no display) → /tmp/mario_ocr_left.png  /tmp/mario_ocr_right.png")

                with self._ocr_lock:
                    self._ocr_epoch += 1
                    self._ocr_in_flight = True

                state = {"remaining": 2, "raw_l": "", "raw_r": ""}

                def _parse_and_llm_after_both():
                    # Snapshot previous score, then update with the freshly OCR'd one.
                    with self._ocr_lock:
                        raw_l, raw_r = state["raw_l"], state["raw_r"]
                        # logger.info(f"[OCR raw] left='{raw_l}' right='{raw_r}'")
                        sl_prev = self.last_reading.score_left
                        sr_prev = self.last_reading.score_right
                        n = _parse_score(raw_l)
                        if n is not None:
                            self.last_reading.score_left = n
                        n = _parse_score(raw_r)
                        if n is not None:
                            self.last_reading.score_right = n
                        sl_new = self.last_reading.score_left
                        sr_new = self.last_reading.score_right
                        # logger.info(f"[OCR] left {sl_new} - right {sr_new}")

                        # Only call the LLM when the score actually changed (real goal).
                        # The per-frame Commentator handles all the other proactive lines.
                        left_scored = sl_prev >= 0 and sl_new > sl_prev
                        right_scored = sr_prev >= 0 and sr_new > sr_prev
                        if not (left_scored or right_scored):
                            return

                        if (
                            self.match_tts is not None
                            and self.mario_config.commentating.deterministic_goal_speech
                        ):
                            line = spoken_line_for_ocr_goal(left_scored, self.mario_config, self.team_map)
                            logger.info(f"[Commentary] {line}")
                            self.match_tts.enqueue_speak(
                                line, urgent=True, source_frame=None, label=L_GOAL
                            )
                            return

                        if left_scored:
                            scorer_side, conceded_side = "left", "right"
                        else:
                            scorer_side, conceded_side = "right", "left"
                        prompt = (
                            "You are a live football play-by-play commentator. Reply with ONE "
                            "short, energetic English sentence, no preamble, no emojis. "
                            f"GOAL! The {scorer_side}-side team just scored against the "
                            f"{conceded_side}-side team. The score is now "
                            f"left {sl_new} - right {sr_new}. Celebrate the goal."
                        )

                    def _on_score_commentary(c: str) -> None:
                        logger.info(f"[LLM] {c}")
                        if self.match_tts is not None:
                            self.match_tts.enqueue_speak(
                                c, urgent=True, source_frame=None, label=L_GOAL
                            )

                    self.llm.llm_call_async(prompt, on_done=_on_score_commentary)

                def _on_ocr_done(fut, side: str):
                    try:
                        raw = fut.result()
                    except Exception as ex:
                        logger.warning("VLM OCR call failed: %s", ex)
                        raw = ""
                    with self._ocr_lock:
                        if side == "left":
                            state["raw_l"] = raw
                        else:
                            state["raw_r"] = raw
                        state["remaining"] -= 1
                        if state["remaining"] != 0:
                            return
                        # Both futures done: allow next OCR cycle to be scheduled.
                        self._ocr_in_flight = False
                    _parse_and_llm_after_both()

                fut_l = self.ocr_vlm.vlm_call_async(prompt_l, crop_left)
                fut_r = self.ocr_vlm.vlm_call_async(prompt_r, crop_right)
                fut_l.add_done_callback(lambda f: _on_ocr_done(f, "left"))
                fut_r.add_done_callback(lambda f: _on_ocr_done(f, "right"))
            except Exception as e:
                logger.warning(f"OCR schedule failed: {e}")
            ocr_schedule_ms = (time.perf_counter() - t_ocr) * 1000.0

        det_sum = (
            video_read_ms
            + yolo_v12_ms
            + yolo_v8_ms
            + nested_suppress_ms
            + ocr_schedule_ms
        )
        self.last_iter_timings = {
            "video_read_ms": video_read_ms,
            "yolo_v12_ms": yolo_v12_ms,
            "yolo_v8_ms": yolo_v8_ms,
            "nested_suppress_ms": nested_suppress_ms,
            "ocr_schedule_ms": ocr_schedule_ms,
            "detector_total_ms": det_sum,
        }

        self._current_frame += 1
        with self._ocr_lock:
            reading_snapshot = OverlayReading(**dataclasses.asdict(self.last_reading))
        return frame, detections, reading_snapshot

    def release(self):
        """Release video capture resource."""
        self._stop_moondream_prefetch()
        if self.video.isOpened():
            self.video.release()
            logger.info("Video capture released")

    def __del__(self):
        """Cleanup on deletion."""
        self.release()

class FrameIteratorFile(FrameIteratorBase):
    """
    Iterator for processing video frames with YOLO detection models.

    Uses YOLOv12 with ByteTrack for robot detection and tracking,
    and YOLOv8 for ball detection.
    """

    def __init__(
        self,
        video_file: Path,
        models: any,
        calibration_frame_offset: int,
        bytetrack_config: Path,
        use_v12_only: bool = False,
        robot_nested_suppression: Optional[dict] = None,
        mario_config: Any = None,
        team_map: Any = None,
    ):
        self.video = cv2.VideoCapture(str(video_file))

        if not self.video.isOpened():
            raise ValueError(f"Cannot open video file: {video_file}")

        self._len = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

        super().__init__(
            models,
            calibration_frame_offset,
            bytetrack_config,
            use_v12_only,
            robot_nested_suppression=robot_nested_suppression,
            mario_config=mario_config,
            team_map=team_map,
        )

        logger.info(f"Initialized FrameIterator for {video_file.name} ({self._len} frames)")

class FrameIteratorStreaming(FrameIteratorBase):

    def __init__(
        self,
        streaming_url,
        time_range,
        models,
        calibration_frame_offset,
        bytetrack_config,
        use_v12_only: bool = False,
        robot_nested_suppression: Optional[dict] = None,
        mario_config: Any = None,
        team_map: Any = None,
    ):
        start, end = parse_deltas(time_range)
        self.video = cap_from_youtube(streaming_url, "best", start)

        if not self.video.isOpened():
            raise ValueError(f"Cannot open video stream at: {streaming_url}")

        duration = (end - start).total_seconds()
        fps = self.video.get(cv2.CAP_PROP_FPS)
        self._len = int(duration * fps)

        super().__init__(
            models,
            calibration_frame_offset,
            bytetrack_config,
            use_v12_only,
            robot_nested_suppression=robot_nested_suppression,
            mario_config=mario_config,
            team_map=team_map,
        )

        logger.info(f"Initialized FrameIterator for {streaming_url} ({self._len} frames)")
