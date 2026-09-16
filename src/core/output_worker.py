"""
Background output pipeline: plan view, annotation, video/CSV write, OpenCV windows.

Runs in a dedicated thread while the main thread stays on detection + geometry + CNN.
Uses a bounded queue so if encoding/UI falls behind, the pipeline applies backpressure
(main blocks on ``queue.put``) — output files stay frame-complete, no silent drops.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import cv2

from ..utils.opencv_qt_fonts import ensure_opencv_qt_fonts
import src.utils.annotation as annotation

ensure_opencv_qt_fonts()

import numpy as np
import numpy.typing as npt

from ..commentator.commentary_display import BallPossessionSnapshot
from .realtime_display import (
    draw_track_dots_and_possession_ring,
    show_debug_window,
    show_field_reprojection_window,
)
from ..utils.drawing import TEAM_COLORS


@dataclass
class OutputJob:
    frame_id: int
    frame: npt.NDArray[np.uint8]
    detections: List[Any]  # List[Detection]
    rows: list
    reading: Any
    # Field reprojection window (optional)
    ball_debug: any
    # Optional: debug window (event + commentary + possession).
    commentary_event_label: Optional[str] = None
    commentary_text: Optional[str] = None
    possession_snapshot: Optional[BallPossessionSnapshot] = None
    ball_motion_debug: Optional[str] = None


def do_output_work(
    job,
    calibration,
    multiwriter,
    config,
    logger,
    *,
    tracker: Any,
    dst_hw: Tuple,
    plan_view_drawer: Any,
) -> None:
    # TODO questa andrebbe spostata nel livello superiore ora, deve essere uno stato persistente
    ui_enabled = config.features.show_preview or config.features.show_ball_debug or config.features.show_commentary_window

    t0 = time.perf_counter()
    plan_view = calibration.antiproject_image(job.frame, dst_hw)
    plan_view_drawer.warped_field(plan_view)
    plan_ms = (time.perf_counter() - t0) * 1000.0

    t_ann = time.perf_counter()
    annotated_frame = annotation.elaborate_track_results_and_annotate(
        job.frame,
        detections=job.detections,
        max_tracks_display=tracker.config.processing.max_tracks_display,
        get_robot_color_name=tracker._robot_color,
    )
    annotate_ms = (time.perf_counter() - t_ann) * 1000.0

    t_draw = time.perf_counter()
    draw_track_dots_and_possession_ring(
        annotated_frame,
        plan_view,
        job.detections,
        plan_view_drawer,
        ball_default_bgr=annotation.BALL_COLOR,
        robot_bgr=lambda track_id: TEAM_COLORS[tracker._robot_color(track_id)],
        possession=job.possession_snapshot,
    )
    draw_overlays_ms = (time.perf_counter() - t_draw) * 1000.0

    field_reproj_ms = 0.0
    if ui_enabled and config.features.show_ball_debug:
        t_fr = time.perf_counter()
        try:
            show_field_reprojection_window(
                plan_view_drawer,
                dst_hw,
                job.frame_id,
                job.ball_debug,
                job.detections,
                tracker._robot_color,
            )
        except cv2.error as ex:
            ui_enabled = False
            logger.warning(
                "OpenCV GUI unavailable (imshow failed: %s). Disabling preview windows and continuing headless.",
                ex,
            )
        field_reproj_ms = (time.perf_counter() - t_fr) * 1000.0

    if ui_enabled and config.features.show_commentary_window:
        try:
            show_debug_window(
                job.commentary_event_label if job.commentary_event_label is not None else "-",
                job.commentary_text if job.commentary_text is not None else "-",
                job.possession_snapshot,
                ball_motion_debug=job.ball_motion_debug,
            )
        except cv2.error as ex:
            ui_enabled = False
            logger.warning(
                "OpenCV GUI unavailable (imshow failed: %s). Disabling preview windows and continuing headless.",
                ex,
            )

    t_out = time.perf_counter()
    multiwriter.multiwrite(annotated_frame, plan_view, job.rows)
    write_out_ms = (time.perf_counter() - t_out) * 1000.0

    t_prev = time.perf_counter()
    if ui_enabled and config.features.show_preview:
        try:
            _af = cv2.resize(annotated_frame, (1280, 720))
            cv2.imshow("Tracking", _af)
            cv2.imshow("plan_view", plan_view)
        except cv2.error as ex:
            ui_enabled = False
            logger.warning(
                "OpenCV GUI unavailable (imshow failed: %s). Disabling preview windows and continuing headless.",
                ex,
            )
    preview_ms = (time.perf_counter() - t_prev) * 1000.0

    t_wk = time.perf_counter()
    if ui_enabled:
        try:
            cv2.waitKey(1)
        except cv2.error as ex:
            ui_enabled = False
            logger.warning(
                "OpenCV GUI unavailable (waitKey failed: %s). Disabling preview windows and continuing headless.",
                ex,
            )
    waitkey_ms = (time.perf_counter() - t_wk) * 1000.0

    total_work_ms = (time.perf_counter() - t0) * 1000.0

    return OutputWorkTimes(total_work_ms, plan_ms, annotate_ms, draw_overlays_ms, field_reproj_ms, write_out_ms, preview_ms, waitkey_ms)

@dataclass
class OutputWorkTimes:
    total_work_ms: float
    plan_ms: float
    annotate_ms: float
    draw_overlays_ms: float
    field_reproj_ms: float
    write_out_ms: float
    preview_ms: float
    waitkey_ms: float
    