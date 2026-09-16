"""
Realtime debug UI and ball-trail state — **not** part of core tracking.

``Tracking`` produces per-frame detections + CSV rows; this module owns anything that is
only for display (OpenCV windows, possession ring, ball speed trail). Heavier temporal
analysis should live in a separate module, not inside :class:`~src.core.tracker.Tracking`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Deque, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from ..analysis.field_reprojection_view import FIELD_REPROJECTION_WINDOW_NAME, render_field_reprojection
from ..commentator.commentary_display import (
    COMMENTARY_DEBUG_WINDOW_NAME,
    BallPossessionSnapshot,
    compute_ball_possession,
    draw_ball_possession_ring,
    render_commentary_panel,
    summarize_event,
)

if TYPE_CHECKING:
    from ..utils.config_loader import MarioConfig
    from ..utils.drawing import Drawer
    from .detection import Detection


@dataclass
class BallDebugFrame:
    """Ball overlay metrics for one frame (field panel + async job)."""

    speed_mms: float
    ball_px: Optional[Tuple[int, int]]
    ball_conf: Optional[float]
    trail: List[Tuple[int, int]]


class RealtimeDisplayState:
    """
    Mutable state for ball trail / last commentary label. Call from the tracking loop
    after detections are finalized for the frame.
    """

    def __init__(self, config: "MarioConfig") -> None:
        self._config = config
        self._ball_trail: Deque[Tuple[int, int]] = deque(
            maxlen=config.features.ball_debug_trail_frames
        )
        self._prev_ball_field: Optional[Tuple[float, float]] = None
        self._prev_ball_frame: Optional[int] = None
        self._last_commentary_event_label: str = "- (no event)"

    def clear_ball_trail(self) -> None:
        """Call when ball is rejected (e.g. OOB) and trail should reset."""
        self._ball_trail.clear()

    def update_ball_debug(
        self,
        frame_id: int,
        fps: float,
        detections: List["Detection"],
        plan_view_drawer: "Drawer",
        show_ball_debug: bool,
    ) -> BallDebugFrame:
        """Update trail / speed; returns metrics for field-reprojection window and jobs."""
        ball_speed_mms = 0.0
        ball_px: Optional[Tuple[int, int]] = None
        ball_conf: Optional[float] = None
        if not show_ball_debug:
            return BallDebugFrame(0.0, None, None, list(self._ball_trail))

        for det in detections:
            if det.cls_name == "ball" and det.position is not None:
                ball_fx, ball_fy = det.position
                ball_conf = float(det.confidence)
                plan_pos = plan_view_drawer.position_to_image_pipeline(ball_fx, ball_fy)
                ball_px = (int(plan_pos[0]), int(plan_pos[1]))
                if self._prev_ball_field is not None and self._prev_ball_frame is not None:
                    df = frame_id - self._prev_ball_frame
                    if df > 0:
                        dt = float(df) / float(fps)
                        p0x, p0y = self._prev_ball_field
                        ball_speed_mms = float(np.hypot(ball_fx - p0x, ball_fy - p0y) / dt)
                self._prev_ball_field = (float(ball_fx), float(ball_fy))
                self._prev_ball_frame = frame_id
                self._ball_trail.append(ball_px)
                break
        else:
            self._prev_ball_field = None
            self._prev_ball_frame = None

        return BallDebugFrame(ball_speed_mms, ball_px, ball_conf, list(self._ball_trail))

    def possession_snapshot(
        self,
        detections: List["Detection"],
        show_window: bool,
        robot_bgr: Callable[[int], Tuple[int, int, int]],
    ) -> Optional[BallPossessionSnapshot]:
        if not show_window:
            return None
        feats = self._config.features
        # TODO this parameter is no longer under this config. match_commentary_tts should be writing it anyway, but it's kind of an antipattern that should be fixed.
        left_lab = str(
            getattr(feats, "left_team_full_name", None) or getattr(feats, "left_team_name", "left")
        )
        right_lab = str(
            getattr(feats, "right_team_full_name", None) or getattr(feats, "right_team_name", "right")
        )
        if self._config.commentating is None:
            return compute_ball_possession(
                detections,
                robot_bgr=robot_bgr,
                left_team_label=left_lab,
                right_team_label=right_lab,
            )
        max_field = self._config.commentating.possession_display_max_field_mm
        max_img = self._config.commentating.possession_display_max_image_px
        contested = self._config.commentating.possession_display_contested_margin_px
        return compute_ball_possession(
            detections,
            robot_bgr=robot_bgr,
            left_team_label=left_lab,
            right_team_label=right_lab,
            max_field_dist_mm=float(max_field),
            max_image_dist_px=float(max_img) if max_img > 0 else None,
            contested_image_margin_px=float(contested) if contested > 0 else None,
        )

    def commentary_labels(
        self,
        *,
        events: Optional[object],
        commentate: bool,
        eventproc: object,
        commentator: Optional[object],
        match_tts: Optional[object],
        llm: Optional[object],
        show_window: bool,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Returns ``(event_label, text, ball_motion_debug)`` for the debug window; all ``None`` if window off."""
        if not show_window:
            return None, None, None
        motion_debug: Optional[str] = None
        if commentate and commentator is not None:
            if events is not None and match_tts is not None:
                match_tts.maybe_llm_commentary(events, llm, eventproc=eventproc, commentator=commentator)
            if events is not None:
                # Always reflect *this* frame — do not stick to the last non-idle label (misleading vs possession / "no ball").
                self._last_commentary_event_label = summarize_event(events)
            motion_debug = eventproc.motion_debug_line()
        label = self._last_commentary_event_label
        text = (
            match_tts.last_commentary_line_for_display()
            if match_tts is not None
            else "-"
        )
        return label, text, motion_debug


def draw_track_dots_and_possession_ring(
    annotated_frame,
    plan_view,
    detections: List["Detection"],
    plan_view_drawer: "Drawer",
    *,
    ball_default_bgr: Tuple[int, int, int],
    robot_bgr: Callable[[int], Tuple[int, int, int]],
    possession: Optional[BallPossessionSnapshot],
) -> None:
    """Foot/center dots on camera + plan view; possession-colored ball + ring. In-place."""
    for det in detections:
        position = (det.box.xc, det.box.y2)
        field_x, field_y = det.position if det.position is not None else (None, None)
        if det.cls_name == "ball":
            color = possession.jersey_bgr if possession is not None else ball_default_bgr
        else:
            color = robot_bgr(det.id)
        cv2.circle(annotated_frame, (int(position[0]), int(position[1])), 7, color, -1)
        if field_x is not None and field_y is not None:
            plan_pos = plan_view_drawer.position_to_image_pipeline(field_x, field_y)
            cv2.circle(plan_view, plan_pos, 7, color, -1)
    draw_ball_possession_ring(annotated_frame, detections, possession)


def show_field_reprojection_window(
    plan_view_drawer: "Drawer",
    calibr_dst_shape_hw: Tuple[int, int],
    frame_id: int,
    ball_debug: BallDebugFrame,
    detections: List["Detection"],
    robot_bgr: Callable[[int], Tuple[int, int, int]],
) -> None:
    robots_field = [
        (float(d.position[0]), float(d.position[1]), robot_bgr(d.id))
        for d in detections
        if d.cls_name == "robot" and d.position is not None
    ]
    panel = render_field_reprojection(
        plan_view_drawer,
        (calibr_dst_shape_hw[0], calibr_dst_shape_hw[1]),
        frame_id=frame_id,
        ball_trail=ball_debug.trail,
        ball_px=ball_debug.ball_px,
        ball_speed_mms=ball_debug.speed_mms,
        ball_conf=ball_debug.ball_conf,
        robots_field=robots_field or None,
    )
    cv2.imshow(FIELD_REPROJECTION_WINDOW_NAME, panel)


def show_debug_window(
    event_label: str,
    commentary_text: str,
    possession: Optional[BallPossessionSnapshot],
    ball_motion_debug: Optional[str] = None,
) -> None:
    comm_panel = render_commentary_panel(
        event_label, commentary_text, possession, ball_motion_debug=ball_motion_debug
    )
    cv2.imshow(COMMENTARY_DEBUG_WINDOW_NAME, comm_panel)