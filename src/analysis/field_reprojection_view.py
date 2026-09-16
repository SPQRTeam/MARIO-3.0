"""
Top-down schematic field view: white background, black pitch lines, overlays (ball trail, robots, …).
Uses the same field→image mapping as plan_view (Drawer with H_inv=None).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import numpy.typing as npt

from ..utils.drawing import Drawer, TEAM_COLORS

FIELD_REPROJECTION_WINDOW_NAME = "Field reprojection"

# BGR
_TEXT_COLOR = (60, 60, 60)
_BALL_TRAIL_POLY = (0, 220, 255)
_BALL_TRAIL_DOTS = (0, 180, 255)
_BALL_RING = (0, 100, 255)  # orange BGR, visible on white


def render_field_reprojection(
    drawer: Drawer,
    dest_hw: Tuple[int, int],
    *,
    frame_id: int,
    ball_trail: Sequence[Tuple[int, int]],
    ball_px: Optional[Tuple[int, int]],
    ball_speed_mms: float,
    ball_conf: Optional[float],
    robots_field: Optional[Sequence[Tuple[float, float, Tuple[int, int, int]]]] = None,
) -> npt.NDArray[np.uint8]:
    """
    Build a BGR image: white canvas, black field lines, then overlays.

    robots_field: optional sequence of (field_x_mm, field_y_mm, bgr) per robot to plot.
    """
    h, w = dest_hw[0], dest_hw[1]
    panel = np.full((h, w, 3), 255, dtype=np.uint8)
    drawer.warped_field(panel, line_color=(0, 0, 0), thickness=2)

    if robots_field:
        for fx, fy, color_name in robots_field:
            px = drawer.position_to_image_pipeline(float(fx), float(fy))
            cv2.circle(panel, (int(px[0]), int(px[1])), 7, TEAM_COLORS[color_name], -1)

    if len(ball_trail) >= 2:
        pts = np.array(list(ball_trail), dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(panel, [pts], False, _BALL_TRAIL_POLY, 2)
    for p in ball_trail:
        cv2.circle(panel, (int(p[0]), int(p[1])), 3, _BALL_TRAIL_DOTS, -1)
    if ball_px is not None:
        cv2.circle(panel, ball_px, 10, _BALL_RING, 2)

    y = 22
    lines: List[str] = [f"frame {frame_id}"]
    if ball_px is not None:
        lines.append(f"speed {ball_speed_mms:.0f} mm/s")
    else:
        lines.append("no ball")
    if ball_conf is not None:
        lines.append(f"conf {ball_conf:.2f}")
    for line in lines:
        cv2.putText(
            panel,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            _TEXT_COLOR,
            2,
            lineType=cv2.LINE_AA,
        )
        y += 24

    return panel
