"""Realtime debug panel: events, last commentary line, ball possession (nearest robot → team)."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ..core.detection import Detection
    from .event_flags import EventFlags

COMMENTARY_DEBUG_WINDOW_NAME = "MARIO Debug"

_PANEL_W, _PANEL_H = 720, 520
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_SCALE = 0.55
_THICK = 1
_LINE_H = 24
_MARGIN = 16
_TEXT_COLOR = (30, 30, 30)
_TITLE_COLOR = (0, 0, 140)
_POSSESSION_TITLE_COLOR = (0, 100, 0)

_RING_RADIUS_PX = 40
_RING_THICKNESS = 4


@dataclass(frozen=True)
class BallPossessionSnapshot:
    """Robot closest to the ball in the **camera image**; ``dist_mm`` is the field gap for that pair."""

    robot_id: int
    dist_mm: float
    side: str  # "left" | "right"
    team_label: str
    jersey_bgr: Tuple[int, int, int]


def compute_ball_possession(
    detections: List["Detection"],
    *,
    robot_bgr: Callable[[int], Tuple[int, int, int]],
    left_team_label: str,
    right_team_label: str,
    max_field_dist_mm: Optional[float] = None,
    max_image_dist_px: Optional[float] = None,
    contested_image_margin_px: Optional[float] = None,
) -> Optional[BallPossessionSnapshot]:
    """
    Pick the robot **visually** closest to the ball (image pixels: ball bbox center vs robot feet).

    Field-only distance was misleading when homography places one robot’s feet wrong on the pitch
    while they still appear next to the ball on the video. Team label still uses field ``x`` of the
    chosen robot (half-line).

    If thresholds are set, returns ``None`` (neutral ball colour — no team ring) when:
    - no robot is within ``max_field_dist_mm`` on the field, and/or
    - no robot is within ``max_image_dist_px`` in the image, and/or
    - the two closest robots in the image are within ``contested_image_margin_px`` (ambiguous).
    """
    balls = [d for d in detections if d.cls_name == "ball" and d.position is not None]
    robots = [d for d in detections if d.cls_name == "robot" and d.position is not None]
    if not balls or not robots:
        return None
    ball = max(balls, key=lambda d: float(d.confidence))
    bx_img, by_img = float(ball.box.xc), float(ball.box.yc)
    img_dists: List[Tuple[float, "Detection"]] = []
    for r in robots:
        rx_img, ry_img = float(r.box.xc), float(r.box.y2)
        d_img = hypot(bx_img - rx_img, by_img - ry_img)
        img_dists.append((d_img, r))
    img_dists.sort(key=lambda t: t[0])
    best_d_img = img_dists[0][0]
    best = img_dists[0][1]
    second_d_img: Optional[float] = img_dists[1][0] if len(img_dists) > 1 else None

    if max_image_dist_px is not None and float(max_image_dist_px) > 0:
        if best_d_img > float(max_image_dist_px):
            return None
    if (
        contested_image_margin_px is not None
        and float(contested_image_margin_px) > 0
        and second_d_img is not None
        and (second_d_img - best_d_img) < float(contested_image_margin_px)
    ):
        return None

    bx, by = float(ball.position[0]), float(ball.position[1])
    rx_field = float(best.position[0])
    ry_field = float(best.position[1])
    best_d = hypot(bx - rx_field, by - ry_field)
    if max_field_dist_mm is not None and float(max_field_dist_mm) > 0:
        if best_d > float(max_field_dist_mm):
            return None
    side = "left" if rx_field < 0.0 else "right"
    team_label = left_team_label if side == "left" else right_team_label
    tid = int(best.id)
    bgr = robot_bgr(tid)
    if not (isinstance(bgr, tuple) and len(bgr) == 3):
        bgr = (128, 128, 128)
    return BallPossessionSnapshot(
        robot_id=tid,
        dist_mm=float(best_d),
        side=side,
        team_label=str(team_label),
        jersey_bgr=(int(bgr[0]), int(bgr[1]), int(bgr[2])),
    )


def draw_ball_possession_ring(
    frame: npt.NDArray[np.uint8],
    detections: List["Detection"],
    possession: Optional[BallPossessionSnapshot],
    *,
    radius_px: int = _RING_RADIUS_PX,
    thickness: int = _RING_THICKNESS,
) -> None:
    """Draw a colored ring around the ball (image center = bbox center). In-place."""
    if possession is None:
        return
    balls = [d for d in detections if d.cls_name == "ball"]
    if not balls:
        return
    ball = max(balls, key=lambda d: float(d.confidence))
    cx, cy = int(ball.box.xc), int(ball.box.yc)
    cv2.circle(frame, (cx, cy), radius_px, possession.jersey_bgr, thickness, lineType=cv2.LINE_AA)


def _display_line(s: Optional[str]) -> str:
    """Safe line for OpenCV; never show Python/JSON ``null``."""
    if s is None:
        return "-"
    t = str(s).strip()
    if not t or t.lower() in ("null", "none"):
        return "-"
    return t


def _ascii_for_opencv(s: str) -> str:
    if not s:
        return "-"
    return (
        s.replace("\u2014", "-")
        .replace("\u2013", "-")
        .replace("\u2026", "...")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )


def summarize_event(events: "EventFlags") -> str:
    """Short label for the strongest event this frame (goal > ball strike > shot > pass)."""
    e = events
    if e.goal_for_left_team:
        return "Goal (left scoreboard)"
    if e.goal_for_right_team:
        return "Goal (right scoreboard)"
    if e.away_team_ball_strike:
        return "Robot–ball strike (away)"
    if e.home_team_ball_strike:
        return "Robot–ball strike (home)"
    if e.away_team_shot:
        return "Shot (away team)"
    if e.home_team_shot:
        return "Shot (home team)"
    if e.away_team_pass:
        return "Pass (away team)"
    if e.home_team_pass:
        return "Pass (home team)"
    return "- (no event)"


def _wrap(text: str, max_chars: int = 92) -> List[str]:
    text = (text or "").strip() or "-"
    words = text.split()
    lines: List[str] = []
    cur = ""
    for w in words:
        cand = (cur + " " + w).strip()
        if len(cand) <= max_chars:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines[:14]


def render_commentary_panel(
    event_label: Optional[str],
    commentary_text: Optional[str],
    possession: Optional[BallPossessionSnapshot] = None,
    ball_motion_debug: Optional[str] = None,
) -> npt.NDArray[np.uint8]:
    """BGR image for ``cv2.imshow`` — event, TTS line, optional ball-motion line, possession."""
    img = np.full((_PANEL_H, _PANEL_W, 3), 255, dtype=np.uint8)
    y = _MARGIN + 20

    cv2.putText(
        img,
        "Event:",
        (_MARGIN, y),
        _FONT,
        0.65,
        _TITLE_COLOR,
        2,
        lineType=cv2.LINE_AA,
    )
    y += _LINE_H + 4
    ev = _ascii_for_opencv(_display_line(event_label)[:120])
    cv2.putText(
        img,
        ev,
        (_MARGIN, y),
        _FONT,
        _SCALE,
        _TEXT_COLOR,
        _THICK,
        lineType=cv2.LINE_AA,
    )
    y += _LINE_H + 18

    cv2.putText(
        img,
        "Commentary:",
        (_MARGIN, y),
        _FONT,
        0.65,
        _TITLE_COLOR,
        2,
        lineType=cv2.LINE_AA,
    )
    y += _LINE_H + 4
    for line in _wrap(_ascii_for_opencv(_display_line(commentary_text)))[:6]:
        cv2.putText(
            img,
            line,
            (_MARGIN, y),
            _FONT,
            _SCALE,
            _TEXT_COLOR,
            _THICK,
            lineType=cv2.LINE_AA,
        )
        y += _LINE_H

    if ball_motion_debug:
        y += 8
        cv2.putText(
            img,
            "Ball motion (analyze_ball):",
            (_MARGIN, y),
            _FONT,
            0.65,
            _TITLE_COLOR,
            2,
            lineType=cv2.LINE_AA,
        )
        y += _LINE_H + 4
        for line in _wrap(_ascii_for_opencv(_display_line(ball_motion_debug)))[:4]:
            cv2.putText(
                img,
                line,
                (_MARGIN, y),
                _FONT,
                _SCALE * 0.95,
                _TEXT_COLOR,
                _THICK,
                lineType=cv2.LINE_AA,
            )
            y += _LINE_H

    y += 8
    cv2.putText(
        img,
        "Possession (nearest robot, gated):",
        (_MARGIN, y),
        _FONT,
        0.65,
        _POSSESSION_TITLE_COLOR,
        2,
        lineType=cv2.LINE_AA,
    )
    y += _LINE_H + 4

    if possession is None:
        pos_lines = ["-- (no ball / no robots / no clear possession)"]
    else:
        pos_lines = [
            f"Team: {possession.team_label}  ({possession.side} half)",
            f"Track id: {possession.robot_id}   dist: {possession.dist_mm:.0f} mm",
            f"Jersey color (BGR): {possession.jersey_bgr}",
        ]
    for pl in pos_lines:
        cv2.putText(
            img,
            _ascii_for_opencv(pl[:100]),
            (_MARGIN, y),
            _FONT,
            _SCALE,
            _TEXT_COLOR,
            _THICK,
            lineType=cv2.LINE_AA,
        )
        y += _LINE_H

    if possession is not None:
        sw = 36
        x0 = _MARGIN
        y0 = min(y + 4, _PANEL_H - sw - 4)
        x1, y1 = x0 + sw, y0 + sw
        cv2.rectangle(img, (x0, y0), (x1, y1), (40, 40, 40), 1, lineType=cv2.LINE_AA)
        cv2.rectangle(
            img,
            (x0 + 2, y0 + 2),
            (x1 - 2, y1 - 2),
            possession.jersey_bgr,
            -1,
            lineType=cv2.LINE_AA,
        )

    return img
