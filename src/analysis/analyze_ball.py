"""
Ball motion: one commentary event — **robot–ball strike** (``home_team_ball_strike`` /
``away_team_ball_strike``). Internally we still branch pass-like vs goal-like direction to decide
*when* to emit, but we do not expose pass vs shot to the rest of the pipeline.

**Home** = left team (attacks the **right** goal at ``+xlimit``). **Away** = right team
(attacks **left** goal at ``-xlimit``).

1. Sliding window + **last_seen** trace across occlusion (see ``ball_window_frames``,
   ``ball_trace_*``, ``max_ball_trace_gap_frames``).
2. **Possessor** at kick origin (nearest robot within ``possession_max_dist_mm``), team from jersey
   vs ``left_team_color`` / ``right_team_color``.
3. While **qualifies** (enough **space + speed**): emit **strike** if displacement aligns toward a
   **teammate** (``pass_direction_min_cos``) **or** toward the **opponent goal**
   (``shot_direction_min_cos`` + ``shot_min_goal_progress_mm``); teammate branch wins if both match.
   Evaluated every frame while ``qualifies``.

Optional: ``ball_loss_shot_infer_enabled`` (default false) — infers a strike after brief ball loss.
"""

from __future__ import annotations

import ast
import json
import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import pandas as pd

from ..commentator.event_flags import EventFlags
from ..utils.drawing import TEAM_COLORS


class SpecializedStore:
    """Append-only (frame, ball row) store."""

    def __init__(self) -> None:
        self.frames: list = []
        self.data: list = []

    def add(self, frame, new_data) -> None:
        self.frames.append(frame)
        self.data.append(new_data)


@dataclass(frozen=True)
class _RobotNear:
    id: int
    dist_mm: float
    team: str  # "left" | "right"


class BallMotionAnalyzer:
    """Stateful ball geometry: shot/pass flags on :class:`EventFlags`."""

    def __init__(self, config) -> None:
        self.fps = config.processing.fps
        self.fldconf = config.field_config
        self.commconf = config.commentating

        self._window_size = self.commconf.ball_window_frames
        self._ball_window: Deque[Tuple[int, float, float]] = deque(maxlen=self._window_size)

        # TODO this parameter is no longer under this config. match_commentary_tts should be writing it anyway, but it's kind of an antipattern that should be fixed.
        self._left_bgr = self._config_color_to_bgr(getattr(config.features, "left_team_color", (255, 0, 0)))
        self._right_bgr = self._config_color_to_bgr(getattr(config.features, "right_team_color", (0, 0, 255)))

        self.balldata = SpecializedStore()
        self.ball_prev: pd.Series | pd.DataFrame = pd.DataFrame()
        self._last_picked_ball: pd.Series | pd.DataFrame = pd.Series(dtype=object)

        self._motion_cooldown = self.commconf.motion_event_cooldown
        self.last_enter_shot_second: float = -1e9
        self.last_pass_second: float = -1e9

        self._last_instant_vx: float = 0.0
        self._last_instant_vy: float = 0.0
        self._persist_vx: float = 0.0
        self._persist_vy: float = 0.0
        self._persist_speed: float = 0.0
        self._ball_lost_streak: int = 0
        self._ghost_shot_fired_for_drop: bool = False

        # After pass/shot, suppress re-fire until the ball has been observed at rest
        # for ``min_idle_frames_between_kicks`` consecutive frames (avoids re-triggering
        # the same rolling motion as multiple "new" kicks).
        self._kick_emit_ok: bool = True
        self._idle_frames: int = 0
        self._last_motion_debug: str = "not relevant"
        # Last ball pose in field (mm) while visible; frozen while the ball row is missing (occlusion).
        self._last_seen_ball_xy: Optional[Tuple[float, float]] = None
        self._last_seen_ball_frame: Optional[int] = None

    @staticmethod
    def _config_color_to_bgr(color_val) -> Tuple[int, int, int]:
        """Accept color names (new config) and BGR triples (legacy config)."""
        if isinstance(color_val, str):
            return TEAM_COLORS.get(color_val.strip().lower(), (100, 100, 100))
        if isinstance(color_val, (list, tuple)) and len(color_val) == 3:
            return (int(color_val[0]), int(color_val[1]), int(color_val[2]))
        return (100, 100, 100)

    def motion_debug_line(self) -> str:
        return self._last_motion_debug

    def _trace_metrics(self, x_now: float, y_now: float, frame: int) -> Tuple[float, float, int]:
        """
        Displacement and mean speed from the frozen **last_seen** anchor to the current ball.
        If the frame gap exceeds ``max_ball_trace_gap_frames``, returns zeros (stale anchor).
        """
        if self._last_seen_ball_xy is None or self._last_seen_ball_frame is None:
            return 0.0, 0.0, 0
        gap_tr = int(frame) - int(self._last_seen_ball_frame)
        if gap_tr < 1:
            return 0.0, 0.0, gap_tr
        max_gap = self.commconf.max_ball_trace_gap_frames
        lx, ly = self._last_seen_ball_xy
        disp_trace = math.hypot(x_now - lx, y_now - ly)
        dt_tr = float(gap_tr) / self.fps
        v_trace = disp_trace / dt_tr if dt_tr > 1e-9 else 0.0
        if gap_tr > max_gap:
            return 0.0, 0.0, gap_tr
        return disp_trace, v_trace, gap_tr

    def _window_endpoint_kinematics(self) -> Tuple[float, float, float, float, float, float]:
        if len(self._ball_window) < 2:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        f0, x0, y0 = self._ball_window[0]
        f1, x1, y1 = self._ball_window[-1]
        return float(f0), float(x0), float(y0), float(f1), float(x1), float(y1)

    def _window_mean_speed(self) -> float:
        if len(self._ball_window) < 2:
            return 0.0
        f0, x0, y0, f1, x1, y1 = self._window_endpoint_kinematics()
        span = float(f1 - f0)
        if span <= 0:
            return 0.0
        dt_w = span / self.fps
        if dt_w <= 1e-9:
            return 0.0
        return math.hypot(x1 - x0, y1 - y0) / dt_w

    @property
    def last_seen_ball_frame(self) -> Optional[int]:
        return self._last_seen_ball_frame

    @property
    def last_instant_vx(self) -> float:
        return self._last_instant_vx

    @property
    def last_instant_vy(self) -> float:
        return self._last_instant_vy

    @property
    def current_ball_row(self) -> pd.Series | pd.DataFrame:
        return self._last_picked_ball

    def _instant_velocity_mm_s(
        self,
        x_prev: float,
        y_prev: float,
        x_now: float,
        y_now: float,
        frame_prev: int,
        frame_now: int,
    ) -> Tuple[float, float, float, float]:
        dt = (float(frame_now) - float(frame_prev)) / self.fps
        if dt <= 1e-9:
            return 0.0, 0.0, 0.0, dt
        vx = (x_now - x_prev) / dt
        vy = (y_now - y_prev) / dt
        return vx, vy, math.hypot(vx, vy), dt

    def _parse_bgr(self, color_val) -> Optional[Tuple[int, int, int]]:
        if color_val is None:
            return None
        if isinstance(color_val, str) and color_val.strip() == "ball":
            return None
        try:
            if pd.isna(color_val):  # type: ignore[arg-type]
                return None
        except TypeError:
            pass
        if isinstance(color_val, (list, tuple)) and len(color_val) == 3:
            return (int(color_val[0]), int(color_val[1]), int(color_val[2]))
        if isinstance(color_val, str):
            name = color_val.strip().lower()
            # Plain color name from CSV color column (e.g. "blue", "red", "unknown")
            if name in TEAM_COLORS:
                return TEAM_COLORS[name]
            try:
                t = ast.literal_eval(name)
                if isinstance(t, (list, tuple)) and len(t) == 3:
                    return (int(t[0]), int(t[1]), int(t[2]))
            except (ValueError, SyntaxError):
                return None
        return None

    def _team_from_color(self, color_val) -> Optional[str]:
        bgr = self._parse_bgr(color_val)
        if bgr is None:
            return None
        dl = sum((int(a) - int(b)) ** 2 for a, b in zip(bgr, self._left_bgr))
        dr = sum((int(a) - int(b)) ** 2 for a, b in zip(bgr, self._right_bgr))
        if min(dl, dr) > self.commconf.possession_color_match_max_sq:
            return None
        return "left" if dl < dr else "right"

    def _is_likely_kicker(
        self, team: str, bx: float, kick_dx: float
    ) -> bool:
        """
        True if the kick direction is consistent with this team kicking the ball
        away from their own goal (positive kick_dx for left team, negative for right).

        A robot that is between the ball and its own goal (defender) would produce
        a kick toward the opponent's side — which matches the kicker heuristic.
        A robot that is on the WRONG side (e.g. left team robot but kick goes left)
        is more likely a defender being beaten, not the kicker.
        """
        if kick_dx == 0.0:
            return True  # ambiguous, don't filter
        if team == "left":
            # Left team attacks right (+X). A left-team kicker moves ball rightward.
            return kick_dx > 0
        else:
            # Right team attacks left (-X). A right-team kicker moves ball leftward.
            return kick_dx < 0

    def _robots_near(
        self,
        frame_df: pd.DataFrame,
        bx: float,
        by: float,
        kick_dx: float = 0.0,
    ) -> Optional[_RobotNear]:
        """Return the nearest robot that is consistent with being the kicker.

        When ``kick_dx`` is provided (non-zero), robots whose team/direction
        is inconsistent with the kick (i.e. likely defenders, not kickers) are
        de-prioritised: first try robots where ``_is_likely_kicker`` is True,
        fall back to all robots only if no consistent candidate is found within
        ``possession_max_dist_mm``.
        """
        rows = frame_df[frame_df["type"] == "robot"]
        poss_max = self.commconf.possession_max_dist_mm

        candidates: list[Tuple[int, float, str]] = []
        for _, row in rows.iterrows():
            try:
                fx = float(row["field_x"])
                fy = float(row["field_y"])
            except (TypeError, ValueError):
                continue
            if math.isnan(fx) or math.isnan(fy):
                continue
            team = self._team_from_color(row.get("color"))
            if team is None:
                continue
            d = math.hypot(bx - fx, by - fy)
            candidates.append((int(row["id"]), d, team))

        if not candidates:
            return None

        # Prefer robots whose kick direction matches the ball displacement direction.
        # This disambiguates when a defender is closer than the actual kicker.
        if kick_dx != 0.0:
            consistent = [(rid, d, t) for rid, d, t in candidates if self._is_likely_kicker(t, bx, kick_dx)]
            inconsistent_in_range = any(
                d <= poss_max and not self._is_likely_kicker(t, bx, kick_dx)
                for _, d, t in candidates
            )
            consistent_in_range = any(d <= poss_max for _, d, _ in consistent)

            if consistent_in_range:
                # At least one kicker-consistent robot is close — use consistent pool only.
                candidates = consistent
            elif inconsistent_in_range:
                # Only inconsistent (defender-like) robots are close — don't trust them as kicker.
                # Return the closest consistent candidate even if far, or None if none exist.
                if consistent:
                    best = min(consistent, key=lambda c: c[1])
                    return _RobotNear(id=best[0], dist_mm=best[1], team=best[2])
                return None
            # else: no robot within range at all — fall through to nearest overall

        best = min(candidates, key=lambda c: c[1])
        return _RobotNear(id=best[0], dist_mm=best[1], team=best[2])

    def _all_robots_near(
        self,
        frame_df: pd.DataFrame,
        bx: float,
        by: float,
        max_radius_mm: float = 2500.0,
        max_count: int = 6,
    ) -> list[dict]:
        """
        Return up to ``max_count`` robots within ``max_radius_mm`` of the ball,
        sorted by distance. Includes robots with unknown team (color misclassified) —
        this preserves raw evidence for the LLM to reason about team assignment.
        """
        rows = frame_df[frame_df["type"] == "robot"]
        out: list[dict] = []
        for _, row in rows.iterrows():
            try:
                fx = float(row["field_x"])
                fy = float(row["field_y"])
            except (TypeError, ValueError):
                continue
            if math.isnan(fx) or math.isnan(fy):
                continue
            d = math.hypot(bx - fx, by - fy)
            if d > max_radius_mm:
                continue
            team = self._team_from_color(row.get("color"))
            color_raw = str(row.get("color", "") or "").strip().lower() or "unknown"
            out.append({
                "id": int(row["id"]),
                "team": team or "unknown",
                "color": color_raw,
                "dist_mm": round(d, 1),
                "x_mm": round(fx, 1),
                "y_mm": round(fy, 1),
                "dx_from_ball_mm": round(fx - bx, 1),
                "dy_from_ball_mm": round(fy - by, 1),
            })
        out.sort(key=lambda r: r["dist_mm"])
        return out[:max_count]

    def _pick_ball_row(self, frame_df: pd.DataFrame) -> pd.Series:
        balls = frame_df[frame_df["type"] == "ball"]
        if len(balls) == 0:
            return pd.Series(dtype=object)
        if len(balls) == 1 or self.ball_prev is None or self.ball_prev.empty:
            return balls.iloc[0]
        px, py = float(self.ball_prev["field_x"]), float(self.ball_prev["field_y"])
        balls = balls.copy()
        balls["_d"] = ((balls["field_x"] - px) ** 2 + (balls["field_y"] - py) ** 2) ** 0.5
        return balls.sort_values("_d").iloc[0]

    def _goal_progress_for_team(self, team: str, x0: float, x1: float) -> float:
        """Positive if the ball moved toward the opponent goal along X."""
        if team == "left":
            return float(x1 - x0)
        return float(x0 - x1)

    def _max_cos_toward_teammate(
        self,
        frame_df: pd.DataFrame,
        x0: float,
        y0: float,
        dx: float,
        dy: float,
        possessor_team: str,
        possessor_id: int,
    ) -> float:
        """Cosine between displacement unit vector and direction from kick origin to each teammate."""
        L = math.hypot(dx, dy)
        if L < 1e-9:
            return -1.0
        ux, uy = dx / L, dy / L
        best = -1.0
        for _, row in frame_df[frame_df["type"] == "robot"].iterrows():
            try:
                tid = int(row["id"])
            except (TypeError, ValueError):
                continue
            tteam = self._team_from_color(row.get("color"))
            if tteam != possessor_team or tid == possessor_id:
                continue
            try:
                tx, ty = float(row["field_x"]), float(row["field_y"])
            except (TypeError, ValueError):
                continue
            if math.isnan(tx) or math.isnan(ty):
                continue
            wx, wy = tx - x0, ty - y0
            wl = math.hypot(wx, wy)
            if wl < 1e-6:
                continue
            c = (ux * wx + uy * wy) / wl
            if c > best:
                best = c
        return best

    def _cos_toward_opponent_goal(
        self, dx: float, dy: float, x0: float, y0: float, team: str
    ) -> float:
        """Cosine between displacement and vector from kick origin toward opponent goal mouth (0, center)."""
        xlim = float(self.fldconf.xlimit)
        L = math.hypot(dx, dy)
        if L < 1e-9:
            return -1.0
        if team == "left":
            gx, gy = xlim - x0, -y0
        else:
            gx, gy = (-xlim) - x0, -y0
        gl = math.hypot(gx, gy)
        if gl < 1e-6:
            return -1.0
        return (dx * gx + dy * gy) / (L * gl)

    def _kick_geometry(
        self,
        x_now: float,
        y_now: float,
        disp_trace: float,
        gap_tr: int,
        x0w: float,
        y0w: float,
        x1w: float,
        y1w: float,
    ) -> Tuple[float, float, float, float]:
        """Kick origin and end: prefer last_seen→current after occlusion, else window, else prev→now."""
        min_trace_gap = self.commconf.ball_trace_min_gap_frames
        min_disp = self.commconf.min_kick_displacement_mm
        max_gap = self.commconf.max_ball_trace_gap_frames
        prefer_trace = (
            self._last_seen_ball_xy is not None
            and self._last_seen_ball_frame is not None
            and min_trace_gap <= gap_tr <= max_gap
            and disp_trace >= min_disp * 0.2
        )
        if prefer_trace:
            lx, ly = self._last_seen_ball_xy
            return float(lx), float(ly), x_now, y_now
        if len(self._ball_window) >= 2:
            return float(x0w), float(y0w), float(x1w), float(y1w)
        return (
            float(self.ball_prev["field_x"]),
            float(self.ball_prev["field_y"]),
            x_now,
            y_now,
        )

    def _attivo_label(
        self,
        second: float,
        frame_df: pd.DataFrame,
        x0: float,
        y0: float,
        x1: float,
        dx_k: float,
        dy_k: float,
        team_start: Optional[str],
        start_nr: Optional[_RobotNear],
        pass_cos_min: float,
        shot_cos_min: float,
        shot_dx: float,
    ) -> str:
        """
        Sotto-classifica ``attivo`` (movimento ok ma nessun evento questo frame):
        cooldown, già emesso nel calcio, tendenza pass vs tiro, progresso X mancante, direzione debole.
        """
        if not self._kick_emit_ok:
            return "attivo_dopo_evento"
        if (
            (second - self.last_enter_shot_second) < self._motion_cooldown
            or (second - self.last_pass_second) < self._motion_cooldown
        ):
            return "attivo_cooldown"
        if team_start is None or start_nr is None:
            return "attivo"
        cos_tm = self._max_cos_toward_teammate(
            frame_df, x0, y0, dx_k, dy_k, team_start, int(start_nr.id)
        )
        cos_goal = self._cos_toward_opponent_goal(
            dx_k, dy_k, x0, y0, team_start
        )
        gprog = self._goal_progress_for_team(team_start, x0, x1)
        pass_ok = cos_tm >= pass_cos_min
        shot_ok = (not pass_ok) and cos_goal >= shot_cos_min and gprog >= shot_dx
        if pass_ok or shot_ok:
            return "attivo"
        if (not pass_ok) and cos_goal >= shot_cos_min and gprog < shot_dx:
            return "attivo_tiro_manca_x"
        if (not pass_ok) and cos_tm >= cos_goal:
            return "attivo_tende_passaggio"
        if cos_goal > cos_tm:
            return "attivo_tende_tiro"
        return "attivo_dir_debole"

    def _try_infer_shot_on_ball_loss(
        self, events: EventFlags, second: float, frame_df: pd.DataFrame
    ) -> None:
        if self._ghost_shot_fired_for_drop:
            return
        if not self.commconf.ball_loss_shot_infer_enabled:
            return
        grace = max(1, self.commconf.ball_loss_shot_grace_frames)
        if self._ball_lost_streak < 1 or self._ball_lost_streak > grace:
            return
        if len(self._ball_window) < 2:
            return

        f0, x0, y0, f1, x1, y1 = self._window_endpoint_kinematics()
        span = float(f1 - f0)
        if span <= 0:
            return
        dt_w = span / self.fps
        if dt_w <= 1e-9:
            return
        wmean = math.hypot(x1 - x0, y1 - y0) / dt_w

        if wmean < self.commconf.min_kick_speed_mms * 0.45:
            return
        if wmean > self.commconf.impossible_speed:
            return

        poss_max = self.commconf.possession_max_dist_mm
        infer_dx = float(x1 - x0)
        start_nr = self._robots_near(frame_df, x0, y0, kick_dx=infer_dx)
        if start_nr is None or start_nr.dist_mm > poss_max or start_nr.team is None:
            return

        shot_dx = self.commconf.shot_min_goal_progress_mm
        if self._goal_progress_for_team(start_nr.team, x0, x1) < shot_dx:
            return

        if (second - self.last_enter_shot_second) < self._motion_cooldown:
            return

        if start_nr.team == "left":
            events.home_team_ball_strike = True
        else:
            events.away_team_ball_strike = True
        self.last_enter_shot_second = second
        self.last_pass_second = second
        self._ghost_shot_fired_for_drop = True

    def apply(self, events: EventFlags, frame_df: pd.DataFrame, second: float) -> None:
        ball_now = self._pick_ball_row(frame_df)
        self._last_picked_ball = ball_now.copy() if not ball_now.empty else pd.Series(dtype=object)
        frame = int(frame_df.iloc[0]["frame"])

        if not ball_now.empty:
            self.balldata.add(frame, ball_now)
            self._ball_window.append(
                (int(frame), float(ball_now["field_x"]), float(ball_now["field_y"]))
            )
            self._ball_lost_streak = 0
            self._ghost_shot_fired_for_drop = False
        else:
            if not self.ball_prev.empty:
                self._ball_lost_streak += 1
                self._try_infer_shot_on_ball_loss(events, second, frame_df)
            else:
                self._ball_lost_streak = 0

        self._last_instant_vx = 0.0
        self._last_instant_vy = 0.0
        if not ball_now.empty and not self.ball_prev.empty:
            xn, yn = float(ball_now["field_x"]), float(ball_now["field_y"])
            xp, yp = float(self.ball_prev["field_x"]), float(self.ball_prev["field_y"])
            fp, fn = int(self.ball_prev["frame"]), int(ball_now["frame"])
            ivx, ivy, _, _ = self._instant_velocity_mm_s(xp, yp, xn, yn, fp, fn)
            self._last_instant_vx = ivx
            self._last_instant_vy = ivy

        if ball_now.empty or self.ball_prev.empty:
            if ball_now.empty and not self.ball_prev.empty:
                infer = int(
                    events.away_team_ball_strike or events.home_team_ball_strike
                )
                self._last_motion_debug = (
                    "robot_ball_strike" if infer else "not relevant"
                )
            elif ball_now.empty and self.ball_prev.empty:
                self._last_motion_debug = "not relevant"
            else:
                self._last_motion_debug = "not relevant"
            if not ball_now.empty:
                self.ball_prev = ball_now
                self._last_seen_ball_xy = (
                    float(ball_now["field_x"]),
                    float(ball_now["field_y"]),
                )
                self._last_seen_ball_frame = frame
            return

        vx, vy, inst_speed, _ = self._instant_velocity_mm_s(
            float(self.ball_prev["field_x"]),
            float(self.ball_prev["field_y"]),
            float(ball_now["field_x"]),
            float(ball_now["field_y"]),
            int(self.ball_prev["frame"]),
            int(ball_now["frame"]),
        )
        self._persist_vx = vx
        self._persist_vy = vy
        self._persist_speed = inst_speed

        x_now = float(ball_now["field_x"])
        y_now = float(ball_now["field_y"])
        disp_trace, v_trace, gap_tr = self._trace_metrics(x_now, y_now, frame)

        f0w, x0w, y0w, f1w, x1w, y1w = self._window_endpoint_kinematics()
        disp_w = (
            math.hypot(x1w - x0w, y1w - y0w) if len(self._ball_window) >= 2 else 0.0
        )
        wmean = self._window_mean_speed()
        disp_eff = max(disp_w, disp_trace)
        v_eff = max(wmean, v_trace)

        min_sp = self.commconf.min_kick_speed_mms
        poss_max = self.commconf.possession_max_dist_mm
        shot_dx = self.commconf.shot_min_goal_progress_mm
        pass_cos_min = self.commconf.pass_direction_min_cos
        shot_cos_min = self.commconf.shot_direction_min_cos
        imp = self.commconf.impossible_speed

        x0, y0, x1, y1 = self._kick_geometry(
            x_now, y_now, disp_trace, gap_tr, x0w, y0w, x1w, y1w
        )
        dx_k = x1 - x0
        dy_k = y1 - y0
        start_nr = self._robots_near(frame_df, x0, y0, kick_dx=dx_k)
        end_nr = self._robots_near(frame_df, x1, y1, kick_dx=dx_k)
        team_start = (
            start_nr.team
            if start_nr is not None and start_nr.dist_mm <= poss_max and start_nr.team
            else None
        )

        has_window_span = len(self._ball_window) >= 2 and float(f1w - f0w) > 0
        has_trace_span = (
            gap_tr >= 1 and disp_trace > 1e-9 and v_trace > 1e-9
        )
        # Trigger on ball movement alone — no team_start required.
        # The LLM will interpret who acted from geometry hints.
        qualifies = (
            (has_window_span or has_trace_span)
            and disp_eff >= self.commconf.min_kick_displacement_mm
            and v_eff >= min_sp
            and v_eff <= imp
        )
        # Only re-arm kick emission once the ball has genuinely come to rest for a while —
        # a single low-speed frame is NOT enough (it would re-fire on the same rolling motion).
        idle_speed_threshold = min_sp * 0.5  # half of kick speed threshold = "ball is slow"
        min_idle_frames = self.commconf.min_idle_frames_between_kicks
        if not qualifies and self._persist_speed < idle_speed_threshold:
            self._idle_frames += 1
        else:
            self._idle_frames = 0
        if self._idle_frames >= min_idle_frames:
            self._kick_emit_ok = True

        # --- Gap-based re-arm ---
        # When the ball reappears after a detection gap >= gap_rearm_frames, the previous
        # motion has ended (ball was occluded / out of view).  Re-arm the emitter and use a
        # reduced cooldown for this new appearance so back-to-back kicks through occlusion
        # (e.g. a clearance at frame 543 followed by a shot at frame 566 with only 0.77s gap)
        # are not silently suppressed.
        reappearance_after_gap = gap_tr >= self.commconf.ball_gap_rearm_frames
        if reappearance_after_gap:
            self._kick_emit_ok = True
            self._idle_frames = min_idle_frames
        effective_cooldown = (
            self._motion_cooldown * self.commconf.ball_gap_cooldown_factor
            if reappearance_after_gap
            else self._motion_cooldown
        )

        # Referee/manual reposition guard:
        # if the ball moved a lot but no robot is near either start or end of the motion,
        # treat it as external reposition (not a robot kick/pass/shot event).
        ext_max_dist = self.commconf.external_reposition_max_robot_dist_mm
        near_start = bool(start_nr is not None and start_nr.dist_mm <= ext_max_dist)
        near_end = bool(end_nr is not None and end_nr.dist_mm <= ext_max_dist)
        external_reposition = qualifies and (not near_start) and (not near_end)

        if (
            qualifies
            and not external_reposition
            and self._kick_emit_ok
            and (second - self.last_enter_shot_second) >= effective_cooldown
            and (second - self.last_pass_second) >= effective_cooldown
        ):
            # Compute cosines toward each goal for the LLM geometry hints.
            cos_toward_right_goal = self._cos_toward_opponent_goal(dx_k, dy_k, x0, y0, "left")
            cos_toward_left_goal = self._cos_toward_opponent_goal(dx_k, dy_k, x0, y0, "right")
            gprog_left = self._goal_progress_for_team("left", x0, x1)
            gprog_right = self._goal_progress_for_team("right", x0, x1)

            # Best-guess team hint from nearest robot (may be None/noisy — LLM decides).
            if team_start is not None and start_nr is not None:
                cos_tm = self._max_cos_toward_teammate(
                    frame_df, x0, y0, dx_k, dy_k, team_start, int(start_nr.id)
                )
                cos_goal = self._cos_toward_opponent_goal(dx_k, dy_k, x0, y0, team_start)
                gprog = self._goal_progress_for_team(team_start, x0, x1)
                pass_ok = cos_tm >= pass_cos_min
                shot_ok = (not pass_ok) and cos_goal >= shot_cos_min and gprog >= shot_dx
                hint_cls = "pass" if pass_ok else ("shot" if shot_ok else "unclear")
            else:
                hint_cls = "unclear"

            # Fallback team guess from ball X-direction when robot proximity is unavailable.
            team_guess = team_start or ("left" if dx_k > 0 else "right")

            # Scene context: all robots near the ball at kick origin (both teams, both known
            # and unknown colors). This lets the LLM reason from geometric patterns rather
            # than trust a single possibly-misclassified nearest-robot signal.
            nearby = self._all_robots_near(frame_df, x0, y0, max_radius_mm=2500.0, max_count=6)

            events.ball_action_geometry = json.dumps({
                "dx_mm": round(dx_k, 1),
                "dy_mm": round(dy_k, 1),
                "disp_mm": round(math.hypot(dx_k, dy_k), 1),
                "speed_mms": round(v_eff, 1),
                "cos_toward_right_goal": round(cos_toward_right_goal, 3),
                "cos_toward_left_goal": round(cos_toward_left_goal, 3),
                "goal_progress_if_left_attacking_mm": round(gprog_left, 1),
                "goal_progress_if_right_attacking_mm": round(gprog_right, 1),
                "nearest_robot_team": team_start or "unknown",
                "nearest_robot_dist_mm": round(start_nr.dist_mm, 1) if start_nr else -1,
                "nearby_robots": nearby,
                "ball_start_x_mm": round(x0, 1),
                "ball_start_y_mm": round(y0, 1),
                "ball_end_x_mm": round(x1, 1),
                "ball_end_y_mm": round(y1, 1),
                "tracker_hint_cls": hint_cls,
                "tracker_hint_team": team_guess,
            })
            events.ball_action_hint = hint_cls
            events.ball_action_team = team_guess

            if team_guess == "left":
                events.home_team_ball_strike = True
            else:
                events.away_team_ball_strike = True
            self.last_pass_second = second
            self.last_enter_shot_second = second
            self._kick_emit_ok = False
            self._idle_frames = 0

        st = bool(
            events.away_team_ball_strike or events.home_team_ball_strike
        )
        if st:
            self._last_motion_debug = "robot_ball_strike"
        elif external_reposition:
            self._last_motion_debug = "external_ball_reposition"
        elif qualifies:
            self._last_motion_debug = self._attivo_label(
                second,
                frame_df,
                x0,
                y0,
                x1,
                dx_k,
                dy_k,
                team_start,
                start_nr,
                pass_cos_min,
                shot_cos_min,
                shot_dx,
            )
        else:
            self._last_motion_debug = "not relevant"

        if not ball_now.empty:
            self.ball_prev = ball_now
            self._last_seen_ball_xy = (x_now, y_now)
            self._last_seen_ball_frame = frame
