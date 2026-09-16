"""
Match commentary: OCR goals + ball motion (shots / passes) from :mod:`src.analysis.analyze_ball`.

Field: left goal at ``x = -xlimit``, right goal at ``+xlimit``. **Home** = ``left_team``
in config; **away** = ``right_team``. ``score_left`` / ``score_right`` are OCR digits.
"""

from __future__ import annotations

import io
import json
from collections import deque
from typing import Any, Optional

import pandas as pd

# Keep enough frames for LLM context windows (e.g. ±5 at 30 fps ≪ cap).
_FRAME_LOG_MAX = 800


def team_labels(features: Any) -> tuple[str, str, str, str]:
    """Return ``(left_name, left_color, right_name, right_color)`` from a features config."""
    # TODO these parameters are no longer under this config. match_commentary_tts should be writing it anyway, but it's kind of an antipattern that should be fixed.
    ln = str(getattr(features, "left_team_name", None) or "Home")
    rn = str(getattr(features, "right_team_name", None) or "Away")
    lc = str(getattr(features, "left_team_color", None) or "left")
    rc = str(getattr(features, "right_team_color", None) or "right")
    return ln, lc, rn, rc


def extract_json_comment(raw: str) -> str:
    """Extract the ``comment`` field from a JSON LLM response, or return the raw text."""
    text = (raw or "").strip()
    try:
        if "{" in text:
            start = text.index("{")
            end = text.rindex("}") + 1
            obj = json.loads(text[start:end])
            return (obj.get("comment") or "").strip() or text
    except (json.JSONDecodeError, ValueError):
        pass
    return text


class Commentator:
    """Per-frame analyser: fills :class:`EventFlags` for fixed-phrase TTS."""

    def __init__(self, config, team_map, eventproc) -> None:
        self.fps = config.processing.fps
        self.featconf = config.features
        self.eventproc = eventproc
        self._left_team_color  = str(team_map.left_color_name)  if team_map is not None else "blue"
        self._right_team_color = str(team_map.right_color_name) if team_map is not None else "red"
        self._left_team_name = team_map.left_short_name
        self._right_team_name = team_map.right_short_name
        self._geometry_llm_sent: deque[str] = deque(maxlen=400)
        self._ball_kinetics_llm_sent: deque[str] = deque(maxlen=400)
        # Timeline of classified ball-kinetics events for tactical narrative.
        # Each entry: {"frame": int, "team": "left"|"right"|"unknown",
        #              "classification": str, "speed_mms": float,
        #              "zone": str, "comment": str}
        self._recent_ball_events: deque[dict] = deque(maxlen=30)
        # Eager (pre-LLM) marker to suppress rapid same-team same-action repeats.
        # Tuple: (frame, acting_team, classification).
        self._last_kinetics_emit: Optional[tuple[int, str, str]] = None
        self._field_w = config.field_config.width
        self._field_h = config.field_config.height

    def csv_excerpt_around_frame(self, center_frame: int, frames_before: int, frames_after: int) -> str:
        """Concatenate CSV blocks for frames in the rolling log around ``center_frame``."""
        rolling_log = self.eventproc.get_snapshots_around_frame(center_frame, frames_before, frames_after)
        parts: list[str] = []
        for f, block in rolling_log:
            parts.append(f"--- frame {f} ---\n{block}")
        return "\n\n".join(parts)


    def ball_motion_summary(
        self,
        trigger_frame: int,
        last_seen_frame: Optional[int],
        history_frames: int = 40,
        post_frames: int = 15,
    ) -> str:
        """
        Compact ball trajectory for LLM context — replaces raw CSV excerpt.

        Shows:
          - last ``history_frames`` ball positions BEFORE the kick origin
          - the detection gap (duration in frames and seconds)
          - first ``post_frames`` ball positions AFTER the gap

        Far more signal-dense than raw CSV rows (no robot clutter).
        """
        positions = self.eventproc.get_ball_positions()
        if not positions:
            return "No ball positions available."

        origin_f = last_seen_frame if last_seen_frame is not None else trigger_frame

        pre = [(f, x, y) for f, x, y in positions if f <= origin_f]
        pre = pre[-history_frames:]

        # First ball frame at or after trigger
        post_start = next((f for f, _, _ in positions if f >= trigger_frame), trigger_frame)
        post = [(f, x, y) for f, x, y in positions if f >= post_start]
        post = post[:post_frames]

        gap_frames = post_start - origin_f if last_seen_frame is not None else 0

        lines: list[str] = []
        if pre:
            lines.append(f"Pre-kick ({len(pre)} ball positions before gap):")
            for f, x, y in pre:
                lines.append(f"  frame {f:4d}:  x={x:8.0f}mm  y={y:7.0f}mm")

        if gap_frames > 0:
            lines.append(
                f"\n  ···  GAP: {gap_frames} frames"
                f" ({gap_frames / max(1.0, self.fps):.2f}s) — ball not visible  ···\n"
            )

        if post:
            lines.append(f"Post-gap ({len(post)} ball positions after gap):")
            for f, x, y in post:
                lines.append(f"  frame {f:4d}:  x={x:8.0f}mm  y={y:7.0f}mm")

        return "\n".join(lines)

    def ball_zone_hint(self, frame: int) -> str:
        """Qualitative ball location near ``frame`` for safer commentary wording."""
        if not self.eventproc.is_ball_positions_valid():
            return "unknown"
        f, x, y = min(self.eventproc.get_ball_positions(), key=lambda t: abs(t[0] - frame))
        _ = f
        return self._zone_from_xy(x, y)

    def field_zone_legend(self) -> str:
        """Text legend of all tactical zones used for commentary prompts."""
        return (
            "Field zones (use these names):\n"
            "- left defensive third / central lane\n"
            "- left defensive third / left wing\n"
            "- left defensive third / right wing\n"
            "- middle third / central lane\n"
            "- middle third / left wing\n"
            "- middle third / right wing\n"
            "- right attacking third / central lane\n"
            "- right attacking third / left wing\n"
            "- right attacking third / right wing\n"
            "- center circle area (strictly around 0,0)\n"
            "- left box edge / central lane\n"
            "- right box edge / central lane"
        )

    def ball_zone_history_hint(self, frame: int, steps: int = 6) -> str:
        """Recent sequence of qualitative zones to give movement context."""
        if not self.eventproc.is_ball_positions_valid():
            return "unknown"
        near = sorted(
            self.eventproc.get_ball_positions(),
            key=lambda t: abs(t[0] - frame),
        )[: max(1, int(steps))]
        near_sorted = sorted(near, key=lambda t: t[0])
        labels: list[str] = []
        for _, x, y in near_sorted:
            z = self._zone_from_xy(x, y)
            if not labels or labels[-1] != z:
                labels.append(z)
        return " -> ".join(labels) if labels else "unknown"

    def _zone_from_xy(self, x: float, y: float) -> str:
        """Map metric field position to tactical zone labels."""
        hw = max(1.0, self._field_w / 2.0)
        hh = max(1.0, self._field_h / 2.0)
        nx = x / hw
        ny = y / hh
        if abs(nx) <= 0.14 and abs(ny) <= 0.24:
            return "center circle area"
        if nx < -0.72 and abs(ny) < 0.35:
            return "left box edge / central lane"
        if nx > 0.72 and abs(ny) < 0.35:
            return "right box edge / central lane"
        if nx < -0.33:
            third = "left defensive third"
        elif nx > 0.33:
            third = "right attacking third"
        else:
            third = "middle third"
        if abs(ny) > 0.58:
            lane = "left wing" if ny < 0 else "right wing"
            return f"{third} / {lane}"
        return f"{third} / central lane"

    def was_geometry_llm_sent(self, key: str) -> bool:
        return key in self._geometry_llm_sent

    def mark_geometry_llm_sent(self, key: str) -> None:
        self._geometry_llm_sent.append(key)

    def was_ball_kinetics_llm_sent(self, key: str) -> bool:
        return key in self._ball_kinetics_llm_sent

    def mark_ball_kinetics_llm_sent(self, key: str) -> None:
        self._ball_kinetics_llm_sent.append(key)

    def should_suppress_kinetics_repeat(
        self,
        *,
        trigger_frame: int,
        team: str,
        classification: str,
        window_seconds: float,
    ) -> bool:
        """Return True if the same (team, classification) was emitted within ``window_seconds``."""
        if self._last_kinetics_emit is None:
            return False
        last_f, last_team, last_cls = self._last_kinetics_emit
        if last_team != team or last_cls != classification:
            return False
        gap_frames = int(trigger_frame) - int(last_f)
        return gap_frames <= int(max(0.0, window_seconds) * max(1.0, self.fps))

    def mark_kinetics_emit(self, *, frame: int, team: str, classification: str) -> None:
        self._last_kinetics_emit = (int(frame), str(team), str(classification))

    def record_ball_event(
        self,
        *,
        frame: int,
        team: str,
        classification: str,
        speed_mms: float = 0.0,
        x_mm: float = 0.0,
        comment: str = "",
    ) -> None:
        """Append a classified ball-kinetics event to the tactical timeline."""
        self._recent_ball_events.append({
            "frame": int(frame),
            "team": str(team),
            "classification": str(classification),
            "speed_mms": float(speed_mms),
            "zone": self._zone_from_xy(float(x_mm), 0.0),
            "comment": (comment or "")[:120],
        })

    def team_formation_hint(self, now_frame: int, *, label_language: str = "en") -> str:
        """
        Compact snapshot of where each team is (defence / midfield / attack).

        Use ``label_language=\"en\"`` for LLM prompts (default for periodic commentary).
        """
        it = str(label_language).lower().startswith("it")
        zd, zm, za = (("difesa", "metacampo", "attacco") if it else ("defence", "midfield", "attack"))
        nv = "non visibili" if it else "not visible"
        na = "posizioni squadre non disponibili" if it else "team positions not available"
        nr = "nessun robot rilevato in questo istante" if it else "no robots detected in this frame"
        nc = "colori robot non disponibili" if it else "robot colors unavailable"
        vis = "robot visibili" if it else "robots visible"

        log = self.eventproc.get_frame_snapshots()
        if not log:
            return na
        _, block = min(log, key=lambda t: abs(t[0] - int(now_frame)))
        try:
            df = pd.read_csv(io.StringIO(block))
        except (pd.errors.ParserError, ValueError):
            return na
        robots = df[df.get("type") == "robot"] if "type" in df.columns else pd.DataFrame()
        if robots.empty or "field_x" not in robots.columns:
            return nr

        xlim = max(1.0, self._field_w / 2.0)
        left_color  = self._left_team_color.lower()
        right_color = self._right_team_color.lower()

        def _bucket(team_goal_side: str, xs: list[float]) -> str:
            if not xs:
                return nv

            def zone_of(x: float) -> str:
                frac = x / xlim
                if team_goal_side == "left":
                    if frac < -0.33:
                        return zd
                    if frac > 0.33:
                        return za
                    return zm
                if frac > 0.33:
                    return zd
                if frac < -0.33:
                    return za
                return zm

            zones = [zone_of(float(x)) for x in xs]
            c = {z: zones.count(z) for z in (zd, zm, za)}
            parts = [f"{v} in {k}" for k, v in c.items() if v > 0]
            return ", ".join(parts) if parts else nv

        color_col = robots["color"].astype(str).str.lower() if "color" in robots.columns else None
        if color_col is None:
            return nc
        left_xs  = robots.loc[color_col == left_color,  "field_x"].dropna().tolist()
        right_xs = robots.loc[color_col == right_color, "field_x"].dropna().tolist()
        ln, rn = self._left_team_name, self._right_team_name
        return (
            f"{ln} ({len(left_xs)} {vis}): {_bucket('left',  left_xs)}. "
            f"{rn} ({len(right_xs)} {vis}): {_bucket('right', right_xs)}."
        )

    def match_narrative_summary(self, now_frame: int, window_seconds: float = 30.0) -> str:
        """
        Tactical digest for the periodic LLM — aggregate what happened recently.

        Text is in **English** so the model is not steered by mixed instruction/output
        languages; TTS language is set separately in the periodic prompt.
        """
        fps = max(1.0, float(self.fps))
        win_frames = int(window_seconds * fps)
        cutoff = int(now_frame) - win_frames

        ln, rn = self._left_team_name, self._right_team_name

        # ── Event aggregation ──────────────────────────────────────────
        ev = [e for e in self._recent_ball_events if int(e.get("frame", 0)) >= cutoff]
        counts: dict[str, dict[str, int]] = {
            "left":  {"shot": 0, "pass": 0, "clearance": 0},
            "right": {"shot": 0, "pass": 0, "clearance": 0},
        }
        for e in ev:
            team = str(e.get("team", "unknown"))
            cls  = str(e.get("classification", "unclear"))
            if team in counts and cls in counts[team]:
                counts[team][cls] += 1

        def _team_line(team_key: str, name: str) -> str:
            c = counts[team_key]
            parts: list[str] = []
            if c["shot"]:
                parts.append(f"{c['shot']} shot" + ("" if c["shot"] == 1 else "s"))
            if c["pass"]:
                parts.append(f"{c['pass']} pass" + ("" if c["pass"] == 1 else "es"))
            if c["clearance"]:
                parts.append(f"{c['clearance']} clearance" + ("" if c["clearance"] == 1 else "s"))
            return f"{name}: " + (", ".join(parts) if parts else "no tagged shots/passes/clearances in this window")

        # ── Zone dominance ─────────────────────────────────────────────
        xs = [x for f, x, _ in self.eventproc.get_ball_positions() if f >= cutoff]
        if xs:
            xlim = self._field_w / 2.0
            in_left  = sum(1 for x in xs if x < -xlim * 0.2)
            in_right = sum(1 for x in xs if x >  xlim * 0.2)
            in_mid   = len(xs) - in_left - in_right
            total = max(1, len(xs))
            pl, pr, pm = int(100 * in_left / total), int(100 * in_right / total), int(100 * in_mid / total)
            if pl >= 55:
                zone_line = f"ball spent ~{pl}% of samples on {ln}'s side"
            elif pr >= 55:
                zone_line = f"ball spent ~{pr}% of samples on {rn}'s side"
            else:
                zone_line = f"contested: left {pl}%, center {pm}%, right {pr}% (rough shares)"
        else:
            zone_line = "ball position trend unavailable (no ball samples in window)"

        # ── Last significant action ────────────────────────────────────
        last_sig = next(
            (e for e in reversed(ev) if e.get("classification") in ("shot", "clearance")),
            None,
        )
        if last_sig:
            t = last_sig.get("team", "unknown")
            name = {"left": ln, "right": rn}.get(t, "unattributed")
            sec_ago = max(0, (int(now_frame) - int(last_sig.get("frame", now_frame))) / fps)
            last_line = f"last significant event: {last_sig.get('classification')} by {name} ~{sec_ago:.1f}s ago"
            if last_sig.get("comment"):
                last_line += f" — note: {last_sig['comment']!r}"
        else:
            last_line = "no shot/clearance event recorded in this window"

        ev_total = sum(
            counts[tk][c] for tk in ("left", "right") for c in ("shot", "pass", "clearance")
        )
        if ev_total == 0:
            action_level = (
                "RECENT ACTION LEVEL: **none** — do not invent attacks, high pressing, or "
                "one-way dominance. Prefer neutral, wide-angle stadium commentary (shape, standoff, "
                "tension, centre circle, re-start feeling) unless the formation snapshot is extreme."
            )
        elif ev_total <= 1:
            action_level = (
                "RECENT ACTION LEVEL: **low** — be cautious; avoid strong tactical claims. "
                "If robots are only loosely spread, describe a careful phase, not a siege."
            )
        else:
            action_level = (
                "RECENT ACTION LEVEL: **active** — formation snapshot and zone hints are more meaningful."
            )

        return (
            f"Window: last {int(window_seconds)}s of tracker narrative.\n"
            f"{_team_line('left',  ln)}.\n"
            f"{_team_line('right', rn)}.\n"
            f"Dominant ball-side trend: {zone_line}.\n"
            f"{last_line}.\n"
            f"{action_level}"
        )
