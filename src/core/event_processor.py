from collections import deque
import io
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from src.analysis.analyze_ball import BallMotionAnalyzer
from src.commentator.event_flags import EventFlags

# Keep enough frames for LLM context windows (e.g. ±5 at 30 fps << cap).
_FRAME_LOG_MAX = 800

class EventProcessor:
    def __init__(self, config, events_output_path, team_map):
        self._symbolic_events_path = events_output_path
        self._last_symbolic_event_key = None
        self.fps = config.processing.fps

        self.prev_frame_df = pd.DataFrame()
        self._frame_snapshots: deque[tuple[int, str]] = deque(maxlen=_FRAME_LOG_MAX)
        self._ball_positions: deque[tuple[int, float, float]] = deque(maxlen=_FRAME_LOG_MAX)

        self._ball = BallMotionAnalyzer(config)

        self._left_team_name = team_map.left_short_name
        self._right_team_name = team_map.right_short_name
        self._left_team_full_name = team_map.left_full_name
        self._right_team_full_name = team_map.right_full_name

    # don't strictly need this, it's just for a little extra "cleanliness"
    # of not doing side-effects in __init__
    def begin_section(self):
        self._symbolic_events_path.parent.mkdir(parents=True, exist_ok=True)
        self._symbolic_events_path.write_text("", encoding="utf-8")

    ####################################################################################################################################################################

    @staticmethod
    def _snapshot_csv_block(frame_df: pd.DataFrame) -> str:
        if frame_df.empty:
            return ""
        buf = io.StringIO()
        frame_df.to_csv(buf, index=False)
        return buf.getvalue().strip()

    def _append_frame_snapshot(self, frame_df: pd.DataFrame) -> None:
        if frame_df.empty:
            return
        frame = int(frame_df.iloc[0]["frame"])
        self._frame_snapshots.append((frame, self._snapshot_csv_block(frame_df)))
        ball_rows = frame_df[frame_df["type"] == "ball"]
        if not ball_rows.empty:
            b = ball_rows.iloc[0]
            bx = b.get("field_x", None)
            by = b.get("field_y", None)
            if bx is not None and by is not None and pd.notna(bx) and pd.notna(by):
                self._ball_positions.append((frame, float(bx), float(by)))

    def _process_events(self, frame_df: pd.DataFrame) -> EventFlags:
        second = int(frame_df.iloc[0]["frame"]) / self.fps
        events = EventFlags()
        events.trigger_frame = int(frame_df.iloc[0]["frame"])
        events.score_left = int(frame_df.iloc[0]["score_left"])
        events.score_right = int(frame_df.iloc[0]["score_right"])
        events.left_team_name = self._left_team_name
        events.right_team_name = self._right_team_name
        events.left_team_full_name = self._left_team_full_name
        events.right_team_full_name = self._right_team_full_name
        events.has_ball = bool((frame_df["type"] == "ball").any())

        self._ball.apply(events, frame_df, second)

        if not self.prev_frame_df.empty:
            prev_left = int(self.prev_frame_df.iloc[0]["score_left"])
            prev_right = int(self.prev_frame_df.iloc[0]["score_right"])
            if prev_left >= 0 and events.score_left > prev_left:
                events.goal_for_left_team = True
            if prev_right >= 0 and events.score_right > prev_right:
                events.goal_for_right_team = True

        self.prev_frame_df = frame_df
        self._append_frame_snapshot(frame_df)
        return events
    
    def _append_symbolic_event(self, events):
        if self._symbolic_events_path is None:
            return
        frame = int(events.trigger_frame)
        if frame < 0:
            return

        event_name = None
        team = None
        payload = {}

        if events.goal_for_left_team:
            event_name = "goal"
            team = "left"
        elif events.goal_for_right_team:
            event_name = "goal"
            team = "right"
        elif events.home_team_ball_strike or events.away_team_ball_strike:
            # Prefer analyzer symbolic class if available (pass/shot/clearance/unclear).
            event_name = str(events.ball_action_hint or "ball_movement").strip().lower()
            if event_name not in ("pass", "shot", "clearance"):
                event_name = "ball_movement"
            team = str(events.ball_action_team or "").strip().lower() or None
            geom = events.ball_action_geometry
            if geom:
                try:
                    payload = json.loads(geom)
                except json.JSONDecodeError:
                    payload = {}
        else:
            return

        key = (frame, event_name, team)
        if key == self._last_symbolic_event_key:
            return
        self._last_symbolic_event_key = key

        row = {
            "frame": frame,
            "time_sec": (float(frame) / self.fps) if self.fps > 0 else 0.0,
            "event": event_name,
            "team": team,
            "score_left": events.score_left,
            "score_right": events.score_right,
            "left_team_name": events.left_team_full_name,
            "right_team_name": events.right_team_full_name,
        }
        if payload:
            row.update(payload)
        with open(self._symbolic_events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def process_frame(self, frame_df: pd.DataFrame) -> EventFlags:
        events = self._process_events(frame_df)
        self._append_symbolic_event(events)
        return events

    ####################################################################################################################################################################

    @property
    def last_seen_ball_frame(self) -> "Optional[int]":
        return self._ball.last_seen_ball_frame

    def motion_debug_line(self) -> str:
        """Last ball-motion diagnostic string from :class:`~src.analysis.analyze_ball.BallMotionAnalyzer`."""
        return self._ball.motion_debug_line()
    
    def csv_block_at_frame(self, frame: int) -> str:
        """Return the logged CSV snapshot for a specific frame, or empty string if not in log."""
        for f, block in self._frame_snapshots:
            if f == frame:
                return block
        return ""

    def get_frame_snapshots(self):
        return list(self._frame_snapshots)

    def get_snapshots_around_frame(self, center_frame: int, frames_before: int, frames_after: int) -> str:
        log = list(self._frame_snapshots)
        idx = next((i for i, (f, _) in enumerate(log) if f == center_frame), None)
        if idx is None:
            return ""
        start = max(0, idx - frames_before)
        end = min(len(log), idx + frames_after + 1)
        return log[start:end]

    def is_ball_positions_valid(self):
        return bool(self._ball_positions)

    def get_ball_positions(self):
        return list(self._ball_positions)
