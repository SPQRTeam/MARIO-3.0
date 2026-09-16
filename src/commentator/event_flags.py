"""Minimal commentary events: OCR goals + unified robot–ball strike (home/away)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Home = left_team in config (defends left goal); away = right_team (defends right goal).


@dataclass
class EventFlags:
    goal_for_left_team: bool = False
    goal_for_right_team: bool = False
    # Unified “robot kicks / strikes the ball” (replaces separate shot/pass from analyze_ball).
    home_team_ball_strike: bool = False
    away_team_ball_strike: bool = False
    # Legacy flags (unused by current ball analyzer; kept for older callers).
    home_team_shot: bool = False
    away_team_shot: bool = False
    home_team_pass: bool = False
    away_team_pass: bool = False
    score_left: Optional[int] = None
    score_right: Optional[int] = None
    left_team_name: Optional[str] = None
    right_team_name: Optional[str] = None
    left_team_full_name: Optional[str] = None
    right_team_full_name: Optional[str] = None
    has_ball: bool = False
    trigger_frame: Optional[int] = None
    # Geometry-computed action hint: "shot", "pass", or "" (set by BallMotionAnalyzer).
    ball_action_hint: str = ""
    # Acting team side for the strike: "left", "right", or "".
    ball_action_team: str = ""
    # Rich geometry JSON for the kinetics LLM (noisy — LLM decides the final interpretation).
    ball_action_geometry: str = ""
