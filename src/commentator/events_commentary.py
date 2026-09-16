"""
EVENTS_COMMENT — deterministic spoken lines from :class:`~src.commentator.event_flags.EventFlags`.

Priority: **goal** (OCR) → **robot–ball strike** → legacy shot → legacy pass.
Uses team names from ``left_*`` / ``right_*`` (home = left, away = right in config).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .event_flags import EventFlags


def _ocr_left_digit_tracks_left_field_team(comm_cfg: Any) -> bool:
    if comm_cfg is None:
        return True
    v = comm_cfg.ocr_left_digit_tracks_left_field_team
    if isinstance(v, bool):
        return v
    s = str(v).lower().strip()
    return s not in ("false", "0", "no", "field_right", "right")


def _scorer_field_side_when_left_score_up(comm_cfg: Any) -> str:
    return "left" if _ocr_left_digit_tracks_left_field_team(comm_cfg) else "right"


def _scorer_field_side_when_right_score_up(comm_cfg: Any) -> str:
    return "right" if _ocr_left_digit_tracks_left_field_team(comm_cfg) else "left"


def spoken_line_for_ocr_goal(left_score_increased: bool, config: Any) -> str:
    """Single line when OCR score changes (same mapping as ``goal_for_*_team``)."""
    comm = config.commentating
    feats = config.features
    # TODO these parameters are no longer under this config. match_commentary_tts should be writing it anyway, but it's kind of an antipattern that should be fixed.
    lf = str(
        getattr(feats, "left_team_full_name", None)
        or getattr(feats, "left_team_name", None)
        or "home"
    )
    rf = str(
        getattr(feats, "right_team_full_name", None)
        or getattr(feats, "right_team_name", None)
        or "away"
    )
    if left_score_increased:
        side = _scorer_field_side_when_left_score_up(comm)
    else:
        side = _scorer_field_side_when_right_score_up(comm)
    name = lf if side == "left" else rf
    return f"Goal! {name} scores!"


def _team_label(events: "EventFlags", side: str) -> str:
    full = events.left_team_full_name if side == "left" else events.right_team_full_name
    if full:
        return full
    short_name = events.left_team_name if side == "left" else events.right_team_name
    if short_name:
        return str(short_name)
    return f"the {side} team"


def goal_line(events: "EventFlags", comm_cfg: Any) -> Optional[str]:
    if not comm_cfg.deterministic_goal_speech:
        return None
    left = _team_label(events, "left")
    right = _team_label(events, "right")
    if events.goal_for_left_team:
        side = _scorer_field_side_when_left_score_up(comm_cfg)
        name = left if side == "left" else right
        return f"Goal! {name} scores!"
    if events.goal_for_right_team:
        side = _scorer_field_side_when_right_score_up(comm_cfg)
        name = left if side == "left" else right
        return f"Goal! {name} scores!"
    return None


def shot_line(events: "EventFlags", comm_cfg: Any) -> Optional[str]:
    if not comm_cfg.deterministic_shot_speech:
        return None
    left = _team_label(events, "left")
    right = _team_label(events, "right")
    if events.away_team_shot:
        return f"{right} shot."
    if events.home_team_shot:
        return f"{left} shot."
    return None


def ball_strike_line(events: "EventFlags", comm_cfg: Any) -> Optional[str]:
    """Single technical line for robot–ball contact (kick / strike), no pass vs shot."""
    if not comm_cfg.deterministic_ball_strike_speech:
        return None
    left = _team_label(events, "left")
    right = _team_label(events, "right")
    if events.away_team_ball_strike:
        return f"{right} robot–ball strike."
    if events.home_team_ball_strike:
        return f"{left} robot–ball strike."
    return None


def pass_line(events: "EventFlags", comm_cfg: Any) -> Optional[str]:
    if not comm_cfg.deterministic_pass_speech:
        return None
    left = _team_label(events, "left")
    right = _team_label(events, "right")
    if events.away_team_pass:
        return f"{right} pass."
    if events.home_team_pass:
        return f"{left} pass."
    return None


def event_comment_line(events: "EventFlags", comm_cfg: Any) -> Optional[str]:
    """One line for this frame: goal, else robot–ball strike, else shot, else pass."""
    line = goal_line(events, comm_cfg)
    if line is not None:
        return line
    line = ball_strike_line(events, comm_cfg)
    if line is not None:
        return line
    line = shot_line(events, comm_cfg)
    if line is not None:
        return line
    return pass_line(events, comm_cfg)


# Back-compat names
deterministic_goal_tts_line = goal_line
deterministic_ball_strike_tts_line = ball_strike_line
deterministic_shot_tts_line = shot_line
deterministic_pass_tts_line = pass_line
deterministic_priority_tts_line = event_comment_line
