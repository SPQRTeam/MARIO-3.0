"""
LLM refinement for geometry-based ball events (strike hints).

Uses a rolling CSV excerpt (frames before/after the event) so the model can
classify pass vs shot vs clearance etc. in plain language for spectators.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from src.commentator.commentator import Commentator, extract_json_comment, team_labels
from src.commentator.events_commentary import _team_label

logger = logging.getLogger(__name__)


def build_system_prompt(
    features: Any,
    field_width_mm: float,
    field_height_mm: float,
    language: str = "en",
) -> str:
    ln, _, rn, _ = team_labels(features)
    lang_note = (
        "Write your commentary in Italian."
        if language.lower().startswith("it")
        else "Write your commentary in English."
    )
    return f"""You are a sports commentator for a RoboCup Small Size League (SSL) robot soccer match.

Facts you can rely on:
- The left side of the field (negative X toward their goal) is "{ln}". The right side is "{rn}".
- The field is {field_width_mm:.0f} mm wide (X / length) and {field_height_mm:.0f} mm tall (Y / width). The origin (0,0) is the center of the field. Distances in the log are in millimeters.
- Each CSV row is one detection: robots and ball with `field_x`, `field_y` in field coordinates. `color` identifies team jersey when `type` is robot.
- `score_left` / `score_right` are match scores when available (may be missing or stale).
- The log is the only ground truth for positions; do not invent events that contradict it.

Your audience knows little about RoboCup: avoid jargon, coordinates, and radians. Prefer team names ({ln}, {rn}), not colors. Be brief and lively, like live TV commentary.

CRITICAL STYLE RULES FOR `comment`:
- Never mention raw numbers, coordinates, axis names, units, or formulas.
- Never refer to teams by colors (no "red", "blue", etc.); use team names only.
- Never write patterns like "x=", "y=", "meters", "millimeters", "radians", or explicit numeric measurements.
- Translate data into football language: "near midfield", "on the wing", "in the box", "under pressure", "quick switch".
- Sound like energetic live play-by-play, not an analytical report.
- Keep it to 1-2 short spoken sentences.

{lang_note}

Respond with a single JSON object only, no markdown:
{{"classification": "<one of: pass, shot, clearance, dribble, contested_ball, other>", "comment": "<one or two short telecast-style sentences, no numbers>"}}"""


def build_user_prompt(
    geometry_hint: str,
    csv_excerpt: str,
    zone_hint: str,
    zone_history_hint: str,
    zone_legend: str,
) -> str:
    return f"""Geometry pipeline hint (may be wrong; use the CSV to decide):
{geometry_hint}

Authoritative spatial hint from tracker (use this, do not guess another area):
{zone_hint}

Recent zone path:
{zone_history_hint}

{zone_legend}

CSV excerpt (frames before and after the event; same columns as the live log):
{csv_excerpt}

Classify what happened (pass, shot toward goal, defensive clearance, etc.) and write the comment for spectators in telecast style. Do not include numeric values or coordinates in the comment.
Do NOT call it midfield/center unless the authoritative hint explicitly says center or middle."""


def maybe_geometry_llm_comment(
    *,
    events: Any,
    features: Any,
    commentator: Commentator,
    llm: Any,
    context_frames: int,
    field_width_mm: float,
    field_height_mm: float,
    language: str,
    on_spoken: Callable[[str], None],
    dedup_key: Optional[str] = None,
    on_async_complete: Optional[Callable[[], None]] = None,
) -> bool:
    """
    If events indicate a ball strike hint, call the LLM asynchronously with CSV context.
    Returns True if a call was scheduled (caller should skip deterministic strike line if desired).
    """
    if not (events.home_team_ball_strike or events.away_team_ball_strike):
        return False

    trigger = events.trigger_frame
    if dedup_key is None:
        dedup_key = f"strike:{trigger}"

    if commentator.was_geometry_llm_sent(dedup_key):
        return False

    side = "left" if events.home_team_ball_strike else "right"
    who = _team_label(events, side)
    hint = (
        f"Geometry suggests {who} had a strong robot–ball interaction "
        f"(possible pass, shot on goal, or clearance — use the CSV excerpt to decide)."
    )

    excerpt = commentator.csv_excerpt_around_frame(trigger, context_frames, context_frames)
    if not excerpt.strip():
        logger.warning("geometry LLM: empty CSV excerpt for frame %s", trigger)
        return False

    system = build_system_prompt(features, field_width_mm, field_height_mm, language)
    zone_hint = commentator.ball_zone_hint(trigger)
    zone_hist = commentator.ball_zone_history_hint(trigger, steps=7)
    zone_legend = commentator.field_zone_legend()
    user = build_user_prompt(hint, excerpt, zone_hint, zone_hist, zone_legend)
    full_prompt = f"{system}\n\n---\n\n{user}"

    def _done(raw: str) -> None:
        try:
            raw_s = raw or ""
            preview = raw_s.strip().replace("\n", " ")[:160]
            print(
                f"[Commentary LLM] raw response ({len(raw_s)} chars): {preview}",
                flush=True,
            )
            text = extract_json_comment(raw_s)
            if text:
                on_spoken(text)
        finally:
            if on_async_complete is not None:
                on_async_complete()

    print(
        f"[Commentary LLM] call geometry refinement | backend={type(llm).__name__} | "
        f"frame={trigger} | ±{context_frames} frames | prompt_chars={len(full_prompt)}",
        flush=True,
    )
    commentator.mark_geometry_llm_sent(dedup_key)
    llm.llm_call_async(full_prompt, on_done=_done)
    return True
