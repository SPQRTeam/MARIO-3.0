"""
Periodic scene commentary when there is no ball-strike / geometry event.

Uses a wider tail of CSV frames plus a short rolling history of prior assistant lines
so the model can stay coherent and occasionally summarize.
"""

from __future__ import annotations

import json
from typing import Any, Callable, List

from src.commentator.commentator import Commentator, extract_json_comment, team_labels


def build_periodic_system_prompt(
    features: Any,
    field_width_mm: float,
    field_height_mm: float,
    language: str = "en",
) -> str:
    _ = field_width_mm, field_height_mm
    ln, _, rn, _ = team_labels(features)
    lang_note = (
        "OUTPUT LANGUAGE: Italian only. One sentence, live TV / radio, natural spoken Italian."
        if language.lower().startswith("it")
        else "OUTPUT LANGUAGE: English only. One sentence, live TV tone."
    )
    return f"""You are the **voice** of a big humanoid-robot soccer match — not a status screen, not a lab report.
Teams: "{ln}" (defend the left goal, attack to the right) and "{rn}" (defend the right goal, attack to the left).

Your job in this **periodic** line: one sentence that **pulls the listener in** — mood, stakes, rhythm, a clear image.
Think **radio energy**: a bit of drama, warmth, or edge. It should feel **lived-in**, not recycled.

Inputs (English): tactical digest, formation, optional phase, your last lines. Use them as **colour**, not as a list to read aloud.

Narrative modes (JSON "mode", pick one):
  - "describe_now"        — something is moving; paint the instant.
  - "recap_phase"         — quick beat on what the last moments felt like.
  - "build_tension"       — breath before the next clash; who blinks first.
  - "emotional_reaction"  — after something big, if the phase allows.
  - "breathe"             — slower beat, still **human** (atmosphere, patience, simmering noise) — never a dead checklist.

**Engaged / "romanzato" (aim for this):**
- Change **how** you open every time: start with a team name, a verb, a short beat, a feeling — not the same scaffold.
- Use **one** concrete picture: lines, midfield, who looks braver, who is waiting — without inventing shots that did not happen.
- Let the listener **feel** whether the game is tight, messy, electric, or cagey.

**Flat (avoid):** two teams + "in midfield" + "watching" + "tense moment" in the same shape as your last lines; generic
  balancing language with no pulse; sounding like a changelog.

STRICT safety (still obey):
- Team names ("{ln}" / "{rn}") only — no jersey colours; do not say the word "left" or "right" as team labels.
- No numbers, coordinates, frame ids, or percentages.
- Do not copy phrasing from your previous lines; **strongly** vary syntax and imagery.
- Length: about **10–24 words** — enough for one strong image; one sentence only.
- If RECENT ACTION LEVEL is **none** or **low**: stay in "breathe" but keep it **alive** (standoff, chess, heart rate of the match),
  not a vague "both sides hold shape" every time. Only assert strong tactical dominance if formation + digest support it.
- After a goal / kickoff-style pause (match_phase): anticipation, reset, nerves — not a fake open-field blitz.
- No ball track: still **never** mention camera, visibility, "out of view", "off screen", "not seen", etc. Talk **people and space**
  only, as if you simply choose not to mention the ball — not as a technical excuse.
- Forbidden openers/templates: "With the ball…", "With the ball out of view", "With no ball in sight", "Con la palla…",
  and close paraphrases of those.

{lang_note}

Output ONE JSON object, no markdown:
{{"mode": "describe_now|recap_phase|build_tension|emotional_reaction|breathe",
  "comment": "<one vivid, engaged sentence in the output language above>"}}"""


def build_periodic_user_prompt(
    *,
    trigger_frame: int,
    fps: float,
    history_lines: List[str],
    narrative_summary: str,
    zone_hint: str,
    score_hint: str,
    recent_events_hint: str,
    has_ball: bool,
    formation_hint: str = "",
    match_phase: str = "live",
    phase_hint: str = "",
    current_snapshot_hint: str = "",
) -> str:
    _ = trigger_frame, fps
    hist = "\n".join(f"- {h}" for h in history_lines) if history_lines else "(none yet)"
    # Do not write "ball not visible" here: LLMs echo it as "with the ball out of view".
    ball_note = (
        "Ball track: OK in this frame — you may refer to the ball if useful."
        if has_ball
        else (
            "Ball track: absent in this frame — speak ONLY about team shape, midfield, tension, "
            "re-start or standoff. Do not mention visibility, camera, or that the ball cannot be seen."
        )
    )
    formation_block = f"\n═══ CURRENT FORMATION (English) ═══\n{formation_hint}\n" if formation_hint else ""
    phase_block = f"\n═══ MATCH PHASE ═══\nmatch_phase = {match_phase}\n{phase_hint}\n" if match_phase != "live" else ""
    snapshot_block = (
        f"\n═══ CURRENT SNAPSHOT (highest priority) ═══\n{current_snapshot_hint}\n"
        if current_snapshot_hint
        else ""
    )
    return f"""═══ TACTICAL DIGEST (English, last few seconds) ═══
{narrative_summary}

Field / zone hint: {zone_hint}
{ball_note}
{formation_block}{phase_block}{snapshot_block}

═══ SCORE ═══
{score_hint}

═══ RECENT FLAGS (from tracker) ═══
{recent_events_hint}

═══ YOUR PREVIOUS LINES (do not copy; move the story forward) ═══
{hist}

Write ONE line for TTS: **engaged**, **varied**, **human** — like you care who wins and you are watching with the crowd.
Avoid repeating the same rhythm or opener as above. JSON with mode + comment as in the system instructions."""


def maybe_periodic_scene_llm(
    *,
    events: Any,
    features: Any,
    comm_cfg: Any,
    eventproc: Any,
    commentator: Commentator,
    llm: Any,
    field_width_mm: float,
    field_height_mm: float,
    language: str,
    tail_frames: int,
    stride: int,
    fps: float,
    history_assistant_lines: List[str],
    history_max_turns: int,
    score_hint: str,
    recent_events_hint: str,
    on_spoken: Callable[[str], None],
    on_async_complete: Callable[[], None],
) -> bool:
    """
    If there is enough log data, call the LLM with a wide tail + history.
    ``history_assistant_lines`` is last-N lines (mutated by caller after speech).
    """
    _ = tail_frames, stride  # kept for backwards-compatible caller signature
    trigger = events.trigger_frame

    # New storytelling path: give the LLM a tactical narrative digest, not raw CSV.
    narrative_summary = commentator.match_narrative_summary(trigger, window_seconds=30.0)
    if not narrative_summary.strip():
        return False

    hist = history_assistant_lines[-history_max_turns:] if history_max_turns > 0 else []

    system = build_periodic_system_prompt(features, field_width_mm, field_height_mm, language)
    zone_hint = commentator.ball_zone_hint(trigger)
    formation_hint = commentator.team_formation_hint(trigger, label_language="en")
    ln, _, rn, _ = team_labels(features)

    # Match phase inference (cheap, purely from existing signals).
    has_ball = events.has_ball
    last_seen = eventproc.last_seen_ball_frame
    ball_gap_frames = (trigger - int(last_seen)) if (last_seen is not None) else -1
    match_phase = "live"
    phase_hint = ""

    # Priority 1: post-goal wait (within N frames from last goal).
    post_goal_window = comm_cfg.post_goal_wait_window_frames
    last_goal_f = getattr(commentator, "_last_goal_frame", None)  # questa getattr è effettivamente giustificata per come è fatto il commentator ad oggi (ma, insomma, si farebbe meglio a metterlo direttamente a None)
    last_goal_team_key = getattr(commentator, "_last_goal_team", None)  # presumo anche questa
    if last_goal_f is not None and (trigger - int(last_goal_f)) <= post_goal_window:
        ln = commentator._left_team_name
        rn = commentator._right_team_name
        scorer = {"left": ln, "right": rn}.get(str(last_goal_team_key or ""), None)
        scorer_note = f" (scorer: {scorer})" if scorer else ""
        match_phase = "post_goal_wait"
        phase_hint = (
            f"A goal was just scored{scorer_note}. Teams are realigning for a central restart. "
            "This is a STOPPAGE: do not describe an ongoing open-field attack, pressing wave, or "
            "sustained one-way dominance. You may use mood, the score, anticipation of the next phase, "
            "or a calm wide shot of the field — not live tactical assault. Do not name the ball."
        )
    # Priority 2: ball absent for a while (occlusion or dead phase).
    elif not has_ball and ball_gap_frames > int(2.0 * max(1.0, fps)):
        match_phase = "ball_absent"
        secs = ball_gap_frames / max(1.0, fps)
        phase_hint = (
            f"Ball not tracked for ~{secs:.0f}s. Do not say the ball is 'lost' or 'missing' on air. "
            "Talk about structure, space, how long the phase feels, or quiet tension instead."
        )

    # Current frame / immediate tracker hints (must dominate over long-window digest).
    act = events.ball_action_hint.strip().lower()
    team = events.ball_action_team.strip().lower()
    team_name = {"left": ln, "right": rn}.get(team, "unknown")
    if match_phase == "post_goal_wait":
        # Formation can still look "attacking" while robots set up; ignore local tracker heuristics.
        current_snapshot_hint = (
            "Override: stoppage / restart after a goal. Ignore short tracker tags that look like 'pass' or 'shot' "
            f"of {team_name} here — the priority is the pause, not open play."
        )
    elif has_ball and act in ("pass", "shot", "clearance") and team in ("left", "right"):
        current_snapshot_hint = (
            f"Tracker immediate action: {act} by {team_name}. "
            "Use as the freshest local signal unless the digest and RECENT ACTION LEVEL say otherwise."
        )
    elif has_ball and team in ("left", "right"):
        current_snapshot_hint = (
            f"Tracker suggests the ball is currently with {team_name}. "
            "If the digest says no recent action, still keep wording soft (not a full assault)."
        )
    elif has_ball:
        current_snapshot_hint = (
            "Ball is visible but team attribution is shaky. Stay neutral; do not assert clear possession."
        )
    else:
        current_snapshot_hint = (
            "No ball track this instant: describe team lines and midfield feel only; "
            "do not explain camera/visibility — same rules as the scene note above."
        )

    user = build_periodic_user_prompt(
        trigger_frame=trigger,
        fps=fps,
        history_lines=hist,
        narrative_summary=narrative_summary,
        zone_hint=zone_hint,
        score_hint=score_hint,
        recent_events_hint=recent_events_hint,
        has_ball=has_ball,
        formation_hint=formation_hint,
        match_phase=match_phase,
        phase_hint=phase_hint,
        current_snapshot_hint=current_snapshot_hint,
    )
    full_prompt = f"{system}\n\n---\n\n{user}"

    def _done(raw: str) -> None:
        try:
            raw_s = raw or ""
            preview = raw_s.strip().replace("\n", " ")[:160]
            print(
                f"[Commentary LLM] periodic raw ({len(raw_s)} chars): {preview}"
                + ("…" if len(raw_s) > 160 else ""),
                flush=True,
            )
            mode = ""
            try:
                if "{" in raw_s:
                    start = raw_s.index("{")
                    end = raw_s.rindex("}") + 1
                    obj = json.loads(raw_s[start:end])
                    mode = str(obj.get("mode", ""))
            except (json.JSONDecodeError, ValueError):
                pass
            if mode:
                print(f"[Commentary LLM] periodic mode={mode}", flush=True)
            text = extract_json_comment(raw_s)
            # Guardrail: one sentence; cap length so periodic does not run over urgent lines.
            if text:
                # First sentence only.
                for sep in (". ", "! ", "? "):
                    idx = text.find(sep)
                    if idx > 0:
                        text = text[: idx + 1].strip()
                        break
                max_chars = 200
                if len(text) > max_chars:
                    text = text[:max_chars].rstrip(" ,;:") + "..."
            if text:
                on_spoken(text)
        finally:
            on_async_complete()

    print(
        f"[Commentary LLM] periodic scene | backend={type(llm).__name__} | frame={trigger} | "
        f"tail_frames={tail_frames} stride={stride} | prompt_chars={len(full_prompt)}",
        flush=True,
    )
    llm.llm_call_async(full_prompt, on_done=_done)
    return True


def maybe_goal_state_llm(
    *,
    events: Any,
    features: Any,
    llm: Any,
    language: str,
    score_hint: str,
    recent_events_hint: str,
    on_spoken: Callable[[str], None],
    on_async_complete: Callable[[], None],
) -> bool:
    """One-shot LLM line right after a goal event with score context."""
    if not (events.goal_for_left_team or events.goal_for_right_team):
        return False
    ln, lc, rn, rc = team_labels(features)
    it = language.lower().startswith("it")
    out = "Italian" if it else "English"
    scorer = "left side" if events.goal_for_left_team else "right side"
    prompt = f"""You are a live sports commentator for RoboCup humanoid robot soccer.
Instructions in English. OUTPUT: one short line in {out} only.

Teams: left side is "{ln}" (jersey colour {lc}); right side is "{rn}" (jersey colour {rc}).

Score context: {score_hint}
Recent flags: {recent_events_hint}
Goal just scored for: {scorer}.

React in one telecast-style sentence; mention score situation if obvious. No coordinates, no jargon, no colours in speech.
Plain text only."""

    def _done(raw: str) -> None:
        try:
            text = (raw or "").strip()
            if text:
                on_spoken(text)
        finally:
            on_async_complete()

    llm.llm_call_async(prompt, on_done=_done)
    return True
