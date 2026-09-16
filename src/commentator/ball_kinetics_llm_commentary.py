"""
Second LLM role: on robot–ball strike (ball clearly moving), classify **pass vs shot**
and **which team** from CSV context only. Used with an attention lead-in and TTS interrupt.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

from src.commentator.commentator import Commentator, extract_json_comment, team_labels

logger = logging.getLogger(__name__)


def _color_to_team_map(features: Any, ln: str, rn: str) -> str:
    """Build a human-readable mapping of jersey color strings to team names for the prompt."""
    # TODO this parameter is no longer under this config. match_commentary_tts should be writing it anyway, but it's kind of an antipattern that should be fixed.
    left_color = str(getattr(features, "left_team_color", "") or "").strip().lower()
    right_color = str(getattr(features, "right_team_color", "") or "").strip().lower()
    lines = []
    if left_color:
        lines.append(f'  color="{left_color}" → team "{ln}" (left)')
    if right_color:
        lines.append(f'  color="{right_color}" → team "{rn}" (right)')
    lines.append('  color="unknown"   → robot team could not be identified (ignore for team attribution)')
    lines.append('  color="ball"      → this row is the ball, not a robot')
    return "\n".join(lines)


def classify_action(
    geometry_json: str,
    pre_kick_positions: Optional[list] = None,
    xlimit_mm: float = 7000.0,
    fps: float = 30.0,
) -> dict:
    """
    Deterministically decide (kicker_team, action_type) from geometry + pre-kick trend.

    Python does the reasoning — the LLM only writes the commentary sentence.
    Returns: {"acting_team": "left"|"right"|"unknown",
              "classification": "shot"|"pass"|"clearance"|"unclear",
              "reasoning": "<short human-readable trace>"}
    """
    try:
        g = json.loads(geometry_json) if isinstance(geometry_json, str) else geometry_json
    except (json.JSONDecodeError, TypeError):
        return {"acting_team": "unknown", "classification": "unclear", "reasoning": "invalid geometry"}

    dx      = float(g.get("dx_mm", 0))
    speed   = float(g.get("speed_mms", 0))
    cos_r   = float(g.get("cos_toward_right_goal", 0))
    cos_l   = float(g.get("cos_toward_left_goal", 0))
    x0      = float(g.get("ball_start_x_mm", 0))
    x1      = float(g.get("ball_end_x_mm", x0))
    nr_team = str(g.get("nearest_robot_team", "unknown"))
    nr_dist = float(g.get("nearest_robot_dist_mm", -1))
    nearby  = g.get("nearby_robots", []) or []

    reasoning_parts: list[str] = []

    # ── STEP 1: identify the kicker ──
    # Priority 1: pose-based — a robot whose motion profile screams "I just kicked this"
    # (approached the ball + heading aligned with kick + post-kick follow-through).
    # This is MORE reliable than color+distance because CNN color detection is noisy.
    likely_kickers = [
        r for r in nearby
        if str(r.get("verdict")) == "likely_kicker"
        and r.get("team") in ("left", "right")
        and float(r.get("dist_mm", 9999)) <= 900
    ]
    kicker = "unknown"
    if likely_kickers:
        best = max(likely_kickers, key=lambda r: float(r.get("likely_kicker_score", 0)))
        kicker = str(best["team"])
        reasoning_parts.append(
            f"pose-based: robot id={best['id']} ({best['team']}) has kicker profile "
            f"(dist={best['dist_mm']}mm, approach={best.get('approach_speed_mms',0):.0f}mm/s, "
            f"pre_align={best.get('pre_align_with_kick',0):.2f}, "
            f"post_align={best.get('post_align_with_kick',0):.2f}) → kicker={kicker}"
        )
    # Priority 2: nearest robot (if very close AND not flagged as moving_away)
    if kicker == "unknown" and nr_team in ("left", "right") and 0 <= nr_dist <= 600:
        # Verify the nearest isn't flagged as moving away
        nr_entry = next((r for r in nearby if r.get("team") == nr_team and float(r.get("dist_mm", 0)) == nr_dist), None)
        if nr_entry is None or nr_entry.get("verdict") != "moving_away":
            kicker = nr_team
            reasoning_parts.append(f"nearest robot ({nr_team}) within {nr_dist:.0f}mm → kicker={nr_team}")
        else:
            reasoning_parts.append(
                f"nearest robot ({nr_team}) is tagged 'moving_away' → NOT the kicker"
            )

    # Priority 3: pre-kick trend
    if kicker == "unknown" and pre_kick_positions and len(pre_kick_positions) >= 4:
        recent = pre_kick_positions[-15:]
        trend_dx = float(recent[-1][1]) - float(recent[0][1])
        n_fr = float(recent[-1][0] - recent[0][0])
        trend_speed = abs(trend_dx) / (n_fr / fps) if n_fr > 0 else 0.0
        if trend_speed > 300:
            if trend_dx > 200:
                kicker = "left"
                reasoning_parts.append(
                    f"pre-kick trend rightward at {trend_speed:.0f}mm/s → left was advancing → kicker=left"
                )
            elif trend_dx < -200:
                kicker = "right"
                reasoning_parts.append(
                    f"pre-kick trend leftward at {trend_speed:.0f}mm/s → right was advancing → kicker=right"
                )

    # Priority 4: direction-only fallback
    if kicker == "unknown":
        if cos_r > 0.7 and x0 < 0:
            kicker = "left"
            reasoning_parts.append("no close robot; ball in left half kicked toward right goal → likely left")
        elif cos_l > 0.7 and x0 > 0:
            kicker = "right"
            reasoning_parts.append("no close robot; ball in right half kicked toward left goal → likely right")
        else:
            reasoning_parts.append("kicker could not be determined with confidence")

    # ── STEP 2: classify the action ──
    # "deep in own end" = within 33% of the field's half-width from the own goal line.
    deep_own_thresh = xlimit_mm * 0.66  # i.e. |x0| > 0.66*xlimit means deep in that end
    classification = "unclear"
    if kicker == "left":
        # left's own goal is at −xlimit; left attacks rightward (+X).
        deep_in_own_end = x0 < -deep_own_thresh
        if deep_in_own_end and dx > 400 and speed > 800:
            # From deep in own end, kicked away → clearance (priority over shot from distance)
            classification = "clearance"
            reasoning_parts.append(
                f"kicked from deep in own end (x={x0:.0f}mm) away from own goal → clearance"
            )
        # Long punt from own territory toward midfield / not deep in opponent end: any +X read looks
        # "goalward" (high cos_r) but it is a relieving clearance, not a shot.
        elif (
            x0 < -0.22 * xlimit_mm
            and x1 < 0.44 * xlimit_mm
            and speed > 1000
            and cos_r > 0.48
            and abs(dx) > 400
        ):
            classification = "clearance"
            reasoning_parts.append(
                f"own-half long release landing short of deep attack zone "
                f"(x0={x0:.0f}mm, x1={x1:.0f}mm) → clearance not shot"
            )
        elif cos_r > 0.7 and speed > 1500 and x1 > 0.36 * xlimit_mm:
            # Require the ball to progress well into the opponent half for a "shot" label
            classification = "shot"
            reasoning_parts.append(
                f"cos_toward_right_goal={cos_r:.2f} & speed={speed:.0f} & end x={x1:.0f}mm → shot"
            )
        elif cos_r > 0.88 and speed > 1800:
            # Exception: laser-like strike may still be a shot even if x1 is noisy
            classification = "shot"
            reasoning_parts.append(f"very tight goal alignment cos={cos_r:.2f} & speed={speed:.0f} → shot")
        elif abs(dx) > 300:
            classification = "pass"
            reasoning_parts.append("forward movement without goal alignment → pass")
    elif kicker == "right":
        # right's own goal is at +xlimit; right attacks leftward (−X).
        deep_in_own_end = x0 > deep_own_thresh
        if deep_in_own_end and dx < -400 and speed > 800:
            classification = "clearance"
            reasoning_parts.append(
                f"kicked from deep in own end (x={x0:.0f}mm) away from own goal → clearance"
            )
        elif (
            x0 > 0.22 * xlimit_mm
            and x1 > -0.44 * xlimit_mm
            and speed > 1000
            and cos_l > 0.48
            and abs(dx) > 400
        ):
            classification = "clearance"
            reasoning_parts.append(
                f"own-half long release landing short of deep attack zone "
                f"(x0={x0:.0f}mm, x1={x1:.0f}mm) → clearance not shot"
            )
        elif cos_l > 0.7 and speed > 1500 and x1 < -0.36 * xlimit_mm:
            classification = "shot"
            reasoning_parts.append(
                f"cos_toward_left_goal={cos_l:.2f} & speed={speed:.0f} & end x={x1:.0f}mm → shot"
            )
        elif cos_l > 0.88 and speed > 1800:
            classification = "shot"
            reasoning_parts.append(f"very tight goal alignment cos={cos_l:.2f} & speed={speed:.0f} → shot")
        elif abs(dx) > 300:
            classification = "pass"
            reasoning_parts.append("forward movement without goal alignment → pass")
    else:
        # Unknown kicker: direction-only (rarely used; kinetics is skipped if team unknown)
        if (
            x0 < -0.22 * xlimit_mm
            and x1 < 0.44 * xlimit_mm
            and speed > 1000
            and cos_r > 0.48
            and abs(dx) > 400
        ):
            classification = "clearance"
            reasoning_parts.append("direction-only: own-half long ball, short of deep attack end → clearance")
        elif (
            x0 > 0.22 * xlimit_mm
            and x1 > -0.44 * xlimit_mm
            and speed > 1000
            and cos_l > 0.48
            and abs(dx) > 400
        ):
            classification = "clearance"
            reasoning_parts.append("direction-only: own-half long ball, short of deep attack end → clearance")
        elif cos_r > 0.7 and speed > 1500 and x1 > 0.36 * xlimit_mm:
            classification = "shot"
            reasoning_parts.append(f"shot-like toward right goal (cos={cos_r:.2f}, end x={x1:.0f}mm)")
        elif cos_l > 0.7 and speed > 1500 and x1 < -0.36 * xlimit_mm:
            classification = "shot"
            reasoning_parts.append(f"shot-like toward left goal (cos={cos_l:.2f}, end x={x1:.0f}mm)")

    return {
        "acting_team": kicker,
        "classification": classification,
        "reasoning": "; ".join(reasoning_parts) or "insufficient signal",
    }


def build_action_narrative(
    geometry_json: str,
    ln: str,
    rn: str,
    xlimit_mm: float,
    fps: float = 30.0,
    pre_kick_positions: Optional[list] = None,
) -> str:
    """
    Convert raw geometry JSON into a plain-English factual description of the action.

    ``pre_kick_positions``: optional list of (frame, x, y) tuples before the kick,
    used to compute the ball's trend direction and infer which team was advancing.
    """
    try:
        g = json.loads(geometry_json) if isinstance(geometry_json, str) else geometry_json
    except (json.JSONDecodeError, TypeError):
        return f"Raw geometry: {geometry_json}"

    dx      = float(g.get("dx_mm", 0))
    dy      = float(g.get("dy_mm", 0))
    disp    = float(g.get("disp_mm", 0))
    speed   = float(g.get("speed_mms", 0))
    cos_r   = float(g.get("cos_toward_right_goal", 0))
    cos_l   = float(g.get("cos_toward_left_goal", 0))
    x0      = float(g.get("ball_start_x_mm", 0))
    y0      = float(g.get("ball_start_y_mm", 0))
    x1      = float(g.get("ball_end_x_mm", 0))
    y1      = float(g.get("ball_end_y_mm", 0))
    nr_team = str(g.get("nearest_robot_team", "unknown"))
    nr_dist = float(g.get("nearest_robot_dist_mm", -1))
    nearby  = g.get("nearby_robots", []) or []

    def _zone(x: float, y: float) -> str:
        # Describe the zone from a SPATIAL perspective, not a team-possession perspective.
        # e.g. "near HTWK's goal" means the ball is physically near that goal,
        # NOT that HTWK is in possession — either team could be acting there.
        if x < -xlimit_mm * 0.66:
            return f"deep in {ln}'s end (near their goal)"
        if x < -xlimit_mm * 0.33:
            return f"in {ln}'s half, attacking third"
        if x > xlimit_mm * 0.66:
            return f"deep in {rn}'s end (near their goal)"
        if x > xlimit_mm * 0.33:
            return f"in {rn}'s half, attacking third"
        return "midfield"

    # Kicker description
    if nr_team == "left" and nr_dist >= 0:
        kicker = f"{ln} robot ({nr_dist:.0f}mm from ball)"
    elif nr_team == "right" and nr_dist >= 0:
        kicker = f"{rn} robot ({nr_dist:.0f}mm from ball)"
    else:
        kicker = "unknown (no robot close enough to identify)"

    # Direction toward goal
    goal_dir_parts = []
    if cos_r > 0.5:
        goal_dir_parts.append(f"toward {rn}'s goal (cos={cos_r:.2f})")
    elif cos_r > 0.25:
        goal_dir_parts.append(f"mildly toward {rn}'s goal (cos={cos_r:.2f})")
    if cos_l > 0.5:
        goal_dir_parts.append(f"toward {ln}'s goal (cos={cos_l:.2f})")
    elif cos_l > 0.25:
        goal_dir_parts.append(f"mildly toward {ln}'s goal (cos={cos_l:.2f})")
    if not goal_dir_parts:
        goal_dir_parts.append("no strong goal alignment (lateral or ambiguous direction)")

    # Horizontal movement
    if abs(dx) > 50:
        horiz = f"{'rightward' if dx > 0 else 'leftward'} by {abs(dx):.0f}mm"
    else:
        horiz = "mostly lateral"

    start_zone = _zone(x0, y0)
    end_zone   = _zone(x1, y1)

    # Distance from ball origin to the two goal lines (±xlimit_mm on X axis)
    dist_to_left_goal  = abs(x0 - (-xlimit_mm))
    dist_to_right_goal = abs(x0 - xlimit_mm)

    lines = [
        f"Ball origin:  ({x0:.0f}, {y0:.0f})mm  — {start_zone}"
        f"  [{dist_to_left_goal:.0f}mm from {ln}'s goal line,"
        f" {dist_to_right_goal:.0f}mm from {rn}'s goal line]",
        f"Ball landing: ({x1:.0f}, {y1:.0f})mm  — {end_zone}",
        f"Displacement: {horiz}, total {disp:.0f}mm at {speed:.0f} mm/s",
        f"Direction:    {'; '.join(goal_dir_parts)}",
        f"Nearest robot at kick origin: {kicker}",
    ]

    # Per-robot scene description: surfaces pose-based evidence (approach, alignment, verdict)
    # so the LLM can reason about WHO kicked even when a single color is misclassified.
    if nearby:
        lines.append("")
        lines.append("Robots near the ball at kick origin (sorted by kicker-likelihood):")
        for r in nearby[:5]:
            team = r.get("team", "unknown")
            team_name = {"left": ln, "right": rn}.get(team, "unknown-team")
            color = r.get("color", "unknown")
            rid = r.get("id", "?")
            d = r.get("dist_mm", -1)
            verdict = r.get("verdict", "unknown")
            approach = r.get("approach_speed_mms", 0.0)
            pre_align = r.get("pre_align_with_kick", 0.0)
            post_align = r.get("post_align_with_kick", 0.0)
            score = r.get("likely_kicker_score", 0.0)

            # Human-readable motion description
            motion_bits = []
            if approach > 200:
                motion_bits.append(f"approached ball at {approach:.0f}mm/s")
            elif approach < -200:
                motion_bits.append(f"moved AWAY from ball at {-approach:.0f}mm/s")
            else:
                motion_bits.append("roughly stationary")

            if pre_align > 0.5:
                motion_bits.append("heading matched kick direction")
            elif pre_align < -0.3:
                motion_bits.append("heading opposite to kick direction")

            if post_align > 0.5:
                motion_bits.append("followed through after kick")
            elif post_align < -0.3:
                motion_bits.append("went opposite to ball after kick")

            lines.append(
                f"  - id={rid} team={team_name} (color={color}) "
                f"dist={d:.0f}mm | verdict={verdict} (score={score:.2f}) | "
                f"{', '.join(motion_bits)}"
            )

    # Pre-kick ball trend: was the ball moving toward the left or right goal before the kick?
    # This reveals which team was advancing — the kicker moves the ball away from their own goal,
    # so if the ball was trending rightward before the kick then reversed leftward, B-Human attacked.
    if pre_kick_positions and len(pre_kick_positions) >= 4:
        # Use a window of the last few positions to compute trend direction
        recent = pre_kick_positions[-8:]
        x_first = float(recent[0][1])
        x_last  = float(recent[-1][1])
        trend_dx = x_last - x_first
        n_frames = float(recent[-1][0] - recent[0][0])
        trend_speed = abs(trend_dx) / (n_frames / fps) if n_frames > 0 else 0.0

        # Only report trend if it's meaningful (not near-stationary)
        if trend_speed > 100:
            if trend_dx > 150:
                trend_team = ln  # was moving rightward = ln advancing
                trend_desc = f"rightward (toward {rn}'s goal) at ~{trend_speed:.0f} mm/s"
            elif trend_dx < -150:
                trend_team = rn  # was moving leftward = rn advancing
                trend_desc = f"leftward (toward {ln}'s goal) at ~{trend_speed:.0f} mm/s"
            else:
                trend_team = None
                trend_desc = "roughly stationary or ambiguous"

            # Direction reversal: if the kick goes opposite to the pre-kick trend,
            # it's likely a tackle/interception followed by a shot in the other direction.
            kick_reverses = (trend_dx > 150 and dx < -150) or (trend_dx < -150 and dx > 150)
            reversal_note = (
                f" — DIRECTION REVERSAL: ball was advancing {trend_desc} then kicked the other way"
                f" (likely {rn if trend_dx > 0 else ln} intercepted and shot)"
                if kick_reverses else ""
            )
            lines.append(
                f"Pre-kick ball trend: moving {trend_desc}"
                + (f" ({trend_team} was advancing)" if trend_team else "")
                + reversal_note
            )
        else:
            lines.append("Pre-kick ball trend: stationary or very slow")

    return "\n".join(lines)



def build_kinetics_system_prompt(
    features: Any,
    field_width_mm: float,
    field_height_mm: float,
    language: str = "it",
    xlimit_mm: Optional[float] = None,
) -> str:
    ln, _, rn, _ = team_labels(features)
    color_map = _color_to_team_map(features, ln, rn)
    it = str(language).lower().startswith("it")
    lang_note = (
        "OUTPUT: one spoken sentence in **Italian** (natural TV/radio style)."
        if it
        else "OUTPUT: one spoken sentence in **English** (natural TV style)."
    )
    half = xlimit_mm if xlimit_mm is not None else field_width_mm / 2.0
    return f"""You are a live-TV commentator for a RoboCup humanoid robot soccer match.
Instructions are in English. {lang_note}

A ball action just happened. The system has ALREADY decided who kicked and whether it was a
shot, pass, or clearance. Your only job is to write ONE matching commentary sentence in the
output language — do not re-classify, do not “fix” the label.

═══ FIELD ═══
- Origin at CENTER. X increases left → right.
- LEFT goal at X ≈ −{half:.0f} mm — defended by "{ln}" (they attack to the right, +X).
- RIGHT goal at X ≈ +{half:.0f} mm — defended by "{rn}" (they attack to the left, −X).

Jersey color string → who it refers to (for reading the log only; do not use colors in speech):
{color_map}

═══ RULES ═══
1. Copy "acting_team" and "classification" from the user message into JSON exactly. Never change them.
2. In the "comment" field, use the team full names ("{ln}" / "{rn}") only — not colors, not "left"/"right".
3. If acting_team is "unknown", describe the action without naming a kicker.
4. Use the narrative only for phrasing, not to override the label.
5. No coordinates, no frame numbers, no mention of “AI”, “model”, or “detection”.

═══ TONE (match the label in the output language) ═══
- shot: finishing energy — strike toward goal, hit from range.
- clearance: relief / defence — hammers it away, row clearance, not a goal attempt.
- pass: build-up — feeds a teammate, switches play.
- unclear: ball moved; stay non-committal on intent.

{lang_note}

One JSON object only, no markdown. Echo classification and acting_team; fill "comment":
{{"classification": "<as given>", "acting_team": "<as given>", "comment": "<one sentence>"}}"""


def build_kinetics_user_prompt(
    action_narrative: str,
    ball_motion_summary: str,
    classification: str,
    acting_team: str,
    classification_reasoning: str = "",
    kick_origin_frame: Optional[int] = None,
    kick_origin_snapshot: str = "",
    ln: str = "left team",
    rn: str = "right team",
    language: str = "it",
) -> str:
    origin_block = ""
    if kick_origin_snapshot.strip():
        origin_block = (
            f"\n\n═══ ROBOTS AT KICK ORIGIN (frame {kick_origin_frame}) ═══\n"
            f"{kick_origin_snapshot}\n"
        )
    acting_team_display = {"left": ln, "right": rn}.get(acting_team, "unknown")
    reasoning_block = (
        f"System reasoning: {classification_reasoning}\n" if classification_reasoning else ""
    )
    it = str(language).lower().startswith("it")
    out_lbl = "Italian" if it else "English"
    return f"""═══ ACTION DESCRIPTION (English) ═══
{action_narrative}

═══ BALL TRAJECTORY (positions before → gap → after) ═══
{ball_motion_summary}{origin_block}

═══ CLASSIFICATION (LOCKED — do not change) ═══
acting_team:    {acting_team}
classification: {classification}
{reasoning_block}
Spoken name to use: "{acting_team_display}".

Write exactly ONE {out_lbl} live-TV sentence in "comment" that matches the classification.

JSON rules:
- "acting_team"    = EXACT machine code "{acting_team}" (not the display name)
- "classification" = EXACTLY "{classification}"
- "comment" = the {out_lbl} line using "{acting_team_display}" when you name the actor"""


def maybe_ball_kinetics_llm_comment(
    *,
    events: Any,  # TODO perché plurale?
    features: Any,
    comm_cfg: Any,
    eventproc: Any,
    commentator: Commentator,
    llm: Any,
    context_frames: int,
    field_width_mm: float,
    field_height_mm: float,
    language: str,
    on_spoken: Callable[[str], None],
    xlimit_mm: Optional[float] = None,
    dedup_key: Optional[str] = None,
    on_async_complete: Optional[Callable[[], None]] = None,
    on_committed: Optional[Callable[[], None]] = None,
) -> bool:
    """
    On ball-strike flags, schedule the kinetics LLM. Caller handles TTS interrupt + lead-in.
    """
    if not (events.home_team_ball_strike or events.away_team_ball_strike):
        return False

    trigger = events.trigger_frame
    if dedup_key is None:
        dedup_key = f"kinetics:{trigger}"

    if commentator.was_ball_kinetics_llm_sent(dedup_key):
        return False

    geometry_json = events.ball_action_geometry

    last_seen_frame = eventproc.last_seen_ball_frame
    eff_xlimit = xlimit_mm if xlimit_mm is not None else field_width_mm / 2.0
    ln, _, rn, _ = team_labels(features)

    # Pre-interpret geometry into a human-readable narrative — the LLM reads facts, not raw numbers.
    # Use only the 30 frames immediately before the kick to avoid contamination from earlier events.
    _last_seen = eventproc.last_seen_ball_frame or trigger
    pre_kick = [
        p for p in eventproc.get_ball_positions()
        if _last_seen - 30 <= p[0] <= _last_seen
    ]
    action_narrative = build_action_narrative(
        geometry_json, ln, rn, eff_xlimit,
        pre_kick_positions=pre_kick,
    )

    # Ball trajectory: compact pre/gap/post positions.
    motion_summary = commentator.ball_motion_summary(
        trigger_frame=trigger,
        last_seen_frame=last_seen_frame,
        history_frames=40,
        post_frames=15,
    )
    if not motion_summary.strip():
        logger.warning("ball kinetics LLM: empty ball motion summary for frame %s", trigger)
        return False

    kick_origin_frame: Optional[int] = None
    kick_origin_snapshot: str = ""
    if last_seen_frame is not None:
        kick_origin_frame = last_seen_frame
        kick_origin_snapshot = eventproc.csv_block_at_frame(last_seen_frame)

    # Python-side deterministic classification — the LLM only writes the sentence.
    decision = classify_action(geometry_json, pre_kick_positions=pre_kick, xlimit_mm=eff_xlimit)

    # Guardrail 1: if we don't know who or what, don't let the LLM fabricate a story.
    dec_team = str(decision.get("acting_team", "unknown"))
    dec_cls  = str(decision.get("classification", "unclear"))
    if dec_team == "unknown" or dec_cls == "unclear":
        print(
            f"[Commentary BALL-KINETICS-LLM] skip unclassified | frame={trigger} | "
            f"team={dec_team} | cls={dec_cls}",
            flush=True,
        )
        commentator.mark_ball_kinetics_llm_sent(dedup_key)
        return False

    # Guardrail 2: suppress kinetics during the post-goal kickoff wait window —
    # spurious kicks right after a goal are almost always tracking artifacts.
    last_goal_f = getattr(commentator, "_last_goal_frame", None)  # questa getattr è effettivamente giustificata per come è fatto il commentator ad oggi (ma, insomma, si farebbe meglio a metterlo direttamente a None)
    post_goal_window = comm_cfg.post_goal_wait_window_frames
    if last_goal_f is not None and (trigger - int(last_goal_f)) <= post_goal_window:
        print(
            f"[Commentary BALL-KINETICS-LLM] skip post-goal wait | frame={trigger} | "
            f"since_goal={trigger - int(last_goal_f)}fr (window={post_goal_window})",
            flush=True,
        )
        commentator.mark_ball_kinetics_llm_sent(dedup_key)
        return False

    # Anti-spam: skip if we just said the same (team, classification) a moment ago.
    repeat_suppress_sec = comm_cfg.ball_kinetics_llm_repeat_suppress_sec
    if repeat_suppress_sec > 0 and commentator.should_suppress_kinetics_repeat(
        trigger_frame=trigger,
        team=dec_team,
        classification=dec_cls,
        window_seconds=repeat_suppress_sec,
    ):
        print(
            f"[Commentary BALL-KINETICS-LLM] suppress repeat | frame={trigger} | "
            f"team={dec_team} | cls={dec_cls} | window={repeat_suppress_sec:.1f}s",
            flush=True,
        )
        commentator.mark_ball_kinetics_llm_sent(dedup_key)
        return False

    system = build_kinetics_system_prompt(
        features, field_width_mm, field_height_mm, language, xlimit_mm=xlimit_mm
    )
    user = build_kinetics_user_prompt(
        action_narrative,
        motion_summary,
        classification=decision["classification"],
        acting_team=decision["acting_team"],
        classification_reasoning=decision["reasoning"],
        kick_origin_frame=kick_origin_frame,
        kick_origin_snapshot=kick_origin_snapshot,
        ln=ln,
        rn=rn,
        language=language,
    )
    full_prompt = f"{system}\n\n---\n\n{user}"

    def _done(raw: str) -> None:
        try:
            raw_s = raw or ""
            preview = raw_s.strip().replace("\n", " ")
            if len(preview) > 160:
                preview = preview[:160] + "…"
            print(
                f"[Commentary BALL-KINETICS-LLM] raw ({len(raw_s)} chars): {preview}",
                flush=True,
            )
            text = extract_json_comment(raw_s)
            if text:
                on_spoken(text)
            # Record classified event in the tactical timeline (used by periodic LLM storytelling).
            try:
                g = json.loads(geometry_json) if isinstance(geometry_json, str) else geometry_json
                commentator.record_ball_event(
                    frame=int(trigger),
                    team=str(decision.get("acting_team", "unknown")),
                    classification=str(decision.get("classification", "unclear")),
                    speed_mms=float(g.get("speed_mms", 0.0)),
                    x_mm=float(g.get("ball_start_x_mm", 0.0)),
                    comment=text or "",
                )
            except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
                pass
        finally:
            if on_async_complete is not None:
                on_async_complete()

    print(
        f"[Commentary BALL-KINETICS-LLM] call | backend={type(llm).__name__} | "
        f"frame={trigger} | prompt_chars={len(full_prompt)}",
        flush=True,
    )
    if on_committed is not None:
        on_committed()
    commentator.mark_ball_kinetics_llm_sent(dedup_key)
    commentator.mark_kinetics_emit(
        frame=trigger,
        team=str(decision["acting_team"]),
        classification=str(decision["classification"]),
    )
    llm.llm_call_async(full_prompt, on_done=_done)
    return True
