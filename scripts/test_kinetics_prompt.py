"""
Standalone test for the ball-kinetics LLM prompt.

Replays the 4 known kick events from mario_1/output.csv and calls the LLM directly,
without running main.py or the full video pipeline.

Usage:
    python scripts/test_kinetics_prompt.py                     # all 4 events
    python scripts/test_kinetics_prompt.py --frame 566         # single event
    python scripts/test_kinetics_prompt.py --dry-run           # print prompts only, no LLM call
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Optional

# ── project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]

# ── OpenAI wrapper (no heavy deps) ────────────────────────────────────────────
import importlib.util as _ilu, types as _types

def _load_bare(rel: str):
    """Load a single .py file without touching sys.path or package __init__ chains."""
    p = ROOT / rel
    spec = _ilu.spec_from_file_location(p.stem, p)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod

_oai = _load_bare("src/ml/models/openai_chat_wrapper.py")
OpenAIChatWrapper = _oai.OpenAIChatWrapper

# Load kinetics prompt builders directly from source, bypassing circular __init__.py chain.
# We stub only the one symbol the module needs (Commentator) since we don't use maybe_ball_kinetics_llm_comment here.
import types as _types
_stub_commentator_mod = _types.ModuleType("src.commentator.commentator")
class _StubCommentator: pass
_stub_commentator_mod.Commentator = _StubCommentator  # type: ignore[attr-defined]
import sys as _sys
_sys.modules["src.commentator.commentator"] = _stub_commentator_mod

_kinetics_mod = _load_bare("src/commentator/ball_kinetics_llm_commentary.py")
build_kinetics_system_prompt = _kinetics_mod.build_kinetics_system_prompt
build_kinetics_user_prompt   = _kinetics_mod.build_kinetics_user_prompt
build_action_narrative       = _kinetics_mod.build_action_narrative
classify_action              = _kinetics_mod.classify_action


# Prompt builders are loaded from source — see _load_kinetics below.
# build_kinetics_system_prompt, build_kinetics_user_prompt, build_action_narrative
# are injected into module scope after the stub setup.

# ── config ────────────────────────────────────────────────────────────────────
CSV_PATH     = ROOT / "data/Final_GO/mario_1/output.csv"
FIELD_W_MM   = 9000.0   # used only for the system prompt field description
FIELD_H_MM   = 6000.0
XLIMIT       = 7000.0   # actual field half-width from field_config.xlimit in mario.yaml
FPS          = 30.0
LANGUAGE     = "it"
MODEL        = "gpt-5.4-nano-2026-03-17"

# Known kick events: (trigger_frame, last_seen_frame, expected)
EVENTS = [
    (393,  367, "clearance/left  — HTWK spazza dalla propria area"),
    (483,  471, "shot/right      — B-Human conclude verso porta sinistra"),
    (543,  527, "shot/right      — B-Human tiro in porta"),
    (566,  556, "shot/right      — B-Human tiro finale verso porta HTWK"),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def ball_positions(rows: list[dict]) -> dict[int, tuple[float, float]]:
    return {
        int(r["frame"]): (float(r["field_x"]), float(r["field_y"]))
        for r in rows
        if r["type"] == "ball" and r["field_x"] and r["field_y"]
    }


def robots_at(rows: list[dict], frame: int) -> list[dict]:
    return [r for r in rows if int(r["frame"]) == frame and r["type"] == "robot"]


def robot_track(rows: list[dict], robot_id: int, f_from: int, f_to: int) -> list[tuple[int, float, float]]:
    """Return (frame, x, y) tuples for a robot across a frame window."""
    out: list[tuple[int, float, float]] = []
    for r in rows:
        if r["type"] != "robot":
            continue
        try:
            if int(r["id"]) != robot_id:
                continue
            f = int(r["frame"])
        except (TypeError, ValueError):
            continue
        if not (f_from <= f <= f_to):
            continue
        try:
            rx = float(r["field_x"]); ry = float(r["field_y"])
        except (TypeError, ValueError):
            continue
        out.append((f, rx, ry))
    out.sort(key=lambda t: t[0])
    return out


def compute_robot_pose(
    rows: list[dict], robot_id: int, kick_frame: int,
    bx_kick: float, by_kick: float,
    kick_dx: float, kick_dy: float,
    pre_window: int = 6,
    post_window: int = 5,
) -> dict:
    """
    Infer a robot's 'kicker likelihood' from its motion profile around the kick.

    Signals computed:
      - pre_heading  : robot's motion vector over the ``pre_window`` frames BEFORE the kick
      - approach_speed_mms : rate at which the robot closed on the ball pre-kick (+ = closing)
      - pre_align    : cos(pre_heading, kick_direction) — was the robot moving the way it then kicked?
      - post_heading : robot's motion vector over the ``post_window`` frames AFTER the kick
      - post_align   : cos(post_heading, kick_direction) — follow-through in the kick direction?
      - verdict      : qualitative label ('likely_kicker' | 'bystander' | 'moving_away' | 'unknown')
      - likely_kicker_score : blended 0..1 score
    """
    pre  = robot_track(rows, robot_id, kick_frame - pre_window, kick_frame)
    post = robot_track(rows, robot_id, kick_frame, kick_frame + post_window)

    def _heading(track):
        if len(track) < 2:
            return (0.0, 0.0, 0.0)
        f0, x0, y0 = track[0]; f1, x1, y1 = track[-1]
        n = f1 - f0
        dt = n / FPS if n > 0 else 1.0
        hdx, hdy = x1 - x0, y1 - y0
        return (hdx, hdy, math.hypot(hdx, hdy) / dt)

    pre_hdx, pre_hdy, pre_speed = _heading(pre)
    post_hdx, post_hdy, post_speed = _heading(post)

    # Approach to ball over the pre-window (d_start - d_end; + = closing in)
    approach_speed = 0.0
    if len(pre) >= 2:
        f0, x0, y0 = pre[0]; f1, x1, y1 = pre[-1]
        dt = (f1 - f0) / FPS if f1 > f0 else 1.0
        d0 = math.hypot(x0 - bx_kick, y0 - by_kick)
        d1 = math.hypot(x1 - bx_kick, y1 - by_kick)
        approach_speed = (d0 - d1) / dt

    # Alignment of motion with kick direction (0 if robot near-stationary)
    pre_align  = cos_toward(pre_hdx,  pre_hdy,  kick_dx, kick_dy) if math.hypot(pre_hdx,  pre_hdy)  > 30 else 0.0
    post_align = cos_toward(post_hdx, post_hdy, kick_dx, kick_dy) if math.hypot(post_hdx, post_hdy) > 30 else 0.0

    # Blended score: approach + pre-alignment + post-alignment all contribute positively.
    # Negative post-alignment (robot moved AGAINST the kick) is a strong "not kicker" signal.
    approach_part = max(-0.3, min(0.4, approach_speed / 1500.0 * 0.4))
    pre_part      = max(0.0, pre_align) * 0.35
    post_part     = max(-0.3, post_align * 0.25)
    score = max(0.0, min(1.0, approach_part + pre_part + post_part))

    # Qualitative verdict (easy for the LLM to read at a glance)
    if len(pre) < 2 and len(post) < 2:
        verdict = "unknown"
    elif score >= 0.45:
        verdict = "likely_kicker"
    elif approach_speed < -300 or post_align < -0.3:
        verdict = "moving_away"
    elif abs(pre_align) < 0.3 and abs(approach_speed) < 200:
        verdict = "bystander"
    else:
        verdict = "uncertain"

    return {
        "pre_heading_mm":    [round(pre_hdx, 1),  round(pre_hdy, 1)],
        "pre_heading_speed_mms": round(pre_speed, 1),
        "post_heading_mm":   [round(post_hdx, 1), round(post_hdy, 1)],
        "post_heading_speed_mms": round(post_speed, 1),
        "approach_speed_mms": round(approach_speed, 1),
        "pre_align_with_kick":  round(pre_align, 3),
        "post_align_with_kick": round(post_align, 3),
        "verdict": verdict,
        "likely_kicker_score": round(score, 3),
    }


def cos_toward(dx: float, dy: float, gx: float, gy: float) -> float:
    d = math.hypot(dx, dy)
    g = math.hypot(gx, gy)
    if d < 1e-9 or g < 1e-9:
        return 0.0
    return (dx * gx + dy * gy) / (d * g)


def build_geometry_json(
    rows: list[dict],
    ball: dict[int, tuple[float, float]],
    last_seen_f: int,
    trigger_f: int,
) -> str:
    x0, y0 = ball[last_seen_f]
    x1, y1 = ball[trigger_f]
    dx, dy = x1 - x0, y1 - y0
    disp = math.hypot(dx, dy)
    dt = (trigger_f - last_seen_f) / FPS
    speed = disp / dt if dt > 1e-9 else 0.0

    cos_right = cos_toward(dx, dy, XLIMIT - x0, -y0)
    cos_left  = cos_toward(dx, dy, -XLIMIT - x0, -y0)
    gprog_left  = x1 - x0          # positive = toward right goal (left team attacks right)
    gprog_right = x0 - x1          # positive = toward left goal  (right team attacks left)

    # nearest robot at kick origin — prefer robots consistent with being the kicker
    # (kick direction matches moving ball away from their own goal)
    robots = robots_at(rows, last_seen_f)
    poss_max = 900.0

    def _is_likely_kicker(team: str, kick_dx: float) -> bool:
        if kick_dx == 0.0:
            return True
        return (team == "left" and kick_dx > 0) or (team == "right" and kick_dx < 0)

    candidates = []
    for r in robots:
        try:
            rx, ry = float(r["field_x"]), float(r["field_y"])
        except (TypeError, ValueError):
            continue
        d = math.hypot(rx - x0, ry - y0)
        color = (r.get("color") or "").strip().lower()
        team = "left" if color == "blue" else ("right" if color == "red" else None)
        if team:
            candidates.append((team, d, r))

    consistent = [(t, d, r) for t, d, r in candidates if _is_likely_kicker(t, dx)]
    inconsistent_in_range = any(d <= poss_max and not _is_likely_kicker(t, dx) for t, d, _ in candidates)
    consistent_in_range   = any(d <= poss_max for _, d, _ in consistent)

    if consistent_in_range:
        pool = consistent
    elif inconsistent_in_range:
        # Only defenders close — use nearest consistent even if far, else unknown
        pool = consistent if consistent else []
    else:
        pool = candidates  # no robot in range at all

    best_team, best_dist = "unknown", -1.0
    for team, d, _ in pool:
        if best_dist < 0 or d < best_dist:
            best_team, best_dist = team, d

    if best_dist < 0 or best_dist > poss_max:
        best_team = "unknown"

    hint_cls = "unclear"
    if best_team != "unknown":
        cos_goal  = cos_right if best_team == "left" else cos_left
        gprog     = gprog_left if best_team == "left" else gprog_right
        if cos_goal > 0.32 and gprog > 150:
            hint_cls = "shot"
        elif cos_goal > 0.32:
            hint_cls = "clearance"
        else:
            hint_cls = "pass"

    team_guess = best_team if best_team != "unknown" else ("left" if dx > 0 else "right")

    # Full nearby-robots context: up to 6 robots within 2500mm (both teams, incl. unknown colors).
    # For each robot we also compute a pose/motion profile to help identify the actual kicker,
    # independent from color classification (which is noisy from the CNN).
    nearby: list[dict] = []
    for r in robots:
        try:
            rx, ry = float(r["field_x"]), float(r["field_y"])
        except (TypeError, ValueError):
            continue
        d = math.hypot(rx - x0, ry - y0)
        if d > 2500.0:
            continue
        color = (r.get("color") or "").strip().lower() or "unknown"
        team = "left" if color == "blue" else ("right" if color == "red" else "unknown")
        pose = compute_robot_pose(rows, int(r["id"]), last_seen_f, x0, y0, dx, dy)
        nearby.append({
            "id": int(r["id"]),
            "team": team,
            "color": color,
            "dist_mm": round(d, 1),
            "x_mm": round(rx, 1),
            "y_mm": round(ry, 1),
            "dx_from_ball_mm": round(rx - x0, 1),
            "dy_from_ball_mm": round(ry - y0, 1),
            **pose,
        })
    nearby.sort(key=lambda r: (-r.get("likely_kicker_score", 0.0), r["dist_mm"]))
    nearby = nearby[:6]

    return json.dumps({
        "dx_mm": round(dx, 1),
        "dy_mm": round(dy, 1),
        "disp_mm": round(disp, 1),
        "speed_mms": round(speed, 1),
        "cos_toward_right_goal": round(cos_right, 3),
        "cos_toward_left_goal":  round(cos_left,  3),
        "goal_progress_if_left_attacking_mm":  round(gprog_left, 1),
        "goal_progress_if_right_attacking_mm": round(gprog_right, 1),
        "nearest_robot_team":     best_team,
        "nearest_robot_dist_mm":  round(best_dist, 1),
        "nearby_robots":          nearby,
        "ball_start_x_mm": round(x0, 1),
        "ball_start_y_mm": round(y0, 1),
        "ball_end_x_mm":   round(x1, 1),
        "ball_end_y_mm":   round(y1, 1),
        "tracker_hint_cls":  hint_cls,
        "tracker_hint_team": team_guess,
    }, indent=2)


def build_motion_summary(
    ball: dict[int, tuple[float, float]],
    last_seen_f: int,
    trigger_f: int,
    history: int = 40,
    post: int = 15,
) -> str:
    sorted_f = sorted(ball.keys())
    pre  = [(f, *ball[f]) for f in sorted_f if f <= last_seen_f][-history:]
    post_start = next((f for f in sorted_f if f >= trigger_f), trigger_f)
    post_rows = [(f, *ball[f]) for f in sorted_f if f >= post_start][:post]
    gap = post_start - last_seen_f

    lines: list[str] = []
    if pre:
        lines.append(f"Pre-kick ({len(pre)} ball positions before gap):")
        for f, x, y in pre:
            lines.append(f"  frame {f:4d}:  x={x:8.0f}mm  y={y:7.0f}mm")
    if gap > 0:
        lines.append(f"\n  ···  GAP: {gap} frames ({gap / FPS:.2f}s) — ball not visible  ···\n")
    if post_rows:
        lines.append(f"Post-gap ({len(post_rows)} ball positions after gap):")
        for f, x, y in post_rows:
            lines.append(f"  frame {f:4d}:  x={x:8.0f}mm  y={y:7.0f}mm")
    return "\n".join(lines)


def kick_origin_snapshot(rows: list[dict], frame: int) -> str:
    block_rows = [r for r in rows if int(r["frame"]) == frame]
    if not block_rows:
        return ""
    lines = [",".join(block_rows[0].keys())]
    for r in block_rows:
        lines.append(",".join(r.values()))
    return "\n".join(lines)


class _FakeFeatures:
    left_team_name      = "HTWK Robots"
    right_team_name     = "B-Human"
    left_team_full_name = "HTWK Robots"
    right_team_full_name= "B-Human"
    left_team_color     = "blue"
    right_team_color    = "red"


# ── main ──────────────────────────────────────────────────────────────────────

def run_event(
    rows: list[dict],
    ball: dict[int, tuple[float, float]],
    trigger_f: int,
    last_seen_f: int,
    expected: str,
    llm: Optional[OpenAIChatWrapper],
    dry_run: bool,
) -> None:
    sep = "═" * 70
    print(f"\n{sep}")
    print(f"  EVENT  frame={trigger_f}  (kick origin frame={last_seen_f})")
    print(f"  Expected: {expected}")
    print(sep)

    geom_json   = build_geometry_json(rows, ball, last_seen_f, trigger_f)
    motion_sum  = build_motion_summary(ball, last_seen_f, trigger_f)
    orig_snap   = kick_origin_snapshot(rows, last_seen_f)
    features    = _FakeFeatures()

    # Pre-kick positions: only the 30 frames immediately before the kick origin,
    # to avoid contamination from previous events in the match.
    sorted_frames = sorted(ball.keys())
    pre_kick_pos = [(f, ball[f][0], ball[f][1])
                    for f in sorted_frames if last_seen_f - 30 <= f <= last_seen_f]
    narrative   = build_action_narrative(
        geom_json, features.left_team_name, features.right_team_name, XLIMIT,
        pre_kick_positions=pre_kick_pos,
    )
    decision    = classify_action(geom_json, pre_kick_positions=pre_kick_pos, xlimit_mm=XLIMIT)
    system = build_kinetics_system_prompt(features, FIELD_W_MM, FIELD_H_MM, LANGUAGE, xlimit_mm=XLIMIT)
    user   = build_kinetics_user_prompt(
        narrative,
        motion_sum,
        classification=decision["classification"],
        acting_team=decision["acting_team"],
        classification_reasoning=decision["reasoning"],
        kick_origin_frame=last_seen_f,
        kick_origin_snapshot=orig_snap,
        ln=features.left_team_name,
        rn=features.right_team_name,
    )
    full = f"{system}\n\n---\n\n{user}"

    print(f"\n── ACTION NARRATIVE ──\n{narrative}\n")
    print(f"── PYTHON CLASSIFICATION ──")
    print(f"  acting_team:    {decision['acting_team']}")
    print(f"  classification: {decision['classification']}")
    print(f"  reasoning:      {decision['reasoning']}\n")
    print(f"── BALL TRAJECTORY (first 10 lines) ──")
    for line in motion_sum.splitlines()[:10]:
        print(f"  {line}")
    print(f"  ... ({len(motion_sum.splitlines())} total lines)")
    print(f"\n── PROMPT SIZE: {len(full)} chars ──")

    if dry_run:
        print("\n[DRY RUN] skipping LLM call.")
        return

    print("\n── LLM RESPONSE ──")
    raw = llm.llm_call(full)
    print(f"  raw: {raw[:300]}")
    try:
        start = raw.index("{")
        end   = raw.rindex("}") + 1
        obj   = json.loads(raw[start:end])
        cls   = obj.get("classification", "?")
        team  = obj.get("acting_team", "?")
        comment = obj.get("comment", "")
        match = "✓" if (
            (cls == "clearance" and "clearance" in expected) or
            (cls == "shot"      and "shot"      in expected) or
            (cls == "pass"      and "pass"      in expected)
        ) else "✗"
        team_ok = "✓" if (
            ("left"  in expected and team == "left")  or
            ("right" in expected and team == "right")
        ) else "✗"
        print(f"\n  classification: {cls}  {match}")
        print(f"  acting_team:    {team}  {team_ok}")
        print(f"  comment:        {comment}")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  [parse error] {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test kinetics LLM prompt on known events")
    parser.add_argument("--frame", type=int, default=None, help="Test only this trigger frame")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts, skip LLM call")
    args = parser.parse_args()

    rows = load_csv(CSV_PATH)
    ball = ball_positions(rows)

    llm: Optional[OpenAIChatWrapper] = None
    if not args.dry_run:
        api_key_path = ROOT / "api.txt"
        api_key = ""
        if api_key_path.is_file():
            for line in api_key_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    api_key = line
                    break
        if not api_key:
            import os
            api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("ERROR: no API key found in api.txt or OPENAI_API_KEY. Use --dry-run to skip.")
            sys.exit(1)
        llm = OpenAIChatWrapper(MODEL, api_key=api_key)

    events = [e for e in EVENTS if args.frame is None or e[0] == args.frame]
    if not events:
        print(f"No event with frame={args.frame}. Available: {[e[0] for e in EVENTS]}")
        sys.exit(1)

    for trigger_f, last_seen_f, expected in events:
        if trigger_f not in ball or last_seen_f not in ball:
            print(f"[SKIP] frame {trigger_f} or {last_seen_f} not in CSV ball positions")
            continue
        run_event(rows, ball, trigger_f, last_seen_f, expected, llm, args.dry_run)

    print("\n" + "═" * 70)
    print("  Done.")


if __name__ == "__main__":
    main()
