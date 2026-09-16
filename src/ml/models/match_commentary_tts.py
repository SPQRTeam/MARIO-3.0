"""TTS queue for commentary. Text comes from :mod:`src.commentator.events_commentary` and optionally :mod:`src.commentator.general_commentary`."""

from __future__ import annotations

import queue
import json
import shutil
import subprocess
import threading
import time
import traceback
import wave
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Tuple

from ...commentator.events_commentary import (
    _scorer_field_side_when_left_score_up,
    _scorer_field_side_when_right_score_up,
    ball_strike_line,
    goal_line,
    pass_line,
    shot_line,
)
from ...commentator.general_commentary import maybe_general_comment
from ...commentator.ball_kinetics_llm_commentary import maybe_ball_kinetics_llm_comment
from ...commentator.geometry_llm_commentary import maybe_geometry_llm_comment
from ...commentator.commentary_labels import (
    L_BALL_KINETICS,
    L_EVENT,
    L_GEOMETRY,
    L_GOAL,
    L_OTHER,
    L_PERIODIC,
    LEGEND as COMMENTARY_LABEL_LEGEND,
    NAMES as COMMENTARY_LABEL_NAMES,
)
from ...commentator.periodic_llm_commentary import (
    maybe_goal_state_llm,
    maybe_periodic_scene_llm,
)
from .tts import generate_speech, resolve_edge_voice, start_audio_playback

if TYPE_CHECKING:
    from ...commentator.event_flags import EventFlags



def spoken_line_for_ocr_goal(left_score_increased: bool, config: Any, team_map: Any) -> str:
    """Fixed line when OCR detects a score change (same mapping as ``goal_for_*_team`` in Commentator)."""
    comm = config.commentating
    if left_score_increased:
        side = _scorer_field_side_when_left_score_up(comm)
    else:
        side = _scorer_field_side_when_right_score_up(comm)
    name = team_map.left_full_name if side == "left" else team_map.right_full_name
    return f"Goal! {name} scores!"


def deterministic_goal_tts_line(events: "EventFlags", comm_cfg: Any) -> Optional[str]:
    """Return a fixed spoken line for goal-like events, or None to fall back to LLM."""
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
    # Geometry only: left net = attacked by *right* field team, etc.
    if events.unconf_goal_at_left:
        return f"Possible goal for {right}! Ball crosses the left goal line."
    if events.unconf_goal_at_right:
        return f"Possible goal for {left}! Ball crosses the right goal line."
    return None


def deterministic_shot_tts_line(events: "EventFlags", comm_cfg: Any) -> Optional[str]:
    """Fixed *Team shot.* / *Team wide.* (same attacking side as the LLM prompts used to use)."""
    if not comm_cfg.deterministic_shot_speech:
        return None
    left = _team_label(events, "left")
    right = _team_label(events, "right")
    # Penalty-area shot: team attacks the goal on that side.
    if events.shot_at_left_goal:
        return f"{right} shot."
    if events.shot_at_right_goal:
        return f"{left} shot."
    # Ball over end line outside the goal mouth. Mention defense only if the ball
    # left from inside the penalty area (stable geometry, not one-off detection).
    if events.failed_shot_at_left:
        if events.failed_shot_from_penalty_area:
            return f"{left} defense holds. {right} wide."
        return f"{right} wide."
    if events.failed_shot_at_right:
        if events.failed_shot_from_penalty_area:
            return f"{right} defense holds. {left} wide."
        return f"{left} wide."
    return None


def deterministic_clearance_tts_line(events: "EventFlags", comm_cfg: Any) -> Optional[str]:
    """Defensive boot out of the penalty area (window-smoothed)."""
    if not comm_cfg.deterministic_clearance_speech:
        return None
    left = _team_label(events, "left")
    right = _team_label(events, "right")
    if events.clearance_from_left_box:
        return f"{left} clear it from the box."
    if events.clearance_from_right_box:
        return f"{right} clear it from the box."
    return None


def deterministic_priority_tts_line(events: "EventFlags", comm_cfg: Any) -> Optional[str]:
    """Goals, shots/wide, then box clearances."""
    line = deterministic_goal_tts_line(events, comm_cfg)
    if line is not None:
        return line
    line = deterministic_shot_tts_line(events, comm_cfg)
    if line is not None:
        return line
    return deterministic_clearance_tts_line(events, comm_cfg)


def _use_llm_transition_bridge(comm_cfg: Any) -> bool:
    """Short LLM line after a fixed TTS cue — fills dead air while OCR / synthesis catch up."""
    if comm_cfg is None:
        return True
    return comm_cfg.llm_transition_bridge


def transition_bridge_prompt_after_fixed(events: "EventFlags", comm_cfg: Any) -> Optional[str]:
    """Follow-up prompt only; does not repeat the deterministic wording.

    No bridge after **clearance** lines: the LLM often repeats \"defender clears\"
    and that clashes when a goal is announced seconds later.
    """
    if not _use_llm_transition_bridge(comm_cfg):
        return None
    base = (
        "You are a live robot-soccer commentator. "
        "Reply with ONE very short English sentence (max 8 words). "
        "Present tense. No emojis, no quotes. "
        "Do not repeat the exact words the previous line just used. "
        "Do not say: clear, clearance, defender, boot, or crowd — only time/scoreboard tension. "
    )
    if events.shot_at_left_goal or events.shot_at_right_goal:
        return (
            f"{base}Context: a shot was just called. "
            f"Say only a bridge: we wait for the goal line or scoreboard to confirm."
        )
    if events.unconf_goal_at_left or events.unconf_goal_at_right:
        return (
            f"{base}Context: a possible goal was already announced. "
            f"Say only a bridge: stay tuned for the official score."
        )
    if events.failed_shot_at_left or events.failed_shot_at_right:
        return (
            f"{base}Context: wide ball already noted. "
            f"One short transitional line; no new facts."
        )
    return None


def _team_label(events: "EventFlags", side: str) -> str:
    """Short, TTS-friendly team name (one or two words at most).

    Prefers the full team name (e.g. ``"HTWK"``), falls back to a plain
    color label (``"the red team"``), then to ``"the left/right team"``.
    Kept short on purpose: long parenthetical labels make spoken English
    hard to follow.
    """
    color = events.left_team_name if side == "left" else events.right_team_name
    full = events.left_team_full_name if side == "left" else events.right_team_full_name
    if full:
        return full
    if color:
        return f"the {color} team"
    return f"the {side} team"


def _score_tag(events: "EventFlags") -> str:
    """Compact ``"Score 0-1."`` tag for context."""
    if events.score_left is None or events.score_right is None:
        return ""
    return f" Score {events.score_left}-{events.score_right}."


def _context_tag(events: "EventFlags") -> str:
    """One short line of state: ball + possession + attack phase (smoothed zone)."""
    parts = []
    if events.ball_visible is False:
        parts.append("ball not visible")
    if events.possession in ("left", "right"):
        parts.append(f"{_team_label(events, events.possession)} has the ball")
    elif events.possession == "loose":
        parts.append("loose ball")
    if events.ball_zone == "left_third":
        parts.append("ball near the left goal")
    elif events.ball_zone == "right_third":
        parts.append("ball near the right goal")
    elif events.ball_zone == "midfield":
        parts.append("ball in midfield")
    if events.phase_hint:
        parts.append(events.phase_hint)
    if not parts:
        return ""
    return " State: " + "; ".join(parts) + "."


# Shared system instruction. Kept short and explicit so the LLM doesn't drift
# into long sentences with rare vocabulary that the TTS makes hard to follow.
_BASE_RULES = (
    "You are a live football commentator for a robot soccer match. "
    "Reply with ONE very short English sentence (max 10 words). "
    "Use simple words. Present tense. No emojis, no preamble, no quotes. "
    "Use the team names exactly as written (e.g. HTWK, BHuman). "
    "State describes who is attacking from field position (smoothed). "
    "Do not say a team is defending unless the Event explicitly mentions defense "
    "after a wide ball from the penalty area."
)


def _approach_event_prompt(
    events: "EventFlags", base: str, score: str, side: str
) -> str:
    """Approaching-the-ball: use **smoothed** zone only — attack side in opponent third, neutral at home."""
    team = _team_label(events, side)
    other = _team_label(events, "right" if side == "left" else "left")
    if events.ball_zone == "left_third":
        if side == "right":
            return f"{base}{score} Event: {team} attack — closing in near the left goal."
        return f"{base}{score} Event: {team} move to the ball in their own half. {other} attacks."
    if events.ball_zone == "right_third":
        if side == "left":
            return f"{base}{score} Event: {team} attack — closing in near the right goal."
        return f"{base}{score} Event: {team} move to the ball in their own half. {other} attacks."
    return f"{base}{score} Event: {team} contest the ball in midfield."


def _llm_detail_salient(comm_cfg: Any) -> bool:
    """``True`` = only high-impact lines (goals, shots, big misses, kickoff after goal)."""
    v = comm_cfg.llm_detail.lower().strip()
    return v in ("salient", "minimal", "highlights", "1", "true", "yes")


def events_to_llm_prompt(events: "EventFlags", comm_cfg: Any = None) -> Optional[str]:
    """Build a single-shot ENGLISH commentary prompt from game events.

    ``commentating.llm_detail``:

    * ``salient`` (default): only goals, possible goals, shots, misses,
      kickoff-after-goal — keeps audio close to what you see (no idle /
      defend / approach chatter).
    * ``full``: also passes, slow attacks, ball in/out, approach, idle filler.

    Returns ``None`` if nothing worth commenting happened on this frame.
    """
    base = _BASE_RULES
    score = _score_tag(events)
    ctx = _context_tag(events)

    left = _team_label(events, "left")
    right = _team_label(events, "right")
    salient = _llm_detail_salient(comm_cfg)

    # 1–2) Goals — LLM only if ``deterministic_goal_speech: false`` (otherwise fixed TTS line).
    if not comm_cfg.deterministic_goal_speech:
        if events.goal_for_left_team:
            return f"{base}{score}{ctx} Event: {left} scored a goal. Say it loudly."
        if events.goal_for_right_team:
            return f"{base}{score}{ctx} Event: {right} scored a goal. Say it loudly."
        if events.unconf_goal_at_left:
            return f"{base}{score}{ctx} Event: maybe a goal by {right}. Wait for confirmation."
        if events.unconf_goal_at_right:
            return f"{base}{score}{ctx} Event: maybe a goal by {left}. Wait for confirmation."

    # 3–4) Shots / wide — LLM only if ``deterministic_shot_speech: false``.
    if not comm_cfg.deterministic_shot_speech:
        if events.shot_at_left_goal:
            return f"{base}{score}{ctx} Event: {right} shoots at the left goal."
        if events.shot_at_right_goal:
            return f"{base}{score}{ctx} Event: {left} shoots at the right goal."
        if events.failed_shot_at_left:
            if events.failed_shot_from_penalty_area:
                return (
                    f"{base}{score}{ctx} Event: wide off the left goal after play in the box; "
                    f"{left} defense involved."
                )
            return f"{base}{score}{ctx} Event: the ball went wide of the left goal."
        if events.failed_shot_at_right:
            if events.failed_shot_from_penalty_area:
                return (
                    f"{base}{score}{ctx} Event: wide off the right goal after play in the box; "
                    f"{right} defense involved."
                )
            return f"{base}{score}{ctx} Event: the ball went wide of the right goal."

    if not comm_cfg.deterministic_clearance_speech:
        if events.clearance_from_left_box:
            return (
                f"{base}{score}{ctx} Event: {left} clears the ball from the penalty area."
            )
        if events.clearance_from_right_box:
            return (
                f"{base}{score}{ctx} Event: {right} clears the ball from the penalty area."
            )

    # 5) Kickoff right after a goal (still salient).
    if events.kickoff_after_goal:
        return f"{base}{score} Event: kickoff after the goal."

    if salient:
        return None

    # ----- ``full`` detail only below ---------------------------------

    # Pass toward goal.
    if events.pass_toward_left_goal:
        return f"{base}{score}{ctx} Event: {right} pushes the ball to the left goal."
    if events.pass_toward_right_goal:
        return f"{base}{score}{ctx} Event: {left} pushes the ball to the right goal."

    # Attack inside penalty area.
    if events.attack_at_left:
        return f"{base}{score}{ctx} Event: {right} attacks near the left goal."
    if events.attack_at_right:
        return f"{base}{score}{ctx} Event: {left} attacks near the right goal."

    # Ball out / back in play.
    if events.ball_out_of_play:
        return f"{base}{score} Event: the ball is out of play."
    if events.ball_back_in_play:
        if events.possession in ("left", "right"):
            holder = _team_label(events, events.possession)
            return f"{base}{score} Event: ball back in play, {holder} has it."
        return f"{base}{score} Event: the ball is back in play."

    # Approach calls (zone-aware: defend/clear vs attack).
    if events.approach_by_left_team:
        return _approach_event_prompt(events, base, score, "left")
    if events.approach_by_right_team:
        return _approach_event_prompt(events, base, score, "right")

    if events.kickoff_imminent:
        return f"{base}{score} Event: ready for the kickoff."

    # Idle filler (every few seconds).
    if events.idle_commentary:
        return (
            f"{base}{score}{ctx} Event: nothing big right now. "
            f"Say one short line from State only — ball, possession, who is attacking "
            f"by field zone. Do not mention defending unless State already does."
        )

    return None


class MatchCommentaryTTS:
    """Background TTS queue; combines EVENTS_COMMENT (fixed lines) and optional GENERAL_COMMENT (LLM)."""

    def _feats_with_names(self, events: Any) -> Any:
        """Return config.features enriched with team names from events (populated via team_map).

        mario.yaml no longer carries left_team_name / left_team_full_name — they come from
        gameinfo → team_map → EventFlags.  This ensures all LLM prompts use real team names
        instead of the "Home"/"Away" hardcoded fallbacks.

        TODO se il problema è questo, probabilmente c'è un modo migliore di risolverlo
        """
        feats = self._config.features
        if feats is None or events is None:
            return feats
        for attr in ("left_team_name", "right_team_name",
                     "left_team_full_name", "right_team_full_name",
                     "left_team_color", "right_team_color"):
            val = getattr(events, attr, None)
            if val:
                setattr(feats, attr, val)
        return feats

    def __init__(self, config: Any):
        self._config = config
        tts_cfg = config.tts
        self._comm_cfg = config.commentating
        # Allow LLM + on-screen line without synthesizing audio (see scripts/main.py).
        self._audio_enabled = bool(tts_cfg.get("enabled", False))
        comm = self._comm_cfg
        if comm is None:
            _comm_lang = "en"
        elif isinstance(comm, dict):
            _comm_lang = str(comm.get("geometry_llm_language", "en") or "en")
        else:
            _comm_lang = comm.geometry_llm_language
        # Edge TTS must match commentary language (same alias "diego" → it-IT vs en-US).
        self._voice = resolve_edge_voice(str(tts_cfg.get("voice", "diego")), _comm_lang)
        _gt = tts_cfg.get("gtts_lang")
        if _gt is None or (isinstance(_gt, str) and not str(_gt).strip()):
            self._gtts_lang = "en" if _comm_lang.lower().startswith("en") else "it"
        else:
            self._gtts_lang = str(_gt)
        self._backend = str(tts_cfg.get("backend", "edge"))
        self._speed = float(tts_cfg.get("speed", 1.0))
        self._use_online = bool(tts_cfg.get("use_edge_tts", True))
        self._min_interval_sec = float(tts_cfg.get("min_interval_sec", 4.0))
        if self._audio_enabled:
            print(
                f"[MatchCommentaryTTS] TTS: commentary_lang={_comm_lang} | "
                f"edge_voice={self._voice} | gtts_lang={self._gtts_lang} | "
                f"use_edge_tts={self._use_online}",
                flush=True,
            )
        self._last_play_monotonic = -1e9
        self._goal_dedup_lock = threading.Lock()
        self._dedup_goal_text = ""
        self._dedup_goal_mono = -1e9
        self._commentary_display_lock = threading.Lock()
        self._last_commentary_line: str = "-"
        _hist = max(1, self._comm_cfg.periodic_llm_history_turns)
        self._llm_assistant_history: deque[str] = deque(maxlen=_hist)
        self._last_periodic_llm_mono: float = -1e9
        self._last_geometry_llm_mono: float = -1e9
        self._last_ball_seen_frame: int = -1
        self._last_kickoff_wait_mono: float = -1e9
        self._llm_in_flight: int = 0
        self._last_geometry_frame: int = -1
        self._last_priority_event_frame: int = -1
        self._periodic_req_seq: int = 0
        self._recent_events: deque[str] = deque(maxlen=24)
        self._score_left: Optional[int] = None
        self._score_right: Optional[int] = None
        self._last_goal_frame: Optional[int] = None
        self._fps = config.processing.fps
        self._export_joined = tts_cfg.export_clean_video_with_commentary
        self._keep_segments = tts_cfg.keep_commentary_audio_segments
        self._section_name: Optional[str] = None
        self._clean_video_path: Optional[Path] = None
        self._audio_mix_path: Optional[Path] = None
        self._joined_video_path: Optional[Path] = None
        self._segments_dir: Optional[Path] = None
        self._label_manifest_path: Optional[Path] = None
        self._segment_idx = 0
        self._timeline_lock = threading.Lock()
        self._timeline_entries: list[tuple[float, Path]] = []
        # Queue items: (text, source_frame, play_next_immediately, label 0-5)
        # play_next_immediately=True makes the worker skip the min-interval after this item.
        self._q: "queue.Queue[Optional[Tuple[str, Optional[int], bool, int]]]" = queue.Queue()
        self._playback_lock = threading.Lock()
        self._current_play_proc: Optional[subprocess.Popen] = None
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def begin_section(self, config: Any, section_name: str, start_frame: int = 0) -> None:
        """Setup per-section timeline export (clean video + commentary audio mux).

        ``start_frame`` is the absolute video frame where this section begins.
        It is subtracted from every ``source_frame`` so audio timestamps are
        relative to the start of the exported clean video (not the whole recording).
        """
        self._section_name = section_name
        self._section_start_frame: int = int(start_frame)
        if not self._export_joined:
            return
        self._clean_video_path = Path(config.video_path(section_name))
        self._audio_mix_path = Path(config.mario_commentary_audio_path(section_name))
        self._joined_video_path = Path(config.mario_clean_with_commentary_video_path(section_name))
        self._segments_dir = Path(config.section_dir(section_name)) / "commentary_segments"
        self._segments_dir.mkdir(parents=True, exist_ok=True)
        self._label_manifest_path = self._segments_dir / "audio_label_manifest.jsonl"
        self._label_manifest_path.write_text("", encoding="utf-8")
        (self._segments_dir / "label_0_to_5_legend.txt").write_text(
            COMMENTARY_LABEL_LEGEND + "\n", encoding="utf-8"
        )
        self._segment_idx = 0
        with self._timeline_lock:
            self._timeline_entries.clear()

    @staticmethod
    def _has_ffmpeg() -> bool:
        return shutil.which("ffmpeg") is not None

    @staticmethod
    def _wav_duration_sec(path: Path) -> float:
        try:
            with wave.open(str(path), "rb") as w:
                frames = w.getnframes()
                rate = w.getframerate()
                if rate <= 0:
                    return 0.1
                return max(0.04, float(frames) / float(rate))
        except (wave.Error, OSError, EOFError):
            return 0.5

    def _save_timeline_segment(
        self,
        src_wav: str,
        source_frame: Optional[int],
        label: int,
        text: str,
    ) -> None:
        if not self._export_joined or self._segments_dir is None:
            return
        self._segment_idx += 1
        lid = int(label) if 0 <= int(label) <= 5 else L_OTHER
        dst = self._segments_dir / f"seg_{self._segment_idx:05d}_L{lid}.wav"
        shutil.copy2(src_wav, dst)
        start_sec = 0.0
        if source_frame is not None and self._fps > 0:
            rel_frame = float(source_frame) - float(self._section_start_frame)
            start_sec = max(0.0, rel_frame / self._fps)
        with self._timeline_lock:
            self._timeline_entries.append((start_sec, dst))
        if self._label_manifest_path is not None:
            row = {
                "id": self._segment_idx,
                "label": lid,
                "label_name": COMMENTARY_LABEL_NAMES.get(lid, "other"),
                "text": (text or "")[:2000],
                "source_frame": source_frame,
                "time_sec": round(start_sec, 4),
                "wav": dst.name,
            }
            with open(self._label_manifest_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _render_commentary_mix(self) -> bool:
        if not self._export_joined or self._audio_mix_path is None:
            return False
        if not self._has_ffmpeg():
            print("[MatchCommentaryTTS] ffmpeg not found; skip commentary audio mix", flush=True)
            return False
        with self._timeline_lock:
            entries = list(self._timeline_entries)
        if not entries:
            return False
        # Order by video time, then by enqueue order if same frame (stable).
        idx_sorted = list(enumerate(entries))
        idx_sorted.sort(key=lambda e: (e[1][0], e[0]))
        ordered: list[Tuple[float, Path]] = [e[1] for e in idx_sorted]
        durs = [self._wav_duration_sec(p) for _, p in ordered]
        if len(ordered) == 1:
            shutil.copy2(ordered[0][1], self._audio_mix_path)
            return True

        # `amix`+`adelay` places each clip at a timeline position. If two clips would
        # overlap in wall-clock (events close in video time), the waveforms add — bad.
        # Shift each segment to start at max(requested, end_of_previous) so the export
        # matches a serial TTS queue without simultaneous speech.
        eff_starts: list[float] = []
        t_end = 0.0
        for i, (req_start, _p) in enumerate(ordered):
            s = max(0.0, float(req_start), t_end)
            eff_starts.append(s)
            t_end = s + durs[i]

        cmd = ["ffmpeg", "-y"]
        for _, seg in ordered:
            cmd.extend(["-i", str(seg)])

        labels = []
        parts = []
        for i, eff in enumerate(eff_starts):
            delay_ms = int(eff * 1000.0)
            lbl = f"a{i}"
            parts.append(f"[{i}:a]adelay={delay_ms}|{delay_ms}[{lbl}]")
            labels.append(f"[{lbl}]")
        parts.append(f"{''.join(labels)}amix=inputs={len(ordered)}:normalize=0[aout]")
        filter_complex = ";".join(parts)
        cmd.extend(
            [
                "-filter_complex",
                filter_complex,
                "-map",
                "[aout]",
                "-ac",
                "2",
                "-ar",
                "48000",
                str(self._audio_mix_path),
            ]
        )
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception as ex:
            print(f"[MatchCommentaryTTS] audio mix failed: {ex}", flush=True)
            return False

    def _recover_video(self, src: Path) -> Path:
        """Try to repair a potentially incomplete MP4 (missing MOOV atom after Ctrl+C).

        Runs ``ffmpeg -c copy`` which rewrites the container header. Returns the
        recovered path on success, or the original path if recovery is not needed
        or fails (caller still tries with the original).
        """
        recovered = src.with_name(src.stem + "_recovered" + src.suffix)
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-fflags", "+genpts+igndts",
                    "-i", str(src),
                    "-c", "copy",
                    "-movflags", "+faststart",
                    str(recovered),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=120,
            )
            if result.returncode == 0 and recovered.exists() and recovered.stat().st_size > 1024:
                print(
                    f"[MatchCommentaryTTS] recovered video: {recovered.name}",
                    flush=True,
                )
                return recovered
        except Exception as ex:
            print(f"[MatchCommentaryTTS] video recovery attempt failed: {ex}", flush=True)
        if recovered.exists():
            recovered.unlink(missing_ok=True)
        return src

    def _mux_clean_video_with_commentary(self) -> None:
        if (
            not self._export_joined
            or self._clean_video_path is None
            or self._audio_mix_path is None
            or self._joined_video_path is None
        ):
            return
        if not self._clean_video_path.exists():
            print("[MatchCommentaryTTS] clean video not found; skip final mux", flush=True)
            return
        if not self._audio_mix_path.exists():
            return
        if not self._has_ffmpeg():
            print("[MatchCommentaryTTS] ffmpeg not found; skip final mux", flush=True)
            return

        # Repair the clean video container in case the VideoWriter was not
        # properly finalised (e.g. Ctrl+C before the MOOV atom was flushed).
        video_src = self._recover_video(self._clean_video_path)
        recovered = video_src != self._clean_video_path

        cmd = [
            "ffmpeg",
            "-y",
            "-fflags", "+genpts",
            "-i", str(video_src),
            "-i", str(self._audio_mix_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            str(self._joined_video_path),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(
                f"[MatchCommentaryTTS] exported joined clean video: {self._joined_video_path}",
                flush=True,
            )
        except Exception as ex:
            print(f"[MatchCommentaryTTS] final mux failed: {ex}", flush=True)
        finally:
            if recovered and video_src.exists():
                video_src.unlink(missing_ok=True)

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                break
            text, source_frame, play_next_immediately, _label = item
            text = text.strip()
            if not text:
                continue
            now = time.monotonic()
            gap = now - self._last_play_monotonic
            if gap < self._min_interval_sec:
                time.sleep(self._min_interval_sec - gap)
            path: Optional[str] = None
            try:
                path = generate_speech(
                    text,
                    voice=self._voice,
                    response_format="wav",
                    speed=self._speed,
                    use_online_tts=self._use_online,
                    gtts_lang=self._gtts_lang,
                )
                self._save_timeline_segment(path, source_frame, _label, text)
                proc = start_audio_playback(path)
                with self._playback_lock:
                    self._current_play_proc = proc
                try:
                    proc.wait()
                finally:
                    with self._playback_lock:
                        if self._current_play_proc is proc:
                            self._current_play_proc = None
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=0.5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
            except Exception as ex:
                print(f"[MatchCommentaryTTS] speak failed: {ex}", flush=True)
                print(traceback.format_exc(), flush=True)
            finally:
                if path:
                    Path(path).unlink(missing_ok=True)
                # If this item was a lead-in (e.g. "Ma attenzione!"), reset the clock
                # so the LLM response that follows plays immediately without min-interval.
                if play_next_immediately:
                    self._last_play_monotonic = -1e9
                else:
                    self._last_play_monotonic = time.monotonic()

    def shutdown(self) -> None:
        self._q.put(None)
        self._worker.join(timeout=2.0)
        mixed = self._render_commentary_mix()
        if mixed:
            self._mux_clean_video_with_commentary()
        if self._segments_dir is not None and not self._keep_segments and self._segments_dir.exists():
            shutil.rmtree(self._segments_dir, ignore_errors=True)

    def enqueue_speak(
        self,
        text: str,
        *,
        urgent: bool = False,
        source_frame: Optional[int] = None,
        play_next_immediately: bool = False,
        label: int = L_OTHER,
    ) -> None:
        """
        ``label`` must be 0-5: see :mod:`src.commentator.commentary_labels` and
        ``commentary_segments/label_0_to_5_legend.txt`` (event, goal, kinetics, geometry, periodic, other).
        """
        lid = int(label) if 0 <= int(label) <= 5 else L_OTHER
        text = (text or "").strip()
        if not text:
            return
        if urgent and text.startswith("Goal!"):
            now = time.monotonic()
            with self._goal_dedup_lock:
                if text == self._dedup_goal_text and (now - self._dedup_goal_mono) < 5.0:
                    return
                self._dedup_goal_text = text
                self._dedup_goal_mono = now
        if self._audio_enabled and urgent:
            while True:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break
        with self._commentary_display_lock:
            self._last_commentary_line = text
        if self._audio_enabled:
            self._q.put((text, source_frame, play_next_immediately, lid))

    def _drain_speech_queue(self) -> None:
        if not self._audio_enabled:
            return
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def interrupt_commentary_playback(self) -> None:
        """Stop current clip (ffplay/aplay) and drop queued TTS items."""
        with self._playback_lock:
            proc = self._current_play_proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=0.6)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._drain_speech_queue()
        # Reset interval so the next item (e.g. lead-in phrase) plays immediately.
        self._last_play_monotonic = -1e9

    def last_commentary_line_for_display(self) -> str:
        with self._commentary_display_lock:
            return self._last_commentary_line

    def _llm_async_done(self) -> None:
        self._llm_in_flight = max(0, self._llm_in_flight - 1)

    def _record_assistant_line(self, text: str) -> None:
        t = (text or "").strip()
        if t:
            self._llm_assistant_history.append(t)

    def _record_event(self, item: str) -> None:
        t = (item or "").strip()
        if t:
            self._recent_events.append(t)

    def _score_hint(self, events: "EventFlags") -> str:
        sl = self._score_left if self._score_left is not None else events.score_left
        sr = self._score_right if self._score_right is not None else events.score_right
        if sl is None or sr is None:
            return "score unavailable"
        left = events.left_team_full_name or events.left_team_name or "left"
        right = events.right_team_full_name or events.right_team_name or "right"
        if sl == sr:
            state = "draw game"
        elif sl > sr:
            state = f"{left} leads"
        else:
            state = f"{right} leads"
        return f"{left} {sl} - {sr} {right} ({state})"

    def _recent_events_hint(self) -> str:
        return " | ".join(self._recent_events) if self._recent_events else "no notable events yet"

    def maybe_llm_commentary(
        self,
        events: "EventFlags",
        llm: object,
        *,
        eventproc: object,
        commentator: Optional[object] = None,
    ) -> None:
        trigger_frame = events.trigger_frame
        comm = self._comm_cfg
        if events.score_left is not None:
            self._score_left = int(events.score_left)
        if events.score_right is not None:
            self._score_right = int(events.score_right)

        if events.goal_for_left_team:
            self._record_event("goal left")
            self._last_goal_frame = trigger_frame
        elif events.goal_for_right_team:
            self._record_event("goal right")
            self._last_goal_frame = trigger_frame
        elif events.home_team_ball_strike or events.away_team_ball_strike:
            self._record_event("ball strike")
        elif not events.has_ball:
            self._record_event("no-ball phase")

        if events.has_ball and trigger_frame is not None:
            self._last_ball_seen_frame = int(trigger_frame)

        goal = goal_line(events, comm) if comm else None
        if goal:
            used_llm_goal = False
            if (
                comm.goal_llm_commentary_enabled
                and llm is not None
            ):
                feats = self._feats_with_names(events)
                lang = comm.geometry_llm_language

                def _on_goal(text: str) -> None:
                    t = (text or "").strip()
                    if not t:
                        return
                    self._record_assistant_line(t)
                    print(f"[Commentary GOAL-LLM] {t}", flush=True)
                    self.enqueue_speak(t, urgent=True, source_frame=trigger_frame, label=L_GOAL)

                used_llm_goal = maybe_goal_state_llm(
                    events=events,
                    features=feats,
                    llm=llm,
                    language=lang,
                    score_hint=self._score_hint(events),
                    recent_events_hint=self._recent_events_hint(),
                    on_spoken=_on_goal,
                    on_async_complete=self._llm_async_done,
                )
                if used_llm_goal:
                    self._llm_in_flight += 1
            if not used_llm_goal:
                print(f"[Commentary EVENT] {goal}", flush=True)
                urgent = bool(events.goal_for_left_team or events.goal_for_right_team)
                self.enqueue_speak(goal, urgent=urgent, source_frame=trigger_frame, label=L_GOAL)
            maybe_general_comment(
                events,
                llm,
                comm,
                enqueue_speak=self.enqueue_speak,
                trigger_frame=trigger_frame,
            )
            return

        kinetics_scheduled = False
        strike_live = events.home_team_ball_strike or events.away_team_ball_strike
        if (
            strike_live
            and comm.ball_kinetics_llm_commentary_enabled
            and commentator is not None
            and llm is not None
        ):
            fw = self._config.field_config.width
            fh = self._config.field_config.height
            fxlimit = self._config.field_config.xlimit
            ctx = comm.ball_kinetics_llm_context_frames
            feats = self._feats_with_names(events)
            lang = comm.ball_kinetics_llm_language
            def _on_kinetics_spoken(t: str) -> None:
                text = (t or "").strip()
                if not text:
                    return
                self._record_assistant_line(text)
                print(f"[Commentary BALL-KINETICS-LLM] {text}", flush=True)
                # Enqueue as urgent so it plays immediately after the interrupt.
                self.enqueue_speak(
                    text,
                    source_frame=trigger_frame,
                    play_next_immediately=True,
                    label=L_BALL_KINETICS,
                )

            def _kinetics_committed() -> None:
                # Interrupt current audio — no lead-in phrase, just change subject directly.
                self.interrupt_commentary_playback()
                print("[Commentary BALL-KINETICS] interrupted, waiting for LLM...", flush=True)

            kinetics_scheduled = maybe_ball_kinetics_llm_comment(
                events=events,
                features=feats,
                comm_cfg=comm,
                eventproc=eventproc,
                commentator=commentator,
                llm=llm,
                context_frames=ctx,
                field_width_mm=fw,
                field_height_mm=fh,
                xlimit_mm=fxlimit,
                language=lang,
                on_spoken=_on_kinetics_spoken,
                on_async_complete=self._llm_async_done,
                on_committed=_kinetics_committed,
            )
            if kinetics_scheduled:
                self._llm_in_flight += 1
                self._last_geometry_llm_mono = time.monotonic()
                if trigger_frame is not None:
                    self._last_geometry_frame = int(trigger_frame)
                    self._last_priority_event_frame = int(trigger_frame)

        geometry_scheduled = False
        if (
            not kinetics_scheduled
            and comm.geometry_llm_commentary_enabled
            and commentator is not None
            and llm is not None
        ):
            fw = self._config.field_config.width
            fh = self._config.field_config.height
            lang = comm.geometry_llm_language
            ctx = comm.geometry_llm_context_frames
            feats = self._feats_with_names(events)

            def _on_spoken(t: str) -> None:
                text = (t or "").strip()
                if not text:
                    return
                self._record_assistant_line(text)
                print(f"[Commentary GEOMETRY-LLM] {text}", flush=True)
                self.enqueue_speak(text, source_frame=trigger_frame, label=L_GEOMETRY)

            geometry_scheduled = maybe_geometry_llm_comment(
                events=events,
                features=feats,
                commentator=commentator,
                llm=llm,
                context_frames=ctx,
                field_width_mm=fw,
                field_height_mm=fh,
                language=lang,
                on_spoken=_on_spoken,
                on_async_complete=self._llm_async_done,
            )
            if geometry_scheduled:
                self._llm_in_flight += 1
                self._last_geometry_llm_mono = time.monotonic()
                if trigger_frame is not None:
                    self._last_geometry_frame = int(trigger_frame)
                    self._last_priority_event_frame = int(trigger_frame)

        skip_geom_det = bool(
            geometry_scheduled
            and comm.geometry_llm_skip_deterministic_ball_strike
        )
        skip_kinetics_det = bool(
            kinetics_scheduled
            and comm.ball_kinetics_llm_skip_deterministic_events
        )
        skip_strike = skip_geom_det or skip_kinetics_det
        skip_shot_pass = skip_kinetics_det

        line = None
        if comm and not skip_strike:
            line = ball_strike_line(events, comm)
        if line is None and comm and not skip_shot_pass:
            line = shot_line(events, comm)
        if line is None and comm and not skip_shot_pass:
            line = pass_line(events, comm)

        if line:
            print(f"[Commentary EVENT] {line}", flush=True)
            urgent = bool(events.goal_for_left_team or events.goal_for_right_team)
            self.enqueue_speak(line, urgent=urgent, source_frame=trigger_frame, label=L_EVENT)
            if "strike" in line.lower():
                self._record_event("deterministic strike line")
            if trigger_frame is not None:
                self._last_priority_event_frame = int(trigger_frame)

        # Post-goal dead-ball: prompt a kickoff reset line when ball is absent.
        if (
            comm.post_goal_kickoff_commentary_enabled
            and not events.has_ball
            and self._last_goal_frame is not None
            and trigger_frame is not None
        ):
            window_frames = comm.post_goal_wait_window_frames
            cooldown_sec = comm.post_goal_wait_comment_cooldown_sec
            now = time.monotonic()
            if (int(trigger_frame) - int(self._last_goal_frame)) <= window_frames:
                if (now - self._last_kickoff_wait_mono) >= cooldown_sec:
                    wait_line = "Teams are reset and waiting for the kickoff restart."
                    print(f"[Commentary RESET] {wait_line}", flush=True)
                    self.enqueue_speak(wait_line, source_frame=trigger_frame, label=L_OTHER)
                    self._record_assistant_line(wait_line)
                    self._record_event("kickoff wait after goal")
                    self._last_kickoff_wait_mono = now

        has_strike_or_geom_llm = bool(
            geometry_scheduled
            or kinetics_scheduled
            or events.home_team_ball_strike
            or events.away_team_ball_strike
        )
        has_deterministic_line = bool(line)

        if (
            comm.periodic_llm_commentary_enabled
            and commentator is not None
            and llm is not None
            and not has_strike_or_geom_llm
            and not has_deterministic_line
            and self._llm_in_flight == 0
        ):
            interval = comm.periodic_llm_interval_sec
            gap_geom = comm.periodic_llm_min_gap_after_geometry_sec
            gap_frames = comm.periodic_llm_min_gap_after_priority_event_frames
            tail = comm.periodic_llm_tail_frames
            stride = max(1, comm.periodic_llm_stride)
            hist_turns = comm.periodic_llm_history_turns
            now = time.monotonic()
            # If ball is absent for a long time, double the normal interval so we
            # don't spam but still break the silence.
            _no_ball_frames = comm.periodic_llm_no_ball_grace_frames
            _ball_absent = (
                not events.has_ball
                and not (
                    trigger_frame is not None
                    and self._last_ball_seen_frame >= 0
                    and (int(trigger_frame) - self._last_ball_seen_frame) <= _no_ball_frames
                )
            )
            effective_interval = interval * 2 if _ball_absent else interval
            if (now - self._last_periodic_llm_mono) >= effective_interval:
                if (now - self._last_geometry_llm_mono) >= gap_geom:
                    if trigger_frame is not None and self._last_priority_event_frame >= 0:
                        # When ball is absent, skip the priority-event gap check — there
                        # won't be new events anyway, so the guard is pointless.
                        if not _ball_absent and (int(trigger_frame) - self._last_priority_event_frame) < gap_frames:
                            maybe_general_comment(
                                events,
                                llm,
                                comm,
                                enqueue_speak=self.enqueue_speak,
                                trigger_frame=trigger_frame,
                            )
                            return
                    fw = self._config.field_config.width
                    fh = self._config.field_config.height
                    fps = self._config.processing.fps
                    feats = self._feats_with_names(events)
                    lang = comm.geometry_llm_language
                    req_seq = self._periodic_req_seq = self._periodic_req_seq + 1
                    request_frame = int(trigger_frame) if trigger_frame is not None else -1

                    def _on_periodic(t: str) -> None:
                        # Drop delayed periodic lines when a newer priority event (kick/pass/shot/goal)
                        # has already happened.
                        if req_seq != self._periodic_req_seq:
                            return
                        if self._last_priority_event_frame > request_frame:
                            return
                        text = (t or "").strip()
                        if not text:
                            return
                        self._record_assistant_line(text)
                        print(f"[Commentary PERIODIC-LLM] {text}", flush=True)
                        self.enqueue_speak(text, source_frame=trigger_frame, label=L_PERIODIC)

                    scheduled = maybe_periodic_scene_llm(
                        events=events,
                        features=feats,
                        comm_cfg=comm,
                        eventproc=eventproc,
                        commentator=commentator,
                        llm=llm,
                        field_width_mm=fw,
                        field_height_mm=fh,
                        language=lang,
                        tail_frames=tail,
                        stride=stride,
                        fps=fps,
                        history_assistant_lines=list(self._llm_assistant_history),
                        history_max_turns=hist_turns,
                        score_hint=self._score_hint(events),
                        recent_events_hint=self._recent_events_hint(),
                        on_spoken=_on_periodic,
                        on_async_complete=self._llm_async_done,
                    )
                    if scheduled:
                        self._llm_in_flight += 1
                        self._last_periodic_llm_mono = now
                    # If not scheduled (e.g. empty narrative digest), do not advance the 40s clock —
                    # so the next frame can try again after other gates pass.

        maybe_general_comment(
            events,
            llm,
            comm,
            enqueue_speak=self.enqueue_speak,
            trigger_frame=trigger_frame,
        )
