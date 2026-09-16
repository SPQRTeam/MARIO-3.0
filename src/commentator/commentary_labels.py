"""
Fixed labels 0..5 for each generated commentary audio segment.

Use the sidecar ``audio_label_manifest.jsonl`` in ``commentary_segments/`` to join clips
with your own review notes (correct/wrong, reason) in another file using ``id`` or ``wav``.
"""

from __future__ import annotations

# Commentary source category (not a quality score — for filtering / bookkeeping).
L_EVENT = 0
L_GOAL = 1
L_BALL_KINETICS = 2
L_GEOMETRY = 3
L_PERIODIC = 4
L_OTHER = 5

NAMES: dict[int, str] = {
    L_EVENT: "event",
    L_GOAL: "goal",
    L_BALL_KINETICS: "ball_kinetics",
    L_GEOMETRY: "geometry",
    L_PERIODIC: "periodic",
    L_OTHER: "other",
}

LEGEND = """commentary label codes (0-5) — each WAV is named seg_NNNNN_L<code>.wav
  0  event         deterministic template: strike / shot / pass from events_commentary
  1  goal          goal line (template, goal LLM, or OCR/vlm goal speech)
  2  ball_kinetics ball kinetics LLM
  3  geometry      geometry / possession LLM
  4  periodic      periodic scene LLM
  5  other         e.g. post-goal wait line, future general_comment, or unspecified

See audio_label_manifest.jsonl for id, text, frame, and filename per segment.
You can add a separate TSV/CSV for manual QA: id, your_rating, notes, ...
""".strip()
