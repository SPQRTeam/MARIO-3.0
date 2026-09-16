"""
GENERAL_COMMENT — richer lines from an LLM (or other generator), separate from
:class:`events_commentary` (fixed goal / shot / pass).

Enable with ``commentating.general_comment_enabled: true`` when you add prompts
and call sites. For now this module is a no-op placeholder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .event_flags import EventFlags


def maybe_general_comment(
    events: "EventFlags",
    llm: Optional[object],
    comm_cfg: Any,
    *,
    enqueue_speak: Any = None,
    trigger_frame: Optional[int] = None,
) -> None:
    """Optional LLM commentary. ``enqueue_speak`` is ``MatchCommentaryTTS.enqueue_speak`` when wired.

    When disabled or ``llm`` is None, returns immediately. Extend this function
    when you decide which ``EventFlags`` should trigger general chat and how
    prompts are built.
    """
    if not comm_cfg.general_comment_enabled:
        return
    if llm is None or enqueue_speak is None:
        return
    # Example future hook:
    # prompt = build_general_prompt(events)
    # if prompt:
    #     llm.llm_call_async(prompt, on_done=lambda line: enqueue_speak(line.strip(), ...))
    return
