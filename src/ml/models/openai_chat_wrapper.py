"""OpenAI Chat Completions API — same text interface as :class:`OllamaWrapper` for commentary."""

from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

import traceback

_executor = ThreadPoolExecutor(max_workers=4)

_API_TXT_NAME = "api.txt"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_openai_api_key_from_api_txt() -> str:
    """First non-empty line from ``api.txt`` in project root (comments with ``#`` skipped)."""
    path = _project_root() / _API_TXT_NAME
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        return line
    return ""


class OpenAIChatWrapper:
    """
    Text-only LLM via OpenAI-compatible HTTP API (``openai`` package).

    Implements ``llm_call`` / ``llm_call_async`` like :class:`OllamaWrapper`.
    Vision (``vlm_*``) is not implemented; OCR/VLM should keep using ``OllamaWrapper``.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: Optional[str] = None,
    ) -> None:
        if not api_key or not str(api_key).strip():
            raise ValueError(
                "OpenAI API key is empty. Set models.openai_api_key, OPENAI_API_KEY, "
                f"or put the key in {_API_TXT_NAME} at the project root."
            )
        from openai import OpenAI

        kwargs = {"api_key": api_key.strip()}
        if base_url and str(base_url).strip():
            kwargs["base_url"] = str(base_url).strip()
        self._client = OpenAI(**kwargs)
        self._model = model

    def llm_call(self, prompt: str) -> str:
        print(
            f"[OpenAI LLM] sync call | model={self._model} | prompt_chars={len(prompt)}",
            flush=True,
        )
        return self._chat(prompt)

    def llm_call_async(
        self,
        prompt: str,
        on_done: Optional[Callable[[str], None]] = None,
    ) -> "Future[str]":
        print(
            f"[OpenAI LLM] async call | model={self._model} | prompt_chars={len(prompt)}",
            flush=True,
        )
        fut = _executor.submit(self._chat, prompt)
        if on_done is not None:

            def _cb(f: "Future[str]") -> None:
                exc = f.exception()
                if exc:
                    print(f"[OpenAIChatWrapper] async call failed: {exc}", flush=True)
                    traceback.format_exc(exc)
                    on_done("")
                else:
                    on_done(f.result())

            fut.add_done_callback(_cb)
        return fut

    def vlm_call(self, prompt: str, img) -> str:
        raise NotImplementedError(
            "OpenAIChatWrapper is text-only; use OllamaWrapper for vision / OCR VLM."
        )

    def vlm_call_async(self, prompt: str, img, on_done: Optional[Callable[[str], None]] = None):
        raise NotImplementedError(
            "OpenAIChatWrapper is text-only; use OllamaWrapper for vision / OCR VLM."
        )

    def _chat(self, prompt: str) -> str:
        r = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        choice = r.choices[0].message
        content = (choice.content or "").strip()
        return content


def commentary_llm_from_config(models_cfg: Any) -> OpenAIChatWrapper:
    """Build :class:`OpenAIChatWrapper` from ``config.models``.

    API key resolution order: ``models.openai_api_key`` in YAML, then ``OPENAI_API_KEY``,
    then first non-comment line of ``api.txt`` in the project root.
    """
    model = models_cfg.openai_model
    key = models_cfg.openai_api_key
    if key is not None and str(key).strip():
        api_key = str(key).strip()
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        api_key = _read_openai_api_key_from_api_txt()
    base = models_cfg.openai_base_url
    return OpenAIChatWrapper(model, api_key=api_key, base_url=base)
