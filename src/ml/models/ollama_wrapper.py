"""Ollama wrapper — blocking and non-blocking LLM/VLM calls."""

import cv2
import numpy as np
import ollama
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, Callable

PADDING_COLOR = (128, 128, 128)
VLM_INPUT_SIZE = (448, 448)

_executor = ThreadPoolExecutor(max_workers=4)


class OllamaWrapper:
    """Wrapper around a local Ollama model.

    llm_call / vlm_call  → blocking, return str
    llm_call_async / vlm_call_async → non-blocking, return Future[str]

    For fire-and-forget with a callback use the async variants and pass
    on_done=callable — the callback is invoked in the background thread
    with the result string as argument.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    # ── blocking ──────────────────────────────────────────────────────────────

    def llm_call(self, prompt: str) -> str:
        return self._chat(prompt, None)

    def vlm_call(self, prompt: str, img: np.ndarray) -> str:
        return self._chat(prompt, self._preprocess(img))

    # ── non-blocking ──────────────────────────────────────────────────────────

    def llm_call_async(self, prompt: str,
                       on_done: Optional[Callable[[str], None]] = None) -> "Future[str]":
        """Submit text prompt in background. Returns Future[str]."""
        return self._submit(prompt, None, on_done)

    def vlm_call_async(self, prompt: str, img: np.ndarray,
                       on_done: Optional[Callable[[str], None]] = None) -> "Future[str]":
        """Submit vision prompt in background. Returns Future[str]."""
        return self._submit(prompt, self._preprocess(img), on_done)

    # ── internals ─────────────────────────────────────────────────────────────

    def _submit(self, prompt: str, img_bytes: Optional[bytes],
                on_done: Optional[Callable[[str], None]]) -> "Future[str]":
        fut = _executor.submit(self._chat, prompt, img_bytes)
        if on_done is not None:
            def _cb(f):
                exc = f.exception()
                if exc:
                    print(f"[OllamaWrapper] async call failed: {exc}", flush=True)
                    on_done("")
                else:
                    on_done(f.result())
            fut.add_done_callback(_cb)
        return fut

    def _preprocess(self, img: np.ndarray) -> bytes:
        h, w = img.shape[:2]
        max_dim = max(h, w)
        pt = (max_dim - h) // 2
        pb = max_dim - h - pt
        pl = (max_dim - w) // 2
        pr = max_dim - w - pl
        img = cv2.copyMakeBorder(img, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=PADDING_COLOR)
        img = cv2.resize(img, VLM_INPUT_SIZE, interpolation=cv2.INTER_NEAREST)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buf.tobytes()

    def _chat(self, prompt: str, img_bytes: Optional[bytes]) -> str:
        message = {"role": "user", "content": prompt}
        if img_bytes is not None:
            message["images"] = [img_bytes]
        response = ollama.chat(model=self.model_name, messages=[message])
        content = response["message"]["content"].strip()
        # strip <think>...</think> blocks (qwen3 reasoning tokens)
        import re
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content
