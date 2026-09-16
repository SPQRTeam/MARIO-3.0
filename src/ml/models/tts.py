from __future__ import annotations

import asyncio
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

try:
    import edge_tts
except ImportError:
    edge_tts = None

try:
    import emoji
except ImportError:
    emoji = None

try:
    from gtts import gTTS
except ImportError:
    gTTS = None

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

# Short aliases (marble.yaml: ``voice: diego``) → full Microsoft Edge TTS names.
# Italian and English are kept in parallel so commentary language and TTS stay aligned.
IT_VOICE_ALIASES: dict[str, str] = {
    "diego": "it-IT-DiegoNeural",
    "fluent": "it-IT-IsabellaNeural",
}
EN_VOICE_ALIASES: dict[str, str] = {
    "diego": "en-US-GuyNeural",
    "fluent": "en-US-JennyNeural",
}

# Back-compat: used by resolve_edge_voice for Italian.
VOICE_MAPPING = IT_VOICE_ALIASES


def resolve_edge_voice(voice: str, commentary_language: str) -> str:
    """
    Map a short alias to the right Edge ``voice=`` for the *commentary* language.

    If ``voice`` is already a full Edge id (e.g. ``en-US-GuyNeural``), it is returned unchanged
    (manual override for power users).
    """
    v = (voice or "diego").strip()
    # Explicit Edge voice id (e.g. ``en-US-GuyNeural``) — do not look up aliases.
    if "-" in v and ("Neural" in v or len(v) > 10):
        return v
    lang = (commentary_language or "en").strip().lower()[:2]
    table = IT_VOICE_ALIASES if lang == "it" else EN_VOICE_ALIASES
    return table.get(v.lower(), table.get("diego", "en-US-GuyNeural"))


def prepare_tts_input_with_context(text: str) -> str:
    if emoji is not None:
        text = emoji.replace_emoji(text, replace="")
    else:
        text = re.sub(r"[\U00010000-\U0010ffff]", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"code snippet: \1", text)
    text = re.sub(r"(\*\*|__|\*|_)", "", text)
    text = re.sub(r"</?[^>]+(>|$)", "", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def speed_to_rate(speed: float) -> str:
    if speed < 0 or speed > 2:
        raise ValueError("Speed must be between 0 and 2 (inclusive).")
    return f"{(speed - 1) * 100:+.0f}%"


def is_ffmpeg_installed() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


async def _generate_audio_edge(text: str, voice: str, response_format: str, speed: float) -> str:
    if edge_tts is None:
        raise RuntimeError("edge_tts is not installed; pip install edge-tts")
    edge_voice = VOICE_MAPPING.get(voice, voice)
    temp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_mp3_path = temp_mp3.name
    temp_mp3.close()
    try:
        communicator = edge_tts.Communicate(text=text, voice=edge_voice, rate=speed_to_rate(speed))
        await communicator.save(temp_mp3_path)
        if response_format == "mp3" or not is_ffmpeg_installed():
            return temp_mp3_path
        out = tempfile.NamedTemporaryFile(delete=False, suffix=f".{response_format}")
        out_path = out.name
        out.close()
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                temp_mp3_path,
                "-c:a",
                {"aac": "aac", "mp3": "libmp3lame", "wav": "pcm_s16le", "opus": "libopus", "flac": "flac"}.get(
                    response_format, "aac"
                ),
                "-f",
                {"aac": "mp4", "mp3": "mp3", "wav": "wav", "opus": "ogg", "flac": "flac"}.get(
                    response_format, response_format
                ),
                "-y",
                out_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        Path(temp_mp3_path).unlink(missing_ok=True)
        return out_path
    except Exception:
        Path(temp_mp3_path).unlink(missing_ok=True)
        raise


def verify_internet_connection() -> bool:
    import socket

    try:
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        return True
    except OSError:
        return False


def generate_speech(
    text: str,
    *,
    voice: str = "diego",
    response_format: str = "wav",
    speed: float = 1.0,
    use_online_tts: bool = True,
    gtts_lang: str = "it",
    **_: object,
) -> str:
    text = prepare_tts_input_with_context(text)
    if not text:
        raise ValueError("Empty text for TTS")
    if use_online_tts and verify_internet_connection() and edge_tts is not None:
        return asyncio.run(_generate_audio_edge(text, voice, response_format, speed))
    if gTTS is None or AudioSegment is None:
        raise RuntimeError("Offline TTS needs gtts and pydub (and ffmpeg).")
    tmp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp_mp3.close()
    try:
        gTTS(text=text, lang=gtts_lang, slow=False).save(tmp_mp3.name)
        audio = AudioSegment.from_mp3(tmp_mp3.name).set_frame_rate(16000).set_channels(1).set_sample_width(2)
        out_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        out_wav.close()
        audio.export(out_wav.name, format="wav")
        return out_wav.name
    finally:
        Path(tmp_mp3.name).unlink(missing_ok=True)


def start_audio_playback(path: str) -> subprocess.Popen:
    """
    Start ffplay or aplay without blocking until the clip ends.
    Caller should ``wait()`` and may ``terminate()`` for interrupt.
    """
    import shutil

    ffplay = shutil.which("ffplay")
    if ffplay:
        return subprocess.Popen(
            [ffplay, "-nodisp", "-autoexit", "-loglevel", "error", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    aplay = shutil.which("aplay")
    if aplay and str(path).lower().endswith(".wav"):
        return subprocess.Popen(
            [aplay, str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    raise RuntimeError("No ffplay or aplay found; install ffmpeg (ffplay) for TTS playback.")


def play_audio_file(path: str, timeout: Optional[float] = None) -> None:
    proc = start_audio_playback(path)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=1.5)
        except subprocess.TimeoutExpired:
            proc.kill()
