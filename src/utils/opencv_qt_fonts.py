"""Populate ``cv2/qt/fonts`` when missing — Qt-backed OpenCV wheels omit fonts (see opencv-python#1205)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterator

_done = False


def _dejavu_ttf_sources() -> Iterator[Path]:
    for base in (
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/dejavu"),
        Path("/usr/share/fonts/TTF"),
    ):
        if base.is_dir():
            yield from sorted(base.glob("DejaVu*.ttf"))
    try:
        import matplotlib

        m = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        if m.is_dir():
            yield from sorted(m.glob("DejaVu*.ttf"))
    except ImportError:
        pass


def ensure_opencv_qt_fonts() -> None:
    """Copy DejaVu TTFs into ``site-packages/cv2/qt/fonts`` if that dir is empty or missing."""
    global _done
    if _done:
        return
    _done = True
    try:
        import cv2
    except ImportError:
        return

    qt_fonts = Path(cv2.__file__).resolve().parent / "qt" / "fonts"
    try:
        if qt_fonts.is_dir() and any(qt_fonts.iterdir()):
            return
    except OSError:
        return

    sources = list(_dejavu_ttf_sources())
    if not sources:
        return

    try:
        qt_fonts.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    for src in sources:
        dst = qt_fonts / src.name
        if dst.exists():
            continue
        try:
            shutil.copy2(src, dst)
        except OSError:
            continue
