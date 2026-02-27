"""ffmpeg subprocess helpers for probing and demuxing audio."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _find_tool(name: str) -> str:
    """Find ffmpeg/ffprobe executable.

    Searches:
    1. System PATH via shutil.which
    2. Common Windows install locations
    """
    found = shutil.which(name)
    if found:
        return found

    if sys.platform == "win32":
        candidates = [
            Path(r"C:\ffmpeg\bin") / f"{name}.exe",
            Path(sys.executable).parent / "ffmpeg" / "bin" / f"{name}.exe",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)

    raise FileNotFoundError(
        f"{name} не найден. Установите {name} и добавьте в PATH.\n"
        f"{name} not found. Install {name} and add it to PATH."
    )


def _subprocess_kwargs() -> dict:
    """Extra kwargs for subprocess on Windows (hide console window)."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def probe_duration(path: Path) -> float:
    """Get media duration in seconds using ffprobe."""
    ffprobe = _find_tool("ffprobe")
    cmd = [
        ffprobe,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(path),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, check=True,
        **_subprocess_kwargs(),
    )
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def demux_audio(path: Path, sample_rate: int = 16000) -> io.BytesIO:
    """Extract audio from video as raw PCM s16le mono via ffmpeg pipe."""
    ffmpeg = _find_tool("ffmpeg")
    cmd = [
        ffmpeg,
        "-i", str(path),
        "-ar", str(sample_rate),
        "-ac", "1",
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "pipe:1",
    ]
    result = subprocess.run(
        cmd, capture_output=True, check=True,
        **_subprocess_kwargs(),
    )
    return io.BytesIO(result.stdout)
