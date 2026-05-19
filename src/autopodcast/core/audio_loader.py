"""Load audio from WAV files or demux from video via ffmpeg."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from autopodcast.utils.ffmpeg import demux_audio

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".aac"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".mxf", ".ts"}


def _resolve_path(path: Path, search_dir: Path | None) -> Path:
    """Resolve a media file path, searching in search_dir if the original doesn't exist."""
    if path.exists():
        return path
    if search_dir is not None:
        found = list(search_dir.rglob(path.name))
        if found:
            return found[0]
    locations = [str(path)]
    if search_dir is not None:
        locations.append(f"{search_dir} (recursive)")
    raise FileNotFoundError(
        f"File not found: {path.name}\nSearched in:\n" + "\n".join(f"  - {loc}" for loc in locations)
    )


def load_audio(path: Path, target_sr: int = 16000, search_dir: Path | None = None) -> np.ndarray:
    """Load audio from a file, returning mono float64 array at target_sr.

    If the file is a video, demuxes audio via ffmpeg first.
    If the file doesn't exist and search_dir is given, looks for it there by filename.
    """
    path = _resolve_path(path, search_dir)
    suffix = path.suffix.lower()

    if suffix in VIDEO_EXTENSIONS:
        return _load_from_video(path, target_sr)

    data, sr = sf.read(str(path), dtype="float64", always_2d=True)
    # Mix to mono
    mono = data.mean(axis=1)

    if sr != target_sr:
        mono = _resample_simple(mono, sr, target_sr)

    return mono


def load_and_align(paths: list[Path], target_sr: int = 16000, search_dir: Path | None = None) -> list[np.ndarray]:
    """Load multiple audio files and pad shorter ones to match the longest."""
    arrays = [load_audio(p, target_sr, search_dir=search_dir) for p in paths]

    max_len = max(len(a) for a in arrays)
    result = []
    for a in arrays:
        if len(a) < max_len:
            padded = np.zeros(max_len, dtype=np.float64)
            padded[: len(a)] = a
            result.append(padded)
        else:
            result.append(a)

    return result


def apply_offset(audio: np.ndarray, offset_s: float, sample_rate: int) -> np.ndarray:
    """Trim the beginning of audio by offset_s seconds."""
    if offset_s <= 0:
        return audio
    samples_to_skip = int(offset_s * sample_rate)
    if samples_to_skip >= len(audio):
        return np.zeros(0, dtype=audio.dtype)
    return audio[samples_to_skip:]


def _load_from_video(path: Path, target_sr: int) -> np.ndarray:
    """Demux audio from video via ffmpeg, then load."""
    raw_bytes = demux_audio(path, target_sr)
    data, _ = sf.read(
        raw_bytes,
        dtype="float64",
        channels=1,
        samplerate=target_sr,
        subtype="PCM_16",
        format="RAW",
    )
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data


def _resample_simple(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Simple resampling via linear interpolation. Good enough for RMS analysis."""
    if src_sr == dst_sr:
        return audio

    duration = len(audio) / src_sr
    n_out = int(duration * dst_sr)
    x_old = np.linspace(0, duration, len(audio), endpoint=False)
    x_new = np.linspace(0, duration, n_out, endpoint=False)
    return np.interp(x_new, x_old, audio)
