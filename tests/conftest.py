"""Test fixtures: synthetic audio generation."""

from __future__ import annotations

import numpy as np
import pytest

from autopodcast.models.project import AudioInput, ProjectConfig
from pathlib import Path


@pytest.fixture
def default_config() -> ProjectConfig:
    """Default ProjectConfig for testing."""
    return ProjectConfig(
        audio_inputs=[
            AudioInput(path=Path("a.wav"), speaker_label="host", camera_index=1),
            AudioInput(path=Path("b.wav"), speaker_label="guest", camera_index=2),
        ],
    )


def make_tone(duration_s: float, level_db: float, sr: int = 16000, freq: float = 440.0) -> np.ndarray:
    """Generate a sine tone at a given level in dB (0 dB = full scale)."""
    n_samples = int(duration_s * sr)
    t = np.arange(n_samples) / sr
    amplitude = 10.0 ** (level_db / 20.0)
    return amplitude * np.sin(2 * np.pi * freq * t)


def make_silence(duration_s: float, sr: int = 16000) -> np.ndarray:
    """Generate silence."""
    return np.zeros(int(duration_s * sr), dtype=np.float64)


def make_speech_pattern(
    segments: list[tuple[float, float, float]],
    total_duration_s: float,
    sr: int = 16000,
) -> np.ndarray:
    """Generate audio with speech-like segments.

    Args:
        segments: List of (start_s, end_s, level_db) tuples.
        total_duration_s: Total duration of the output.
        sr: Sample rate.

    Returns:
        numpy array with tones at specified positions, silence elsewhere.
    """
    audio = np.zeros(int(total_duration_s * sr), dtype=np.float64)

    for start_s, end_s, level_db in segments:
        start_idx = int(start_s * sr)
        end_idx = int(end_s * sr)
        n = end_idx - start_idx
        t = np.arange(n) / sr
        amplitude = 10.0 ** (level_db / 20.0)
        audio[start_idx:end_idx] = amplitude * np.sin(2 * np.pi * 200.0 * t)

    return audio
