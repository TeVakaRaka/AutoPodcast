"""Auto-calibration of speech detection thresholds from mic noise floor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from autopodcast.core.analyzer import apply_highpass, compute_rms


@dataclass
class CalibrationResult:
    noise_floor_db: float
    speech_threshold_db: float
    release_threshold_db: float
    label: str


def estimate_noise_floor(
    audio: np.ndarray,
    sample_rate: int,
    window_s: float = 3.0,
    percentile: float = 95.0,
    window_ms: float = 30.0,
    hop_ms: float = 10.0,
    highpass_hz: float = 0.0,
) -> float:
    """Estimate noise floor from the full audio.

    Algorithm:
    1. Optionally apply highpass filter.
    2. Compute RMS over the entire file with a coarse 50 ms hop.
    3. Split RMS frames into 1-second windows.
    4. Take the median RMS within each window.
    5. Return the 10th percentile of those medians.

    This is robust against speech at the start and transient bursts.
    Falls back to -96 dB for empty/silent audio.

    The window_s and percentile parameters are kept for backward compatibility
    but are no longer used.
    """
    if len(audio) == 0:
        return -96.0

    if highpass_hz > 0.0:
        audio = apply_highpass(audio, highpass_hz, sample_rate)

    _COARSE_HOP_MS = 50.0
    _ANALYSIS_WINDOW_S = 1.0
    _NOISE_PERCENTILE = 10.0

    coarse_window_size = int(sample_rate * _COARSE_HOP_MS / 1000.0)
    coarse_hop_size = coarse_window_size  # non-overlapping

    rms_db = compute_rms(audio, coarse_window_size, coarse_hop_size)
    if len(rms_db) == 0:
        return -96.0

    # Number of RMS frames per 1-second analysis window
    frames_per_window = max(1, int(_ANALYSIS_WINDOW_S * 1000.0 / _COARSE_HOP_MS))

    # Split into 1-second windows and take median of each
    medians = []
    for start in range(0, len(rms_db), frames_per_window):
        chunk = rms_db[start : start + frames_per_window]
        if len(chunk) > 0:
            medians.append(float(np.median(chunk)))

    if not medians:
        return -96.0

    return float(np.percentile(medians, _NOISE_PERCENTILE))


def calibrate(
    audio: np.ndarray,
    sample_rate: int,
    label: str,
    window_s: float = 3.0,
    percentile: float = 95.0,
    margin_db: float = 12.0,
    max_speech_threshold_db: float = -12.0,
    highpass_hz: float = 0.0,
) -> CalibrationResult:
    """Calibrate speech thresholds from audio noise floor.

    speech_threshold = noise_floor + margin (clamped to max_speech_threshold_db).
    release_threshold = speech_threshold - 6 dB.
    """
    noise_floor = estimate_noise_floor(
        audio, sample_rate, window_s, percentile,
        highpass_hz=highpass_hz,
    )
    speech_threshold = min(noise_floor + margin_db, max_speech_threshold_db)
    release_threshold = speech_threshold - 6.0

    return CalibrationResult(
        noise_floor_db=noise_floor,
        speech_threshold_db=speech_threshold,
        release_threshold_db=release_threshold,
        label=label,
    )
