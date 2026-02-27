"""RMS analysis and envelope smoothing."""

from __future__ import annotations

import numpy as np

from autopodcast.models.domain import AnalysisFrame, SpeakerActivity
from autopodcast.models.project import ProjectConfig


def compute_rms(audio: np.ndarray, window_size: int, hop_size: int) -> np.ndarray:
    """Compute RMS energy in dB for each frame.

    Returns array of shape (n_frames,) with values in dB.
    """
    n_samples = len(audio)
    n_frames = max(0, (n_samples - window_size) // hop_size + 1)

    if n_frames == 0:
        return np.array([], dtype=np.float64)

    rms_values = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_size
        chunk = audio[start : start + window_size]
        rms_linear = np.sqrt(np.mean(chunk ** 2))
        rms_values[i] = 20.0 * np.log10(rms_linear + 1e-10)

    return rms_values


def smooth_envelope(rms_db: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply exponential moving average smoothing.

    EMA provides fast attack and smooth decay, natural for speech envelopes.
    """
    if len(rms_db) == 0:
        return rms_db.copy()

    alpha = 2.0 / (kernel_size + 1)
    envelope = np.empty_like(rms_db)
    envelope[0] = rms_db[0]

    for i in range(1, len(rms_db)):
        envelope[i] = alpha * rms_db[i] + (1 - alpha) * envelope[i - 1]

    return envelope


def analyze_speaker(
    audio: np.ndarray, speaker_label: str, config: ProjectConfig
) -> SpeakerActivity:
    """Full analysis pipeline for one speaker: RMS -> envelope -> frames."""
    rms_db = compute_rms(audio, config.window_samples, config.hop_samples)
    envelope_db = smooth_envelope(rms_db, config.smoothing_kernel_size)

    hop_s = config.hop_ms / 1000.0
    frames = []
    for i in range(len(rms_db)):
        frames.append(
            AnalysisFrame(
                time_s=i * hop_s,
                rms_db=float(rms_db[i]),
                envelope_db=float(envelope_db[i]),
                is_active=False,  # filled by detector
            )
        )

    return SpeakerActivity(speaker_label=speaker_label, frames=frames)
