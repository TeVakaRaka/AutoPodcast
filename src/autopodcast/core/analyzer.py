"""RMS analysis and envelope smoothing."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from autopodcast.models.domain import AnalysisFrame, SpeakerActivity
from autopodcast.models.project import ProjectConfig


def apply_highpass(audio: np.ndarray, cutoff_hz: float, sample_rate: int, order: int = 4) -> np.ndarray:
    """Apply a Butterworth highpass filter. Returns audio unchanged if cutoff_hz <= 0."""
    if cutoff_hz <= 0.0:
        return audio
    sos = butter(order, cutoff_hz, btype="high", fs=sample_rate, output="sos")
    return sosfilt(sos, audio)


def apply_lowpass(audio: np.ndarray, cutoff_hz: float, sample_rate: int, order: int = 4) -> np.ndarray:
    """Apply a Butterworth lowpass filter. Returns audio unchanged if cutoff_hz <= 0."""
    if cutoff_hz <= 0.0:
        return audio
    sos = butter(order, cutoff_hz, btype="low", fs=sample_rate, output="sos")
    return sosfilt(sos, audio)


def compute_peak_db(audio: np.ndarray, window_size: int, hop_size: int) -> np.ndarray:
    """Compute peak level in dB for each frame.

    Returns array of shape (n_frames,) with values in dB.
    """
    n_samples = len(audio)
    n_frames = max(0, (n_samples - window_size) // hop_size + 1)

    if n_frames == 0:
        return np.array([], dtype=np.float64)

    peak_values = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_size
        chunk = audio[start : start + window_size]
        peak_linear = np.max(np.abs(chunk))
        peak_values[i] = 20.0 * np.log10(peak_linear + 1e-10)

    return peak_values


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


def smooth_envelope(rms_db: np.ndarray, attack_kernel: int, decay_kernel: int) -> np.ndarray:
    """Apply asymmetric EMA smoothing (fast attack, slow decay).

    When the signal rises (attack), a smaller kernel gives faster tracking.
    When the signal falls (decay), a larger kernel holds the envelope longer.
    """
    if len(rms_db) == 0:
        return rms_db.copy()

    alpha_attack = 2.0 / (attack_kernel + 1)
    alpha_decay = 2.0 / (decay_kernel + 1)
    envelope = np.empty_like(rms_db)
    envelope[0] = rms_db[0]

    for i in range(1, len(rms_db)):
        alpha = alpha_attack if rms_db[i] > envelope[i - 1] else alpha_decay
        envelope[i] = alpha * rms_db[i] + (1 - alpha) * envelope[i - 1]

    return envelope


def analyze_speaker(
    audio: np.ndarray, speaker_label: str, config: ProjectConfig,
    gain_db: float | None = None,
) -> SpeakerActivity:
    """Full analysis pipeline for one speaker: highpass -> lowpass -> RMS -> envelope -> frames."""
    if config.highpass_hz > 0.0:
        audio = apply_highpass(audio, config.highpass_hz, config.sample_rate)
    if config.lowpass_hz > 0.0:
        audio = apply_lowpass(audio, config.lowpass_hz, config.sample_rate)

    rms_db = compute_rms(audio, config.window_samples, config.hop_samples)
    peak_db_arr = compute_peak_db(audio, config.window_samples, config.hop_samples)
    effective_gain = gain_db if gain_db is not None else config.input_gain_db
    if effective_gain != 0.0:
        rms_db = rms_db + effective_gain
        peak_db_arr = peak_db_arr + effective_gain
    envelope_db = smooth_envelope(rms_db, config.smoothing_attack_kernel, config.smoothing_decay_kernel)

    hop_s = config.hop_ms / 1000.0
    frames = []
    for i in range(len(rms_db)):
        frames.append(
            AnalysisFrame(
                time_s=i * hop_s,
                rms_db=float(rms_db[i]),
                envelope_db=float(envelope_db[i]),
                is_active=False,  # filled by detector
                peak_db=float(peak_db_arr[i]),
            )
        )

    return SpeakerActivity(speaker_label=speaker_label, frames=frames)
