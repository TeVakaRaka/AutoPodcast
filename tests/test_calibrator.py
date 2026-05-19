"""Tests for auto-calibration of speech thresholds."""

from __future__ import annotations

import numpy as np

from autopodcast.core.calibrator import CalibrationResult, calibrate, estimate_noise_floor


def test_estimate_noise_floor_silence():
    """Silence should give a very low noise floor."""
    audio = np.zeros(16000 * 5, dtype=np.float64)
    nf = estimate_noise_floor(audio, 16000)
    assert nf < -80.0


def test_estimate_noise_floor_noise():
    """Constant noise at ~-30 dB should give noise floor near -30 dB."""
    rng = np.random.default_rng(42)
    # -30 dB RMS ≈ amplitude 0.0316
    amplitude = 10.0 ** (-30.0 / 20.0)
    audio = rng.normal(0, amplitude, 16000 * 5)
    nf = estimate_noise_floor(audio, 16000)
    assert -35.0 < nf < -25.0


def test_calibrate_ordering():
    """speech > release > noise_floor."""
    rng = np.random.default_rng(123)
    amplitude = 10.0 ** (-40.0 / 20.0)
    audio = rng.normal(0, amplitude, 16000 * 5)
    result = calibrate(audio, 16000, "test")
    assert result.speech_threshold_db > result.release_threshold_db
    assert result.release_threshold_db > result.noise_floor_db
    assert result.label == "test"


def test_calibrate_empty_audio():
    """Empty audio should fall back to -96 dB noise floor."""
    audio = np.array([], dtype=np.float64)
    result = calibrate(audio, 16000, "empty")
    assert result.noise_floor_db == -96.0


def test_calibrate_clamp_noisy():
    """Very noisy mic should clamp speech threshold to -12 dB max."""
    # -5 dB noise = very loud
    amplitude = 10.0 ** (-5.0 / 20.0)
    rng = np.random.default_rng(99)
    audio = rng.normal(0, amplitude, 16000 * 5)
    result = calibrate(audio, 16000, "noisy")
    assert result.speech_threshold_db <= -12.0


def test_estimate_noise_floor_ignores_speech_at_start():
    """Speech at the beginning of audio should not inflate noise floor estimate."""
    sr = 16000
    rng = np.random.default_rng(77)

    # 2 seconds of loud speech-like tone at -10 dB
    speech_amp = 10.0 ** (-10.0 / 20.0)
    t = np.arange(sr * 2) / sr
    speech = speech_amp * np.sin(2 * np.pi * 300.0 * t)

    # 8 seconds of quiet noise at -40 dB
    noise_amp = 10.0 ** (-40.0 / 20.0)
    noise = rng.normal(0, noise_amp, sr * 8)

    audio = np.concatenate([speech, noise])
    nf = estimate_noise_floor(audio, sr)

    # Noise floor should be near -40 dB, not pulled up by the speech
    assert nf < -30.0


def test_estimate_noise_floor_uses_full_audio():
    """A quiet tail at the end of the file should be found by the estimator."""
    sr = 16000
    rng = np.random.default_rng(88)

    # 7 seconds of moderate noise at -20 dB
    moderate_amp = 10.0 ** (-20.0 / 20.0)
    moderate = rng.normal(0, moderate_amp, sr * 7)

    # 3 seconds of very quiet noise at -50 dB
    quiet_amp = 10.0 ** (-50.0 / 20.0)
    quiet = rng.normal(0, quiet_amp, sr * 3)

    audio = np.concatenate([moderate, quiet])
    nf = estimate_noise_floor(audio, sr)

    # Should find the quiet section — noise floor should be well below -20
    assert nf < -30.0


def test_estimate_noise_floor_transient_bursts_do_not_dominate():
    """Short loud bursts should not inflate the noise floor estimate."""
    sr = 16000
    rng = np.random.default_rng(55)

    # 10 seconds of quiet noise at -45 dB
    noise_amp = 10.0 ** (-45.0 / 20.0)
    audio = rng.normal(0, noise_amp, sr * 10)

    # Add 3 short 100 ms bursts at -5 dB
    burst_amp = 10.0 ** (-5.0 / 20.0)
    for offset_s in [1.0, 4.0, 7.0]:
        start = int(offset_s * sr)
        end = start + int(0.1 * sr)
        t = np.arange(end - start) / sr
        audio[start:end] = burst_amp * np.sin(2 * np.pi * 500.0 * t)

    nf = estimate_noise_floor(audio, sr)

    # Should still reflect the background noise, not the bursts
    assert nf < -35.0
