"""Tests for RMS analysis and envelope smoothing."""

import numpy as np
import pytest

from autopodcast.core.analyzer import compute_rms, smooth_envelope, analyze_speaker
from autopodcast.models.project import ProjectConfig
from tests.conftest import make_tone, make_silence


class TestComputeRMS:
    def test_known_amplitude(self):
        """A sine at 0 dB (amplitude=1.0) should have RMS ≈ -3 dB."""
        audio = make_tone(0.5, level_db=0.0)
        rms = compute_rms(audio, window_size=480, hop_size=160)
        # RMS of sine = amplitude / sqrt(2) ≈ 0.707 → -3.01 dB
        assert len(rms) > 0
        mean_rms = np.mean(rms)
        assert -4.0 < mean_rms < -2.0

    def test_silence_is_very_low(self):
        """Silence should give very low dB values."""
        audio = make_silence(0.5)
        rms = compute_rms(audio, window_size=480, hop_size=160)
        assert len(rms) > 0
        assert np.all(rms < -90.0)

    def test_level_scales(self):
        """A -20 dB tone should have RMS ~20 dB lower than 0 dB tone."""
        rms_loud = compute_rms(make_tone(0.5, 0.0), 480, 160)
        rms_quiet = compute_rms(make_tone(0.5, -20.0), 480, 160)
        diff = np.mean(rms_loud) - np.mean(rms_quiet)
        assert 18.0 < diff < 22.0

    def test_empty_audio(self):
        """Empty audio should return empty array."""
        rms = compute_rms(np.array([]), 480, 160)
        assert len(rms) == 0


class TestSmoothEnvelope:
    def test_smooths_step(self):
        """A step function should be smoothed by EMA."""
        rms = np.concatenate([np.full(50, -40.0), np.full(50, -10.0)])
        envelope = smooth_envelope(rms, kernel_size=10)
        # The transition should be gradual
        assert envelope[50] < -10.0  # not instant jump
        assert envelope[55] > -40.0  # but rising
        # End should converge
        assert abs(envelope[-1] - (-10.0)) < 1.0

    def test_empty(self):
        rms = np.array([])
        envelope = smooth_envelope(rms, 10)
        assert len(envelope) == 0

    def test_kernel_1_is_identity(self):
        """Kernel size 1 means alpha=1, so envelope = rms."""
        rms = np.random.randn(100)
        envelope = smooth_envelope(rms, kernel_size=1)
        np.testing.assert_allclose(envelope, rms)


class TestAnalyzeSpeaker:
    def test_returns_correct_frame_count(self, default_config):
        audio = make_tone(1.0, -10.0, sr=default_config.sample_rate)
        activity = analyze_speaker(audio, "test", default_config)
        expected_frames = (len(audio) - default_config.window_samples) // default_config.hop_samples + 1
        assert len(activity.frames) == expected_frames
        assert activity.speaker_label == "test"
