"""Tests for RMS analysis and envelope smoothing."""

import numpy as np
import pytest

from autopodcast.core.analyzer import compute_rms, smooth_envelope, analyze_speaker, apply_highpass
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
        envelope = smooth_envelope(rms, attack_kernel=10, decay_kernel=10)
        # The transition should be gradual
        assert envelope[50] < -10.0  # not instant jump
        assert envelope[55] > -40.0  # but rising
        # End should converge
        assert abs(envelope[-1] - (-10.0)) < 1.0

    def test_empty(self):
        rms = np.array([])
        envelope = smooth_envelope(rms, 10, 10)
        assert len(envelope) == 0

    def test_kernel_1_is_identity(self):
        """Kernel size 1 means alpha=1, so envelope = rms."""
        rms = np.random.randn(100)
        envelope = smooth_envelope(rms, attack_kernel=1, decay_kernel=1)
        np.testing.assert_allclose(envelope, rms)


class TestApplyHighpass:
    def test_disabled_is_noop(self):
        """cutoff_hz=0 should return audio unchanged."""
        audio = make_tone(0.5, -10.0)
        result = apply_highpass(audio, cutoff_hz=0.0, sample_rate=16000)
        np.testing.assert_array_equal(result, audio)

    def test_attenuates_dc(self):
        """DC offset (0 Hz) should be heavily attenuated."""
        sr = 16000
        audio = np.ones(sr, dtype=np.float64) * 0.5  # pure DC
        filtered = apply_highpass(audio, cutoff_hz=100.0, sample_rate=sr)
        # After transient, the output should be near zero
        tail_rms = np.sqrt(np.mean(filtered[sr // 2 :] ** 2))
        assert tail_rms < 0.01

    def test_passes_1khz(self):
        """A 1 kHz tone should pass through a 100 Hz highpass mostly intact."""
        sr = 16000
        audio = make_tone(0.5, level_db=0.0, sr=sr, freq=1000.0)
        filtered = apply_highpass(audio, cutoff_hz=100.0, sample_rate=sr)
        # Compare RMS of tail (skip transient)
        orig_rms = np.sqrt(np.mean(audio[sr // 4 :] ** 2))
        filt_rms = np.sqrt(np.mean(filtered[sr // 4 :] ** 2))
        # Should retain >90% of energy
        assert filt_rms / orig_rms > 0.90

    def test_analyze_speaker_uses_highpass(self, default_config):
        """analyze_speaker should apply highpass when configured."""
        sr = default_config.sample_rate
        # Low-frequency tone at 50 Hz should be attenuated by 100 Hz highpass
        audio = make_tone(1.0, level_db=-10.0, sr=sr, freq=50.0)

        default_config.highpass_hz = 0.0
        activity_no_hp = analyze_speaker(audio, "test", default_config)

        default_config.highpass_hz = 100.0
        activity_hp = analyze_speaker(audio, "test", default_config)

        rms_no_hp = np.mean([f.rms_db for f in activity_no_hp.frames])
        rms_hp = np.mean([f.rms_db for f in activity_hp.frames])
        # Highpass should significantly reduce RMS for 50 Hz tone
        assert rms_hp < rms_no_hp - 10.0


class TestAsymmetricSmoothing:
    def test_fast_attack_converges_quickly(self):
        """With small attack kernel, rising signal should converge fast."""
        rms = np.concatenate([np.full(50, -40.0), np.full(50, -10.0)])
        envelope = smooth_envelope(rms, attack_kernel=3, decay_kernel=50)
        # After just 5 frames of attack, should be close to -10
        assert envelope[55] > -15.0

    def test_slow_decay_holds_level(self):
        """With large decay kernel, falling signal should decay slowly."""
        rms = np.concatenate([np.full(50, -10.0), np.full(50, -40.0)])
        envelope = smooth_envelope(rms, attack_kernel=3, decay_kernel=50)
        # 10 frames after drop, envelope should still be well above -40
        assert envelope[60] > -25.0

    def test_attack_faster_than_decay(self):
        """Attack with small kernel should converge faster than decay with large kernel."""
        n = 100
        # Rising step
        rms_up = np.concatenate([np.full(n, -40.0), np.full(n, -10.0)])
        env_up = smooth_envelope(rms_up, attack_kernel=3, decay_kernel=50)
        # How many frames to reach within 3 dB of target (-10)?
        attack_frames = 0
        for i in range(n, 2 * n):
            if env_up[i] > -13.0:
                attack_frames = i - n
                break

        # Falling step
        rms_down = np.concatenate([np.full(n, -10.0), np.full(n, -40.0)])
        env_down = smooth_envelope(rms_down, attack_kernel=3, decay_kernel=50)
        # How many frames to reach within 3 dB of target (-40)?
        decay_frames = 0
        for i in range(n, 2 * n):
            if env_down[i] < -37.0:
                decay_frames = i - n
                break

        # If decay never reaches target in 100 frames, that's fine - it's slow
        if decay_frames == 0:
            decay_frames = n

        assert attack_frames < decay_frames


class TestAnalyzeSpeaker:
    def test_returns_correct_frame_count(self, default_config):
        audio = make_tone(1.0, -10.0, sr=default_config.sample_rate)
        activity = analyze_speaker(audio, "test", default_config)
        expected_frames = (len(audio) - default_config.window_samples) // default_config.hop_samples + 1
        assert len(activity.frames) == expected_frames
        assert activity.speaker_label == "test"

    def test_per_mic_gain_shifts_rms(self, default_config):
        """Different gain_db values should shift RMS by exactly that amount."""
        audio = make_tone(1.0, -20.0, sr=default_config.sample_rate)
        activity_0 = analyze_speaker(audio, "a", default_config, gain_db=0.0)
        activity_10 = analyze_speaker(audio, "a", default_config, gain_db=10.0)

        rms_0 = np.array([f.rms_db for f in activity_0.frames])
        rms_10 = np.array([f.rms_db for f in activity_10.frames])
        np.testing.assert_allclose(rms_10 - rms_0, 10.0, atol=1e-10)

    def test_gain_db_none_falls_back_to_config(self, default_config):
        """gain_db=None should use config.input_gain_db."""
        default_config.input_gain_db = 5.0
        audio = make_tone(1.0, -20.0, sr=default_config.sample_rate)

        activity_none = analyze_speaker(audio, "a", default_config, gain_db=None)
        activity_explicit = analyze_speaker(audio, "a", default_config, gain_db=5.0)

        rms_none = [f.rms_db for f in activity_none.frames]
        rms_explicit = [f.rms_db for f in activity_explicit.frames]
        np.testing.assert_allclose(rms_none, rms_explicit, atol=1e-10)
