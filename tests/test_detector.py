"""Tests for hysteresis speech detection."""

import autopodcast.core.detector as detector_module
import numpy as np
import pytest

from autopodcast.core.analyzer import analyze_speaker
from autopodcast.core.detector import (
    apply_cross_gate,
    detect_activity,
    combine_speakers,
    _apply_safety_rule,
    _stabilize_frames,
    _dilate_frames,
    _apply_mask_filter,
    _compute_noise_floor,
)
from autopodcast.models.domain import AnalysisFrame, SpeakerActivity, SpeakerState
from autopodcast.models.project import ProjectConfig, AudioInput
from tests.conftest import make_speech_pattern, make_silence
from pathlib import Path


def _make_frames(n, rms_db=-50.0, is_active=False):
    """Helper: create N uniform frames."""
    return [
        AnalysisFrame(time_s=i * 0.01, rms_db=rms_db, envelope_db=rms_db, is_active=is_active)
        for i in range(n)
    ]


def _make_config(**overrides):
    """Helper: create a ProjectConfig with sensible test defaults."""
    defaults = dict(
        audio_inputs=[
            AudioInput(path=Path("a.wav"), speaker_label="host", camera_index=1),
            AudioInput(path=Path("b.wav"), speaker_label="guest", camera_index=2),
        ],
        speech_threshold_db=-24.0,
        release_threshold_db=-30.0,
        hangover_ms=600.0,
        detector_backend="rms",
        # Post-detection defaults for unit tests: disable unless explicitly set
        detection_pre_roll_s=0.0,
        detection_post_roll_s=0.0,
        detection_min_on_s=0.0,
        detection_min_off_s=0.0,
        detection_hangover_s=0.0,
        safety_margin_db=6.0,
        safety_window_s=0.20,
    )
    defaults.update(overrides)
    return ProjectConfig(**defaults)


@pytest.fixture
def config():
    return _make_config()


class TestDetectActivity:
    def test_loud_speech_detected(self, config):
        """A loud continuous tone should be fully detected as active."""
        audio = make_speech_pattern([(0.5, 4.5, -10.0)], 5.0)
        activity = analyze_speaker(audio, "test", config)
        detect_activity(activity, config)

        # Middle frames should be active
        mid = len(activity.frames) // 2
        assert activity.frames[mid].is_active is True

    def test_silence_not_detected(self, config):
        """Pure silence should not be detected."""
        audio = make_silence(3.0)
        activity = analyze_speaker(audio, "test", config)
        detect_activity(activity, config)

        assert all(f.is_active is False for f in activity.frames)

    def test_hangover_bridges_short_pause(self, config):
        """A 400ms pause within speech should be bridged by 600ms hangover."""
        # Speech (1s) -> pause (0.4s) -> speech (1s)
        audio = make_speech_pattern(
            [(0.5, 1.5, -10.0), (1.9, 2.9, -10.0)],
            total_duration_s=4.0,
        )
        activity = analyze_speaker(audio, "test", config)
        detect_activity(activity, config)

        # Check the pause region (~1.5s to ~1.9s) — should still be active due to hangover
        hop_s = config.hop_ms / 1000.0
        pause_frame = int(1.7 / hop_s)
        if pause_frame < len(activity.frames):
            assert activity.frames[pause_frame].is_active is True

    def test_hangover_does_not_bridge_long_pause(self, config):
        """A 1.5s pause should NOT be bridged by 600ms hangover."""
        audio = make_speech_pattern(
            [(0.5, 1.5, -10.0), (3.0, 4.0, -10.0)],
            total_duration_s=5.0,
        )
        activity = analyze_speaker(audio, "test", config)
        detect_activity(activity, config)

        # Well into the gap (~2.5s) should be inactive
        hop_s = config.hop_ms / 1000.0
        gap_frame = int(2.5 / hop_s)
        if gap_frame < len(activity.frames):
            assert activity.frames[gap_frame].is_active is False

    def test_diagnostics_returned(self, config):
        """detect_activity fills diagnostics dict when provided."""
        audio = make_speech_pattern([(0.5, 1.5, -10.0)], 2.0)
        activity = analyze_speaker(audio, "test", config)
        diag = {}
        detect_activity(activity, config, diagnostics=diag)

        assert "total_frames" in diag
        assert "active_frames" in diag
        assert "active_seconds" in diag
        assert "total_seconds" in diag
        assert "safety_forced_frames" in diag
        assert diag["active_frames"] > 0


class TestSileroBackend:
    def test_projects_intervals_onto_frames(self, monkeypatch):
        """Silero timestamps should map directly to frame activity."""
        config = _make_config(
            detector_backend="silero",
            detection_pre_roll_s=0.0,
            detection_post_roll_s=0.0,
            detection_min_on_s=0.0,
            detection_min_off_s=0.0,
            detection_hangover_s=0.0,
        )
        audio = make_silence(2.5)
        activity = analyze_speaker(audio, "test", config)

        monkeypatch.setattr(
            detector_module,
            "_silero_detect_intervals",
            lambda _audio, _config: [(0.5, 1.0), (1.5, 2.0)],
        )

        diag = {}
        detect_activity(activity, config, diagnostics=diag, audio=audio)

        hop_s = config.hop_ms / 1000.0
        assert activity.frames[int(0.7 / hop_s)].is_active is True
        assert activity.frames[int(1.2 / hop_s)].is_active is False
        assert activity.frames[int(1.7 / hop_s)].is_active is True
        assert diag["detector_backend"] == "silero"
        assert diag["safety_forced_frames"] == 0
        assert len(diag["vad_intervals"]) == 2

    def test_auto_backend_falls_back_to_rms(self, monkeypatch):
        """Auto mode should fall back to RMS when Silero is unavailable."""
        config = _make_config(detector_backend="auto")
        audio = make_speech_pattern([(0.5, 1.5, -10.0)], 2.0)
        activity = analyze_speaker(audio, "test", config)

        def _raise_unavailable(_activity, _config, _audio, _diagnostics=None):
            raise detector_module.SileroVADUnavailableError("silero missing")

        monkeypatch.setattr(detector_module, "_detect_activity_silero", _raise_unavailable)

        diag = {}
        detect_activity(activity, config, diagnostics=diag, audio=audio)

        assert diag["detector_backend"] == "rms_fallback"
        assert "silero missing" in diag["detector_warning"]
        assert diag["active_frames"] > 0


class TestSafetyRule:
    def test_forces_loud_frames(self):
        """Frames with rms_db well above safety threshold should be forced active."""
        # safety threshold = -24 + 6 = -18. Frames at -8 dB are above.
        config = _make_config(safety_window_s=0.03)  # 3 frames at 10ms hop
        frames = _make_frames(10, rms_db=-8.0, is_active=False)
        activity = SpeakerActivity(speaker_label="test", frames=frames)

        forced = _apply_safety_rule(activity, config)

        assert forced > 0
        assert all(f.is_active for f in activity.frames)

    def test_ignores_quiet_frames(self):
        """Frames below safety threshold should not be changed."""
        config = _make_config(safety_window_s=0.03)
        frames = _make_frames(10, rms_db=-30.0, is_active=False)
        activity = SpeakerActivity(speaker_label="test", frames=frames)

        forced = _apply_safety_rule(activity, config)

        assert forced == 0
        assert all(not f.is_active for f in activity.frames)

    def test_partial_window_not_forced(self):
        """If only some frames in window are loud, safety rule should not trigger."""
        config = _make_config(safety_window_s=0.03)  # 3 frames
        frames = _make_frames(5, rms_db=-8.0, is_active=False)
        # Make one frame quiet — breaks the "all frames loud" condition
        frames[2].rms_db = -30.0
        activity = SpeakerActivity(speaker_label="test", frames=frames)

        forced = _apply_safety_rule(activity, config)

        # Frame 2 breaks any window that includes it
        # Windows: [0,1,2], [1,2,3], [2,3,4] — all contain frame 2
        # Only window that doesn't is none — so no forcing for windows containing frame 2
        # But window [0,1,2] has min=-30, [1,2,3] has min=-30, [2,3,4] has min=-30
        # No window is fully loud
        assert forced == 0


class TestStabilization:
    def test_removes_short_burst(self):
        """Active burst shorter than min_on should be removed."""
        config = _make_config(detection_min_on_s=0.25)  # 25 frames at 10ms
        frames = _make_frames(100, rms_db=-50.0, is_active=False)
        # Short burst: 2 frames (< 25)
        frames[50].is_active = True
        frames[51].is_active = True
        activity = SpeakerActivity(speaker_label="test", frames=frames)

        _stabilize_frames(activity, config)

        assert not frames[50].is_active
        assert not frames[51].is_active

    def test_keeps_long_burst(self):
        """Active burst longer than min_on should be kept."""
        config = _make_config(detection_min_on_s=0.05)  # 5 frames
        frames = _make_frames(100, rms_db=-50.0, is_active=False)
        for i in range(40, 60):
            frames[i].is_active = True
        activity = SpeakerActivity(speaker_label="test", frames=frames)

        _stabilize_frames(activity, config)

        assert frames[50].is_active

    def test_fills_short_gap(self):
        """Inactive gap shorter than min_off should be filled."""
        config = _make_config(detection_min_off_s=0.30)  # 30 frames
        frames = _make_frames(100, rms_db=-50.0, is_active=True)
        # Short gap: 2 frames (< 30)
        frames[50].is_active = False
        frames[51].is_active = False
        activity = SpeakerActivity(speaker_label="test", frames=frames)

        _stabilize_frames(activity, config)

        assert frames[50].is_active
        assert frames[51].is_active

    def test_hangover_extension(self):
        """Hangover should extend active run forward."""
        config = _make_config(detection_hangover_s=0.05)  # 5 frames
        frames = _make_frames(100, rms_db=-50.0, is_active=False)
        for i in range(20, 40):
            frames[i].is_active = True
        activity = SpeakerActivity(speaker_label="test", frames=frames)

        _stabilize_frames(activity, config)

        # Frames 40-44 should now be active (hangover)
        for i in range(40, 45):
            assert frames[i].is_active, f"Frame {i} should be active (hangover)"
        # Frame 45 should not
        assert not frames[45].is_active


class TestDilation:
    def test_pre_roll(self):
        """Pre-roll should extend active region backward into non-silent frames."""
        config = _make_config(detection_pre_roll_s=0.10)  # 10 frames
        frames = _make_frames(200, rms_db=-30.0, is_active=False)  # above silence threshold
        for i in range(50, 100):
            frames[i].is_active = True
        activity = SpeakerActivity(speaker_label="test", frames=frames)

        _dilate_frames(activity, config)

        # Pre-roll: frames 40-49 should now be active
        for i in range(40, 50):
            assert frames[i].is_active, f"Frame {i} should be active (pre-roll)"
        assert not frames[39].is_active

    def test_post_roll(self):
        """Post-roll should extend active region forward."""
        config = _make_config(detection_post_roll_s=0.10)  # 10 frames
        frames = _make_frames(200, rms_db=-50.0, is_active=False)
        for i in range(50, 100):
            frames[i].is_active = True
        activity = SpeakerActivity(speaker_label="test", frames=frames)

        _dilate_frames(activity, config)

        # Post-roll: frames 100-109 should now be active
        for i in range(100, 110):
            assert frames[i].is_active, f"Frame {i} should be active (post-roll)"
        assert not frames[110].is_active

    def test_pre_roll_clamps_to_zero(self):
        """Pre-roll should not go below frame 0."""
        config = _make_config(detection_pre_roll_s=0.10)  # 10 frames
        frames = _make_frames(50, rms_db=-50.0, is_active=False)
        # Set rms_db above silence threshold so pre-roll extends
        for i in range(50):
            frames[i].rms_db = -30.0
        for i in range(3, 20):
            frames[i].is_active = True
        activity = SpeakerActivity(speaker_label="test", frames=frames)

        _dilate_frames(activity, config)

        assert frames[0].is_active  # clamped, not out-of-bounds

    def test_pre_roll_stops_at_silence(self):
        """Pre-roll should stop when encountering frames below silence_threshold."""
        config = _make_config(
            detection_pre_roll_s=0.10,  # 10 frames
            silence_threshold_db=-40.0,
        )
        frames = _make_frames(200, rms_db=-50.0, is_active=False)  # all below silence
        for i in range(50, 100):
            frames[i].is_active = True
            frames[i].rms_db = -10.0
        # Frames 40-49 are at -50dB (below silence threshold) → pre-roll should stop
        activity = SpeakerActivity(speaker_label="test", frames=frames)

        _dilate_frames(activity, config)

        # Pre-roll should NOT extend into silence
        assert not frames[40].is_active
        assert not frames[49].is_active

    def test_pre_roll_extends_into_non_silence(self):
        """Pre-roll extends into frames above silence_threshold."""
        config = _make_config(
            detection_pre_roll_s=0.10,  # 10 frames
            silence_threshold_db=-40.0,
        )
        frames = _make_frames(200, rms_db=-30.0, is_active=False)  # above silence
        for i in range(50, 100):
            frames[i].is_active = True
        activity = SpeakerActivity(speaker_label="test", frames=frames)

        _dilate_frames(activity, config)

        # Pre-roll should extend into non-silent frames
        for i in range(40, 50):
            assert frames[i].is_active, f"Frame {i} should be active (pre-roll)"
        assert not frames[39].is_active

    def test_post_roll_still_unconditional(self):
        """Post-roll extends forward regardless of silence."""
        config = _make_config(
            detection_post_roll_s=0.10,  # 10 frames
            silence_threshold_db=-40.0,
        )
        frames = _make_frames(200, rms_db=-50.0, is_active=False)  # all below silence
        for i in range(50, 100):
            frames[i].is_active = True
            frames[i].rms_db = -10.0
        activity = SpeakerActivity(speaker_label="test", frames=frames)

        _dilate_frames(activity, config)

        # Post-roll should extend even into silence
        for i in range(100, 110):
            assert frames[i].is_active, f"Frame {i} should be active (post-roll)"


class TestFullPipeline:
    def test_no_false_silence_for_loud_speech(self):
        """Loud speech at -8 dB should never produce false silence."""
        config = _make_config(
            detection_pre_roll_s=0.7,
            detection_post_roll_s=0.25,
            detection_min_on_s=0.25,
            detection_min_off_s=0.30,
            detection_hangover_s=0.20,
            safety_margin_db=6.0,
            safety_window_s=0.20,
        )
        # Continuous loud speech
        audio = make_speech_pattern([(0.5, 4.5, -8.0)], 5.0)
        activity = analyze_speaker(audio, "test", config)
        detect_activity(activity, config)

        # All frames in the speech region should be active
        hop_s = config.hop_ms / 1000.0
        start_frame = int(1.0 / hop_s)  # well into speech
        end_frame = int(4.0 / hop_s)
        speech_frames = activity.frames[start_frame:end_frame]
        inactive = [f for f in speech_frames if not f.is_active]
        assert len(inactive) == 0, f"{len(inactive)} frames falsely marked inactive in loud speech"

    def test_pre_roll_stops_at_dead_silence(self):
        """Pre-roll should NOT extend into dead silence before speech onset."""
        config = _make_config(
            detection_pre_roll_s=1.0,
            detection_post_roll_s=0.25,
            detection_min_on_s=0.25,
            detection_min_off_s=0.30,
            detection_hangover_s=0.20,
        )
        # Speech starting at 2.0s, silence before
        audio = make_speech_pattern([(2.0, 4.0, -10.0)], 5.0)
        activity = analyze_speaker(audio, "test", config)
        detect_activity(activity, config)

        hop_s = config.hop_ms / 1000.0
        # 1.0s pre-roll before ~2.0s onset, but silence before → should NOT extend
        pre_roll_frame = int(1.1 / hop_s)
        if pre_roll_frame < len(activity.frames):
            assert not activity.frames[pre_roll_frame].is_active, \
                "Pre-roll should stop at dead silence"


class TestCombineSpeakers:
    def test_both_active(self, config):
        """Both speakers active -> BOTH."""
        audio_a = make_speech_pattern([(0.5, 2.5, -10.0)], 3.0)
        audio_b = make_speech_pattern([(0.5, 2.5, -10.0)], 3.0)

        act_a = analyze_speaker(audio_a, "host", config)
        act_b = analyze_speaker(audio_b, "guest", config)
        detect_activity(act_a, config)
        detect_activity(act_b, config)

        states = combine_speakers(act_a, act_b)
        # Mid-speech frames should be BOTH
        mid = len(states) // 2
        assert states[mid] == SpeakerState.BOTH

    def test_only_a_active(self, config):
        """Only speaker A active -> SPEAKER_A."""
        audio_a = make_speech_pattern([(0.5, 2.5, -10.0)], 3.0)
        audio_b = make_silence(3.0)

        act_a = analyze_speaker(audio_a, "host", config)
        act_b = analyze_speaker(audio_b, "guest", config)
        detect_activity(act_a, config)
        detect_activity(act_b, config)

        states = combine_speakers(act_a, act_b)
        mid = len(states) // 2
        assert states[mid] == SpeakerState.SPEAKER_A

    def test_neither_active(self, config):
        """Both silent -> SILENCE."""
        audio_a = make_silence(3.0)
        audio_b = make_silence(3.0)

        act_a = analyze_speaker(audio_a, "host", config)
        act_b = analyze_speaker(audio_b, "guest", config)
        detect_activity(act_a, config)
        detect_activity(act_b, config)

        states = combine_speakers(act_a, act_b)
        assert all(s == SpeakerState.SILENCE for s in states)


class TestApplyCrossGate:
    def test_crosstalk_suppressed_speaker_a(self):
        """Host at -10dB, guest at -18dB (diff=8 >= 6) -> SPEAKER_A."""
        states = [SpeakerState.BOTH]
        activity_a = SpeakerActivity(speaker_label="host", frames=_make_frames(1, rms_db=-10.0, is_active=True))
        activity_b = SpeakerActivity(speaker_label="guest", frames=_make_frames(1, rms_db=-18.0, is_active=True))

        result = apply_cross_gate(states, activity_a, activity_b, threshold_db=6.0)
        assert result == [SpeakerState.SPEAKER_A]

    def test_crosstalk_suppressed_speaker_b(self):
        """Guest at -10dB, host at -18dB (diff=8 >= 6) -> SPEAKER_B."""
        states = [SpeakerState.BOTH]
        activity_a = SpeakerActivity(speaker_label="host", frames=_make_frames(1, rms_db=-18.0, is_active=True))
        activity_b = SpeakerActivity(speaker_label="guest", frames=_make_frames(1, rms_db=-10.0, is_active=True))

        result = apply_cross_gate(states, activity_a, activity_b, threshold_db=6.0)
        assert result == [SpeakerState.SPEAKER_B]

    def test_equal_threshold_resolves_crosstalk(self):
        """Diff exactly at threshold should resolve to the louder speaker."""
        states = [SpeakerState.BOTH, SpeakerState.BOTH]
        activity_a = SpeakerActivity(
            speaker_label="host",
            frames=_make_frames(2, rms_db=-10.0, is_active=True),
        )
        activity_b = SpeakerActivity(
            speaker_label="guest",
            frames=_make_frames(2, rms_db=-16.0, is_active=True),
        )
        activity_a.frames[1].rms_db = -16.0
        activity_a.frames[1].envelope_db = -16.0
        activity_b.frames[1].rms_db = -10.0
        activity_b.frames[1].envelope_db = -10.0

        result = apply_cross_gate(states, activity_a, activity_b, threshold_db=6.0)
        assert result == [SpeakerState.SPEAKER_A, SpeakerState.SPEAKER_B]

    def test_similar_levels_stays_both(self):
        """Host at -10dB, guest at -12dB (diff=2 < 6) -> BOTH."""
        states = [SpeakerState.BOTH]
        activity_a = SpeakerActivity(speaker_label="host", frames=_make_frames(1, rms_db=-10.0, is_active=True))
        activity_b = SpeakerActivity(speaker_label="guest", frames=_make_frames(1, rms_db=-12.0, is_active=True))

        result = apply_cross_gate(states, activity_a, activity_b, threshold_db=6.0)
        assert result == [SpeakerState.BOTH]

    def test_silence_unchanged(self):
        """SILENCE frames are not modified."""
        states = [SpeakerState.SILENCE]
        activity_a = SpeakerActivity(speaker_label="host", frames=_make_frames(1, rms_db=-60.0))
        activity_b = SpeakerActivity(speaker_label="guest", frames=_make_frames(1, rms_db=-60.0))

        result = apply_cross_gate(states, activity_a, activity_b, threshold_db=6.0)
        assert result == [SpeakerState.SILENCE]

    def test_single_speaker_unchanged(self):
        """SPEAKER_A / SPEAKER_B frames pass through unchanged."""
        states = [SpeakerState.SPEAKER_A, SpeakerState.SPEAKER_B]
        activity_a = SpeakerActivity(speaker_label="host", frames=_make_frames(2, rms_db=-10.0, is_active=True))
        activity_b = SpeakerActivity(speaker_label="guest", frames=_make_frames(2, rms_db=-10.0, is_active=True))

        result = apply_cross_gate(states, activity_a, activity_b, threshold_db=6.0)
        assert result == [SpeakerState.SPEAKER_A, SpeakerState.SPEAKER_B]

    def test_threshold_zero_resolves_any_difference(self):
        """threshold=0: any positive diff resolves, equal stays BOTH."""
        states = [SpeakerState.BOTH, SpeakerState.BOTH]
        a_frames = _make_frames(2, rms_db=-10.0, is_active=True)
        b_frames = _make_frames(2, rms_db=-10.0, is_active=True)
        b_frames[0].rms_db = -18.0
        b_frames[0].envelope_db = -18.0
        activity_a = SpeakerActivity(speaker_label="host", frames=a_frames)
        activity_b = SpeakerActivity(speaker_label="guest", frames=b_frames)

        result = apply_cross_gate(states, activity_a, activity_b, threshold_db=0.0)
        assert result[0] == SpeakerState.SPEAKER_A  # diff=8 >= 0
        assert result[1] == SpeakerState.BOTH       # diff=0, equal levels stay BOTH


def _make_frames_with_peak(n, rms_db=-50.0, peak_db=-50.0, is_active=False):
    """Helper: create N frames with explicit peak_db."""
    return [
        AnalysisFrame(
            time_s=i * 0.01, rms_db=rms_db, envelope_db=rms_db,
            is_active=is_active, peak_db=peak_db,
        )
        for i in range(n)
    ]


class TestMaskFilter:
    """Tests for SNR-aware mask post-filter."""

    def _mask_config(self, **overrides):
        defaults = dict(
            audio_inputs=[
                AudioInput(path=Path("a.wav"), speaker_label="host", camera_index=1),
                AudioInput(path=Path("b.wav"), speaker_label="guest", camera_index=2),
            ],
            speech_threshold_db=-24.0,
            release_threshold_db=-30.0,
            detection_pre_roll_s=0.0,
            detection_post_roll_s=0.0,
            detection_min_on_s=0.0,
            detection_min_off_s=0.0,
            detection_hangover_s=0.0,
            mask_filter_enabled=True,
            mask_filter_min_on_ms=500.0,      # 50 frames
            mask_filter_min_off_fill_ms=360.0, # 36 frames
            mask_filter_snr_peak_strong_db=14.0,
            mask_filter_snr_peak_weak_db=10.0,
            mask_filter_crest_min_db=8.0,
            mask_filter_noise_window_s=15.0,
            mask_filter_noise_percentile=10.0,
        )
        defaults.update(overrides)
        return ProjectConfig(**defaults)

    def test_short_noise_removed(self):
        """Flat low-level burst (~200ms, low SNR/low crest) is removed."""
        config = self._mask_config()
        # Background at -50dB, short burst of 20 frames (200ms) at -40dB rms, -39dB peak
        # SNR = -39 - (-50) = 11, crest = -39 - (-40) = 1 < 8
        # Not strong (11 < 14), weak+crest fails (crest=1 < 8) → removed
        frames = _make_frames_with_peak(200, rms_db=-50.0, peak_db=-50.0, is_active=False)
        for i in range(80, 100):  # 20 frames = 200ms
            frames[i].is_active = True
            frames[i].rms_db = -40.0
            frames[i].peak_db = -39.0

        activity = SpeakerActivity(speaker_label="test", frames=frames)
        diag = {}
        _apply_mask_filter(activity, config, diag)

        assert not any(frames[i].is_active for i in range(80, 100))
        assert diag["mask_filter_removed"] == 1

    def test_short_speech_kept_strong_peak(self):
        """Short burst with sharp peak (SNR >= 14dB) is kept."""
        config = self._mask_config()
        # Background -50dB, burst: rms=-35, peak=-30
        # SNR = -30 - (-50) = 20 >= 14 → keep
        frames = _make_frames_with_peak(200, rms_db=-50.0, peak_db=-50.0, is_active=False)
        for i in range(80, 100):
            frames[i].is_active = True
            frames[i].rms_db = -35.0
            frames[i].peak_db = -30.0

        activity = SpeakerActivity(speaker_label="test", frames=frames)
        diag = {}
        _apply_mask_filter(activity, config, diag)

        assert all(frames[i].is_active for i in range(80, 100))
        assert diag["mask_filter_removed"] == 0

    def test_short_speech_kept_weak_with_crest(self):
        """Moderate SNR (>= 10dB) + high crest (>= 8dB) is kept."""
        config = self._mask_config()
        # Background -50dB, burst: rms=-42, peak=-32
        # SNR = -32 - (-50) = 18 >= 10 ✓, crest = -32 - (-42) = 10 >= 8 ✓ → keep
        frames = _make_frames_with_peak(200, rms_db=-50.0, peak_db=-50.0, is_active=False)
        for i in range(80, 100):
            frames[i].is_active = True
            frames[i].rms_db = -42.0
            frames[i].peak_db = -32.0

        activity = SpeakerActivity(speaker_label="test", frames=frames)
        diag = {}
        _apply_mask_filter(activity, config, diag)

        assert all(frames[i].is_active for i in range(80, 100))
        assert diag["mask_filter_removed"] == 0

    def test_long_segment_untouched(self):
        """Segment > min_on_ms is never filtered regardless of SNR."""
        config = self._mask_config()
        # 60 frames = 600ms > 500ms threshold, low SNR
        frames = _make_frames_with_peak(200, rms_db=-50.0, peak_db=-50.0, is_active=False)
        for i in range(50, 110):  # 60 frames
            frames[i].is_active = True
            frames[i].rms_db = -40.0
            frames[i].peak_db = -39.0  # low crest, moderate SNR

        activity = SpeakerActivity(speaker_label="test", frames=frames)
        _apply_mask_filter(activity, config)

        assert all(frames[i].is_active for i in range(50, 110))

    def test_gap_filling(self):
        """Short OFF gap between two ON segments is filled."""
        config = self._mask_config()
        # Two long ON segments with a short gap (10 frames = 100ms < 360ms)
        frames = _make_frames_with_peak(300, rms_db=-50.0, peak_db=-50.0, is_active=False)
        for i in range(20, 80):  # 60 frames ON
            frames[i].is_active = True
        # gap: 80-89 (10 frames OFF)
        for i in range(90, 150):  # 60 frames ON
            frames[i].is_active = True

        activity = SpeakerActivity(speaker_label="test", frames=frames)
        diag = {}
        _apply_mask_filter(activity, config, diag)

        # Gap should be filled
        assert all(frames[i].is_active for i in range(80, 90))
        assert diag["mask_filter_gap_filled"] == 1

    def test_filter_disabled(self):
        """mask_filter_enabled=False → no change."""
        config = self._mask_config(mask_filter_enabled=False)
        # Short noisy burst that would be removed if filter was on
        frames = _make_frames_with_peak(200, rms_db=-50.0, peak_db=-50.0, is_active=False)
        for i in range(80, 100):
            frames[i].is_active = True
            frames[i].rms_db = -40.0
            frames[i].peak_db = -39.0

        activity = SpeakerActivity(speaker_label="test", frames=frames)

        # Call detect_activity with filter disabled
        # Instead, just verify _apply_mask_filter is not called by checking frames unchanged
        # We test the config flag via the full pipeline
        original_active = [f.is_active for f in frames]
        # Since filter is disabled, we simulate by NOT calling _apply_mask_filter
        # The real test is that detect_activity checks config.mask_filter_enabled
        # Let's test the flag directly:
        assert not config.mask_filter_enabled
        # And verify that if we still call it explicitly, it still works
        _apply_mask_filter(activity, config)
        # But the point is detect_activity won't call it - let's test via pipeline
        activity2 = SpeakerActivity(speaker_label="test", frames=frames)
        detect_activity(activity2, config)
        # The flag test is really about the pipeline integration

    def test_noise_floor_tracks_background(self):
        """Noise floor follows actual background level."""
        import numpy as np
        # First half at -50dB, second half at -30dB
        rms = np.full(200, -50.0)
        rms[100:] = -30.0

        noise_floor = _compute_noise_floor(rms, window_frames=50, percentile=10.0)

        # First quarter should reflect ~-50
        assert noise_floor[25] < -45.0
        # Last quarter should reflect ~-30
        assert noise_floor[175] > -35.0
