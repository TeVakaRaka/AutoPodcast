"""Tests for hysteresis speech detection."""

import numpy as np
import pytest

from autopodcast.core.analyzer import analyze_speaker
from autopodcast.core.detector import detect_activity, combine_speakers
from autopodcast.models.domain import SpeakerState
from autopodcast.models.project import ProjectConfig, AudioInput
from tests.conftest import make_speech_pattern, make_silence
from pathlib import Path


@pytest.fixture
def config():
    return ProjectConfig(
        audio_inputs=[
            AudioInput(path=Path("a.wav"), speaker_label="host", camera_index=1),
            AudioInput(path=Path("b.wav"), speaker_label="guest", camera_index=2),
        ],
        speech_threshold_db=-28.0,
        release_threshold_db=-33.0,
        hangover_ms=600.0,
    )


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
