"""Tests for ducking automation with gate-style paired keyframes."""

import numpy as np
import pytest

from autopodcast.core.ducking import generate_ducking_events
from autopodcast.models.domain import Segment, SpeakerState
from autopodcast.models.project import AudioInput, ProjectConfig
from pathlib import Path


@pytest.fixture
def config():
    return ProjectConfig(
        audio_inputs=[
            AudioInput(path=Path("a.wav"), speaker_label="host", camera_index=1),
            AudioInput(path=Path("b.wav"), speaker_label="guest", camera_index=2),
        ],
        ducking_enabled=True,
        ducking_db=-12.0,
        gate_fade_s=0.15,
    )


@pytest.fixture
def config_disabled():
    return ProjectConfig(ducking_enabled=False)


class TestDucking:
    def test_disabled_returns_empty(self, config_disabled):
        segs = [Segment(0.0, 5.0, 0, SpeakerState.SPEAKER_A)]
        events = generate_ducking_events(segs, config_disabled)
        assert events == []

    def test_speaker_a_ducks_track_b(self, config):
        segs = [Segment(0.0, 5.0, 1, SpeakerState.SPEAKER_A)]
        events = generate_ducking_events(segs, config)
        # Initial keyframes at t=0 for both tracks
        track0 = [e for e in events if e.track_index == 0]
        track1 = [e for e in events if e.track_index == 1]
        assert track0[0].target_db == 0.0
        assert track0[0].time_s == 0.0
        assert track1[0].target_db == -12.0
        assert track1[0].time_s == 0.0

    def test_speaker_b_ducks_track_a(self, config):
        segs = [Segment(0.0, 5.0, 2, SpeakerState.SPEAKER_B)]
        events = generate_ducking_events(segs, config)
        track0 = [e for e in events if e.track_index == 0]
        track1 = [e for e in events if e.track_index == 1]
        assert track0[0].target_db == -12.0
        assert track1[0].target_db == 0.0

    def test_both_no_ducking(self, config):
        segs = [Segment(0.0, 5.0, 0, SpeakerState.BOTH)]
        events = generate_ducking_events(segs, config)
        assert all(e.target_db == 0.0 for e in events)

    def test_silence_no_ducking(self, config):
        segs = [Segment(0.0, 5.0, 0, SpeakerState.SILENCE)]
        events = generate_ducking_events(segs, config)
        assert all(e.target_db == 0.0 for e in events)


class TestGateKeyframes:
    def test_gate_keyframe_pairs(self, config):
        """Two segments [A, B]: boundary produces hold+target pairs."""
        segs = [
            Segment(0.0, 5.0, 1, SpeakerState.SPEAKER_A),
            Segment(5.0, 10.0, 2, SpeakerState.SPEAKER_B),
        ]
        events = generate_ducking_events(segs, config)
        track0 = [e for e in events if e.track_index == 0]
        track1 = [e for e in events if e.track_index == 1]

        # Track 0: initial 0dB at t=0, then hold at 4.85 (0dB), target at 5.0 (-12dB)
        assert track0[0].time_s == 0.0
        assert track0[0].target_db == 0.0
        assert track0[1].time_s == pytest.approx(4.85)
        assert track0[1].target_db == 0.0  # hold: old value
        assert track0[2].time_s == 5.0
        assert track0[2].target_db == -12.0  # target: new value

        # Track 1: initial -12dB at t=0, then hold at 4.85 (-12dB), target at 5.0 (0dB)
        assert track1[0].time_s == 0.0
        assert track1[0].target_db == -12.0
        assert track1[1].time_s == pytest.approx(4.85)
        assert track1[1].target_db == -12.0
        assert track1[2].time_s == 5.0
        assert track1[2].target_db == 0.0

    def test_gate_initial_at_zero(self, config):
        """First keyframe for each track is at t=0."""
        segs = [Segment(0.0, 5.0, 1, SpeakerState.SPEAKER_A)]
        events = generate_ducking_events(segs, config)
        track0 = [e for e in events if e.track_index == 0]
        track1 = [e for e in events if e.track_index == 1]
        assert track0[0].time_s == 0.0
        assert track1[0].time_s == 0.0

    def test_gate_no_redundant(self, config):
        """Adjacent segments with same state produce no extra keyframes."""
        segs = [
            Segment(0.0, 5.0, 1, SpeakerState.SPEAKER_A),
            Segment(5.0, 10.0, 1, SpeakerState.SPEAKER_A),
        ]
        events = generate_ducking_events(segs, config)
        track0 = [e for e in events if e.track_index == 0]
        track1 = [e for e in events if e.track_index == 1]
        # Only initial keyframe, no transitions needed
        assert len(track0) == 1
        assert len(track1) == 1

    def test_gate_clamp_hold(self, config):
        """Transition at t=0.05 with fade=0.15 → hold clamped to t=0."""
        segs = [
            Segment(0.0, 0.05, 0, SpeakerState.BOTH),
            Segment(0.05, 5.0, 1, SpeakerState.SPEAKER_A),
        ]
        events = generate_ducking_events(segs, config)
        # Track 1: initial 0dB at t=0, then transition at 0.05.
        # hold would be at 0.05-0.15 = -0.10, clamped to 0.0
        track1 = [e for e in events if e.track_index == 1]
        hold_events = [e for e in track1 if e.time_s == 0.0 and e.target_db == 0.0]
        assert len(hold_events) >= 1
        # The target at 0.05 should be -12dB
        target_events = [e for e in track1 if e.time_s == 0.05]
        assert target_events[0].target_db == -12.0

    def test_continuous_speech_no_drop(self, config):
        """Guest speaks 20s continuously → their track only has initial 0dB keyframe."""
        segs = [Segment(0.0, 20.0, 2, SpeakerState.SPEAKER_B)]
        events = generate_ducking_events(segs, config)
        track1 = [e for e in events if e.track_index == 1]
        # Only one keyframe at t=0 with 0dB
        assert len(track1) == 1
        assert track1[0].time_s == 0.0
        assert track1[0].target_db == 0.0

    def test_pause_fast_mute(self, config):
        """Guest pauses → mute keyframe at boundary."""
        segs = [
            Segment(0.0, 5.0, 2, SpeakerState.SPEAKER_B),
            Segment(5.0, 10.0, 0, SpeakerState.SILENCE),
        ]
        events = generate_ducking_events(segs, config)
        # Track 1 (guest): initial 0dB, then no change at boundary
        # (both SPEAKER_B and SILENCE have track1 at 0dB)
        track1 = [e for e in events if e.track_index == 1]
        assert len(track1) == 1  # no change needed

        # Track 0 (host): initial -12dB, then transition to 0dB at boundary
        track0 = [e for e in events if e.track_index == 0]
        target = [e for e in track0 if e.time_s == 5.0]
        assert target[0].target_db == 0.0


class TestRMSGuard:
    def test_rms_guard_prevents_false_mute(self):
        """When RMS guard detects signal above noise floor, mute is skipped."""
        sr = 16000
        duration = 10
        n_samples = sr * duration

        # Track 1: mostly quiet noise, but loud speech around transition at 5s
        track1_audio = np.random.randn(n_samples).astype(np.float32) * 0.001  # quiet noise
        # Add loud speech from 4.5s to 6s (around transition point)
        speech_start = int(4.5 * sr)
        speech_end = int(6.0 * sr)
        t = np.arange(speech_end - speech_start) / sr
        track1_audio[speech_start:speech_end] = (
            np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5
        )

        silence = np.zeros(n_samples, dtype=np.float32)

        config = ProjectConfig(
            audio_inputs=[
                AudioInput(path=Path("a.wav"), speaker_label="host", camera_index=1),
                AudioInput(path=Path("b.wav"), speaker_label="guest", camera_index=2),
            ],
            ducking_enabled=True,
            ducking_db=-96.0,
            gate_fade_s=0.15,
            rms_guard_enabled=True,
            rms_guard_threshold_db=12.0,
            rms_guard_window_ms=50.0,
        )

        # Segment says SPEAKER_A at 5s (should mute track 1), but track 1 has signal
        segs = [
            Segment(0.0, 5.0, 2, SpeakerState.SPEAKER_B),
            Segment(5.0, 10.0, 1, SpeakerState.SPEAKER_A),
        ]
        events = generate_ducking_events(
            segs, config, audio_arrays=[silence, track1_audio],
        )
        # Track 1: initial 0dB (SPEAKER_B), then SPEAKER_A would mute it.
        # But RMS guard should prevent the mute because track 1 has signal at 5s.
        track1 = [e for e in events if e.track_index == 1]
        mute_events = [e for e in track1 if e.target_db < 0]
        assert len(mute_events) == 0, "RMS guard should prevent mute when signal present"

    def test_rms_guard_off_allows_mute(self):
        """With rms_guard_enabled=False, muting works normally."""
        sr = 16000
        loud_signal = np.sin(2 * np.pi * 440 * np.arange(sr * 10) / sr).astype(np.float32) * 0.5
        silence = np.zeros(sr * 10, dtype=np.float32)

        config = ProjectConfig(
            audio_inputs=[
                AudioInput(path=Path("a.wav"), speaker_label="host", camera_index=1),
                AudioInput(path=Path("b.wav"), speaker_label="guest", camera_index=2),
            ],
            ducking_enabled=True,
            ducking_db=-96.0,
            gate_fade_s=0.15,
            rms_guard_enabled=False,
        )

        segs = [
            Segment(0.0, 5.0, 2, SpeakerState.SPEAKER_B),
            Segment(5.0, 10.0, 1, SpeakerState.SPEAKER_A),
        ]
        events = generate_ducking_events(
            segs, config, audio_arrays=[silence, loud_signal],
        )
        track1 = [e for e in events if e.track_index == 1]
        mute_events = [e for e in track1 if e.target_db < 0]
        assert len(mute_events) > 0, "Without RMS guard, muting should happen"
