"""Tests for ducking automation."""

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
        assert len(events) == 2
        # Track 0 (A's mic) at 0 dB
        track0 = [e for e in events if e.track_index == 0]
        assert track0[0].target_db == 0.0
        # Track 1 (B's mic) ducked
        track1 = [e for e in events if e.track_index == 1]
        assert track1[0].target_db == -12.0

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
