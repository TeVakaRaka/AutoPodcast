"""Tests for camera assignment."""

import pytest

from autopodcast.core.switcher import assign_cameras
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
        default_camera=0,
        both_speaking_camera=0,
    )


class TestAssignCameras:
    def test_speaker_a_gets_camera_1(self, config):
        segs = [Segment(0.0, 5.0, 0, SpeakerState.SPEAKER_A)]
        result = assign_cameras(segs, config)
        assert result[0].camera_index == 1
        assert result[0].speaker_label == "host"

    def test_speaker_b_gets_camera_2(self, config):
        segs = [Segment(0.0, 5.0, 0, SpeakerState.SPEAKER_B)]
        result = assign_cameras(segs, config)
        assert result[0].camera_index == 2
        assert result[0].speaker_label == "guest"

    def test_both_gets_wide(self, config):
        segs = [Segment(0.0, 5.0, 0, SpeakerState.BOTH)]
        result = assign_cameras(segs, config)
        assert result[0].camera_index == 0
        assert result[0].speaker_label is None

    def test_silence_gets_wide(self, config):
        segs = [Segment(0.0, 5.0, 0, SpeakerState.SILENCE)]
        result = assign_cameras(segs, config)
        assert result[0].camera_index == 0

    def test_multiple_segments(self, config):
        segs = [
            Segment(0.0, 5.0, 0, SpeakerState.SPEAKER_A),
            Segment(5.0, 10.0, 0, SpeakerState.SPEAKER_B),
            Segment(10.0, 12.0, 0, SpeakerState.BOTH),
            Segment(12.0, 15.0, 0, SpeakerState.SILENCE),
        ]
        result = assign_cameras(segs, config)
        assert [s.camera_index for s in result] == [1, 2, 0, 0]
