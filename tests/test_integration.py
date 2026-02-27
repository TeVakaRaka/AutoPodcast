"""End-to-end integration test: synthetic audio -> full pipeline -> timeline."""

import numpy as np
import pytest

from autopodcast.core.analyzer import analyze_speaker
from autopodcast.core.detector import combine_speakers, detect_activity
from autopodcast.core.ducking import generate_ducking_events
from autopodcast.core.segmenter import segment_timeline
from autopodcast.core.switcher import assign_cameras
from autopodcast.export.json_export import timeline_to_dict, dict_to_timeline
from autopodcast.export.fcp7xml import generate_fcp7xml
from autopodcast.models.domain import SpeakerState, Timeline
from autopodcast.models.project import AudioInput, ProjectConfig
from tests.conftest import make_speech_pattern
from pathlib import Path


@pytest.fixture
def two_speaker_config():
    return ProjectConfig(
        audio_inputs=[
            AudioInput(path=Path("host.wav"), speaker_label="host", camera_index=1),
            AudioInput(path=Path("guest.wav"), speaker_label="guest", camera_index=2),
        ],
        default_camera=0,
        both_speaking_camera=0,
        ducking_enabled=True,
        ducking_db=-12.0,
        min_segment_ms=500.0,
        debounce_ms=100.0,
    )


class TestEndToEnd:
    def test_two_speakers_alternating(self, two_speaker_config):
        """Host speaks, then guest speaks — should produce 2+ segments."""
        config = two_speaker_config
        duration = 10.0

        # Host speaks 1-4s, Guest speaks 5-9s
        audio_a = make_speech_pattern([(1.0, 4.0, -10.0)], duration)
        audio_b = make_speech_pattern([(5.0, 9.0, -10.0)], duration)

        # Analyze
        act_a = analyze_speaker(audio_a, "host", config)
        act_b = analyze_speaker(audio_b, "guest", config)

        # Detect
        detect_activity(act_a, config)
        detect_activity(act_b, config)

        # Combine
        states = combine_speakers(act_a, act_b)
        assert len(states) > 0

        # Segment
        segments = segment_timeline(states, config)
        assert len(segments) >= 2

        # Camera assignment
        segments = assign_cameras(segments, config)

        # Check that we have host and guest segments
        speaker_states = {s.speaker_state for s in segments}
        assert SpeakerState.SPEAKER_A in speaker_states
        assert SpeakerState.SPEAKER_B in speaker_states

        # Check camera assignments
        for seg in segments:
            if seg.speaker_state == SpeakerState.SPEAKER_A:
                assert seg.camera_index == 1
            elif seg.speaker_state == SpeakerState.SPEAKER_B:
                assert seg.camera_index == 2

        # Ducking
        ducking = generate_ducking_events(segments, config)
        assert len(ducking) > 0

        # Build timeline
        timeline = Timeline(
            segments=segments,
            ducking_events=ducking,
            total_duration_s=duration,
        )

        # JSON round-trip
        data = timeline_to_dict(timeline)
        restored = dict_to_timeline(data)
        assert len(restored.segments) == len(timeline.segments)
        assert restored.total_duration_s == timeline.total_duration_s

        # FCP 7 XML generation
        xml_str = generate_fcp7xml(
            timeline,
            camera_paths=["/cam/wide.mp4", "/cam/host.mp4", "/cam/guest.mp4"],
            audio_paths=["/mic/host.wav", "/mic/guest.wav"],
        )
        assert 'version="5"' in xml_str
        assert "clipitem" in xml_str
        assert "<timecode>" in xml_str
        assert "<enabled>TRUE</enabled>" in xml_str

    def test_simultaneous_speech(self, two_speaker_config):
        """Both speakers at the same time -> BOTH segments."""
        config = two_speaker_config
        duration = 5.0

        audio_a = make_speech_pattern([(0.5, 4.0, -10.0)], duration)
        audio_b = make_speech_pattern([(0.5, 4.0, -10.0)], duration)

        act_a = analyze_speaker(audio_a, "host", config)
        act_b = analyze_speaker(audio_b, "guest", config)
        detect_activity(act_a, config)
        detect_activity(act_b, config)

        states = combine_speakers(act_a, act_b)
        segments = segment_timeline(states, config)
        segments = assign_cameras(segments, config)

        # Should contain BOTH state
        both_segs = [s for s in segments if s.speaker_state == SpeakerState.BOTH]
        assert len(both_segs) > 0
        # BOTH -> wide camera
        assert all(s.camera_index == 0 for s in both_segs)
