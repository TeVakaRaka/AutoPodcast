"""Tests for segmentation: RLE, debounce, min-length, merge."""

import pytest

from autopodcast.core.segmenter import (
    run_length_encode,
    debounce_segments,
    enforce_min_length,
    merge_adjacent,
    segment_timeline,
)
from autopodcast.models.domain import Segment, SpeakerState
from autopodcast.models.project import ProjectConfig, AudioInput
from pathlib import Path


class TestRunLengthEncode:
    def test_single_state(self):
        states = [SpeakerState.SPEAKER_A] * 100
        segs = run_length_encode(states, hop_s=0.01)
        assert len(segs) == 1
        assert segs[0].speaker_state == SpeakerState.SPEAKER_A
        assert abs(segs[0].end_s - 1.0) < 0.001

    def test_alternating(self):
        states = (
            [SpeakerState.SPEAKER_A] * 50
            + [SpeakerState.SPEAKER_B] * 50
            + [SpeakerState.SILENCE] * 50
        )
        segs = run_length_encode(states, hop_s=0.01)
        assert len(segs) == 3
        assert segs[0].speaker_state == SpeakerState.SPEAKER_A
        assert segs[1].speaker_state == SpeakerState.SPEAKER_B
        assert segs[2].speaker_state == SpeakerState.SILENCE

    def test_empty(self):
        assert run_length_encode([], 0.01) == []


class TestDebounceSegments:
    def test_short_segment_removed(self):
        """A 0.1s segment should be merged into the previous 1s segment."""
        segs = [
            Segment(0.0, 1.0, 0, SpeakerState.SPEAKER_A),
            Segment(1.0, 1.1, 0, SpeakerState.SPEAKER_B),  # too short
            Segment(1.1, 3.0, 0, SpeakerState.SPEAKER_A),
        ]
        result = debounce_segments(segs, min_duration_s=0.3)
        # Short segment merged into previous
        assert len(result) == 2
        assert result[0].end_s == 1.1
        assert result[0].speaker_state == SpeakerState.SPEAKER_A

    def test_long_segments_kept(self):
        segs = [
            Segment(0.0, 2.0, 0, SpeakerState.SPEAKER_A),
            Segment(2.0, 4.0, 0, SpeakerState.SPEAKER_B),
        ]
        result = debounce_segments(segs, min_duration_s=0.3)
        assert len(result) == 2


class TestEnforceMinLength:
    def test_short_merged(self):
        segs = [
            Segment(0.0, 3.0, 0, SpeakerState.SPEAKER_A),
            Segment(3.0, 3.5, 0, SpeakerState.SPEAKER_B),  # only 0.5s
            Segment(3.5, 6.0, 0, SpeakerState.SPEAKER_A),
        ]
        result = enforce_min_length(segs, min_duration_s=2.0)
        assert len(result) == 2
        assert result[0].end_s == 3.5


class TestMergeAdjacent:
    def test_same_state_merged(self):
        segs = [
            Segment(0.0, 1.0, 0, SpeakerState.SPEAKER_A),
            Segment(1.0, 2.0, 0, SpeakerState.SPEAKER_A),
            Segment(2.0, 3.0, 0, SpeakerState.SPEAKER_B),
        ]
        result = merge_adjacent(segs)
        assert len(result) == 2
        assert result[0].end_s == 2.0
        assert result[0].speaker_state == SpeakerState.SPEAKER_A

    def test_different_states_kept(self):
        segs = [
            Segment(0.0, 1.0, 0, SpeakerState.SPEAKER_A),
            Segment(1.0, 2.0, 0, SpeakerState.SPEAKER_B),
        ]
        result = merge_adjacent(segs)
        assert len(result) == 2


class TestSegmentTimeline:
    def test_full_pipeline(self):
        config = ProjectConfig(
            audio_inputs=[
                AudioInput(path=Path("a.wav"), speaker_label="host", camera_index=1),
                AudioInput(path=Path("b.wav"), speaker_label="guest", camera_index=2),
            ],
            debounce_ms=100.0,
            min_segment_ms=500.0,
        )
        # 300 frames of A, 5 frames of B (too short), 300 frames of A
        states = (
            [SpeakerState.SPEAKER_A] * 300
            + [SpeakerState.SPEAKER_B] * 5
            + [SpeakerState.SPEAKER_A] * 300
        )
        segments = segment_timeline(states, config)
        # The 5-frame B blip should be absorbed, leaving one A segment
        assert len(segments) == 1
        assert segments[0].speaker_state == SpeakerState.SPEAKER_A
