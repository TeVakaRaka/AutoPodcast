"""Tests for segmentation: RLE, debounce, min-length, merge."""

from pathlib import Path

import pytest

from autopodcast.core.segmenter import (
    build_audio_mute_segments,
    build_camera_segments,
    run_length_encode,
    debounce_segments,
    enforce_min_length,
    merge_adjacent,
    collapse_both_between_same,
    segment_timeline,
)
from autopodcast.models.domain import Segment, SpeakerState
from autopodcast.models.project import ProjectConfig, AudioInput


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


class TestAudioMuteDebounce:
    """Simulate the audio mute pipeline: RLE → debounce → merge."""

    def test_short_both_bleed_removed(self):
        """Short 'both' segments (crosstalk bleed) between speaker_a segments
        should be absorbed by debounce, producing cleaner audio mute output."""
        # Pattern: A(2s) → both(0.08s) → A(1s) → both(0.05s) → B(2s)
        # At hop_s=0.01: 200 A, 8 both, 100 A, 5 both, 200 B
        states = (
            [SpeakerState.SPEAKER_A] * 200
            + [SpeakerState.BOTH] * 8       # 80ms bleed — should be absorbed
            + [SpeakerState.SPEAKER_A] * 100
            + [SpeakerState.BOTH] * 5       # 50ms bleed — should be absorbed
            + [SpeakerState.SPEAKER_B] * 200
        )
        hop_s = 0.01
        segs = run_length_encode(states, hop_s)
        assert len(segs) == 5  # A, both, A, both, B

        segs = debounce_segments(segs, min_duration_s=0.3)
        segs = merge_adjacent(segs)

        # Both bleed segments absorbed into A, leaving A + B
        assert len(segs) == 2
        assert segs[0].speaker_state == SpeakerState.SPEAKER_A
        assert segs[1].speaker_state == SpeakerState.SPEAKER_B

    def test_long_both_segment_preserved(self):
        """A genuine 'both speaking' segment (>300ms) should survive debounce."""
        states = (
            [SpeakerState.SPEAKER_A] * 200
            + [SpeakerState.BOTH] * 50      # 500ms — real overlap, keep it
            + [SpeakerState.SPEAKER_B] * 200
        )
        hop_s = 0.01
        segs = run_length_encode(states, hop_s)
        segs = debounce_segments(segs, min_duration_s=0.3)
        segs = merge_adjacent(segs)

        assert len(segs) == 3
        assert segs[1].speaker_state == SpeakerState.BOTH


class TestAudioMuteMinLength:
    """Audio mute pipeline should enforce min_length to remove short interruptions."""

    def test_short_mute_interruption_absorbed(self):
        """Short speaker switch (e.g. 0.6s 'ugu') should not mute the other track.

        Pattern: A(3s) → B(0.6s) → A(3s)
        Without min_length: 3 segments, track 0 muted during B(0.6s)
        With min_length(2.0s): B absorbed into A → single A segment, no mute.
        """
        states = (
            [SpeakerState.SPEAKER_A] * 300   # 3s
            + [SpeakerState.SPEAKER_B] * 60  # 0.6s — short interruption
            + [SpeakerState.SPEAKER_A] * 300 # 3s
        )
        hop_s = 0.01
        segs = run_length_encode(states, hop_s)
        segs = debounce_segments(segs, min_duration_s=0.3)
        segs = enforce_min_length(segs, min_duration_s=2.0)
        segs = collapse_both_between_same(segs)
        segs = merge_adjacent(segs)

        # The 0.6s B segment should be absorbed → single A segment
        assert len(segs) == 1
        assert segs[0].speaker_state == SpeakerState.SPEAKER_A

    def test_long_speaker_switch_preserved(self):
        """A genuine speaker switch (>2s) should survive min_length enforcement."""
        states = (
            [SpeakerState.SPEAKER_A] * 300   # 3s
            + [SpeakerState.SPEAKER_B] * 300 # 3s — real switch
            + [SpeakerState.SPEAKER_A] * 300 # 3s
        )
        hop_s = 0.01
        segs = run_length_encode(states, hop_s)
        segs = debounce_segments(segs, min_duration_s=0.3)
        segs = enforce_min_length(segs, min_duration_s=2.0)
        segs = collapse_both_between_same(segs)
        segs = merge_adjacent(segs)

        assert len(segs) == 3
        assert segs[0].speaker_state == SpeakerState.SPEAKER_A
        assert segs[1].speaker_state == SpeakerState.SPEAKER_B
        assert segs[2].speaker_state == SpeakerState.SPEAKER_A


class TestBuildAudioMuteSegments:
    def _config(self, **overrides):
        defaults = dict(
            audio_inputs=[
                AudioInput(path=Path("a.wav"), speaker_label="host", camera_index=1),
                AudioInput(path=Path("b.wav"), speaker_label="guest", camera_index=2),
            ],
            audio_mute_debounce_ms=300.0,
            audio_mute_min_segment_ms=500.0,
            audio_mute_collapse_both_between_same=False,
        )
        defaults.update(overrides)
        return ProjectConfig(**defaults)

    def test_preserves_overlap_between_same_speaker_runs(self):
        """Audio mute should keep a real BOTH segment between A runs."""
        config = self._config()
        states = (
            [SpeakerState.SPEAKER_A] * 300
            + [SpeakerState.BOTH] * 80
            + [SpeakerState.SPEAKER_A] * 300
        )

        segs = build_audio_mute_segments(states, config)

        assert [seg.speaker_state for seg in segs] == [
            SpeakerState.SPEAKER_A,
            SpeakerState.BOTH,
            SpeakerState.SPEAKER_A,
        ]

    def test_audio_pipeline_uses_shorter_min_segment_than_video(self):
        """Short interjection should survive audio mute segmentation."""
        config = self._config(audio_mute_min_segment_ms=500.0)
        states = (
            [SpeakerState.SPEAKER_A] * 300
            + [SpeakerState.SPEAKER_B] * 60
            + [SpeakerState.SPEAKER_A] * 300
        )

        segs = build_audio_mute_segments(states, config)

        assert [seg.speaker_state for seg in segs] == [
            SpeakerState.SPEAKER_A,
            SpeakerState.SPEAKER_B,
            SpeakerState.SPEAKER_A,
        ]

    def test_can_opt_in_to_collapse_both(self):
        """Legacy collapse can still be enabled explicitly."""
        config = self._config(audio_mute_collapse_both_between_same=True)
        states = (
            [SpeakerState.SPEAKER_A] * 300
            + [SpeakerState.BOTH] * 80
            + [SpeakerState.SPEAKER_A] * 300
        )

        segs = build_audio_mute_segments(states, config)

        assert len(segs) == 1
        assert segs[0].speaker_state == SpeakerState.SPEAKER_A


class TestBuildCameraSegments:
    def _config(self, **overrides):
        defaults = dict(
            audio_inputs=[
                AudioInput(path=Path("a.wav"), speaker_label="host", camera_index=1),
                AudioInput(path=Path("b.wav"), speaker_label="guest", camera_index=2),
            ],
            camera_debounce_ms=300.0,
            camera_min_segment_ms=800.0,
            camera_takeover_min_segment_ms=700.0,
            camera_takeover_context_ms=3000.0,
            camera_both_min_segment_ms=900.0,
            camera_silence_min_segment_ms=1600.0,
        )
        defaults.update(overrides)
        return ProjectConfig(**defaults)

    def test_short_overlap_does_not_force_wide_camera_segment(self):
        """Short BOTH island should be absorbed into the surrounding speaker."""
        config = self._config()
        states = (
            [SpeakerState.SPEAKER_A] * 300
            + [SpeakerState.BOTH] * 80
            + [SpeakerState.SPEAKER_A] * 300
        )

        segs = build_camera_segments(states, config)

        assert len(segs) == 1
        assert segs[0].speaker_state == SpeakerState.SPEAKER_A

    def test_real_overlap_survives_for_camera_logic(self):
        """Long BOTH segment should still survive and be available for wide camera."""
        config = self._config()
        states = (
            [SpeakerState.SPEAKER_A] * 300
            + [SpeakerState.BOTH] * 120
            + [SpeakerState.SPEAKER_A] * 300
        )

        segs = build_camera_segments(states, config)

        assert [seg.speaker_state for seg in segs] == [
            SpeakerState.SPEAKER_A,
            SpeakerState.BOTH,
            SpeakerState.SPEAKER_A,
        ]

    def test_one_second_speaker_turn_survives_for_camera_switch(self):
        """Video should react to a real short speaker turn faster than legacy 2s pipeline."""
        config = self._config()
        states = (
            [SpeakerState.SPEAKER_A] * 300
            + [SpeakerState.SPEAKER_B] * 100
            + [SpeakerState.SPEAKER_A] * 300
        )

        segs = build_camera_segments(states, config)

        assert [seg.speaker_state for seg in segs] == [
            SpeakerState.SPEAKER_A,
            SpeakerState.SPEAKER_B,
            SpeakerState.SPEAKER_A,
        ]

    def test_short_takeover_survives_when_flanked_by_long_opposite_runs(self):
        """A 0.7s question after a long monologue should still reach camera logic."""
        config = self._config()
        states = (
            [SpeakerState.SPEAKER_A] * 300
            + [SpeakerState.SPEAKER_B] * 70
            + [SpeakerState.SPEAKER_A] * 300
        )

        segs = build_camera_segments(states, config)

        assert [seg.speaker_state for seg in segs] == [
            SpeakerState.SPEAKER_A,
            SpeakerState.SPEAKER_B,
            SpeakerState.SPEAKER_A,
        ]

    def test_short_takeover_is_ignored_in_fast_back_and_forth(self):
        """The takeover rule should not turn every rapid exchange into a cut."""
        config = self._config()
        states = (
            [SpeakerState.SPEAKER_A] * 120
            + [SpeakerState.SPEAKER_B] * 70
            + [SpeakerState.SPEAKER_A] * 120
        )

        segs = build_camera_segments(states, config)

        assert len(segs) == 1
        assert segs[0].speaker_state == SpeakerState.SPEAKER_A

    def test_short_silence_is_more_conservative_than_both(self):
        """Brief pause should not force a camera reset to wide."""
        config = self._config()
        states = (
            [SpeakerState.SPEAKER_A] * 300
            + [SpeakerState.SILENCE] * 120
            + [SpeakerState.SPEAKER_A] * 300
        )

        segs = build_camera_segments(states, config)

        assert len(segs) == 1
        assert segs[0].speaker_state == SpeakerState.SPEAKER_A

    def test_dense_overlap_cluster_is_preserved_for_camera(self):
        """Repeated BOTH islands with only short turns between them should stay wide-friendly."""
        config = self._config()
        states = (
            [SpeakerState.SPEAKER_A] * 200
            + [SpeakerState.BOTH] * 120
            + [SpeakerState.SPEAKER_A] * 60
            + [SpeakerState.BOTH] * 40
            + [SpeakerState.SPEAKER_B] * 70
            + [SpeakerState.SPEAKER_A] * 80
            + [SpeakerState.BOTH] * 100
            + [SpeakerState.SPEAKER_B] * 200
        )

        segs = build_camera_segments(states, config)

        assert [seg.speaker_state for seg in segs] == [
            SpeakerState.SPEAKER_A,
            SpeakerState.BOTH,
            SpeakerState.SPEAKER_B,
        ]
        assert segs[1].duration_s == pytest.approx(4.7, abs=0.01)

    def test_dense_overlap_cluster_breaks_on_long_clear_turn(self):
        """A long clean turn between overlaps should break the wide-friendly cluster."""
        config = self._config()
        states = (
            [SpeakerState.SPEAKER_A] * 200
            + [SpeakerState.BOTH] * 120
            + [SpeakerState.SPEAKER_A] * 60
            + [SpeakerState.BOTH] * 40
            + [SpeakerState.SPEAKER_B] * 300
            + [SpeakerState.SPEAKER_A] * 80
            + [SpeakerState.BOTH] * 100
            + [SpeakerState.SPEAKER_B] * 200
        )

        segs = build_camera_segments(states, config)

        both_durations = [seg.duration_s for seg in segs if seg.speaker_state == SpeakerState.BOTH]
        assert both_durations
        assert max(both_durations) < 2.5


class TestCollapseBothBetweenSame:
    """collapse_both_between_same: replace bleed 'both' with neighbor speaker."""

    def test_a_both_a(self):
        """A→both→A collapses to single A after merge."""
        segs = [
            Segment(0, 1, 0, SpeakerState.SPEAKER_A),
            Segment(1, 2, 0, SpeakerState.BOTH),
            Segment(2, 3, 0, SpeakerState.SPEAKER_A),
        ]
        result = merge_adjacent(collapse_both_between_same(segs))
        assert len(result) == 1
        assert result[0].speaker_state == SpeakerState.SPEAKER_A
        assert result[0].start_s == 0
        assert result[0].end_s == 3

    def test_b_both_b(self):
        """B→both→B collapses to single B after merge."""
        segs = [
            Segment(0, 1, 0, SpeakerState.SPEAKER_B),
            Segment(1, 2, 0, SpeakerState.BOTH),
            Segment(2, 3, 0, SpeakerState.SPEAKER_B),
        ]
        result = merge_adjacent(collapse_both_between_same(segs))
        assert len(result) == 1
        assert result[0].speaker_state == SpeakerState.SPEAKER_B

    def test_preserve_a_both_b(self):
        """A→both→B is a real transition, preserved."""
        segs = [
            Segment(0, 1, 0, SpeakerState.SPEAKER_A),
            Segment(1, 2, 0, SpeakerState.BOTH),
            Segment(2, 3, 0, SpeakerState.SPEAKER_B),
        ]
        result = collapse_both_between_same(segs)
        assert len(result) == 3
        assert result[1].speaker_state == SpeakerState.BOTH

    def test_chained_a_both_a_both_a(self):
        """A→both→A→both→A collapses to single A after merge."""
        segs = [
            Segment(0, 1, 0, SpeakerState.SPEAKER_A),
            Segment(1, 2, 0, SpeakerState.BOTH),
            Segment(2, 3, 0, SpeakerState.SPEAKER_A),
            Segment(3, 4, 0, SpeakerState.BOTH),
            Segment(4, 5, 0, SpeakerState.SPEAKER_A),
        ]
        result = merge_adjacent(collapse_both_between_same(segs))
        assert len(result) == 1
        assert result[0].speaker_state == SpeakerState.SPEAKER_A
        assert result[0].end_s == 5

    def test_short_input(self):
        """Fewer than 3 segments returned as-is."""
        segs = [Segment(0, 1, 0, SpeakerState.BOTH)]
        assert len(collapse_both_between_same(segs)) == 1

        segs2 = [
            Segment(0, 1, 0, SpeakerState.SPEAKER_A),
            Segment(1, 2, 0, SpeakerState.BOTH),
        ]
        assert len(collapse_both_between_same(segs2)) == 2


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
