"""Tests for camera_scheduler: wide cutaway on long monologues."""

from pathlib import Path

import pytest

from autopodcast.core.segmenter import build_camera_segments
from autopodcast.core.camera_scheduler import schedule_camera_events
from autopodcast.models.domain import CameraEvent, Segment, SpeakerState
from autopodcast.models.project import AudioInput, ProjectConfig


@pytest.fixture
def config():
    return ProjectConfig(
        audio_inputs=[
            AudioInput(path=Path("host.wav"), speaker_label="host", camera_index=1),
            AudioInput(path=Path("guest.wav"), speaker_label="guest", camera_index=2),
        ],
        default_camera=0,
        both_speaking_camera=0,
        long_talk_threshold_sec=15.0,
        wide_duration_sec=5.0,
        long_talk_mode="once",
    )


def _seg(start, end, state):
    return Segment(start_s=start, end_s=end, camera_index=0, speaker_state=state)


class TestCameraScheduler:
    def test_ac1_short_segment_no_cutaway(self, config):
        """AC1: A: 0-14.9s -> [CamA 0-14.9]"""
        segments = [_seg(0, 14.9, SpeakerState.SPEAKER_A)]
        events = schedule_camera_events(segments, config)

        assert len(events) == 1
        assert events[0].camera_index == 1
        assert events[0].start_s == 0
        assert events[0].end_s == 14.9

    def test_ac2_long_segment_cutaway_at_end(self, config):
        """AC2: A: 0-20s -> [CamA 0-15, Wide 15-20]"""
        segments = [_seg(0, 20, SpeakerState.SPEAKER_A)]
        events = schedule_camera_events(segments, config)

        assert len(events) == 2
        assert events[0].camera_index == 1
        assert events[0].start_s == 0
        assert events[0].end_s == 15
        assert events[1].camera_index == 0
        assert events[1].start_s == 15
        assert events[1].end_s == 20

    def test_ac3_long_segment_once_mode(self, config):
        """AC3: A: 0-40s (once) -> [CamA 0-15, Wide 15-20, CamA 20-40]"""
        segments = [_seg(0, 40, SpeakerState.SPEAKER_A)]
        events = schedule_camera_events(segments, config)

        assert len(events) == 3
        assert events[0].camera_index == 1
        assert events[0].start_s == 0
        assert events[0].end_s == 15
        assert events[1].camera_index == 0
        assert events[1].start_s == 15
        assert events[1].end_s == 20
        assert events[2].camera_index == 1
        assert events[2].start_s == 20
        assert events[2].end_s == 40

    def test_ac4_speaker_change_clips_cutaway(self, config):
        """AC4: A: 0-18s, B: 18-30s -> [CamA 0-15, Wide 15-18, CamB 18-30]"""
        segments = [
            _seg(0, 18, SpeakerState.SPEAKER_A),
            _seg(18, 30, SpeakerState.SPEAKER_B),
        ]
        events = schedule_camera_events(segments, config)

        assert len(events) == 3
        assert events[0].camera_index == 1
        assert events[0].end_s == 15
        assert events[1].camera_index == 0
        assert events[1].start_s == 15
        assert events[1].end_s == 18
        assert events[2].camera_index == 2
        assert events[2].start_s == 18
        assert events[2].end_s == 30

    def test_ac5_cutaway_carries_through_both(self, config):
        """AC5: A: 0-17, BOTH: 17-19, A: 19-25 -> [CamA 0-15, Wide 15-20, CamA 20-25]"""
        segments = [
            _seg(0, 17, SpeakerState.SPEAKER_A),
            _seg(17, 19, SpeakerState.BOTH),
            _seg(19, 25, SpeakerState.SPEAKER_A),
        ]
        events = schedule_camera_events(segments, config)

        assert len(events) == 3
        assert events[0].camera_index == 1
        assert events[0].start_s == 0
        assert events[0].end_s == 15
        assert events[1].camera_index == 0
        assert events[1].start_s == 15
        assert events[1].end_s == 20
        assert events[2].camera_index == 1
        assert events[2].start_s == 20
        assert events[2].end_s == 25

    def test_ac6_silence_covers_cutaway(self, config):
        """AC6: A: 0-16, SILENCE: 16-22, A: 22-25 -> [CamA 0-16, Wide 16-22, CamA 22-25]"""
        segments = [
            _seg(0, 16, SpeakerState.SPEAKER_A),
            _seg(16, 22, SpeakerState.SILENCE),
            _seg(22, 25, SpeakerState.SPEAKER_A),
        ]
        events = schedule_camera_events(segments, config)

        assert len(events) == 3
        assert events[0].camera_index == 1
        assert events[0].start_s == 0
        assert events[0].end_s == 16
        assert events[1].camera_index == 0
        assert events[1].start_s == 16
        assert events[1].end_s == 22
        assert events[2].camera_index == 1
        assert events[2].start_s == 22
        assert events[2].end_s == 25

    def test_ac7_adjacent_same_camera_merged(self, config):
        """AC7: Adjacent events with same camera are merged."""
        segments = [
            _seg(0, 5, SpeakerState.SILENCE),
            _seg(5, 10, SpeakerState.BOTH),
        ]
        events = schedule_camera_events(segments, config)

        # Both SILENCE and BOTH map to wide (cam 0) → merged
        assert len(events) == 1
        assert events[0].camera_index == 0
        assert events[0].start_s == 0
        assert events[0].end_s == 10

    def test_both_always_wide(self, config):
        """BOTH segments always get wide camera."""
        segments = [_seg(0, 10, SpeakerState.BOTH)]
        events = schedule_camera_events(segments, config)

        assert len(events) == 1
        assert events[0].camera_index == 0

    def test_silence_always_wide(self, config):
        """SILENCE segments always get wide camera."""
        segments = [_seg(0, 10, SpeakerState.SILENCE)]
        events = schedule_camera_events(segments, config)

        assert len(events) == 1
        assert events[0].camera_index == 0

    def test_speaker_b_long_talk(self, config):
        """Speaker B long talk also gets cutaway."""
        segments = [_seg(0, 25, SpeakerState.SPEAKER_B)]
        events = schedule_camera_events(segments, config)

        assert len(events) == 3
        assert events[0].camera_index == 2
        assert events[0].end_s == 15
        assert events[1].camera_index == 0
        assert events[1].start_s == 15
        assert events[1].end_s == 20
        assert events[2].camera_index == 2
        assert events[2].start_s == 20

    def test_short_overlap_is_absorbed_before_camera_wide(self, config):
        """Short BOTH should not create a separate wide shot when camera segments are built."""
        config.camera_debounce_ms = 300.0
        config.camera_min_segment_ms = 800.0
        config.camera_both_min_segment_ms = 900.0
        config.camera_silence_min_segment_ms = 1600.0
        config.min_camera_event_s = 0.8
        states = (
            [SpeakerState.SPEAKER_A] * 300
            + [SpeakerState.BOTH] * 80
            + [SpeakerState.SPEAKER_A] * 300
        )

        segments = build_camera_segments(states, config)
        events = schedule_camera_events(segments, config)

        assert len(events) == 1
        assert events[0].camera_index == 1

    def test_one_second_turn_can_reach_camera_scheduler(self, config):
        """A real 1s speaker turn should remain visible to camera scheduling."""
        config.camera_debounce_ms = 300.0
        config.camera_min_segment_ms = 800.0
        config.camera_takeover_min_segment_ms = 700.0
        config.camera_takeover_context_ms = 3000.0
        config.camera_both_min_segment_ms = 900.0
        config.camera_silence_min_segment_ms = 1600.0
        config.min_camera_event_s = 0.8
        states = (
            [SpeakerState.SPEAKER_A] * 300
            + [SpeakerState.SPEAKER_B] * 100
            + [SpeakerState.SPEAKER_A] * 300
        )

        segments = build_camera_segments(states, config)
        events = schedule_camera_events(segments, config)

        assert [event.camera_index for event in events] == [1, 2, 1]

    def test_short_takeover_can_survive_camera_event_floor(self, config):
        """A 0.7s interruption should still get a camera when it punctuates a long turn."""
        config.camera_debounce_ms = 300.0
        config.camera_min_segment_ms = 800.0
        config.camera_takeover_min_segment_ms = 700.0
        config.camera_takeover_context_ms = 3000.0
        config.camera_both_min_segment_ms = 900.0
        config.camera_silence_min_segment_ms = 1600.0
        config.min_camera_event_s = 0.8
        states = (
            [SpeakerState.SPEAKER_A] * 300
            + [SpeakerState.SPEAKER_B] * 70
            + [SpeakerState.SPEAKER_A] * 300
        )

        segments = build_camera_segments(states, config)
        events = schedule_camera_events(segments, config)

        assert [event.camera_index for event in events] == [1, 2, 1]

    def test_one_second_overlap_can_reach_wide_camera(self, config):
        """A genuine overlap should now survive long enough to produce wide."""
        config.camera_debounce_ms = 300.0
        config.camera_min_segment_ms = 800.0
        config.camera_takeover_min_segment_ms = 700.0
        config.camera_takeover_context_ms = 3000.0
        config.camera_both_min_segment_ms = 900.0
        config.camera_silence_min_segment_ms = 1600.0
        config.min_camera_event_s = 0.8
        states = (
            [SpeakerState.SPEAKER_A] * 300
            + [SpeakerState.BOTH] * 100
            + [SpeakerState.SPEAKER_A] * 300
        )

        segments = build_camera_segments(states, config)
        events = schedule_camera_events(segments, config)

        assert [event.camera_index for event in events] == [1, 0, 1]

    def test_dialogue_reestablishing_wide_appears_after_active_exchanges(self, config):
        """Back-and-forth dialogue should get an occasional re-establishing wide."""
        config.dialogue_wide_interval_sec = 24.0
        config.dialogue_wide_duration_sec = 2.0
        config.dialogue_wide_min_turns = 3
        config.long_talk_threshold_sec = 60.0
        segments = [
            _seg(0, 10, SpeakerState.SPEAKER_A),
            _seg(10, 18, SpeakerState.SPEAKER_B),
            _seg(18, 26, SpeakerState.SPEAKER_A),
            _seg(26, 34, SpeakerState.SPEAKER_B),
        ]

        events = schedule_camera_events(segments, config)

        assert [(event.start_s, event.end_s, event.camera_index) for event in events] == [
            (0, 10, 1),
            (10, 18, 2),
            (18, 26, 1),
            (26, 28.0, 0),
            (28.0, 34, 2),
        ]

    def test_dialogue_reestablishing_wide_waits_for_interval(self, config):
        """Wide should not appear too early even if speakers are alternating."""
        config.dialogue_wide_interval_sec = 24.0
        config.dialogue_wide_duration_sec = 2.0
        config.dialogue_wide_min_turns = 3
        config.long_talk_threshold_sec = 60.0
        segments = [
            _seg(0, 6, SpeakerState.SPEAKER_A),
            _seg(6, 12, SpeakerState.SPEAKER_B),
            _seg(12, 18, SpeakerState.SPEAKER_A),
            _seg(18, 24, SpeakerState.SPEAKER_B),
        ]

        events = schedule_camera_events(segments, config)

        assert [event.camera_index for event in events] == [1, 2, 1, 2]

    def test_sticky_wide_bridges_dense_overlap_cluster(self, config):
        """Dense BOTH/A/B/A/BOTH cluster should hold the wide longer."""
        config.long_talk_threshold_sec = 60.0
        config.dialogue_wide_interval_sec = 999.0
        config.sticky_wide_max_bridge_sec = 8.0
        config.sticky_wide_max_turn_sec = 2.4
        config.sticky_wide_min_turns = 3
        segments = [
            _seg(0.0, 3.64, SpeakerState.BOTH),
            _seg(3.64, 5.40, SpeakerState.SPEAKER_A),
            _seg(5.40, 6.71, SpeakerState.SPEAKER_B),
            _seg(6.71, 8.37, SpeakerState.SPEAKER_A),
            _seg(8.37, 10.80, SpeakerState.BOTH),
            _seg(10.80, 13.39, SpeakerState.SPEAKER_A),
        ]

        events = schedule_camera_events(segments, config)

        assert [(event.start_s, event.end_s, event.camera_index) for event in events] == [
            (0.0, 10.8, 0),
            (10.8, 13.39, 1),
        ]

    def test_sticky_wide_does_not_bridge_through_long_turn(self, config):
        """A long closeup turn should break the sticky wide cluster."""
        config.long_talk_threshold_sec = 60.0
        config.dialogue_wide_interval_sec = 999.0
        config.sticky_wide_max_bridge_sec = 8.0
        config.sticky_wide_max_turn_sec = 2.4
        config.sticky_wide_min_turns = 3
        segments = [
            _seg(0.0, 3.64, SpeakerState.BOTH),
            _seg(3.64, 6.40, SpeakerState.SPEAKER_A),
            _seg(6.40, 7.71, SpeakerState.SPEAKER_B),
            _seg(7.71, 9.37, SpeakerState.SPEAKER_A),
            _seg(9.37, 11.80, SpeakerState.BOTH),
        ]

        events = schedule_camera_events(segments, config)

        assert [event.camera_index for event in events] == [0, 1, 2, 1, 0]

    def test_dialogue_cluster_extends_wide_through_rapid_turns(self, config):
        """A wide anchor should hold through a burst of short alternating dialogue."""
        config.long_talk_threshold_sec = 60.0
        config.dialogue_wide_interval_sec = 999.0
        config.dialogue_cluster_max_span_sec = 9.0
        config.dialogue_cluster_max_turn_sec = 4.0
        config.dialogue_cluster_min_turns = 4
        segments = [
            _seg(0.0, 0.9, SpeakerState.BOTH),
            _seg(0.9, 2.59, SpeakerState.SPEAKER_A),
            _seg(2.59, 3.66, SpeakerState.SPEAKER_B),
            _seg(3.66, 6.98, SpeakerState.SPEAKER_A),
            _seg(6.98, 8.25, SpeakerState.SPEAKER_B),
            _seg(8.25, 10.85, SpeakerState.SPEAKER_A),
        ]

        events = schedule_camera_events(segments, config)

        assert [(event.start_s, event.end_s, event.camera_index) for event in events] == [
            (0.0, 8.25, 0),
            (8.25, 10.85, 1),
        ]

    def test_dialogue_cluster_does_not_extend_through_long_turn(self, config):
        """Dense-dialogue wide should stop if one closeup turn gets too long."""
        config.long_talk_threshold_sec = 60.0
        config.dialogue_wide_interval_sec = 999.0
        config.dialogue_cluster_max_span_sec = 9.0
        config.dialogue_cluster_max_turn_sec = 4.0
        config.dialogue_cluster_min_turns = 4
        segments = [
            _seg(0.0, 0.9, SpeakerState.BOTH),
            _seg(0.9, 2.59, SpeakerState.SPEAKER_A),
            _seg(2.59, 3.66, SpeakerState.SPEAKER_B),
            _seg(3.66, 7.86, SpeakerState.SPEAKER_A),
            _seg(7.86, 9.13, SpeakerState.SPEAKER_B),
        ]

        events = schedule_camera_events(segments, config)

        assert [event.camera_index for event in events] == [0, 1, 2, 1, 2]


class TestRepeatMode:
    @pytest.fixture
    def config_repeat(self):
        return ProjectConfig(
            audio_inputs=[
                AudioInput(path=Path("host.wav"), speaker_label="host", camera_index=1),
                AudioInput(path=Path("guest.wav"), speaker_label="guest", camera_index=2),
            ],
            default_camera=0,
            both_speaking_camera=0,
            long_talk_threshold_sec=15.0,
            wide_duration_sec=5.0,
            long_talk_mode="repeat",
            wide_cooldown_sec=20.0,
            max_segment_sec=100.0,
        )

    def test_repeat_mode_40s(self, config_repeat):
        """A speaks 40s: [CamA 0-15, Wide 15-20, CamA 20-40]."""
        segments = [_seg(0, 40, SpeakerState.SPEAKER_A)]
        events = schedule_camera_events(segments, config_repeat)

        assert len(events) == 3
        assert events[0].camera_index == 1
        assert events[0].start_s == 0
        assert events[0].end_s == 15
        assert events[1].camera_index == 0
        assert events[1].start_s == 15
        assert events[1].end_s == 20
        assert events[2].camera_index == 1
        assert events[2].start_s == 20
        assert events[2].end_s == 40

    def test_repeat_mode_80s(self, config_repeat):
        """A speaks 80s: wide inserts repeat with cooldown."""
        segments = [_seg(0, 80, SpeakerState.SPEAKER_A)]
        events = schedule_camera_events(segments, config_repeat)

        # Expected: [CamA 0-15, Wide 15-20, CamA 20-40, Wide 40-45, CamA 45-65, Wide 65-70, CamA 70-80]
        wide_events = [e for e in events if e.camera_index == 0]
        speaker_events = [e for e in events if e.camera_index == 1]
        assert len(wide_events) >= 2  # at least 2 wide inserts
        # Check that wide inserts are spaced by at least cooldown
        for i in range(1, len(wide_events)):
            gap = wide_events[i].start_s - wide_events[i - 1].end_s
            assert gap >= config_repeat.wide_cooldown_sec - 0.01

    def test_once_mode_preserved(self):
        """long_talk_mode='once' still works as before."""
        config = ProjectConfig(
            audio_inputs=[
                AudioInput(path=Path("host.wav"), speaker_label="host", camera_index=1),
                AudioInput(path=Path("guest.wav"), speaker_label="guest", camera_index=2),
            ],
            default_camera=0,
            both_speaking_camera=0,
            long_talk_threshold_sec=15.0,
            wide_duration_sec=5.0,
            long_talk_mode="once",
            max_segment_sec=100.0,
        )
        segments = [_seg(0, 40, SpeakerState.SPEAKER_A)]
        events = schedule_camera_events(segments, config)

        assert len(events) == 3
        assert events[0].camera_index == 1
        assert events[0].end_s == 15
        assert events[1].camera_index == 0
        assert events[1].start_s == 15
        assert events[1].end_s == 20
        assert events[2].camera_index == 1
        assert events[2].start_s == 20
        assert events[2].end_s == 40

    def test_wide_cooldown(self, config_repeat):
        """With cooldown=30, wide inserts are spaced at least 30s apart."""
        config_repeat.wide_cooldown_sec = 30.0
        segments = [_seg(0, 80, SpeakerState.SPEAKER_A)]
        events = schedule_camera_events(segments, config_repeat)

        wide_events = [e for e in events if e.camera_index == 0]
        for i in range(1, len(wide_events)):
            gap = wide_events[i].start_s - wide_events[i - 1].end_s
            assert gap >= 29.99


class TestSplitAlternatesWide:
    def test_split_alternates_wide(self):
        """50s speaker segment, max_segment=20: alternates cam/wide."""
        config = ProjectConfig(
            audio_inputs=[
                AudioInput(path=Path("host.wav"), speaker_label="host", camera_index=1),
                AudioInput(path=Path("guest.wav"), speaker_label="guest", camera_index=2),
            ],
            default_camera=0,
            both_speaking_camera=0,
            long_talk_threshold_sec=100.0,  # no wide cutaway
            wide_duration_sec=5.0,
            long_talk_mode="once",
            max_segment_sec=20.0,
        )
        segments = [_seg(0, 50, SpeakerState.SPEAKER_A)]
        events = schedule_camera_events(segments, config)

        # 50s / 20s = 3 chunks: [0-20 cam1, 20-40 wide, 40-50 cam1]
        assert len(events) == 3
        assert events[0].camera_index == 1
        assert events[1].camera_index == 0  # wide
        assert events[2].camera_index == 1


class TestMinCameraEventDuration:
    @pytest.fixture
    def config_min_cam(self):
        return ProjectConfig(
            audio_inputs=[
                AudioInput(path=Path("host.wav"), speaker_label="host", camera_index=1),
                AudioInput(path=Path("guest.wav"), speaker_label="guest", camera_index=2),
            ],
            default_camera=0,
            both_speaking_camera=0,
            long_talk_threshold_sec=100.0,  # no wide cutaway
            wide_duration_sec=5.0,
            long_talk_mode="once",
            max_segment_sec=100.0,
            min_camera_event_s=2.0,
        )

    def test_short_both_absorbed(self, config_min_cam):
        """Short BOTH (0.5s) between two SPEAKER_A segments is absorbed."""
        segments = [
            _seg(0, 10, SpeakerState.SPEAKER_A),
            _seg(10, 10.5, SpeakerState.BOTH),
            _seg(10.5, 20, SpeakerState.SPEAKER_A),
        ]
        events = schedule_camera_events(segments, config_min_cam)

        # The 0.5s BOTH (wide cam) should be absorbed → single SPEAKER_A event
        assert len(events) == 1
        assert events[0].camera_index == 1
        assert events[0].start_s == 0
        assert events[0].end_s == 20

    def test_long_segments_unchanged(self, config_min_cam):
        """Segments longer than min_camera_event_s are not affected."""
        segments = [
            _seg(0, 10, SpeakerState.SPEAKER_A),
            _seg(10, 15, SpeakerState.BOTH),
            _seg(15, 25, SpeakerState.SPEAKER_B),
        ]
        events = schedule_camera_events(segments, config_min_cam)

        assert len(events) == 3
        assert events[0].camera_index == 1
        assert events[1].camera_index == 0
        assert events[2].camera_index == 2

    def test_disabled_when_zero(self):
        """min_camera_event_s=0 disables the filter."""
        config = ProjectConfig(
            audio_inputs=[
                AudioInput(path=Path("host.wav"), speaker_label="host", camera_index=1),
                AudioInput(path=Path("guest.wav"), speaker_label="guest", camera_index=2),
            ],
            default_camera=0,
            both_speaking_camera=0,
            long_talk_threshold_sec=100.0,
            wide_duration_sec=5.0,
            long_talk_mode="once",
            max_segment_sec=100.0,
            min_camera_event_s=0.0,
        )
        segments = [
            _seg(0, 10, SpeakerState.SPEAKER_A),
            _seg(10, 10.5, SpeakerState.BOTH),
            _seg(10.5, 20, SpeakerState.SPEAKER_B),
        ]
        events = schedule_camera_events(segments, config)

        # Short BOTH should remain since filter is disabled
        assert len(events) == 3
