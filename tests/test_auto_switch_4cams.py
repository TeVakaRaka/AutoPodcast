"""Tests for the 4-cam roundtable auto-switching planner."""

from __future__ import annotations

from autopodcast.core.auto_switch_4cams import (
    ConversationPhaseType,
    ParticipantSpec,
    Roundtable4CamConfig,
    build_roundtable_plan,
)
from autopodcast.core.camera_motion import CameraMotionPlan, MotionInterval
from autopodcast.models.domain import AnalysisFrame, SpeakerActivity


def _activity(
    label: str,
    active_runs: list[tuple[float, float]],
    total_s: float,
    hop_s: float = 0.1,
    active_db: float = -18.0,
    inactive_db: float = -60.0,
) -> SpeakerActivity:
    frames = []
    n_frames = int(total_s / hop_s)
    for idx in range(n_frames):
        time_s = idx * hop_s
        is_active = any(start <= time_s < end for start, end in active_runs)
        level = active_db if is_active else inactive_db
        frames.append(
            AnalysisFrame(
                time_s=time_s,
                rms_db=level,
                envelope_db=level,
                is_active=is_active,
                peak_db=level + 2.0,
            )
        )
    return SpeakerActivity(speaker_label=label, frames=frames)


def _participants() -> list[ParticipantSpec]:
    return [
        ParticipantSpec(key="host", label="host", role="host", audio_track_index=0),
        ParticipantSpec(key="guest_1", label="guest_1", role="guest", audio_track_index=1),
        ParticipantSpec(key="guest_2", label="guest_2", role="guest", audio_track_index=2),
        ParticipantSpec(key="guest_3", label="guest_3", role="guest", audio_track_index=3),
    ]


def _config(**overrides) -> Roundtable4CamConfig:
    config = Roundtable4CamConfig(
        camera_all_wide=0,
        camera_guests_wide=1,
        camera_guest_close=2,
        camera_host_close=3,
        reestablish_all_wide_interval_s=100.0,
    )
    return Roundtable4CamConfig(**(config.__dict__ | overrides))


def _motion_plan(*intervals: tuple[float, float]) -> CameraMotionPlan:
    return CameraMotionPlan(
        moving_intervals=tuple(
            MotionInterval(start_s=start_s, end_s=end_s)
            for start_s, end_s in intervals
        )
    )


class TestRoundtable4CamPlan:
    def test_guest_cluster_defaults_to_guest_master(self):
        participants = _participants()
        config = _config()
        activities = {
            "host": _activity("host", [], 3.0),
            "guest_1": _activity("guest_1", [(0.0, 0.9)], 3.0),
            "guest_2": _activity("guest_2", [(0.9, 1.8)], 3.0),
            "guest_3": _activity("guest_3", [], 3.0),
        }

        plan = build_roundtable_plan(participants, activities, 0.1, config)

        assert [(round(seg.start_s, 1), round(seg.end_s, 1), seg.camera_index, seg.reason) for seg in plan.camera_segments] == [
            (0.0, 1.8, 1, "guest_cluster"),
            (1.8, 3.0, 0, "pause_reset"),
        ]
        assert plan.conversation_phases[0].phase == ConversationPhaseType.GUEST_CLUSTER

    def test_long_single_guest_answer_uses_master_then_guest_accent(self):
        participants = _participants()
        config = _config()
        activities = {
            "host": _activity("host", [], 4.0),
            "guest_1": _activity("guest_1", [(0.0, 3.5)], 4.0),
            "guest_2": _activity("guest_2", [], 4.0),
            "guest_3": _activity("guest_3", [], 4.0),
        }

        plan = build_roundtable_plan(participants, activities, 0.1, config)

        assert [(round(seg.start_s, 1), round(seg.end_s, 1), seg.camera_index, seg.reason) for seg in plan.camera_segments] == [
            (0.0, 0.5, 1, "guest_cluster"),
            (0.5, 4.0, 2, "guest_accent"),
        ]

    def test_motion_guard_blocks_guest_close_when_camera_is_moving(self):
        participants = _participants()
        config = _config()
        activities = {
            "host": _activity("host", [], 4.0),
            "guest_1": _activity("guest_1", [(0.0, 3.5)], 4.0),
            "guest_2": _activity("guest_2", [], 4.0),
            "guest_3": _activity("guest_3", [], 4.0),
        }

        plan = build_roundtable_plan(
            participants,
            activities,
            0.1,
            config,
            motion_plans={
                2: _motion_plan((0.4, 4.0)),
            },
        )

        assert not any(seg.camera_index == 2 for seg in plan.camera_segments)
        assert any(event.reason == "planned_camera_blocked" for event in plan.motion_events)

    def test_motion_guard_escapes_when_current_camera_starts_moving(self):
        participants = _participants()
        config = _config()
        activities = {
            "host": _activity("host", [], 3.0),
            "guest_1": _activity("guest_1", [(0.0, 0.9)], 3.0),
            "guest_2": _activity("guest_2", [(0.9, 1.8)], 3.0),
            "guest_3": _activity("guest_3", [], 3.0),
        }

        plan = build_roundtable_plan(
            participants,
            activities,
            0.1,
            config,
            motion_plans={
                1: _motion_plan((1.6, 2.2)),
            },
        )

        assert any(seg.reason == "motion_escape" and seg.camera_index == 0 for seg in plan.camera_segments)
        assert any(event.reason == "escape_from_motion" for event in plan.motion_events)

    def test_short_host_question_inside_guest_cluster_stays_on_guest_master(self):
        participants = _participants()
        config = _config()
        activities = {
            "host": _activity("host", [(1.0, 1.6)], 3.0),
            "guest_1": _activity("guest_1", [(0.0, 1.0)], 3.0),
            "guest_2": _activity("guest_2", [(1.6, 2.5)], 3.0),
            "guest_3": _activity("guest_3", [], 3.0),
        }

        plan = build_roundtable_plan(participants, activities, 0.1, config)

        assert [(round(seg.start_s, 1), round(seg.end_s, 1), seg.camera_index, seg.reason) for seg in plan.camera_segments] == [
            (0.0, 3.0, 1, "guest_cluster"),
        ]

    def test_long_host_followup_returns_to_host_close(self):
        participants = _participants()
        config = _config()
        activities = {
            "host": _activity("host", [(1.0, 2.5)], 4.0),
            "guest_1": _activity("guest_1", [(0.0, 1.0)], 4.0),
            "guest_2": _activity("guest_2", [(2.5, 3.6)], 4.0),
            "guest_3": _activity("guest_3", [], 4.0),
        }

        plan = build_roundtable_plan(participants, activities, 0.1, config)

        assert [(round(seg.start_s, 1), round(seg.end_s, 1), seg.camera_index, seg.reason) for seg in plan.camera_segments] == [
            (0.0, 1.0, 1, "guest_cluster"),
            (1.0, 2.5, 3, "host_followup"),
            (2.5, 4.0, 1, "guest_cluster"),
        ]

    def test_dense_overlap_cluster_uses_full_wide(self):
        participants = _participants()
        config = _config()
        activities = {
            "host": _activity("host", [(0.6, 1.7), (1.9, 3.0)], 4.0),
            "guest_1": _activity("guest_1", [(0.0, 1.7)], 4.0),
            "guest_2": _activity("guest_2", [(1.3, 3.0)], 4.0),
            "guest_3": _activity("guest_3", [], 4.0),
        }

        plan = build_roundtable_plan(participants, activities, 0.1, config)

        all_wide_segments = [seg for seg in plan.camera_segments if seg.reason == "all_overlap"]
        assert all_wide_segments
        assert all_wide_segments[0].camera_index == 0
        assert sum(seg.duration_s for seg in all_wide_segments) >= 1.0

    def test_short_overlap_stays_on_guest_side_and_allows_accent(self):
        participants = _participants()
        config = _config()
        activities = {
            "host": _activity("host", [(1.0, 1.7)], 4.0),
            "guest_1": _activity("guest_1", [(0.0, 3.2)], 4.0),
            "guest_2": _activity("guest_2", [], 4.0),
            "guest_3": _activity("guest_3", [], 4.0),
        }

        plan = build_roundtable_plan(participants, activities, 0.1, config)

        assert not any(seg.reason == "all_overlap" for seg in plan.camera_segments)
        assert any(seg.reason == "guest_accent" for seg in plan.camera_segments)

    def test_host_share_cap_suppresses_repeated_short_host_cuts(self):
        participants = _participants()
        config = _config()
        activities = {
            "host": _activity("host", [(2.0, 3.0), (5.0, 6.0)], 8.5),
            "guest_1": _activity("guest_1", [(0.0, 2.0), (6.0, 8.0)], 8.5),
            "guest_2": _activity("guest_2", [(3.0, 5.0)], 8.5),
            "guest_3": _activity("guest_3", [], 8.5),
        }

        plan = build_roundtable_plan(participants, activities, 0.1, config)

        host_segments = [seg for seg in plan.camera_segments if seg.camera_index == 3]
        assert len(host_segments) == 1
        assert round(host_segments[0].start_s, 1) == 2.0
        assert round(host_segments[0].end_s, 1) == 3.0
        assert plan.camera_balance_stats["global_host_share"] < 0.45

    def test_audio_false_micro_open_is_removed(self):
        participants = _participants()
        config = _config()
        activities = {
            "host": _activity("host", [(0.0, 1.0)], 2.0),
            "guest_1": _activity("guest_1", [(0.4, 0.6)], 2.0),
            "guest_2": _activity("guest_2", [], 2.0),
            "guest_3": _activity("guest_3", [], 2.0),
        }

        plan = build_roundtable_plan(participants, activities, 0.1, config)

        assert plan.audio_open_intervals_s[1] == [(1.0, 2.0)]
        assert tuple(round(value, 1) for value in plan.audio_open_intervals_raw_s[1][0]) == (0.4, 0.6)

    def test_audio_small_gap_is_merged(self):
        participants = _participants()
        config = _config()
        activities = {
            "host": _activity("host", [(1.0, 1.5)], 2.0),
            "guest_1": _activity("guest_1", [(0.0, 0.4), (0.5, 1.0)], 2.0),
            "guest_2": _activity("guest_2", [], 2.0),
            "guest_3": _activity("guest_3", [], 2.0),
        }

        plan = build_roundtable_plan(participants, activities, 0.1, config)

        assert plan.audio_open_intervals_s[1][0] == (0.0, 1.0)

    def test_audio_min_open_after_trigger_extends_short_real_phrase(self):
        participants = _participants()
        config = _config()
        activities = {
            "host": _activity("host", [(0.4, 1.5)], 2.0),
            "guest_1": _activity("guest_1", [(0.0, 0.3)], 2.0),
            "guest_2": _activity("guest_2", [], 2.0),
            "guest_3": _activity("guest_3", [], 2.0),
        }

        plan = build_roundtable_plan(participants, activities, 0.1, config)

        assert plan.audio_open_intervals_s[1][0] == (0.0, 0.4)

    def test_audio_other_tracks_stay_closed_without_chatter(self):
        participants = _participants()
        config = _config()
        activities = {
            "host": _activity("host", [(0.0, 1.5)], 2.0),
            "guest_1": _activity("guest_1", [(0.2, 0.3), (0.6, 0.65)], 2.0),
            "guest_2": _activity("guest_2", [], 2.0),
            "guest_3": _activity("guest_3", [], 2.0),
        }

        plan = build_roundtable_plan(participants, activities, 0.1, config)

        assert plan.audio_open_intervals_s[1] == [(1.5, 2.0)]
        assert plan.audio_open_intervals_s[2] == [(1.5, 2.0)]
        assert plan.audio_open_intervals_s[3] == [(1.5, 2.0)]
