"""Tests for the generic configurable ("Конструктор") planner."""

from __future__ import annotations

import pytest

from autopodcast.core.auto_switch_custom import (
    CustomPerson,
    CustomSwitchConfig,
    build_custom_plan,
)
from autopodcast.models.domain import AnalysisFrame, SpeakerActivity

HOP_S = 0.1


def _activity(
    label: str,
    active_runs: list[tuple[float, float]],
    total_s: float,
    hop_s: float = HOP_S,
    active_db: float = -18.0,
    inactive_db: float = -60.0,
) -> SpeakerActivity:
    frames = []
    n_frames = int(round(total_s / hop_s))
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


def _people() -> list[CustomPerson]:
    """The user's setup: 1 host + 3 guests; guests 1 & 2 share one camera."""
    return [
        CustomPerson(key="host", label="ведущий", audio_track_index=0, camera_angle=2),
        CustomPerson(key="g1", label="гость1", audio_track_index=1, camera_angle=3),
        CustomPerson(key="g2", label="гость2", audio_track_index=2, camera_angle=3),
        CustomPerson(key="g3", label="гость3", audio_track_index=3, camera_angle=4),
    ]


def _config(**overrides) -> CustomSwitchConfig:
    base = CustomSwitchConfig(wide_camera=1, shot_hold_s=1.4)
    return CustomSwitchConfig(**(base.__dict__ | overrides))


def _summary(plan) -> list[tuple[int, float, float]]:
    return [
        (seg.camera_index, round(seg.start_s, 1), round(seg.end_s, 1))
        for seg in plan.camera_segments
    ]


class TestCameraRule:
    def test_solo_uses_persons_camera(self):
        people = _people()
        activities = {
            "host": _activity("host", [(0.0, 4.0)], 4.0),
            "g1": _activity("g1", [], 4.0),
            "g2": _activity("g2", [], 4.0),
            "g3": _activity("g3", [], 4.0),
        }
        plan = build_custom_plan(people, activities, HOP_S, _config())
        assert _summary(plan) == [(2, 0.0, 4.0)]

    def test_two_people_sharing_a_camera_stay_on_that_camera(self):
        """Guests 1 & 2 both talk -> their shared camera (3), NOT the wide."""
        people = _people()
        activities = {
            "host": _activity("host", [], 4.0),
            "g1": _activity("g1", [(0.0, 4.0)], 4.0),
            "g2": _activity("g2", [(0.0, 4.0)], 4.0),
            "g3": _activity("g3", [], 4.0),
        }
        plan = build_custom_plan(people, activities, HOP_S, _config())
        assert _summary(plan) == [(3, 0.0, 4.0)]

    def test_overlap_across_cameras_goes_wide(self):
        """Host (cam2) and guest 3 (cam4) overlap -> общак (wide=1)."""
        people = _people()
        activities = {
            "host": _activity("host", [(0.0, 4.0)], 4.0),
            "g1": _activity("g1", [], 4.0),
            "g2": _activity("g2", [], 4.0),
            "g3": _activity("g3", [(0.0, 4.0)], 4.0),
        }
        plan = build_custom_plan(people, activities, HOP_S, _config())
        assert _summary(plan) == [(1, 0.0, 4.0)]

    def test_silence_goes_wide(self):
        people = _people()
        activities = {p.key: _activity(p.key, [], 4.0) for p in people}
        plan = build_custom_plan(people, activities, HOP_S, _config())
        assert _summary(plan) == [(1, 0.0, 4.0)]


class TestUserScenario:
    def test_full_four_person_sequence(self):
        people = _people()
        total = 12.0
        activities = {
            "host": _activity("host", [(0.0, 2.0), (8.0, 10.0)], total),
            "g1": _activity("g1", [(2.0, 6.0)], total),
            "g2": _activity("g2", [(4.0, 6.0)], total),
            "g3": _activity("g3", [(6.0, 8.0), (8.0, 10.0)], total),
        }
        plan = build_custom_plan(people, activities, HOP_S, _config())
        # host solo -> 2; g1 then g1+g2 (same cam) -> 3; g3 solo -> 4;
        # host+g3 overlap then silence -> wide 1.
        assert _summary(plan) == [
            (2, 0.0, 2.0),
            (3, 2.0, 6.0),
            (4, 6.0, 8.0),
            (1, 8.0, 12.0),
        ]


class TestStability:
    def test_brief_interjection_is_held_through(self):
        """A 0.2 s cross-camera blip is absorbed: the camera holds on the host."""
        people = _people()
        activities = {
            "host": _activity("host", [(0.0, 3.0)], 3.0),
            "g1": _activity("g1", [], 3.0),
            "g2": _activity("g2", [], 3.0),
            "g3": _activity("g3", [(1.0, 1.2)], 3.0),
        }
        plan = build_custom_plan(people, activities, HOP_S, _config())
        assert _summary(plan) == [(2, 0.0, 3.0)]

    def test_long_overlap_is_not_absorbed(self):
        """A 2 s overlap is real -> a genuine wide segment appears."""
        people = _people()
        activities = {
            "host": _activity("host", [(0.0, 5.0)], 5.0),
            "g1": _activity("g1", [], 5.0),
            "g2": _activity("g2", [], 5.0),
            "g3": _activity("g3", [(1.5, 3.5)], 5.0),
        }
        plan = build_custom_plan(people, activities, HOP_S, _config())
        cams = [c for c, _s, _e in _summary(plan)]
        assert cams == [2, 1, 2]


class TestAudioPlan:
    def test_open_intervals_keyed_by_track(self):
        people = _people()
        total = 10.0
        activities = {
            "host": _activity("host", [(0.0, 2.0), (8.0, 10.0)], total),
            "g1": _activity("g1", [(2.0, 4.0)], total),
            "g2": _activity("g2", [(4.0, 6.0)], total),
            "g3": _activity("g3", [(6.0, 8.0)], total),
        }
        plan = build_custom_plan(people, activities, HOP_S, _config())
        # one entry per audio track
        assert set(plan.audio_open_intervals_s.keys()) == {0, 1, 2, 3}
        # host track (0) is open during its speech
        host_intervals = plan.audio_open_intervals_s[0]
        assert any(s <= 1.0 <= e for s, e in host_intervals)
        assert any(s <= 9.0 <= e for s, e in host_intervals)


class TestValidation:
    def test_empty_people_raises(self):
        with pytest.raises(ValueError):
            build_custom_plan([], {}, HOP_S, _config())

    def test_duplicate_audio_tracks_raise(self):
        people = [
            CustomPerson(key="a", label="a", audio_track_index=0, camera_angle=2),
            CustomPerson(key="b", label="b", audio_track_index=0, camera_angle=3),
        ]
        activities = {
            "a": _activity("a", [(0.0, 1.0)], 1.0),
            "b": _activity("b", [], 1.0),
        }
        with pytest.raises(ValueError):
            build_custom_plan(people, activities, HOP_S, _config())


class TestStudioCleanMode:
    """The 'studio' leak-matrix attribution (reused from sakha) wired into custom.

    Asserts correct attribution on clean cases; the bleed-suppression advantage
    over plain loudness is a production property that depends on the studio
    internals and is not asserted with a tiny synthetic.
    """

    def test_monologue_attributed_to_speaker(self):
        # total > active run so the per-mic floor sees real silence (as in any
        # real recording); otherwise the studio SNR floor has nothing to anchor to.
        people = _people()
        activities = {
            "host": _activity("host", [(0.0, 4.0)], 5.0),
            "g1": _activity("g1", [], 5.0),
            "g2": _activity("g2", [], 5.0),
            "g3": _activity("g3", [], 5.0),
        }
        plan = build_custom_plan(people, activities, HOP_S, _config(clean_mode="studio"))
        cams = {c for c, _s, _e in _summary(plan)}
        # the host's camera is shown, and no other person's camera is falsely opened
        assert 2 in cams
        assert 3 not in cams and 4 not in cams

    def test_clean_alternation_attributes_each_speaker(self):
        people = _people()
        activities = {
            "host": _activity("host", [(0.0, 3.0)], 9.0),
            "g3": _activity("g3", [(3.0, 6.0)], 9.0),
            "g1": _activity("g1", [(6.0, 9.0)], 9.0),
            "g2": _activity("g2", [], 9.0),
        }
        plan = build_custom_plan(people, activities, HOP_S, _config(clean_mode="studio"))
        cams = [c for c, _s, _e in _summary(plan)]
        assert cams == [2, 4, 3]  # host -> cam2, guest3 -> cam4, guest1 -> cam3

    def test_genuine_cross_camera_overlap_goes_wide(self):
        people = _people()
        activities = {
            "host": _activity("host", [(0.0, 4.0)], 4.0),
            "g3": _activity("g3", [(0.0, 4.0)], 4.0),  # host (cam2) + guest3 (cam4), both real
            "g1": _activity("g1", [], 4.0),
            "g2": _activity("g2", [], 4.0),
        }
        plan = build_custom_plan(people, activities, HOP_S, _config(clean_mode="studio"))
        cams = {c for c, _s, _e in _summary(plan)}
        assert 1 in cams  # the wide / общак appears for the genuine overlap

    def test_studio_emits_audio_plan(self):
        people = _people()
        activities = {
            "host": _activity("host", [(0.0, 4.0)], 4.0),
            "g1": _activity("g1", [], 4.0),
            "g2": _activity("g2", [], 4.0),
            "g3": _activity("g3", [], 4.0),
        }
        plan = build_custom_plan(people, activities, HOP_S, _config(clean_mode="studio"))
        assert set(plan.audio_open_intervals_s.keys()) == {0, 1, 2, 3}

    def test_invalid_clean_mode_raises(self):
        people = _people()
        activities = {p.key: _activity(p.key, [], 2.0) for p in people}
        with pytest.raises(ValueError, match="clean_mode"):
            build_custom_plan(people, activities, HOP_S, _config(clean_mode="bogus"))
