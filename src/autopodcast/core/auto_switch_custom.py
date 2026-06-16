"""Generic configurable multicam planner — the "Конструктор" (custom) mode.

Unlike the fixed presets (4cams / sakha / monologue), this planner takes an
*arbitrary* cast: each person is mapped to a camera angle, and one angle is the
wide / "общак". The camera rule is a direct generalization of every preset:

  - nobody speaking            -> wide
  - all speakers share a camera -> that camera (solo close-up OR a shared shot,
                                   e.g. two guests framed together)
  - speakers span >= 2 cameras  -> wide (cross-camera overlap = общак)

The mapping is static (no auto-moving close-up). Short blips — a brief silence
between turns, a 0.2 s interjection, a momentary overlap — are absorbed by the
minimum-shot-hold merge, so the camera holds through them instead of flickering.

This module is pure (numpy-free, no I/O): it takes per-person ``SpeakerActivity``
in and returns a :class:`CustomPlan`. The audio side is delegated to the shared,
role-agnostic ``build_audio_plan`` already used by the roundtable planner.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from autopodcast.core.auto_switch_4cams import AudioLevelSegment, build_audio_plan
from autopodcast.models.domain import SpeakerActivity


@dataclass(frozen=True)
class CustomPerson:
    """One participant: a mic (audio track) and the camera that frames them."""

    key: str                 # stable internal id, e.g. "person_0"
    label: str               # display name, e.g. "ведущий"
    audio_track_index: int   # 0-based index into the sequence audio tracks
    camera_angle: int        # camera index shown when this person solos (patcher convention;
                             # the CLI converts the user's 1-based angle to 0-based)


@dataclass(frozen=True)
class CustomSwitchConfig:
    """Tuning for the custom planner.

    The ``audio_*`` fields mirror :class:`Roundtable4CamConfig` so the shared
    ``build_audio_plan`` can consume this config directly (duck-typed).
    """

    wide_camera: int = 0                 # camera index of the общак / wide shot (same base as camera_angle)
    dominance_delta_db: float = 6.0      # 2nd+ speakers within this many dB of the loudest stay "active"
    shot_hold_s: float = 1.4             # minimum camera duration; bridges brief pauses/overlaps
    reestablish_interval_s: float = 0.0  # 0 = off; else cut to wide after this long on one camera
    reestablish_hold_s: float = 1.5      # duration of an inserted re-establishing wide

    # --- audio plan (names/defaults match Roundtable4CamConfig) ---
    audio_silence_opens_all_tracks: bool = True
    audio_keep_recently_active_open: bool = False
    audio_recent_hold_s: float = 0.30
    audio_merge_gap_s: float = 0.22
    audio_min_on_s: float = 0.28
    audio_min_open_after_trigger_s: float = 0.55
    audio_release_hold_s: float = 0.30
    audio_min_closed_s: float = 0.35
    audio_pre_roll_s: float = 0.24
    audio_post_roll_s: float = 0.12
    audio_hard_mute_db: float = -96.0

    def validate(self) -> None:
        for name, value in [
            ("shot_hold_s", self.shot_hold_s),
            ("audio_recent_hold_s", self.audio_recent_hold_s),
            ("audio_merge_gap_s", self.audio_merge_gap_s),
            ("audio_min_on_s", self.audio_min_on_s),
            ("audio_min_open_after_trigger_s", self.audio_min_open_after_trigger_s),
            ("audio_release_hold_s", self.audio_release_hold_s),
            ("audio_min_closed_s", self.audio_min_closed_s),
            ("audio_pre_roll_s", self.audio_pre_roll_s),
            ("audio_post_roll_s", self.audio_post_roll_s),
        ]:
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.reestablish_interval_s < 0:
            raise ValueError(f"reestablish_interval_s must be >= 0, got {self.reestablish_interval_s}")
        if self.wide_camera < 0:
            raise ValueError(f"wide_camera must be >= 0, got {self.wide_camera}")


@dataclass(frozen=True)
class CustomFrameState:
    time_s: float
    active_keys: tuple[str, ...]
    camera_index: int


@dataclass(frozen=True)
class CustomCameraSegment:
    start_s: float
    end_s: float
    camera_index: int
    reason: str = ""

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class CustomPlan:
    frame_states: list[CustomFrameState] = field(default_factory=list)
    camera_segments: list[CustomCameraSegment] = field(default_factory=list)
    audio_open_intervals_raw_s: dict[int, list[tuple[float, float]]] = field(default_factory=dict)
    audio_open_intervals_s: dict[int, list[tuple[float, float]]] = field(default_factory=dict)
    audio_level_segments: list[AudioLevelSegment] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def build_custom_plan(
    people: list[CustomPerson],
    activities: dict[str, SpeakerActivity],
    hop_s: float,
    config: CustomSwitchConfig,
) -> CustomPlan:
    """Build a full custom switching plan from per-person speech activities."""
    config.validate()
    _validate_people(people)

    frame_states = build_frame_states(people, activities, hop_s, config)
    camera_segments = build_camera_segments(frame_states, hop_s, config)

    # The audio plan is role-agnostic: it only reads frame.active_keys / time_s
    # and per-person audio_track_index, so the roundtable builder works as-is.
    raw_audio, stable_audio, level_segments = build_audio_plan(
        people, frame_states, hop_s, config
    )

    diagnostics = {
        "people": [
            {"key": p.key, "label": p.label,
             "audio_track_index": p.audio_track_index, "camera_angle": p.camera_angle}
            for p in people
        ],
        "wide_camera": config.wide_camera,
        "camera_counts": _camera_counts(camera_segments),
        "frame_count": len(frame_states),
    }

    return CustomPlan(
        frame_states=frame_states,
        camera_segments=camera_segments,
        audio_open_intervals_raw_s=raw_audio,
        audio_open_intervals_s=stable_audio,
        audio_level_segments=level_segments,
        diagnostics=diagnostics,
    )


def build_frame_states(
    people: list[CustomPerson],
    activities: dict[str, SpeakerActivity],
    hop_s: float,
    config: CustomSwitchConfig,
) -> list[CustomFrameState]:
    """Per-frame active set (dominance-filtered) -> resolved camera index."""
    if not people:
        return []

    frame_count = min(len(activities[p.key].frames) for p in people)
    if frame_count <= 0:
        return []

    camera_by_key = {p.key: p.camera_angle for p in people}
    base_key = people[0].key

    states: list[CustomFrameState] = []
    for idx in range(frame_count):
        active_keys = _active_keys_for_frame(people, activities, idx, config)
        camera = _camera_for_active(active_keys, camera_by_key, config.wide_camera)
        states.append(
            CustomFrameState(
                time_s=activities[base_key].frames[idx].time_s,
                active_keys=tuple(active_keys),
                camera_index=camera,
            )
        )
    return states


def build_camera_segments(
    frame_states: list[CustomFrameState],
    hop_s: float,
    config: CustomSwitchConfig,
) -> list[CustomCameraSegment]:
    """Run-length encode resolved cameras, then stabilize."""
    segments = _run_length_encode(frame_states, hop_s)
    segments = _merge_adjacent(segments)
    segments = _enforce_min_duration(segments, config.shot_hold_s)
    segments = _merge_adjacent(segments)
    if config.reestablish_interval_s > 0:
        segments = _insert_reestablishing_wide(segments, config)
        segments = _merge_adjacent(segments)
    return segments


# --------------------------------------------------------------------------- #
# Camera rule
# --------------------------------------------------------------------------- #

def _camera_for_active(
    active_keys: list[str],
    camera_by_key: dict[str, int],
    wide_camera: int,
) -> int:
    """The heart of the mode: map the active speaker set to one camera angle."""
    if not active_keys:
        return wide_camera
    cameras = {camera_by_key[key] for key in active_keys}
    if len(cameras) == 1:
        return next(iter(cameras))
    return wide_camera


def _active_keys_for_frame(
    people: list[CustomPerson],
    activities: dict[str, SpeakerActivity],
    idx: int,
    config: CustomSwitchConfig,
) -> list[str]:
    """Active speakers at frame ``idx``, dropping ones far below the loudest.

    Mirrors the roundtable dominance filter: keep the loudest speaker plus any
    other within ``dominance_delta_db`` of it (so quiet bleed doesn't count as a
    real overlap).
    """
    candidates = [p.key for p in people if activities[p.key].frames[idx].is_active]
    if len(candidates) <= 1:
        return candidates

    levels = {key: activities[key].frames[idx].envelope_db for key in candidates}
    strongest = max(candidates, key=lambda key: levels[key])
    keep = [strongest]
    for key in candidates:
        if key == strongest:
            continue
        if levels[strongest] - levels[key] < config.dominance_delta_db:
            keep.append(key)
    return sorted(set(keep))


# --------------------------------------------------------------------------- #
# Segmentation / stability
# --------------------------------------------------------------------------- #

def _run_length_encode(
    frame_states: list[CustomFrameState],
    hop_s: float,
) -> list[CustomCameraSegment]:
    if not frame_states:
        return []

    segments: list[CustomCameraSegment] = []
    start_idx = 0
    current = frame_states[0].camera_index
    for i in range(1, len(frame_states)):
        if frame_states[i].camera_index != current:
            segments.append(
                CustomCameraSegment(
                    start_s=frame_states[start_idx].time_s,
                    end_s=frame_states[i].time_s,
                    camera_index=current,
                    reason="rle",
                )
            )
            current = frame_states[i].camera_index
            start_idx = i

    segments.append(
        CustomCameraSegment(
            start_s=frame_states[start_idx].time_s,
            end_s=frame_states[-1].time_s + hop_s,
            camera_index=current,
            reason="rle",
        )
    )
    return segments


def _merge_adjacent(segments: list[CustomCameraSegment]) -> list[CustomCameraSegment]:
    if len(segments) <= 1:
        return list(segments)
    result = [segments[0]]
    for seg in segments[1:]:
        prev = result[-1]
        if seg.camera_index == prev.camera_index:
            result[-1] = CustomCameraSegment(prev.start_s, seg.end_s, prev.camera_index, prev.reason)
        else:
            result.append(seg)
    return result


def _enforce_min_duration(
    segments: list[CustomCameraSegment],
    min_duration_s: float,
) -> list[CustomCameraSegment]:
    """Merge segments shorter than ``min_duration_s`` into the previous one.

    This is what makes the camera hold through brief pauses, interjections and
    momentary overlaps: a sub-threshold wide/other blip is absorbed by whatever
    shot preceded it.
    """
    if len(segments) <= 1:
        return list(segments)
    result = [segments[0]]
    for seg in segments[1:]:
        if seg.duration_s < min_duration_s:
            prev = result[-1]
            result[-1] = CustomCameraSegment(prev.start_s, seg.end_s, prev.camera_index, prev.reason)
        else:
            result.append(seg)
    return result


def _insert_reestablishing_wide(
    segments: list[CustomCameraSegment],
    config: CustomSwitchConfig,
) -> list[CustomCameraSegment]:
    """Optionally cut back to wide after a long unbroken hold on one close shot."""
    interval = config.reestablish_interval_s
    hold = config.reestablish_hold_s
    result: list[CustomCameraSegment] = []
    for seg in segments:
        if (
            seg.camera_index == config.wide_camera
            or seg.duration_s < interval + hold
        ):
            result.append(seg)
            continue
        cut_at = seg.start_s + interval
        result.append(CustomCameraSegment(seg.start_s, cut_at, seg.camera_index, seg.reason))
        result.append(CustomCameraSegment(cut_at, cut_at + hold, config.wide_camera, "reestablish"))
        result.append(CustomCameraSegment(cut_at + hold, seg.end_s, seg.camera_index, seg.reason))
    return result


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _validate_people(people: list[CustomPerson]) -> None:
    if not people:
        raise ValueError("At least one person is required")
    tracks = [p.audio_track_index for p in people]
    if len(set(tracks)) != len(tracks):
        raise ValueError(f"Audio track indices must be unique, got {tracks}")
    keys = [p.key for p in people]
    if len(set(keys)) != len(keys):
        raise ValueError(f"Person keys must be unique, got {keys}")


def _camera_counts(segments: list[CustomCameraSegment]) -> dict[int, float]:
    counts: dict[int, float] = {}
    for seg in segments:
        counts[seg.camera_index] = counts.get(seg.camera_index, 0.0) + seg.duration_s
    return counts
