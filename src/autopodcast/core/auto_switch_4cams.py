"""Offline auto-switching planner for 1 host + 3 guests + 4 cameras."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from math import ceil

from autopodcast.core.camera_motion import CameraMotionPlan, MotionInterval
from autopodcast.models.domain import SpeakerActivity


EPS = 1e-9
ROUNDTABLE_MOTION_ENTRY_LOOKAHEAD_S = 1.5


class RoundtableState(str, Enum):
    SILENCE = "silence"
    HOST_ONLY = "host_only"
    SINGLE_GUEST = "single_guest"
    MULTI_GUEST = "multi_guest"
    HOST_PLUS_GUEST = "host_plus_guest"
    GROUP_OVERLAP = "group_overlap"


class ConversationPhaseType(str, Enum):
    HOST_INTRO = "host_intro"
    GUEST_CLUSTER = "guest_cluster"
    GUEST_ACCENT = "guest_accent"
    HOST_FOLLOWUP = "host_followup"
    ALL_OVERLAP = "all_overlap"
    PAUSE_RESET = "pause_reset"


@dataclass(frozen=True)
class ParticipantSpec:
    key: str
    label: str
    role: str  # "host" | "guest"
    audio_track_index: int


@dataclass(frozen=True)
class Roundtable4CamConfig:
    camera_all_wide: int
    camera_guests_wide: int
    camera_guest_close: int
    camera_host_close: int
    dominance_delta_db: float = 6.0
    shot_hold_time_s: float = 1.4
    cooldown_s: float = 0.9
    host_return_min_s: float = 1.0
    overlap_min_hold_s: float = 0.8
    silence_timeout_s: float = 0.9
    cam3_wait_timeout_s: float = 0.0
    cam3_min_useful_after_ready_s: float = 0.75
    cam3_recent_turn_window_s: float = 3.0
    cam3_max_recent_turns: int = 2
    cam3_cut_in_grace_s: float = 0.15
    reestablish_all_wide_interval_s: float = 25.0
    reestablish_all_wide_duration_s: float = 1.8
    reestablish_min_turns: int = 3
    guest_close_min_domination_s: float = 1.0
    guest_close_min_duration_s: float = 0.8
    guest_close_max_continuous_s: float = 4.0
    guest_close_cooldown_s: float = 1.8
    guest_close_overlap_clear_s: float = 0.35
    guest_close_side_settle_s: float = 0.45
    guest_cluster_bridge_host_s: float = 0.9
    guest_cluster_window_s: float = 4.5
    host_screen_share_cap_window_s: float = 30.0
    host_screen_share_cap_ratio: float = 0.45
    host_screen_share_cap_min_guest_s: float = 8.0
    all_overlap_trigger_s: float = 1.0
    all_overlap_sticky_s: float = 0.7
    all_overlap_window_s: float = 5.0
    all_overlap_min_islands: int = 3
    all_overlap_min_turns: int = 5
    audio_recent_hold_s: float = 0.30
    audio_hard_mute_timeout_s: float = 2.0
    audio_min_on_s: float = 0.28
    audio_merge_gap_s: float = 0.22
    audio_min_open_after_trigger_s: float = 0.55
    audio_release_hold_s: float = 0.30
    audio_min_closed_s: float = 0.35
    audio_pre_roll_s: float = 0.24
    audio_post_roll_s: float = 0.12
    audio_standby_db: float = -9.0
    audio_inactive_db: float = -18.0
    audio_hard_mute_db: float = -96.0
    audio_keep_recently_active_open: bool = False
    audio_silence_opens_all_tracks: bool = True

    def validate(self) -> None:
        for name, value in [
            ("shot_hold_time_s", self.shot_hold_time_s),
            ("cooldown_s", self.cooldown_s),
            ("host_return_min_s", self.host_return_min_s),
            ("overlap_min_hold_s", self.overlap_min_hold_s),
            ("silence_timeout_s", self.silence_timeout_s),
            ("cam3_min_useful_after_ready_s", self.cam3_min_useful_after_ready_s),
            ("cam3_recent_turn_window_s", self.cam3_recent_turn_window_s),
            ("cam3_cut_in_grace_s", self.cam3_cut_in_grace_s),
            ("reestablish_all_wide_interval_s", self.reestablish_all_wide_interval_s),
            ("reestablish_all_wide_duration_s", self.reestablish_all_wide_duration_s),
            ("guest_close_min_domination_s", self.guest_close_min_domination_s),
            ("guest_close_min_duration_s", self.guest_close_min_duration_s),
            ("guest_close_max_continuous_s", self.guest_close_max_continuous_s),
            ("guest_close_cooldown_s", self.guest_close_cooldown_s),
            ("guest_close_overlap_clear_s", self.guest_close_overlap_clear_s),
            ("guest_close_side_settle_s", self.guest_close_side_settle_s),
            ("guest_cluster_bridge_host_s", self.guest_cluster_bridge_host_s),
            ("guest_cluster_window_s", self.guest_cluster_window_s),
            ("host_screen_share_cap_window_s", self.host_screen_share_cap_window_s),
            ("host_screen_share_cap_ratio", self.host_screen_share_cap_ratio),
            ("host_screen_share_cap_min_guest_s", self.host_screen_share_cap_min_guest_s),
            ("all_overlap_trigger_s", self.all_overlap_trigger_s),
            ("all_overlap_sticky_s", self.all_overlap_sticky_s),
            ("all_overlap_window_s", self.all_overlap_window_s),
            ("audio_recent_hold_s", self.audio_recent_hold_s),
            ("audio_hard_mute_timeout_s", self.audio_hard_mute_timeout_s),
            ("audio_min_on_s", self.audio_min_on_s),
            ("audio_merge_gap_s", self.audio_merge_gap_s),
            ("audio_min_open_after_trigger_s", self.audio_min_open_after_trigger_s),
            ("audio_release_hold_s", self.audio_release_hold_s),
            ("audio_min_closed_s", self.audio_min_closed_s),
            ("audio_pre_roll_s", self.audio_pre_roll_s),
            ("audio_post_roll_s", self.audio_post_roll_s),
        ]:
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.cam3_wait_timeout_s < 0:
            raise ValueError(f"cam3_wait_timeout_s must be >= 0, got {self.cam3_wait_timeout_s}")
        if self.cam3_max_recent_turns < 0:
            raise ValueError("cam3_max_recent_turns must be >= 0")
        if self.reestablish_min_turns < 1:
            raise ValueError("reestablish_min_turns must be >= 1")
        if self.all_overlap_min_islands < 1:
            raise ValueError("all_overlap_min_islands must be >= 1")
        if self.all_overlap_min_turns < 1:
            raise ValueError("all_overlap_min_turns must be >= 1")


@dataclass(frozen=True)
class RoundtableFrameState:
    time_s: float
    state: RoundtableState
    active_keys: tuple[str, ...]
    focus_key: str | None = None


@dataclass(frozen=True)
class RoundtableSpeechSegment:
    start_s: float
    end_s: float
    state: RoundtableState
    active_keys: tuple[str, ...]
    focus_key: str | None = None

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class ConversationPhase:
    start_s: float
    end_s: float
    phase: ConversationPhaseType
    focus_key: str | None = None
    active_keys: tuple[str, ...] = ()
    segment_start_idx: int = 0
    segment_end_idx: int = 0

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class RoundtableCameraSegment:
    start_s: float
    end_s: float
    camera_index: int
    reason: str
    focus_key: str | None = None

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class AudioLevelSegment:
    start_s: float
    end_s: float
    levels_db: dict[int, float]


@dataclass(frozen=True)
class RoundtableMotionEvent:
    reason: str
    time_s: float
    from_camera_index: int | None = None
    to_camera_index: int | None = None
    desired_camera_index: int | None = None
    moving_from_s: float | None = None
    moving_to_s: float | None = None


@dataclass
class RoundtablePlan:
    frame_states: list[RoundtableFrameState] = field(default_factory=list)
    speech_segments: list[RoundtableSpeechSegment] = field(default_factory=list)
    conversation_phases: list[ConversationPhase] = field(default_factory=list)
    camera_segments: list[RoundtableCameraSegment] = field(default_factory=list)
    motion_events: list[RoundtableMotionEvent] = field(default_factory=list)
    audio_open_intervals_raw_s: dict[int, list[tuple[float, float]]] = field(default_factory=dict)
    audio_open_intervals_s: dict[int, list[tuple[float, float]]] = field(default_factory=dict)
    audio_level_segments: list[AudioLevelSegment] = field(default_factory=list)
    camera_balance_stats: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)


def build_roundtable_plan(
    participants: list[ParticipantSpec],
    activities: dict[str, SpeakerActivity],
    hop_s: float,
    config: Roundtable4CamConfig,
    motion_plans: dict[int, CameraMotionPlan] | None = None,
) -> RoundtablePlan:
    """Build full 4-cam switching plan from per-participant speech activities."""
    config.validate()
    _validate_participants(participants)
    frame_states = build_frame_states(participants, activities, config)
    speech_segments = build_speech_segments(frame_states, hop_s)
    conversation_phases = build_conversation_phases(speech_segments, config)
    camera_segments = build_camera_segments_from_phases(conversation_phases, speech_segments, config)
    camera_segments = _insert_reestablishing_wide(camera_segments, config)
    camera_segments = _apply_overlap_sticky(camera_segments, speech_segments, config)
    camera_segments = _merge_camera_segments(camera_segments)
    camera_segments = _enforce_min_camera_duration(camera_segments, config)
    camera_segments = _merge_camera_segments(camera_segments)
    motion_events: list[RoundtableMotionEvent] = []
    if motion_plans:
        camera_segments, motion_events = _apply_motion_guard_to_camera_segments(
            camera_segments,
            config,
            motion_plans,
        )
        camera_segments = _merge_camera_segments(camera_segments)
    raw_audio_open_intervals_s, audio_open_intervals_s, audio_level_segments = build_audio_plan(
        participants,
        frame_states,
        hop_s,
        config,
    )
    camera_balance_stats = _compute_camera_balance_stats(camera_segments, conversation_phases, config)
    diagnostics = {
        "participants": [
            {
                "key": part.key,
                "label": part.label,
                "role": part.role,
                "audio_track_index": part.audio_track_index,
            }
            for part in participants
        ],
        "state_counts": _count_states(frame_states),
        "phase_counts": _count_phases(conversation_phases),
        "motion_enabled": bool(motion_plans),
        "motion_events": len(motion_events),
        "motion_entry_lookahead_s": ROUNDTABLE_MOTION_ENTRY_LOOKAHEAD_S,
    }
    return RoundtablePlan(
        frame_states=frame_states,
        speech_segments=speech_segments,
        conversation_phases=conversation_phases,
        camera_segments=camera_segments,
        motion_events=motion_events,
        audio_open_intervals_raw_s=raw_audio_open_intervals_s,
        audio_open_intervals_s=audio_open_intervals_s,
        audio_level_segments=audio_level_segments,
        camera_balance_stats=camera_balance_stats,
        diagnostics=diagnostics,
    )


def build_frame_states(
    participants: list[ParticipantSpec],
    activities: dict[str, SpeakerActivity],
    config: Roundtable4CamConfig,
) -> list[RoundtableFrameState]:
    """Combine per-speaker activity into generalized 4-person states."""
    if not participants:
        return []

    frame_count = min(len(activities[part.key].frames) for part in participants)
    if frame_count <= 0:
        return []

    result: list[RoundtableFrameState] = []
    for idx in range(frame_count):
        active_keys = _active_keys_for_frame(participants, activities, idx, config)
        host_active = [key for key in active_keys if _role_for_key(participants, key) == "host"]
        guest_active = [key for key in active_keys if _role_for_key(participants, key) == "guest"]

        if not active_keys:
            state = RoundtableState.SILENCE
            focus_key = None
        elif host_active and not guest_active:
            state = RoundtableState.HOST_ONLY
            focus_key = host_active[0]
        elif not host_active and len(guest_active) == 1:
            state = RoundtableState.SINGLE_GUEST
            focus_key = guest_active[0]
        elif not host_active and len(guest_active) >= 2:
            state = RoundtableState.MULTI_GUEST
            focus_key = _dominant_guest(guest_active, activities, idx)
        elif host_active and len(guest_active) == 1:
            state = RoundtableState.HOST_PLUS_GUEST
            focus_key = guest_active[0]
        else:
            state = RoundtableState.GROUP_OVERLAP
            focus_key = _dominant_guest(guest_active, activities, idx)

        result.append(
            RoundtableFrameState(
                time_s=activities[participants[0].key].frames[idx].time_s,
                state=state,
                active_keys=tuple(active_keys),
                focus_key=focus_key,
            )
        )

    return result


def build_speech_segments(
    frame_states: list[RoundtableFrameState],
    hop_s: float,
) -> list[RoundtableSpeechSegment]:
    """Run-length encode frame states into longer segments."""
    if not frame_states:
        return []

    segments: list[RoundtableSpeechSegment] = []
    start_idx = 0
    cur_state = frame_states[0].state
    cur_focus = frame_states[0].focus_key
    cur_active = frame_states[0].active_keys

    for idx in range(1, len(frame_states)):
        frame = frame_states[idx]
        if (
            frame.state != cur_state
            or frame.focus_key != cur_focus
            or frame.active_keys != cur_active
        ):
            segments.append(
                RoundtableSpeechSegment(
                    start_s=start_idx * hop_s,
                    end_s=idx * hop_s,
                    state=cur_state,
                    active_keys=cur_active,
                    focus_key=cur_focus,
                )
            )
            start_idx = idx
            cur_state = frame.state
            cur_focus = frame.focus_key
            cur_active = frame.active_keys

    segments.append(
        RoundtableSpeechSegment(
            start_s=start_idx * hop_s,
            end_s=len(frame_states) * hop_s,
            state=cur_state,
            active_keys=cur_active,
            focus_key=cur_focus,
        )
    )
    return segments


def build_conversation_phases(
    speech_segments: list[RoundtableSpeechSegment],
    config: Roundtable4CamConfig,
) -> list[ConversationPhase]:
    """Collapse raw speech segments into broader conversation phases."""
    if not speech_segments:
        return []

    phases: list[ConversationPhase] = []
    idx = 0
    while idx < len(speech_segments):
        seg = speech_segments[idx]

        if seg.state == RoundtableState.SILENCE:
            if seg.duration_s < config.silence_timeout_s and phases:
                phases[-1] = _extend_phase(phases[-1], seg.end_s, idx + 1)
            else:
                phases.append(
                    ConversationPhase(
                        start_s=seg.start_s,
                        end_s=seg.end_s,
                        phase=ConversationPhaseType.PAUSE_RESET,
                        segment_start_idx=idx,
                        segment_end_idx=idx + 1,
                    )
                )
            idx += 1
            continue

        if _is_overlap_phase_segment(seg, config):
            phases.append(
                ConversationPhase(
                    start_s=seg.start_s,
                    end_s=seg.end_s,
                    phase=ConversationPhaseType.ALL_OVERLAP,
                    focus_key=seg.focus_key,
                    active_keys=seg.active_keys,
                    segment_start_idx=idx,
                    segment_end_idx=idx + 1,
                )
            )
            idx += 1
            continue

        if _is_guest_cluster_start(speech_segments, idx, config):
            start_idx = idx
            end_idx = idx + 1
            while end_idx < len(speech_segments) and _can_extend_guest_cluster(
                speech_segments,
                start_idx,
                end_idx,
                config,
            ):
                end_idx += 1
            cluster_segments = speech_segments[start_idx:end_idx]
            phases.append(
                ConversationPhase(
                    start_s=cluster_segments[0].start_s,
                    end_s=cluster_segments[-1].end_s,
                    phase=ConversationPhaseType.GUEST_CLUSTER,
                    focus_key=_dominant_focus_for_segments(cluster_segments),
                    active_keys=_combined_active_keys(cluster_segments),
                    segment_start_idx=start_idx,
                    segment_end_idx=end_idx,
                )
            )
            idx = end_idx
            continue

        if seg.state == RoundtableState.HOST_ONLY:
            prev_phase = phases[-1].phase if phases else None
            phase_type = (
                ConversationPhaseType.HOST_FOLLOWUP
                if prev_phase in {ConversationPhaseType.GUEST_CLUSTER, ConversationPhaseType.ALL_OVERLAP}
                else ConversationPhaseType.HOST_INTRO
            )
            phases.append(
                ConversationPhase(
                    start_s=seg.start_s,
                    end_s=seg.end_s,
                    phase=phase_type,
                    focus_key=seg.focus_key,
                    active_keys=seg.active_keys,
                    segment_start_idx=idx,
                    segment_end_idx=idx + 1,
                )
            )
            idx += 1
            continue

        fallback_phase = (
            ConversationPhaseType.ALL_OVERLAP
            if seg.state in {RoundtableState.HOST_PLUS_GUEST, RoundtableState.GROUP_OVERLAP}
            else ConversationPhaseType.GUEST_CLUSTER
        )
        phases.append(
            ConversationPhase(
                start_s=seg.start_s,
                end_s=seg.end_s,
                phase=fallback_phase,
                focus_key=seg.focus_key,
                active_keys=seg.active_keys,
                segment_start_idx=idx,
                segment_end_idx=idx + 1,
            )
        )
        idx += 1

    return _merge_conversation_phases(phases)


def build_camera_segments(
    speech_segments: list[RoundtableSpeechSegment],
    config: Roundtable4CamConfig,
) -> list[RoundtableCameraSegment]:
    """Public convenience wrapper for planning video cameras from speech segments."""
    phases = build_conversation_phases(speech_segments, config)
    camera_segments = build_camera_segments_from_phases(phases, speech_segments, config)
    camera_segments = _insert_reestablishing_wide(camera_segments, config)
    camera_segments = _apply_overlap_sticky(camera_segments, speech_segments, config)
    camera_segments = _merge_camera_segments(camera_segments)
    camera_segments = _enforce_min_camera_duration(camera_segments, config)
    return _merge_camera_segments(camera_segments)


def build_camera_segments_from_phases(
    conversation_phases: list[ConversationPhase],
    speech_segments: list[RoundtableSpeechSegment],
    config: Roundtable4CamConfig,
) -> list[RoundtableCameraSegment]:
    """Plan video cameras from high-level conversation phases."""
    if not conversation_phases:
        return []

    planned: list[RoundtableCameraSegment] = []
    for idx, phase in enumerate(conversation_phases):
        prev_phase = conversation_phases[idx - 1] if idx > 0 else None
        next_phase = conversation_phases[idx + 1] if idx + 1 < len(conversation_phases) else None

        if phase.phase == ConversationPhaseType.PAUSE_RESET:
            planned.append(_cam_segment(phase.start_s, phase.end_s, config.camera_all_wide, "pause_reset"))
            continue

        if phase.phase == ConversationPhaseType.ALL_OVERLAP:
            planned.append(
                _cam_segment(
                    phase.start_s,
                    phase.end_s,
                    config.camera_all_wide,
                    "all_overlap",
                    phase.focus_key,
                )
            )
            continue

        if phase.phase in {ConversationPhaseType.HOST_INTRO, ConversationPhaseType.HOST_FOLLOWUP}:
            if _should_keep_guest_side_for_host_phase(
                phase,
                prev_phase,
                next_phase,
                planned,
                conversation_phases,
                config,
            ):
                planned.append(
                    _cam_segment(
                        phase.start_s,
                        phase.end_s,
                        config.camera_guests_wide,
                        "guest_cluster",
                        _phase_focus_for_neighbors(prev_phase, next_phase),
                    )
                )
                continue

            if next_phase and next_phase.phase == ConversationPhaseType.GUEST_CLUSTER and phase.duration_s > 1.6:
                host_end = min(phase.end_s, phase.start_s + 1.6)
                planned.append(
                    _cam_segment(
                        phase.start_s,
                        host_end,
                        config.camera_host_close,
                        phase.phase.value,
                        phase.focus_key,
                    )
                )
                if host_end < phase.end_s:
                    planned.append(
                        _cam_segment(
                            host_end,
                            phase.end_s,
                            config.camera_guests_wide,
                            "guest_cluster_prepare",
                            next_phase.focus_key,
                        )
                    )
                continue

            planned.append(
                _cam_segment(
                    phase.start_s,
                    phase.end_s,
                    config.camera_host_close,
                    phase.phase.value,
                    phase.focus_key,
                )
            )
            continue

        phase_segments = speech_segments[phase.segment_start_idx:phase.segment_end_idx]
        planned.extend(_plan_guest_cluster_phase(phase, phase_segments, config))

    return _merge_camera_segments(planned)


def build_audio_plan(
    participants: list[ParticipantSpec],
    frame_states: list[RoundtableFrameState],
    hop_s: float,
    config: Roundtable4CamConfig,
) -> tuple[dict[int, list[tuple[float, float]]], dict[int, list[tuple[float, float]]], list[AudioLevelSegment]]:
    """Build raw/stabilized audible intervals and target audio levels per track."""
    if not frame_states:
        return {}, {}, []

    last_active_time: dict[str, float | None] = {part.key: None for part in participants}
    raw_flags: dict[int, list[bool]] = {part.audio_track_index: [] for part in participants}
    competing_flags: dict[int, list[bool]] = {part.audio_track_index: [] for part in participants}

    for frame in frame_states:
        active_any = bool(frame.active_keys)
        active_key_set = set(frame.active_keys)
        for part in participants:
            if part.key in frame.active_keys:
                last_active_time[part.key] = frame.time_s
                open_flag = True
            elif not active_any and config.audio_silence_opens_all_tracks:
                open_flag = True
            else:
                last_seen = last_active_time[part.key]
                since_last = float("inf") if last_seen is None else frame.time_s - last_seen
                open_flag = bool(
                    config.audio_keep_recently_active_open and since_last <= config.audio_recent_hold_s
                )
            raw_flags[part.audio_track_index].append(open_flag)
            competing_flags[part.audio_track_index].append(
                any(key != part.key for key in active_key_set)
            )

    stable_flags = {
        track_idx: stabilize_audio_open_flags(flags, competing_flags[track_idx], hop_s, config)
        for track_idx, flags in raw_flags.items()
    }
    raw_intervals = {
        track_idx: _bool_runs_to_intervals(flags, hop_s)
        for track_idx, flags in raw_flags.items()
    }
    stable_intervals = {
        track_idx: _bool_runs_to_intervals(flags, hop_s)
        for track_idx, flags in stable_flags.items()
    }
    level_segments = _build_audio_level_segments(stable_flags, hop_s, config)
    return raw_intervals, stable_intervals, level_segments


def stabilize_audio_open_flags(
    flags: list[bool],
    competing_flags: list[bool],
    hop_s: float,
    config: Roundtable4CamConfig,
) -> list[bool]:
    """Suppress false micro-open/micro-close audio intervals without changing detector output."""
    if not flags:
        return []

    result = list(flags)
    merge_gap_frames = max(1, ceil(config.audio_merge_gap_s / hop_s))
    min_on_frames = max(1, ceil(config.audio_min_on_s / hop_s))
    min_open_frames = max(1, ceil(config.audio_min_open_after_trigger_s / hop_s))
    release_frames = max(1, ceil(config.audio_release_hold_s / hop_s))
    min_closed_frames = max(1, ceil(config.audio_min_closed_s / hop_s))

    _fill_short_false_gaps(result, merge_gap_frames, competing_flags)
    _remove_short_true_runs(result, min_on_frames)
    _extend_true_runs_to_min(result, min_open_frames, competing_flags)
    _extend_true_runs_tail(result, release_frames, competing_flags)
    _fill_short_false_gaps(result, merge_gap_frames, competing_flags)
    _fill_short_false_gaps(result, min_closed_frames, competing_flags)
    _remove_short_true_runs(result, min_on_frames)
    return result


def _validate_participants(participants: list[ParticipantSpec]) -> None:
    host_count = sum(1 for part in participants if part.role == "host")
    guest_count = sum(1 for part in participants if part.role == "guest")
    if host_count != 1:
        raise ValueError(f"Expected exactly one host participant, got {host_count}")
    if guest_count != 3:
        raise ValueError(f"Expected exactly three guest participants, got {guest_count}")


def _active_keys_for_frame(
    participants: list[ParticipantSpec],
    activities: dict[str, SpeakerActivity],
    idx: int,
    config: Roundtable4CamConfig,
) -> list[str]:
    candidates = [part.key for part in participants if activities[part.key].frames[idx].is_active]
    if len(candidates) <= 1:
        return candidates

    levels = {
        key: activities[key].frames[idx].envelope_db
        for key in candidates
    }
    strongest = max(candidates, key=lambda key: levels[key])
    keep: list[str] = [strongest]
    for key in sorted(candidates, key=lambda item: levels[item], reverse=True):
        if key == strongest:
            continue
        if levels[strongest] - levels[key] < config.dominance_delta_db:
            keep.append(key)
    return sorted(set(keep))


def _dominant_guest(
    guest_keys: list[str],
    activities: dict[str, SpeakerActivity],
    idx: int,
) -> str | None:
    if not guest_keys:
        return None
    return max(guest_keys, key=lambda key: activities[key].frames[idx].envelope_db)


def _role_for_key(participants: list[ParticipantSpec], key: str) -> str:
    for part in participants:
        if part.key == key:
            return part.role
    raise KeyError(key)


def _is_overlap_phase_segment(seg: RoundtableSpeechSegment, config: Roundtable4CamConfig) -> bool:
    if seg.state == RoundtableState.GROUP_OVERLAP:
        return True
    return seg.state == RoundtableState.HOST_PLUS_GUEST and seg.duration_s >= config.all_overlap_trigger_s


def _is_guest_cluster_start(
    speech_segments: list[RoundtableSpeechSegment],
    idx: int,
    config: Roundtable4CamConfig,
) -> bool:
    seg = speech_segments[idx]
    if seg.state in {RoundtableState.SINGLE_GUEST, RoundtableState.MULTI_GUEST}:
        return True
    if (
        seg.state == RoundtableState.HOST_PLUS_GUEST
        and seg.duration_s < config.all_overlap_trigger_s
        and idx + 1 < len(speech_segments)
        and speech_segments[idx + 1].state in {RoundtableState.SINGLE_GUEST, RoundtableState.MULTI_GUEST}
    ):
        return True
    return False


def _can_extend_guest_cluster(
    speech_segments: list[RoundtableSpeechSegment],
    start_idx: int,
    candidate_idx: int,
    config: Roundtable4CamConfig,
) -> bool:
    seg = speech_segments[candidate_idx]
    if seg.state in {RoundtableState.SINGLE_GUEST, RoundtableState.MULTI_GUEST}:
        return True
    if seg.state == RoundtableState.HOST_ONLY and seg.duration_s < config.guest_cluster_bridge_host_s:
        return True
    if seg.state == RoundtableState.HOST_PLUS_GUEST and seg.duration_s < config.all_overlap_trigger_s:
        return True
    if seg.state == RoundtableState.SILENCE and seg.duration_s < min(0.25, config.silence_timeout_s):
        return True
    return False


def _extend_phase(phase: ConversationPhase, new_end_s: float, new_end_idx: int) -> ConversationPhase:
    return replace(phase, end_s=new_end_s, segment_end_idx=new_end_idx)


def _merge_conversation_phases(phases: list[ConversationPhase]) -> list[ConversationPhase]:
    if not phases:
        return []

    merged = [phases[0]]
    for phase in phases[1:]:
        prev = merged[-1]
        if (
            phase.phase == prev.phase
            and abs(phase.start_s - prev.end_s) < EPS
            and phase.focus_key == prev.focus_key
        ):
            merged[-1] = ConversationPhase(
                start_s=prev.start_s,
                end_s=phase.end_s,
                phase=prev.phase,
                focus_key=prev.focus_key,
                active_keys=tuple(sorted(set(prev.active_keys) | set(phase.active_keys))),
                segment_start_idx=prev.segment_start_idx,
                segment_end_idx=phase.segment_end_idx,
            )
        else:
            merged.append(phase)
    return merged


def _dominant_focus_for_segments(segments: list[RoundtableSpeechSegment]) -> str | None:
    durations: dict[str, float] = {}
    for seg in segments:
        if seg.focus_key is None:
            continue
        durations[seg.focus_key] = durations.get(seg.focus_key, 0.0) + seg.duration_s
    if not durations:
        return None
    return max(durations.items(), key=lambda item: item[1])[0]


def _combined_active_keys(segments: list[RoundtableSpeechSegment]) -> tuple[str, ...]:
    combined: set[str] = set()
    for seg in segments:
        combined.update(seg.active_keys)
    return tuple(sorted(combined))


def _phase_focus_for_neighbors(
    prev_phase: ConversationPhase | None,
    next_phase: ConversationPhase | None,
) -> str | None:
    if next_phase and next_phase.focus_key:
        return next_phase.focus_key
    if prev_phase and prev_phase.focus_key:
        return prev_phase.focus_key
    return None


def _should_keep_guest_side_for_host_phase(
    phase: ConversationPhase,
    prev_phase: ConversationPhase | None,
    next_phase: ConversationPhase | None,
    planned: list[RoundtableCameraSegment],
    phases: list[ConversationPhase],
    config: Roundtable4CamConfig,
) -> bool:
    adjacent_guest_cluster = (
        (prev_phase and prev_phase.phase == ConversationPhaseType.GUEST_CLUSTER)
        or (next_phase and next_phase.phase == ConversationPhaseType.GUEST_CLUSTER)
    )
    if not adjacent_guest_cluster:
        return False

    if phase.duration_s < config.guest_cluster_bridge_host_s:
        return True

    if phase.duration_s < 1.2:
        last_host_end = _last_host_close_end(planned, config.camera_host_close)
        if last_host_end is not None and phase.start_s - last_host_end < 6.0:
            return True
        if _recent_host_share_exceeded(planned, phases, phase.start_s, config):
            return True
    return False


def _last_host_close_end(
    planned: list[RoundtableCameraSegment],
    host_camera_index: int,
) -> float | None:
    for seg in reversed(planned):
        if seg.camera_index == host_camera_index:
            return seg.end_s
    return None


def _recent_host_share_exceeded(
    planned: list[RoundtableCameraSegment],
    phases: list[ConversationPhase],
    now_s: float,
    config: Roundtable4CamConfig,
) -> bool:
    window_start = max(0.0, now_s - config.host_screen_share_cap_window_s)
    window_len = max(now_s - window_start, EPS)
    host_time = sum(
        _overlap_duration(seg.start_s, seg.end_s, window_start, now_s)
        for seg in planned
        if seg.camera_index == config.camera_host_close
    )
    guest_activity_time = sum(
        _overlap_duration(phase.start_s, phase.end_s, window_start, now_s)
        for phase in phases
        if phase.phase == ConversationPhaseType.GUEST_CLUSTER
    )
    if guest_activity_time < config.host_screen_share_cap_min_guest_s:
        return False
    return host_time / window_len > config.host_screen_share_cap_ratio


def _plan_guest_cluster_phase(
    phase: ConversationPhase,
    phase_segments: list[RoundtableSpeechSegment],
    config: Roundtable4CamConfig,
) -> list[RoundtableCameraSegment]:
    accents = _build_guest_accent_segments(phase, phase_segments, config)
    if not accents:
        return [
            _cam_segment(
                phase.start_s,
                phase.end_s,
                config.camera_guests_wide,
                "guest_cluster",
                phase.focus_key,
            )
        ]

    result: list[RoundtableCameraSegment] = []
    cursor = phase.start_s
    for accent in accents:
        if accent.start_s - cursor > EPS:
            result.append(
                _cam_segment(
                    cursor,
                    accent.start_s,
                    config.camera_guests_wide,
                    "guest_cluster",
                    phase.focus_key,
                )
            )
        result.append(accent)
        cursor = accent.end_s

    if phase.end_s - cursor > EPS:
        result.append(
            _cam_segment(
                cursor,
                phase.end_s,
                config.camera_guests_wide,
                "guest_cluster",
                phase.focus_key,
            )
        )
    return result


def _build_guest_accent_segments(
    phase: ConversationPhase,
    phase_segments: list[RoundtableSpeechSegment],
    config: Roundtable4CamConfig,
) -> list[RoundtableCameraSegment]:
    runs: list[tuple[str, float, float, float]] = []
    last_nonclean_end = phase.start_s
    run_focus: str | None = None
    run_start: float | None = None
    run_end = phase.start_s
    run_last_nonclean_end = phase.start_s

    for seg in phase_segments:
        if seg.state == RoundtableState.SINGLE_GUEST and seg.focus_key:
            if run_focus == seg.focus_key and run_start is not None and abs(seg.start_s - run_end) < EPS:
                run_end = seg.end_s
            else:
                if run_focus is not None and run_start is not None:
                    runs.append((run_focus, run_start, run_end, run_last_nonclean_end))
                run_focus = seg.focus_key
                run_start = seg.start_s
                run_end = seg.end_s
                run_last_nonclean_end = last_nonclean_end
        else:
            if run_focus is not None and run_start is not None:
                runs.append((run_focus, run_start, run_end, run_last_nonclean_end))
                run_focus = None
                run_start = None
            last_nonclean_end = seg.end_s

    if run_focus is not None and run_start is not None:
        runs.append((run_focus, run_start, run_end, run_last_nonclean_end))

    result: list[RoundtableCameraSegment] = []
    last_accent_end = float("-inf")
    min_useful = max(config.cam3_min_useful_after_ready_s, config.guest_close_min_duration_s)

    for focus_key, start_s, end_s, nonclean_end_s in runs:
        if end_s - start_s < config.guest_close_min_domination_s:
            continue

        accent_start = max(
            start_s,
            nonclean_end_s + config.guest_close_overlap_clear_s,
            phase.start_s + config.guest_close_side_settle_s + config.cam3_wait_timeout_s,
            last_accent_end + config.guest_close_cooldown_s if last_accent_end > 0 else start_s,
        )
        accent_end = min(end_s, accent_start + config.guest_close_max_continuous_s)
        if accent_end - accent_start < min_useful:
            continue

        result.append(
            _cam_segment(
                accent_start,
                accent_end,
                config.camera_guest_close,
                ConversationPhaseType.GUEST_ACCENT.value,
                focus_key,
            )
        )
        last_accent_end = accent_end

    return result


def _cam_segment(
    start_s: float,
    end_s: float,
    camera_index: int,
    reason: str,
    focus_key: str | None = None,
) -> RoundtableCameraSegment:
    return RoundtableCameraSegment(
        start_s=start_s,
        end_s=end_s,
        camera_index=camera_index,
        reason=reason,
        focus_key=focus_key,
    )


def _merge_camera_segments(
    segments: list[RoundtableCameraSegment],
) -> list[RoundtableCameraSegment]:
    if not segments:
        return []
    merged: list[RoundtableCameraSegment] = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        if (
            seg.camera_index == prev.camera_index
            and seg.reason == prev.reason
            and seg.focus_key == prev.focus_key
            and abs(seg.start_s - prev.end_s) < EPS
        ):
            merged[-1] = RoundtableCameraSegment(
                start_s=prev.start_s,
                end_s=seg.end_s,
                camera_index=prev.camera_index,
                reason=prev.reason,
                focus_key=prev.focus_key,
            )
        else:
            merged.append(seg)
    return merged


def _apply_overlap_sticky(
    segments: list[RoundtableCameraSegment],
    speech_segments: list[RoundtableSpeechSegment],
    config: Roundtable4CamConfig,
) -> list[RoundtableCameraSegment]:
    if not segments:
        return []

    result: list[RoundtableCameraSegment] = []
    max_end_s = speech_segments[-1].end_s if speech_segments else 0.0

    for seg in segments:
        if result and seg.end_s <= result[-1].end_s + EPS:
            continue
        if result and seg.start_s < result[-1].end_s - EPS:
            seg = replace(seg, start_s=result[-1].end_s)

        if seg.reason == "all_overlap" and _has_dense_overlap_cluster(
            speech_segments,
            seg.start_s,
            seg.end_s,
            config,
        ):
            seg = replace(seg, end_s=min(max_end_s, seg.end_s + config.all_overlap_sticky_s))

        if result and seg.start_s < result[-1].end_s - EPS:
            if seg.end_s <= result[-1].end_s + EPS:
                continue
            seg = replace(seg, start_s=result[-1].end_s)

        result.append(seg)

    return _merge_camera_segments(result)


def _has_dense_overlap_cluster(
    speech_segments: list[RoundtableSpeechSegment],
    start_s: float,
    end_s: float,
    config: Roundtable4CamConfig,
) -> bool:
    window_end = end_s + config.all_overlap_window_s
    overlap_islands = 0
    turns = 0
    prev_actor: str | None = None

    for seg in speech_segments:
        if seg.end_s <= start_s or seg.start_s >= window_end:
            continue
        if seg.state in {RoundtableState.GROUP_OVERLAP, RoundtableState.HOST_PLUS_GUEST}:
            overlap_islands += 1
        actor = _primary_actor_for_segment(seg)
        if actor and actor != prev_actor:
            if prev_actor is not None:
                turns += 1
            prev_actor = actor

    return overlap_islands >= config.all_overlap_min_islands or turns >= config.all_overlap_min_turns


def _primary_actor_for_segment(seg: RoundtableSpeechSegment) -> str | None:
    if seg.state == RoundtableState.HOST_ONLY:
        return "host"
    if seg.state in {RoundtableState.HOST_PLUS_GUEST, RoundtableState.GROUP_OVERLAP}:
        return "overlap"
    if seg.focus_key:
        return seg.focus_key
    return None


def _insert_reestablishing_wide(
    segments: list[RoundtableCameraSegment],
    config: Roundtable4CamConfig,
) -> list[RoundtableCameraSegment]:
    if len(segments) < 2:
        return segments

    result: list[RoundtableCameraSegment] = []
    last_all_wide_end = segments[0].end_s if segments[0].camera_index == config.camera_all_wide else 0.0
    turns_since_wide = 0
    prev_focus = segments[0].focus_key if segments[0].camera_index != config.camera_all_wide else None

    for idx, seg in enumerate(segments):
        if idx == 0:
            result.append(seg)
            continue

        if seg.camera_index == config.camera_all_wide:
            last_all_wide_end = seg.end_s
            turns_since_wide = 0
            prev_focus = None
            result.append(seg)
            continue

        if seg.focus_key and prev_focus and seg.focus_key != prev_focus:
            turns_since_wide += 1
        elif prev_focus is None and seg.focus_key:
            turns_since_wide += 1
        prev_focus = seg.focus_key

        enough_time = seg.start_s - last_all_wide_end >= config.reestablish_all_wide_interval_s
        enough_turns = turns_since_wide >= config.reestablish_min_turns
        enough_room = seg.duration_s >= config.reestablish_all_wide_duration_s + config.host_return_min_s

        if enough_time and enough_turns and enough_room:
            wide_end = seg.start_s + config.reestablish_all_wide_duration_s
            result.append(
                RoundtableCameraSegment(
                    start_s=seg.start_s,
                    end_s=wide_end,
                    camera_index=config.camera_all_wide,
                    reason="reestablish_wide",
                )
            )
            result.append(
                RoundtableCameraSegment(
                    start_s=wide_end,
                    end_s=seg.end_s,
                    camera_index=seg.camera_index,
                    reason=seg.reason,
                    focus_key=seg.focus_key,
                )
            )
            last_all_wide_end = wide_end
            turns_since_wide = 0
        else:
            result.append(seg)

    return result


def _segment_min_duration(
    seg: RoundtableCameraSegment,
    config: Roundtable4CamConfig,
) -> float:
    if seg.reason in {"all_overlap", "reestablish_wide"}:
        return config.overlap_min_hold_s
    if seg.reason == "pause_reset":
        return config.silence_timeout_s
    if seg.reason in {"host_intro", "host_followup"}:
        return config.host_return_min_s
    if seg.reason in {"guest_cluster_prepare", "guest_cluster"}:
        return config.shot_hold_time_s
    if seg.reason == ConversationPhaseType.GUEST_ACCENT.value:
        return config.guest_close_min_duration_s
    return config.shot_hold_time_s


def _enforce_min_camera_duration(
    segments: list[RoundtableCameraSegment],
    config: Roundtable4CamConfig,
) -> list[RoundtableCameraSegment]:
    if len(segments) <= 1:
        return segments

    result: list[RoundtableCameraSegment] = [segments[0]]
    for idx, seg in enumerate(segments[1:], start=1):
        min_dur = _segment_min_duration(seg, config)
        if seg.duration_s >= min_dur:
            result.append(seg)
            continue

        prev = result[-1] if result else None
        nxt = segments[idx + 1] if idx + 1 < len(segments) else None
        if prev is not None and (nxt is None or prev.duration_s >= (nxt.duration_s if nxt else 0.0)):
            result[-1] = RoundtableCameraSegment(
                start_s=prev.start_s,
                end_s=seg.end_s,
                camera_index=prev.camera_index,
                reason=prev.reason,
                focus_key=prev.focus_key,
            )
        elif nxt is not None:
            segments[idx + 1] = RoundtableCameraSegment(
                start_s=seg.start_s,
                end_s=nxt.end_s,
                camera_index=nxt.camera_index,
                reason=nxt.reason,
                focus_key=nxt.focus_key,
            )
        else:
            result.append(seg)
    return result


def _apply_motion_guard_to_camera_segments(
    segments: list[RoundtableCameraSegment],
    config: Roundtable4CamConfig,
    motion_plans: dict[int, CameraMotionPlan],
) -> tuple[list[RoundtableCameraSegment], list[RoundtableMotionEvent]]:
    if not segments:
        return [], []

    result: list[RoundtableCameraSegment] = []
    events: list[RoundtableMotionEvent] = []
    current_segment: RoundtableCameraSegment | None = None

    def _flush(end_s: float) -> None:
        nonlocal current_segment
        if current_segment is None:
            return
        if end_s <= current_segment.start_s + EPS:
            current_segment = None
            return
        result.append(
            RoundtableCameraSegment(
                start_s=current_segment.start_s,
                end_s=end_s,
                camera_index=current_segment.camera_index,
                reason=current_segment.reason,
                focus_key=current_segment.focus_key,
            )
        )
        current_segment = None

    def _open(start_s: float, camera_index: int, reason: str, focus_key: str | None) -> None:
        nonlocal current_segment
        current_segment = RoundtableCameraSegment(
            start_s=start_s,
            end_s=start_s,
            camera_index=camera_index,
            reason=reason,
            focus_key=focus_key,
        )

    current_camera: int | None = None

    for planned in segments:
        boundary_s = planned.start_s
        if current_segment is None:
            chosen_camera, chosen_reason, chosen_focus, boundary_events = _select_roundtable_camera_for_boundary(
                desired_segment=planned,
                current_camera=None,
                time_s=boundary_s,
                config=config,
                motion_plans=motion_plans,
            )
            events.extend(boundary_events)
            _open(boundary_s, chosen_camera, chosen_reason, chosen_focus)
            current_camera = chosen_camera
        else:
            chosen_camera, chosen_reason, chosen_focus, boundary_events = _select_roundtable_camera_for_boundary(
                desired_segment=planned,
                current_camera=current_camera,
                time_s=boundary_s,
                config=config,
                motion_plans=motion_plans,
            )
            events.extend(boundary_events)
            if (
                chosen_camera != current_camera
                or chosen_reason != current_segment.reason
                or chosen_focus != current_segment.focus_key
            ):
                _flush(boundary_s)
                _open(boundary_s, chosen_camera, chosen_reason, chosen_focus)
                current_camera = chosen_camera

        cursor_s = boundary_s
        while cursor_s < planned.end_s - EPS and current_camera is not None:
            current_motion = _find_camera_motion_interval_between(
                current_camera,
                cursor_s,
                planned.end_s,
                motion_plans,
            )
            if current_motion is None:
                break

            escape_s = max(cursor_s, current_motion.start_s)
            fallback_camera = _pick_roundtable_fallback_camera(
                time_s=escape_s,
                window_end_s=planned.end_s,
                config=config,
                motion_plans=motion_plans,
                preferred_cameras=(
                    planned.camera_index,
                    config.camera_all_wide,
                    config.camera_guests_wide,
                    config.camera_host_close,
                    config.camera_guest_close,
                ),
                exclude_cameras={current_camera},
            )
            if fallback_camera is None:
                events.append(
                    RoundtableMotionEvent(
                        reason="no_static_camera_available",
                        time_s=escape_s,
                        from_camera_index=current_camera,
                        desired_camera_index=planned.camera_index,
                        moving_from_s=current_motion.start_s,
                        moving_to_s=current_motion.end_s,
                    )
                )
                break

            events.append(
                RoundtableMotionEvent(
                    reason="escape_from_motion",
                    time_s=escape_s,
                    from_camera_index=current_camera,
                    to_camera_index=fallback_camera,
                    desired_camera_index=planned.camera_index,
                    moving_from_s=current_motion.start_s,
                    moving_to_s=current_motion.end_s,
                )
            )
            _flush(escape_s)
            _open(
                escape_s,
                fallback_camera,
                "motion_escape",
                planned.focus_key,
            )
            current_camera = fallback_camera
            cursor_s = escape_s

        if current_segment is not None:
            current_segment = replace(current_segment, end_s=planned.end_s)

    if current_segment is not None:
        _flush(current_segment.end_s)

    return result, events


def _select_roundtable_camera_for_boundary(
    *,
    desired_segment: RoundtableCameraSegment,
    current_camera: int | None,
    time_s: float,
    config: Roundtable4CamConfig,
    motion_plans: dict[int, CameraMotionPlan],
) -> tuple[int, str, str | None, list[RoundtableMotionEvent]]:
    desired_camera = desired_segment.camera_index
    desired_end_s = desired_segment.end_s
    events: list[RoundtableMotionEvent] = []

    desired_block = _find_roundtable_blocking_interval(
        desired_camera,
        time_s,
        desired_end_s,
        motion_plans,
    )
    if desired_block is None:
        return desired_camera, desired_segment.reason, desired_segment.focus_key, events

    if (
        current_camera is not None
        and _find_roundtable_blocking_interval(current_camera, time_s, desired_end_s, motion_plans) is None
    ):
        events.append(
            RoundtableMotionEvent(
                reason="planned_camera_blocked",
                time_s=time_s,
                from_camera_index=current_camera,
                to_camera_index=current_camera,
                desired_camera_index=desired_camera,
                moving_from_s=desired_block.start_s,
                moving_to_s=desired_block.end_s,
            )
        )
        return current_camera, "motion_hold", desired_segment.focus_key, events

    fallback_camera = _pick_roundtable_fallback_camera(
        time_s=time_s,
        window_end_s=desired_end_s,
        config=config,
        motion_plans=motion_plans,
        preferred_cameras=(
            config.camera_all_wide,
            config.camera_guests_wide,
            config.camera_host_close,
            config.camera_guest_close,
        ),
        exclude_cameras={desired_camera},
    )
    if fallback_camera is not None:
        events.append(
            RoundtableMotionEvent(
                reason="fallback_to_static_camera",
                time_s=time_s,
                from_camera_index=current_camera,
                to_camera_index=fallback_camera,
                desired_camera_index=desired_camera,
                moving_from_s=desired_block.start_s,
                moving_to_s=desired_block.end_s,
            )
        )
        return fallback_camera, "motion_fallback", desired_segment.focus_key, events

    events.append(
        RoundtableMotionEvent(
            reason="no_static_camera_available",
            time_s=time_s,
            from_camera_index=current_camera,
            to_camera_index=current_camera,
            desired_camera_index=desired_camera,
            moving_from_s=desired_block.start_s,
            moving_to_s=desired_block.end_s,
        )
    )
    fallback_camera = current_camera if current_camera is not None else desired_camera
    fallback_reason = desired_segment.reason if fallback_camera == desired_camera else "motion_forced_hold"
    return fallback_camera, fallback_reason, desired_segment.focus_key, events


def _find_roundtable_blocking_interval(
    camera_index: int,
    time_s: float,
    window_end_s: float,
    motion_plans: dict[int, CameraMotionPlan],
) -> MotionInterval | None:
    plan = motion_plans.get(camera_index)
    if plan is None:
        return None
    interval = plan.find_interval_at(time_s)
    if interval is not None:
        return interval
    lookahead_end_s = min(window_end_s, time_s + ROUNDTABLE_MOTION_ENTRY_LOOKAHEAD_S)
    if lookahead_end_s <= time_s + EPS:
        return None
    return plan.first_interval_between(time_s + EPS, lookahead_end_s)


def _find_camera_motion_interval_between(
    camera_index: int,
    start_s: float,
    end_s: float,
    motion_plans: dict[int, CameraMotionPlan],
) -> MotionInterval | None:
    plan = motion_plans.get(camera_index)
    if plan is None:
        return None
    interval = plan.find_interval_at(start_s)
    if interval is not None:
        return interval
    return plan.first_interval_between(start_s + EPS, end_s)


def _pick_roundtable_fallback_camera(
    *,
    time_s: float,
    window_end_s: float,
    config: Roundtable4CamConfig,
    motion_plans: dict[int, CameraMotionPlan],
    preferred_cameras: tuple[int, ...],
    exclude_cameras: set[int] | None = None,
) -> int | None:
    exclude_cameras = exclude_cameras or set()
    seen: set[int] = set()
    for camera_index in preferred_cameras:
        if camera_index in seen or camera_index in exclude_cameras:
            continue
        seen.add(camera_index)
        if _find_roundtable_blocking_interval(camera_index, time_s, window_end_s, motion_plans) is None:
            return camera_index
    return None


def _fill_short_false_gaps(flags: list[bool], max_gap_frames: int, competing_flags: list[bool]) -> None:
    idx = 0
    while idx < len(flags):
        if flags[idx]:
            idx += 1
            continue
        start = idx
        while idx < len(flags) and not flags[idx]:
            idx += 1
        end = idx
        if start == 0 or end >= len(flags):
            continue
        if (
            end - start <= max_gap_frames
            and flags[start - 1]
            and flags[end]
            and not any(competing_flags[start:end])
        ):
            for pos in range(start, end):
                flags[pos] = True


def _remove_short_true_runs(flags: list[bool], min_frames: int) -> None:
    for start, end in _true_runs(flags):
        if end - start < min_frames:
            for pos in range(start, end):
                flags[pos] = False


def _extend_true_runs_to_min(flags: list[bool], min_frames: int, competing_flags: list[bool]) -> None:
    runs = _true_runs(flags)
    for start, end in runs:
        if end - start >= min_frames:
            continue
        target_end = min(len(flags), start + min_frames)
        target_end = _cap_target_end_at_competing(target_end, end, competing_flags)
        for pos in range(end, target_end):
            flags[pos] = True


def _extend_true_runs_tail(flags: list[bool], add_frames: int, competing_flags: list[bool]) -> None:
    runs = _true_runs(flags)
    for _start, end in runs:
        target_end = min(len(flags), end + add_frames)
        target_end = _cap_target_end_at_competing(target_end, end, competing_flags)
        for pos in range(end, target_end):
            flags[pos] = True


def _true_runs(flags: list[bool]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    active = False
    start = 0
    for idx, flag in enumerate(flags):
        if flag and not active:
            start = idx
            active = True
        elif not flag and active:
            runs.append((start, idx))
            active = False
    if active:
        runs.append((start, len(flags)))
    return runs


def _cap_target_end_at_competing(target_end: int, start_idx: int, competing_flags: list[bool]) -> int:
    for idx in range(start_idx, min(target_end, len(competing_flags))):
        if competing_flags[idx]:
            return idx
    return target_end


def _build_audio_level_segments(
    stable_flags: dict[int, list[bool]],
    hop_s: float,
    config: Roundtable4CamConfig,
) -> list[AudioLevelSegment]:
    if not stable_flags:
        return []

    frame_count = max((len(flags) for flags in stable_flags.values()), default=0)
    if frame_count <= 0:
        return []

    current_levels: dict[int, float] | None = None
    start_idx = 0
    segments: list[AudioLevelSegment] = []

    for idx in range(frame_count):
        levels = {
            track_idx: (0.0 if idx < len(flags) and flags[idx] else config.audio_hard_mute_db)
            for track_idx, flags in stable_flags.items()
        }
        if current_levels is None:
            current_levels = levels
            start_idx = idx
        elif levels != current_levels:
            segments.append(
                AudioLevelSegment(
                    start_s=start_idx * hop_s,
                    end_s=idx * hop_s,
                    levels_db=current_levels,
                )
            )
            current_levels = levels
            start_idx = idx

    if current_levels is not None:
        segments.append(
            AudioLevelSegment(
                start_s=start_idx * hop_s,
                end_s=frame_count * hop_s,
                levels_db=current_levels,
            )
        )
    return segments


def _bool_runs_to_intervals(flags: list[bool], hop_s: float) -> list[tuple[float, float]]:
    if not flags:
        return []
    intervals: list[tuple[float, float]] = []
    active = False
    start_idx = 0
    for idx, flag in enumerate(flags):
        if flag and not active:
            start_idx = idx
            active = True
        elif not flag and active:
            intervals.append((start_idx * hop_s, idx * hop_s))
            active = False
    if active:
        intervals.append((start_idx * hop_s, len(flags) * hop_s))
    return intervals


def _count_states(frame_states: list[RoundtableFrameState]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for frame in frame_states:
        counts[frame.state.value] = counts.get(frame.state.value, 0) + 1
    return counts


def _count_phases(phases: list[ConversationPhase]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for phase in phases:
        counts[phase.phase.value] = counts.get(phase.phase.value, 0) + 1
    return counts


def _compute_camera_balance_stats(
    camera_segments: list[RoundtableCameraSegment],
    phases: list[ConversationPhase],
    config: Roundtable4CamConfig,
) -> dict:
    if not camera_segments:
        return {
            "by_angle_seconds": {},
            "by_reason_seconds": {},
            "global_host_share": 0.0,
            "global_guest_side_share": 0.0,
            "rolling_window_s": config.host_screen_share_cap_window_s,
            "rolling_host_share_max": 0.0,
            "rolling_host_share_avg": 0.0,
            "rolling_guest_side_share_max": 0.0,
            "rolling_guest_side_share_avg": 0.0,
        }

    total_s = camera_segments[-1].end_s
    by_angle: dict[str, float] = {}
    by_reason: dict[str, float] = {}
    for seg in camera_segments:
        duration = seg.duration_s
        by_angle[str(seg.camera_index)] = by_angle.get(str(seg.camera_index), 0.0) + duration
        by_reason[seg.reason] = by_reason.get(seg.reason, 0.0) + duration

    host_total = by_angle.get(str(config.camera_host_close), 0.0)
    guest_total = by_angle.get(str(config.camera_guests_wide), 0.0) + by_angle.get(str(config.camera_guest_close), 0.0)
    anchors = sorted({seg.end_s for seg in camera_segments})
    host_shares: list[float] = []
    guest_shares: list[float] = []
    for anchor in anchors:
        window_start = max(0.0, anchor - config.host_screen_share_cap_window_s)
        window_len = max(anchor - window_start, EPS)
        host_s = sum(
            _overlap_duration(seg.start_s, seg.end_s, window_start, anchor)
            for seg in camera_segments
            if seg.camera_index == config.camera_host_close
        )
        guest_s = sum(
            _overlap_duration(seg.start_s, seg.end_s, window_start, anchor)
            for seg in camera_segments
            if seg.camera_index in {config.camera_guests_wide, config.camera_guest_close}
        )
        host_shares.append(host_s / window_len)
        guest_shares.append(guest_s / window_len)

    return {
        "by_angle_seconds": {key: round(value, 3) for key, value in sorted(by_angle.items())},
        "by_reason_seconds": {key: round(value, 3) for key, value in sorted(by_reason.items())},
        "global_host_share": round(host_total / max(total_s, EPS), 4),
        "global_guest_side_share": round(guest_total / max(total_s, EPS), 4),
        "rolling_window_s": config.host_screen_share_cap_window_s,
        "rolling_host_share_max": round(max(host_shares, default=0.0), 4),
        "rolling_host_share_avg": round(sum(host_shares) / max(len(host_shares), 1), 4),
        "rolling_guest_side_share_max": round(max(guest_shares, default=0.0), 4),
        "rolling_guest_side_share_avg": round(sum(guest_shares) / max(len(guest_shares), 1), 4),
        "guest_activity_seconds": round(
            sum(
                phase.duration_s
                for phase in phases
                if phase.phase in {ConversationPhaseType.GUEST_CLUSTER, ConversationPhaseType.ALL_OVERLAP}
            ),
            3,
        ),
    }


def _overlap_duration(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))
