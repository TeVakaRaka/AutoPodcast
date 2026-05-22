"""Camera and audio planner for the SAKHA AYMAKH 2-host + 1-guest mode."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field, replace
from enum import Enum
from math import ceil

import numpy as np

from autopodcast.core.camera_motion import CameraMotionPlan
from autopodcast.models.domain import SpeakerActivity

EPS = 1e-9
SAKHA_MOTION_ENTRY_LOOKAHEAD_S = 1.5


class SakhaAymakhState(str, Enum):
    SILENCE = "silence"
    MAIN_HOST_ONLY = "main_host_only"
    COHOST_ONLY = "cohost_only"
    GUEST_ONLY = "guest_only"
    MAIN_HOST_COHOST = "main_host_cohost"
    MAIN_HOST_GUEST = "main_host_guest"
    COHOST_GUEST = "cohost_guest"
    ALL_OVERLAP = "all_overlap"


@dataclass(frozen=True)
class SakhaParticipantSpec:
    key: str
    label: str
    role: str  # "main_host" | "cohost" | "guest"
    audio_track_index: int


@dataclass(frozen=True)
class SakhaAymakhConfig:
    camera_main_host_close: int
    camera_guest_close: int
    camera_pair_wide: int
    camera_all_wide: int
    dominance_delta_db: float = 6.0
    shot_hold_time_s: float = 1.2
    overlap_min_hold_s: float = 0.8
    silence_timeout_s: float = 0.8
    max_solo_hold_s: float = 60.0
    solo_cutaway_interval_s: float = 45.0
    guest_cutaway_interval_s: float = 24.0
    cutaway_duration_s: float = 2.2
    cut_search_window_s: float = 5.0
    forced_min_drop_db: float = 3.0
    reestablish_all_wide_interval_s: float = 25.0
    reestablish_all_wide_duration_s: float = 1.8
    reestablish_min_turns: int = 3
    audio_min_on_s: float = 0.28
    audio_merge_gap_s: float = 0.22
    audio_min_open_after_trigger_s: float = 0.55
    audio_release_hold_s: float = 0.30
    audio_min_closed_s: float = 0.35
    audio_pre_roll_s: float = 0.24
    audio_post_roll_s: float = 0.12
    audio_hard_mute_db: float = -96.0
    audio_silence_policy: str = "hold-last"
    debleed_enabled: bool = True
    debleed_overlap_margin_db: float = 3.0
    debleed_min_leader_score_db: float = -10.0
    debleed_score_snr_weight: float = 0.20
    debleed_guest_rescue_snr_db: float = 18.0
    debleed_guest_ambiguous_snr_margin_db: float = 8.0
    debleed_cohost_confidence_s: float = 0.60
    audio_clean_mode: str = "strict"
    waveform_window_s: float = 0.32
    waveform_step_s: float = 0.50
    waveform_max_lag_s: float = 0.012
    waveform_corr_bleed_threshold: float = 0.62
    waveform_independent_threshold: float = 0.35
    waveform_overlap_snr_db: float = 22.0
    waveform_downsample_hz: int = 1000
    strict_min_speaker_hold_s: float = 0.80
    strict_switch_margin_db: float = 6.0
    waveform_lead_tolerance_s: float = 0.001
    source_owner_bleed_residual_db: float = -9.0
    source_owner_residual_voice_snr_db: float = 16.0
    source_owner_min_corr: float = 0.42
    source_owner_tie_margin_db: float = 3.0
    # "calibrated" mode: loudness comparison after normalising every channel
    # to its own reference level. switch_margin is the hysteresis band — an
    # already-open channel keeps priority until a rival is louder by at least
    # this many dB. overlap_floor is how far below its own reference a second
    # channel may sit and still count as a real (overlapping) speaker.
    calibrated_switch_margin_db: float = 2.5
    calibrated_overlap_floor_db: float = 4.0

    def validate(self) -> None:
        for name, value in [
            ("shot_hold_time_s", self.shot_hold_time_s),
            ("overlap_min_hold_s", self.overlap_min_hold_s),
            ("silence_timeout_s", self.silence_timeout_s),
            ("max_solo_hold_s", self.max_solo_hold_s),
            ("solo_cutaway_interval_s", self.solo_cutaway_interval_s),
            ("guest_cutaway_interval_s", self.guest_cutaway_interval_s),
            ("cutaway_duration_s", self.cutaway_duration_s),
            ("cut_search_window_s", self.cut_search_window_s),
            ("reestablish_all_wide_interval_s", self.reestablish_all_wide_interval_s),
            ("reestablish_all_wide_duration_s", self.reestablish_all_wide_duration_s),
            ("audio_min_on_s", self.audio_min_on_s),
            ("audio_merge_gap_s", self.audio_merge_gap_s),
            ("audio_min_open_after_trigger_s", self.audio_min_open_after_trigger_s),
            ("audio_release_hold_s", self.audio_release_hold_s),
            ("audio_min_closed_s", self.audio_min_closed_s),
            ("audio_pre_roll_s", self.audio_pre_roll_s),
            ("audio_post_roll_s", self.audio_post_roll_s),
            ("debleed_cohost_confidence_s", self.debleed_cohost_confidence_s),
            ("waveform_window_s", self.waveform_window_s),
            ("waveform_step_s", self.waveform_step_s),
            ("waveform_max_lag_s", self.waveform_max_lag_s),
            ("waveform_overlap_snr_db", self.waveform_overlap_snr_db),
            ("strict_min_speaker_hold_s", self.strict_min_speaker_hold_s),
            ("strict_switch_margin_db", self.strict_switch_margin_db),
            ("source_owner_residual_voice_snr_db", self.source_owner_residual_voice_snr_db),
            ("source_owner_tie_margin_db", self.source_owner_tie_margin_db),
            ("calibrated_switch_margin_db", self.calibrated_switch_margin_db),
            ("calibrated_overlap_floor_db", self.calibrated_overlap_floor_db),
        ]:
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.audio_clean_mode not in {"strict", "balanced", "legacy", "calibrated"}:
            raise ValueError(
                "audio_clean_mode must be 'strict', 'balanced', 'legacy', or 'calibrated'"
            )
        if self.dominance_delta_db < 0:
            raise ValueError(f"dominance_delta_db must be >= 0, got {self.dominance_delta_db}")
        if self.forced_min_drop_db < 0:
            raise ValueError(f"forced_min_drop_db must be >= 0, got {self.forced_min_drop_db}")
        if self.reestablish_min_turns < 1:
            raise ValueError("reestablish_min_turns must be >= 1")
        if self.audio_silence_policy not in {"mute-all", "hold-last"}:
            raise ValueError("audio_silence_policy must be 'mute-all' or 'hold-last'")
        if self.debleed_overlap_margin_db < 0:
            raise ValueError(
                f"debleed_overlap_margin_db must be >= 0, got {self.debleed_overlap_margin_db}"
            )
        for name, value in [
            ("debleed_score_snr_weight", self.debleed_score_snr_weight),
            ("debleed_guest_rescue_snr_db", self.debleed_guest_rescue_snr_db),
            ("debleed_guest_ambiguous_snr_margin_db", self.debleed_guest_ambiguous_snr_margin_db),
            ("waveform_corr_bleed_threshold", self.waveform_corr_bleed_threshold),
            ("waveform_independent_threshold", self.waveform_independent_threshold),
            ("waveform_lead_tolerance_s", self.waveform_lead_tolerance_s),
            ("source_owner_min_corr", self.source_owner_min_corr),
        ]:
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if self.source_owner_bleed_residual_db > 0:
            raise ValueError("source_owner_bleed_residual_db must be <= 0")
        if self.waveform_corr_bleed_threshold > 1.0:
            raise ValueError("waveform_corr_bleed_threshold must be <= 1")
        if self.waveform_independent_threshold > 1.0:
            raise ValueError("waveform_independent_threshold must be <= 1")
        if self.source_owner_min_corr > 1.0:
            raise ValueError("source_owner_min_corr must be <= 1")
        if self.waveform_independent_threshold > self.waveform_corr_bleed_threshold:
            raise ValueError("waveform_independent_threshold must be <= waveform_corr_bleed_threshold")
        if self.waveform_downsample_hz < 200:
            raise ValueError("waveform_downsample_hz must be >= 200")
        for name, value in [
            ("camera_main_host_close", self.camera_main_host_close),
            ("camera_guest_close", self.camera_guest_close),
            ("camera_pair_wide", self.camera_pair_wide),
            ("camera_all_wide", self.camera_all_wide),
        ]:
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")


@dataclass(frozen=True)
class SakhaFrameState:
    time_s: float
    state: SakhaAymakhState
    active_keys: tuple[str, ...]
    focus_key: str | None = None
    debleed_reason: str | None = None


@dataclass(frozen=True)
class SakhaSpeechSegment:
    start_s: float
    end_s: float
    state: SakhaAymakhState
    active_keys: tuple[str, ...]
    focus_key: str | None = None

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class SakhaCameraSegment:
    start_s: float
    end_s: float
    camera_index: int
    reason: str
    focus_key: str | None = None

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class SakhaAudioLevelSegment:
    start_s: float
    end_s: float
    levels_db: dict[int, float]


@dataclass(frozen=True)
class SakhaDebleedProfile:
    key: str
    floor_db: float
    reference_db: float


@dataclass(frozen=True)
class SakhaDebleedFrameScore:
    key: str
    envelope_db: float
    normalized_db: float
    snr_db: float
    score_db: float


@dataclass(frozen=True)
class SakhaWaveformSimilarity:
    key_a: str
    key_b: str
    correlation: float
    lag_s: float
    available: bool = True


@dataclass(frozen=True)
class SakhaResidualEvidence:
    owner_key: str
    target_key: str
    correlation: float
    lag_s: float
    scale: float
    target_rms_db: float
    residual_rms_db: float
    residual_to_target_db: float
    available: bool = True


@dataclass(frozen=True)
class SakhaMotionEvent:
    reason: str
    time_s: float
    from_camera_index: int | None = None
    to_camera_index: int | None = None
    desired_camera_index: int | None = None
    moving_from_s: float | None = None
    moving_to_s: float | None = None


@dataclass
class SakhaAymakhPlan:
    frame_states: list[SakhaFrameState] = field(default_factory=list)
    speech_segments: list[SakhaSpeechSegment] = field(default_factory=list)
    camera_segments: list[SakhaCameraSegment] = field(default_factory=list)
    motion_events: list[SakhaMotionEvent] = field(default_factory=list)
    audio_open_intervals_raw_s: dict[int, list[tuple[float, float]]] = field(default_factory=dict)
    audio_open_intervals_s: dict[int, list[tuple[float, float]]] = field(default_factory=dict)
    audio_level_segments: list[SakhaAudioLevelSegment] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def config_from_controls(
    *,
    camera_main_host_close: int,
    camera_guest_close: int,
    camera_pair_wide: int,
    camera_all_wide: int,
    temperature: float = 50.0,
    cut_intensity: float = 50.0,
    max_solo_hold_s: float = 60.0,
) -> SakhaAymakhConfig:
    """Create a conservative config from human-facing 0..100 controls."""
    temp = _clamp01(temperature / 100.0)
    intensity = _clamp01(cut_intensity / 100.0)
    return SakhaAymakhConfig(
        camera_main_host_close=camera_main_host_close,
        camera_guest_close=camera_guest_close,
        camera_pair_wide=camera_pair_wide,
        camera_all_wide=camera_all_wide,
        dominance_delta_db=_lerp(8.0, 4.0, temp),
        shot_hold_time_s=_lerp(1.8, 0.75, intensity),
        silence_timeout_s=_lerp(1.1, 0.55, temp),
        max_solo_hold_s=max_solo_hold_s,
        solo_cutaway_interval_s=min(max_solo_hold_s, _lerp(60.0, 28.0, intensity)),
        guest_cutaway_interval_s=min(max_solo_hold_s, _lerp(42.0, 14.0, intensity)),
        cutaway_duration_s=_lerp(1.6, 2.8, intensity),
        cut_search_window_s=_lerp(3.0, 8.0, temp),
        forced_min_drop_db=_lerp(5.0, 1.5, temp),
        audio_min_on_s=_lerp(0.36, 0.18, temp),
        audio_min_open_after_trigger_s=_lerp(0.65, 0.42, temp),
        debleed_overlap_margin_db=_lerp(2.0, 4.0, temp),
        debleed_min_leader_score_db=_lerp(-8.0, -12.0, temp),
        debleed_guest_rescue_snr_db=_lerp(16.0, 20.0, temp),
        debleed_guest_ambiguous_snr_margin_db=_lerp(10.0, 6.0, temp),
        debleed_cohost_confidence_s=_lerp(0.80, 0.45, temp),
    )


def build_sakha_aimakh_plan(
    participants: list[SakhaParticipantSpec],
    activities: dict[str, SpeakerActivity],
    hop_s: float,
    config: SakhaAymakhConfig,
    motion_plans: dict[int, CameraMotionPlan] | None = None,
    audio_arrays_by_key: dict[str, np.ndarray] | None = None,
    audio_sample_rate: int | None = None,
) -> SakhaAymakhPlan:
    config.validate()
    _validate_participants(participants)
    debleed_profiles = _build_debleed_profiles(participants, activities, config)
    waveform_gate = _build_waveform_gate_context(
        audio_arrays_by_key,
        audio_sample_rate,
        config,
    )
    frame_states = build_frame_states(
        participants,
        activities,
        config,
        hop_s=hop_s,
        debleed_profiles=debleed_profiles,
        waveform_gate=waveform_gate,
    )
    speech_segments = build_speech_segments(frame_states, hop_s)
    camera_segments = build_camera_segments_from_speech(
        speech_segments,
        activities,
        config,
    )
    camera_segments = _insert_reestablishing_wide(camera_segments, config)
    camera_segments = _merge_camera_segments(camera_segments)
    camera_segments = _enforce_min_camera_duration(camera_segments, config)
    camera_segments = _merge_camera_segments(camera_segments)
    camera_segments = _enforce_max_visible_hold(camera_segments, config)
    camera_segments = _merge_camera_segments(camera_segments)

    motion_events: list[SakhaMotionEvent] = []
    if motion_plans:
        camera_segments, motion_events = _apply_motion_guard(
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
    debleed_diagnostics = _build_debleed_diagnostics(
        participants,
        activities,
        frame_states,
        debleed_profiles,
        audio_open_intervals_s,
        hop_s,
        config,
    )

    return SakhaAymakhPlan(
        frame_states=frame_states,
        speech_segments=speech_segments,
        camera_segments=camera_segments,
        motion_events=motion_events,
        audio_open_intervals_raw_s=raw_audio_open_intervals_s,
        audio_open_intervals_s=audio_open_intervals_s,
        audio_level_segments=audio_level_segments,
        diagnostics={
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
            "motion_enabled": bool(motion_plans),
            "motion_events": len(motion_events),
            "motion_entry_lookahead_s": SAKHA_MOTION_ENTRY_LOOKAHEAD_S,
            "debleed": debleed_diagnostics,
            "waveform_gate": _build_waveform_gate_diagnostics(frame_states, hop_s, config, waveform_gate),
            "source_owner": _build_source_owner_diagnostics(
                participants,
                activities,
                frame_states,
                hop_s,
                config,
                waveform_gate,
            ),
            "leak_matrix": {} if waveform_gate is None else waveform_gate.leak_matrix(),
        },
    )


def build_frame_states(
    participants: list[SakhaParticipantSpec],
    activities: dict[str, SpeakerActivity],
    config: SakhaAymakhConfig,
    hop_s: float = 0.1,
    debleed_profiles: dict[str, SakhaDebleedProfile] | None = None,
    waveform_gate: "SakhaWaveformGateContext | None" = None,
) -> list[SakhaFrameState]:
    if not participants:
        return []

    frame_count = min(len(activities[part.key].frames) for part in participants)
    if frame_count <= 0:
        return []

    result: list[SakhaFrameState] = []
    profiles = debleed_profiles or _build_debleed_profiles(participants, activities, config)
    if config.audio_clean_mode == "calibrated":
        return _build_calibrated_frame_states(
            participants, activities, config, hop_s, profiles,
        )
    if config.audio_clean_mode != "legacy":
        if waveform_gate is not None and waveform_gate.available:
            return _build_source_owner_frame_states(
                participants,
                activities,
                config,
                hop_s,
                profiles,
                waveform_gate,
            )
        return _build_strict_frame_states(
            participants,
            activities,
            config,
            hop_s,
            profiles,
            waveform_gate,
        )

    for idx in range(frame_count):
        active_keys, debleed_reason = _active_keys_for_frame_with_reason(
            participants,
            activities,
            idx,
            config,
            profiles,
        )
        result.append(_make_frame_state(participants, activities, idx, active_keys, debleed_reason))

    return _apply_cohost_guest_hysteresis(result, participants, activities, profiles, hop_s, config)


def build_speech_segments(
    frame_states: list[SakhaFrameState],
    hop_s: float,
) -> list[SakhaSpeechSegment]:
    if not frame_states:
        return []

    segments: list[SakhaSpeechSegment] = []
    start_idx = 0
    cur_state = frame_states[0].state
    cur_focus = frame_states[0].focus_key
    cur_active = frame_states[0].active_keys

    for idx in range(1, len(frame_states)):
        frame = frame_states[idx]
        if frame.state != cur_state or frame.focus_key != cur_focus or frame.active_keys != cur_active:
            segments.append(
                SakhaSpeechSegment(
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
        SakhaSpeechSegment(
            start_s=start_idx * hop_s,
            end_s=len(frame_states) * hop_s,
            state=cur_state,
            active_keys=cur_active,
            focus_key=cur_focus,
        )
    )
    return segments


def build_camera_segments_from_speech(
    speech_segments: list[SakhaSpeechSegment],
    activities: dict[str, SpeakerActivity],
    config: SakhaAymakhConfig,
) -> list[SakhaCameraSegment]:
    result: list[SakhaCameraSegment] = []
    for seg in speech_segments:
        if seg.state == SakhaAymakhState.SILENCE:
            result.append(_cam_segment(seg.start_s, seg.end_s, config.camera_all_wide, "silence"))
        elif seg.state == SakhaAymakhState.MAIN_HOST_ONLY:
            result.extend(
                _plan_solo_segment(
                    seg,
                    activities,
                    primary_camera=config.camera_main_host_close,
                    cutaway_camera=config.camera_all_wide,
                    primary_reason="main_host_close",
                    cutaway_reason="main_host_cutaway",
                    interval_s=config.solo_cutaway_interval_s,
                    config=config,
                )
            )
        elif seg.state == SakhaAymakhState.COHOST_ONLY:
            result.extend(
                _plan_solo_segment(
                    seg,
                    activities,
                    primary_camera=config.camera_pair_wide,
                    cutaway_camera=config.camera_all_wide,
                    primary_reason="cohost_pair_wide",
                    cutaway_reason="cohost_cutaway",
                    interval_s=config.solo_cutaway_interval_s,
                    config=config,
                )
            )
        elif seg.state == SakhaAymakhState.GUEST_ONLY:
            result.extend(
                _plan_solo_segment(
                    seg,
                    activities,
                    primary_camera=config.camera_guest_close,
                    cutaway_camera=config.camera_pair_wide,
                    primary_reason="guest_close",
                    cutaway_reason="guest_pair_cutaway",
                    interval_s=config.guest_cutaway_interval_s,
                    config=config,
                )
            )
        elif seg.state == SakhaAymakhState.COHOST_GUEST:
            result.append(
                _cam_segment(
                    seg.start_s,
                    seg.end_s,
                    config.camera_pair_wide,
                    "cohost_guest_pair_wide",
                    seg.focus_key,
                )
            )
        else:
            result.append(
                _cam_segment(
                    seg.start_s,
                    seg.end_s,
                    config.camera_all_wide,
                    seg.state.value,
                    seg.focus_key,
                )
            )

    return _merge_camera_segments(result)


def build_audio_plan(
    participants: list[SakhaParticipantSpec],
    frame_states: list[SakhaFrameState],
    hop_s: float,
    config: SakhaAymakhConfig,
) -> tuple[dict[int, list[tuple[float, float]]], dict[int, list[tuple[float, float]]], list[SakhaAudioLevelSegment]]:
    if not frame_states:
        return {}, {}, []

    raw_flags: dict[int, list[bool]] = {part.audio_track_index: [] for part in participants}
    competing_flags: dict[int, list[bool]] = {part.audio_track_index: [] for part in participants}

    for frame in frame_states:
        active_key_set = set(frame.active_keys)
        for part in participants:
            raw_flags[part.audio_track_index].append(part.key in active_key_set)
            competing_flags[part.audio_track_index].append(
                any(key != part.key for key in active_key_set)
            )

    stable_flags = {
        track_idx: stabilize_audio_open_flags(flags, competing_flags[track_idx], hop_s, config)
        for track_idx, flags in raw_flags.items()
    }
    if config.audio_silence_policy == "hold-last":
        _apply_hold_last_audio(stable_flags, frame_states, participants)
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
    config: SakhaAymakhConfig,
) -> list[bool]:
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


def _plan_solo_segment(
    seg: SakhaSpeechSegment,
    activities: dict[str, SpeakerActivity],
    *,
    primary_camera: int,
    cutaway_camera: int,
    primary_reason: str,
    cutaway_reason: str,
    interval_s: float,
    config: SakhaAymakhConfig,
) -> list[SakhaCameraSegment]:
    interval_s = min(interval_s, config.max_solo_hold_s)
    if seg.duration_s <= interval_s + config.cutaway_duration_s + config.shot_hold_time_s:
        return [_cam_segment(seg.start_s, seg.end_s, primary_camera, primary_reason, seg.focus_key)]

    result: list[SakhaCameraSegment] = []
    cursor = seg.start_s
    activity = activities.get(seg.focus_key or "")
    while seg.end_s - cursor > interval_s + EPS:
        latest_cutaway_start_s = seg.end_s - config.cutaway_duration_s - config.shot_hold_time_s
        if latest_cutaway_start_s <= cursor + config.shot_hold_time_s + EPS:
            break
        target_s = min(cursor + interval_s, latest_cutaway_start_s)
        cutaway_start_s = _find_low_energy_time(
            activity,
            target_s,
            cursor + config.shot_hold_time_s,
            latest_cutaway_start_s,
            config,
        )
        result.append(
            _cam_segment(
                cursor,
                cutaway_start_s,
                primary_camera,
                primary_reason,
                seg.focus_key,
            )
        )
        cutaway_end_s = min(seg.end_s, cutaway_start_s + config.cutaway_duration_s)
        result.append(
            _cam_segment(
                cutaway_start_s,
                cutaway_end_s,
                cutaway_camera,
                cutaway_reason,
                seg.focus_key,
            )
        )
        cursor = cutaway_end_s

    if seg.end_s - cursor > EPS:
        result.append(
            _cam_segment(cursor, seg.end_s, primary_camera, primary_reason, seg.focus_key)
        )
    return result


def _find_low_energy_time(
    activity: SpeakerActivity | None,
    target_s: float,
    earliest_s: float,
    latest_s: float,
    config: SakhaAymakhConfig,
) -> float:
    target_s = min(max(target_s, earliest_s), latest_s)
    if activity is None or not activity.frames:
        return target_s

    frame_times = [frame.time_s for frame in activity.frames]
    window_start_s = max(earliest_s, target_s - config.cut_search_window_s * 0.25)
    window_end_s = min(latest_s, target_s + config.cut_search_window_s)
    start_idx = bisect_left(frame_times, window_start_s)
    end_idx = bisect_right(frame_times, window_end_s)
    if start_idx >= end_idx:
        return target_s

    frames = activity.frames[start_idx:end_idx]
    min_frame = min(frames, key=lambda frame: frame.envelope_db)
    peak_db = max(frame.envelope_db for frame in frames)
    if peak_db - min_frame.envelope_db >= config.forced_min_drop_db:
        return min(max(min_frame.time_s, earliest_s), latest_s)
    return target_s


def _validate_participants(participants: list[SakhaParticipantSpec]) -> None:
    expected = {"main_host", "cohost", "guest"}
    roles = [part.role for part in participants]
    for role in expected:
        count = roles.count(role)
        if count != 1:
            raise ValueError(f"Expected exactly one {role} participant, got {count}")
    if len(participants) != 3:
        raise ValueError(f"Expected exactly three participants, got {len(participants)}")


class SakhaWaveformGateContext:
    def __init__(
        self,
        audio_arrays_by_key: dict[str, np.ndarray],
        sample_rate: int,
        config: SakhaAymakhConfig,
    ) -> None:
        self.audio_arrays_by_key = audio_arrays_by_key
        self.sample_rate = sample_rate
        self.config = config
        self._cache: dict[tuple[str, str, int], SakhaWaveformSimilarity] = {}
        self._requested_pairs: dict[tuple[str, str], int] = {}

    @property
    def available(self) -> bool:
        return bool(self.audio_arrays_by_key) and self.sample_rate > 0

    def similarity(self, key_a: str, key_b: str, time_s: float) -> SakhaWaveformSimilarity | None:
        if not self.available or key_a == key_b:
            return None
        if key_a not in self.audio_arrays_by_key or key_b not in self.audio_arrays_by_key:
            return None

        first, second = sorted((key_a, key_b))
        bucket = int(max(0.0, time_s) / self.config.waveform_step_s)
        cache_key = (first, second, bucket)
        self._requested_pairs[(first, second)] = self._requested_pairs.get((first, second), 0) + 1
        if cache_key not in self._cache:
            center_s = (bucket + 0.5) * self.config.waveform_step_s
            self._cache[cache_key] = self._compute_similarity(first, second, center_s)
        metric = self._cache[cache_key]
        if key_a == first:
            return metric
        return SakhaWaveformSimilarity(
            key_a=key_a,
            key_b=key_b,
            correlation=metric.correlation,
            lag_s=-metric.lag_s,
            available=metric.available,
        )

    def _compute_similarity(self, key_a: str, key_b: str, center_s: float) -> SakhaWaveformSimilarity:
        arr_a = self.audio_arrays_by_key[key_a]
        arr_b = self.audio_arrays_by_key[key_b]
        x, y = self._window_pair(arr_a, arr_b, center_s)
        if len(x) < 16 or len(y) < 16:
            return SakhaWaveformSimilarity(key_a, key_b, 0.0, 0.0, available=False)

        x = x - float(np.mean(x))
        y = y - float(np.mean(y))
        x_norm = float(np.linalg.norm(x))
        y_norm = float(np.linalg.norm(y))
        if x_norm <= EPS or y_norm <= EPS:
            return SakhaWaveformSimilarity(key_a, key_b, 0.0, 0.0, available=False)

        stride = self._stride()
        max_lag = max(1, int(round(self.config.waveform_max_lag_s * self.sample_rate / stride)))
        best_corr = 0.0
        best_lag = 0
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                xs = x[-lag:]
                ys = y[:len(y) + lag]
            elif lag > 0:
                xs = x[:len(x) - lag]
                ys = y[lag:]
            else:
                xs = x
                ys = y
            if len(xs) < 16 or len(ys) < 16:
                continue
            denom = float(np.linalg.norm(xs) * np.linalg.norm(ys))
            if denom <= EPS:
                continue
            corr = float(np.dot(xs, ys) / denom)
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag

        lag_s = best_lag * stride / self.sample_rate
        return SakhaWaveformSimilarity(
            key_a=key_a,
            key_b=key_b,
            correlation=abs(best_corr),
            lag_s=lag_s,
            available=True,
        )

    def residual_evidence(
        self,
        owner_key: str,
        target_key: str,
        time_s: float,
    ) -> SakhaResidualEvidence | None:
        if not self.available or owner_key == target_key:
            return None
        if owner_key not in self.audio_arrays_by_key or target_key not in self.audio_arrays_by_key:
            return None
        metric = self.similarity(owner_key, target_key, time_s)
        if metric is None or not metric.available:
            return None

        bucket = int(max(0.0, time_s) / self.config.waveform_step_s)
        center_s = (bucket + 0.5) * self.config.waveform_step_s
        owner = self.audio_arrays_by_key[owner_key]
        target = self.audio_arrays_by_key[target_key]
        x, y = self._window_pair(owner, target, center_s)
        if len(x) < 16 or len(y) < 16:
            return SakhaResidualEvidence(
                owner_key,
                target_key,
                metric.correlation,
                metric.lag_s,
                0.0,
                -120.0,
                -120.0,
                0.0,
                available=False,
            )

        lag_samples = int(round(metric.lag_s * self.sample_rate / self._stride()))
        x, y = _aligned_by_lag(x, y, lag_samples)
        if len(x) < 16 or len(y) < 16:
            return SakhaResidualEvidence(
                owner_key,
                target_key,
                metric.correlation,
                metric.lag_s,
                0.0,
                -120.0,
                -120.0,
                0.0,
                available=False,
            )

        x = x - float(np.mean(x))
        y = y - float(np.mean(y))
        denom = float(np.dot(x, x))
        if denom <= EPS:
            return SakhaResidualEvidence(
                owner_key,
                target_key,
                metric.correlation,
                metric.lag_s,
                0.0,
                _signal_rms_db(y),
                _signal_rms_db(y),
                0.0,
                available=False,
            )

        scale = float(np.dot(y, x) / denom)
        residual = y - scale * x
        target_rms_db = _signal_rms_db(y)
        residual_rms_db = _signal_rms_db(residual)
        residual_to_target_db = residual_rms_db - target_rms_db
        return SakhaResidualEvidence(
            owner_key=owner_key,
            target_key=target_key,
            correlation=metric.correlation,
            lag_s=metric.lag_s,
            scale=scale,
            target_rms_db=target_rms_db,
            residual_rms_db=residual_rms_db,
            residual_to_target_db=residual_to_target_db,
            available=True,
        )

    def _stride(self) -> int:
        return max(1, int(round(self.sample_rate / self.config.waveform_downsample_hz)))

    def _window_pair(
        self,
        arr_a: np.ndarray,
        arr_b: np.ndarray,
        center_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        start = max(0, int(round((center_s - self.config.waveform_window_s * 0.5) * self.sample_rate)))
        end = min(len(arr_a), len(arr_b), int(round((center_s + self.config.waveform_window_s * 0.5) * self.sample_rate)))
        if end - start < max(16, int(self.sample_rate * 0.04)):
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        stride = self._stride()
        return (
            np.asarray(arr_a[start:end:stride], dtype=np.float64),
            np.asarray(arr_b[start:end:stride], dtype=np.float64),
        )

    def leak_matrix(self) -> dict:
        matrix: dict[str, dict] = {}
        by_pair: dict[tuple[str, str], list[SakhaWaveformSimilarity]] = {}
        for key_a, key_b, _bucket in self._cache:
            pair = (key_a, key_b)
            by_pair.setdefault(pair, []).append(self._cache[(key_a, key_b, _bucket)])
        for pair, metrics in sorted(by_pair.items()):
            corrs = [metric.correlation for metric in metrics if metric.available]
            if not corrs:
                continue
            pair_name = f"{pair[0]}:{pair[1]}"
            matrix[pair_name] = {
                "samples": len(corrs),
                "requests": self._requested_pairs.get(pair, 0),
                "avg_corr": round(sum(corrs) / len(corrs), 4),
                "max_corr": round(max(corrs), 4),
                "correlated_samples": sum(
                    1 for corr in corrs if corr >= _waveform_corr_bleed_threshold(self.config)
                ),
                "independent_samples": sum(
                    1 for corr in corrs if corr <= _waveform_independent_threshold(self.config)
                ),
            }
        return matrix


def _build_waveform_gate_context(
    audio_arrays_by_key: dict[str, np.ndarray] | None,
    audio_sample_rate: int | None,
    config: SakhaAymakhConfig,
) -> SakhaWaveformGateContext | None:
    if config.audio_clean_mode == "legacy":
        return None
    if not audio_arrays_by_key or not audio_sample_rate:
        return None
    return SakhaWaveformGateContext(audio_arrays_by_key, audio_sample_rate, config)


def _signal_rms_db(values: np.ndarray) -> float:
    if len(values) == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(values))))
    return 20.0 * float(np.log10(max(rms, 1e-10)))


def _aligned_by_lag(
    source: np.ndarray,
    target: np.ndarray,
    lag_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    if lag_samples < 0:
        return source[-lag_samples:], target[:len(target) + lag_samples]
    if lag_samples > 0:
        return source[:len(source) - lag_samples], target[lag_samples:]
    return source, target


def _build_strict_frame_states(
    participants: list[SakhaParticipantSpec],
    activities: dict[str, SpeakerActivity],
    config: SakhaAymakhConfig,
    hop_s: float,
    profiles: dict[str, SakhaDebleedProfile],
    waveform_gate: SakhaWaveformGateContext | None,
) -> list[SakhaFrameState]:
    frame_count = min(len(activities[part.key].frames) for part in participants)
    result: list[SakhaFrameState] = []
    current_focus: str | None = None
    current_focus_start_idx = 0

    for idx in range(frame_count):
        active_keys, reason, scores = _strict_active_keys_for_frame(
            participants,
            activities,
            idx,
            config,
            profiles,
            waveform_gate,
            current_focus,
        )
        active_keys, reason = _apply_strict_sticky_focus(
            active_keys,
            reason,
            scores,
            current_focus,
            current_focus_start_idx,
            idx,
            hop_s,
            config,
        )
        frame = _make_frame_state(participants, activities, idx, active_keys, reason)
        result.append(frame)

        if len(frame.active_keys) == 1:
            key = frame.active_keys[0]
            if key != current_focus:
                current_focus = key
                current_focus_start_idx = idx
        elif len(frame.active_keys) > 1:
            focus = frame.focus_key
            if focus is not None and focus != current_focus:
                current_focus = focus
                current_focus_start_idx = idx

    return result


def _calibrated_normalized_db(
    key: str,
    activities: dict[str, SpeakerActivity],
    idx: int,
    profiles: dict[str, SakhaDebleedProfile],
) -> float:
    """Channel level relative to that speaker's own reference level.

    Subtracting each channel's own 90th-percentile speaking level cancels any
    fixed gain offset: a permanently hot microphone gets a high reference, so
    its bleed still lands well below 0 and cannot win on raw loudness.
    """
    envelope_db = activities[key].frames[idx].envelope_db
    profile = profiles.get(key)
    if profile is None:
        return envelope_db
    return envelope_db - profile.reference_db


def _pick_calibrated_owner(
    raw_candidates: list[str],
    normalized: dict[str, float],
    current_owner: str | None,
    config: SakhaAymakhConfig,
) -> tuple[list[str], str]:
    """Resolve simultaneous raw detections by normalised loudness + hysteresis."""
    best = max(raw_candidates, key=lambda key: normalized[key])
    owner = best
    # Hysteresis: the already-open channel keeps priority until a rival beats
    # it by a confident margin — this stops the edit chattering on every small
    # loudness wobble or breath.
    if (
        current_owner in normalized
        and normalized[best] - normalized[current_owner] < config.calibrated_switch_margin_db
    ):
        owner = current_owner
    # Genuine overlap: any other channel still close to its own reference is a
    # person actually talking, not bleed (bleed sits far below 0).
    overlap = [
        key
        for key in raw_candidates
        if key != owner and normalized[key] >= -config.calibrated_overlap_floor_db
    ]
    if overlap:
        return sorted({owner, *overlap}), "calibrated_overlap"
    return [owner], "calibrated_bleed_suppressed"


def _build_calibrated_frame_states(
    participants: list[SakhaParticipantSpec],
    activities: dict[str, SpeakerActivity],
    config: SakhaAymakhConfig,
    hop_s: float,
    profiles: dict[str, SakhaDebleedProfile],
) -> list[SakhaFrameState]:
    """Frame states for the "calibrated" loudness mode.

    Stage 1 — per-channel speech detection (the VAD result already carried in
              ``activities``).
    Stage 2 — when several channels fire at once, the owner is the channel
              loudest relative to its own reference, with hysteresis so the
              open channel keeps priority until a rival is clearly louder.
    Stage 3 — seam clean-up runs later in :func:`build_audio_plan` and the
              camera-segment post-processing, shared with every mode.
    """
    frame_count = min(len(activities[part.key].frames) for part in participants)
    all_keys = [part.key for part in participants]
    result: list[SakhaFrameState] = []
    current_owner: str | None = None

    for idx in range(frame_count):
        raw_candidates = [
            key for key in all_keys if activities[key].frames[idx].is_active
        ]
        if not raw_candidates:
            result.append(_make_frame_state(participants, activities, idx, [], None))
            continue

        if len(raw_candidates) == 1:
            active_keys: list[str] = list(raw_candidates)
            reason = "calibrated_single_raw"
        else:
            normalized = {
                key: _calibrated_normalized_db(key, activities, idx, profiles)
                for key in raw_candidates
            }
            active_keys, reason = _pick_calibrated_owner(
                raw_candidates, normalized, current_owner, config,
            )

        frame = _make_frame_state(participants, activities, idx, active_keys, reason)
        result.append(frame)

        if len(frame.active_keys) == 1:
            current_owner = frame.active_keys[0]
        elif len(frame.active_keys) > 1 and frame.focus_key is not None:
            current_owner = frame.focus_key

    return result


def _build_source_owner_frame_states(
    participants: list[SakhaParticipantSpec],
    activities: dict[str, SpeakerActivity],
    config: SakhaAymakhConfig,
    hop_s: float,
    profiles: dict[str, SakhaDebleedProfile],
    waveform_gate: SakhaWaveformGateContext,
) -> list[SakhaFrameState]:
    frame_count = min(len(activities[part.key].frames) for part in participants)
    provisional: list[SakhaFrameState] = []
    current_focus: str | None = None
    current_focus_start_idx = 0

    for idx in range(frame_count):
        active_keys, reason, scores = _source_owner_active_keys_for_frame(
            participants,
            activities,
            idx,
            config,
            profiles,
            waveform_gate,
            current_focus,
        )
        active_keys, reason = _apply_strict_sticky_focus(
            active_keys,
            reason,
            scores,
            current_focus,
            current_focus_start_idx,
            idx,
            hop_s,
            config,
        )
        frame = _make_frame_state(participants, activities, idx, active_keys, reason)
        provisional.append(frame)

        if len(frame.active_keys) == 1:
            key = frame.active_keys[0]
            if key != current_focus:
                current_focus = key
                current_focus_start_idx = idx
        elif len(frame.active_keys) > 1:
            focus = frame.focus_key
            if focus is not None and focus != current_focus:
                current_focus = focus
                current_focus_start_idx = idx

    return _stabilize_source_owner_overlaps(provisional, participants, activities, hop_s, config)


def _source_owner_active_keys_for_frame(
    participants: list[SakhaParticipantSpec],
    activities: dict[str, SpeakerActivity],
    idx: int,
    config: SakhaAymakhConfig,
    profiles: dict[str, SakhaDebleedProfile],
    waveform_gate: SakhaWaveformGateContext,
    current_focus: str | None,
) -> tuple[list[str], str | None, dict[str, SakhaDebleedFrameScore]]:
    raw_candidates = [part.key for part in participants if activities[part.key].frames[idx].is_active]
    scores = _debleed_frame_scores([part.key for part in participants], activities, idx, profiles, config)
    if not raw_candidates:
        return [], None, scores
    if len(raw_candidates) == 1:
        return raw_candidates, "source_owner_single_raw", scores

    global_snr = max((scores.get(key).snr_db for key in raw_candidates if scores.get(key)), default=-120.0)
    if global_snr < max(8.0, config.debleed_guest_rescue_snr_db * 0.5):
        return [], "source_owner_global_silence", scores

    time_s = activities[raw_candidates[0]].frames[idx].time_s
    owner = _pick_source_owner(
        participants,
        raw_candidates,
        scores,
        waveform_gate,
        time_s,
        current_focus,
        config,
    )
    if owner is None:
        return [], "source_owner_no_owner", scores

    overlap_keys = _source_owner_overlap_keys(
        participants,
        raw_candidates,
        owner,
        scores,
        profiles,
        waveform_gate,
        time_s,
        config,
    )
    if overlap_keys:
        return sorted({owner, *overlap_keys}), "source_owner_true_overlap", scores

    reason = "source_owner_bleed_suppressed" if len(raw_candidates) > 1 else "source_owner_single"
    return [owner], reason, scores


def _pick_source_owner(
    participants: list[SakhaParticipantSpec],
    raw_candidates: list[str],
    scores: dict[str, SakhaDebleedFrameScore],
    waveform_gate: SakhaWaveformGateContext,
    time_s: float,
    current_focus: str | None,
    config: SakhaAymakhConfig,
) -> str | None:
    owner_scores: dict[str, float] = {}
    lead_votes: dict[str, int] = {key: 0 for key in raw_candidates}
    explain_votes: dict[str, int] = {key: 0 for key in raw_candidates}
    for key in raw_candidates:
        score = scores.get(key)
        if score is None:
            continue
        owner_scores[key] = score.score_db + 0.08 * min(score.snr_db, 60.0)

    for left_idx, key_a in enumerate(raw_candidates):
        for key_b in raw_candidates[left_idx + 1:]:
            metric = waveform_gate.similarity(key_a, key_b, time_s)
            if metric is None or not metric.available:
                continue
            if metric.correlation < _source_owner_min_corr(config):
                continue
            if metric.lag_s > config.waveform_lead_tolerance_s:
                lead_votes[key_a] += 1
            elif metric.lag_s < -config.waveform_lead_tolerance_s:
                lead_votes[key_b] += 1

            evidence_ab = waveform_gate.residual_evidence(key_a, key_b, time_s)
            evidence_ba = waveform_gate.residual_evidence(key_b, key_a, time_s)
            if _residual_is_bleed(evidence_ab, config):
                explain_votes[key_a] += 1
            if _residual_is_bleed(evidence_ba, config):
                explain_votes[key_b] += 1

    for key in list(owner_scores):
        owner_scores[key] += 24.0 * lead_votes.get(key, 0) + 2.0 * explain_votes.get(key, 0)
        if key == current_focus:
            owner_scores[key] += 2.0

    if not owner_scores:
        return None

    best_score = max(owner_scores.values())
    contenders = [
        key
        for key, value in owner_scores.items()
        if best_score - value <= config.source_owner_tie_margin_db
    ]
    return _source_owner_tiebreak(participants, contenders, owner_scores, current_focus)


def _source_owner_tiebreak(
    participants: list[SakhaParticipantSpec],
    contenders: list[str],
    owner_scores: dict[str, float],
    current_focus: str | None,
) -> str:
    role_by_key = {part.key: part.role for part in participants}
    cohost = next((key for key in contenders if role_by_key.get(key) == "cohost"), None)
    guest = next((key for key in contenders if role_by_key.get(key) == "guest"), None)
    if cohost is not None and guest is not None:
        return guest
    if current_focus in contenders:
        return current_focus
    priority = {"main_host": 0, "guest": 1, "cohost": 2}
    return min(
        contenders,
        key=lambda key: (
            priority.get(role_by_key.get(key, ""), 10),
            -owner_scores.get(key, -999.0),
        ),
    )


def _source_owner_overlap_keys(
    participants: list[SakhaParticipantSpec],
    raw_candidates: list[str],
    owner: str,
    scores: dict[str, SakhaDebleedFrameScore],
    profiles: dict[str, SakhaDebleedProfile],
    waveform_gate: SakhaWaveformGateContext,
    time_s: float,
    config: SakhaAymakhConfig,
) -> list[str]:
    overlap: list[str] = []
    for key in raw_candidates:
        if key == owner:
            continue
        score = scores.get(key)
        profile = profiles.get(key)
        evidence = waveform_gate.residual_evidence(owner, key, time_s)
        if score is None or profile is None or evidence is None or not evidence.available:
            continue
        residual_snr = evidence.residual_rms_db - profile.floor_db
        metric = waveform_gate.similarity(owner, key, time_s)
        correlation = 1.0 if metric is None else metric.correlation
        if (
            residual_snr >= _source_owner_residual_voice_snr_db(config)
            and evidence.residual_to_target_db >= _source_owner_overlap_residual_db(config)
            and correlation <= _source_owner_overlap_corr_threshold(config)
            and score.snr_db >= _waveform_overlap_snr_db(config)
        ):
            overlap.append(key)
    return overlap


def _stabilize_source_owner_overlaps(
    frame_states: list[SakhaFrameState],
    participants: list[SakhaParticipantSpec],
    activities: dict[str, SpeakerActivity],
    hop_s: float,
    config: SakhaAymakhConfig,
) -> list[SakhaFrameState]:
    min_frames = max(1, ceil(config.strict_min_speaker_hold_s / hop_s))
    result = list(frame_states)
    idx = 0
    while idx < len(result):
        if result[idx].debleed_reason != "source_owner_true_overlap" or len(result[idx].active_keys) <= 1:
            idx += 1
            continue
        start_idx = idx
        active = result[idx].active_keys
        while (
            idx < len(result)
            and result[idx].debleed_reason == "source_owner_true_overlap"
            and result[idx].active_keys == active
        ):
            idx += 1
        if idx - start_idx >= min_frames:
            continue
        owner = result[start_idx].focus_key or active[0]
        for pos in range(start_idx, idx):
            result[pos] = _make_frame_state(
                participants,
                activities,
                pos,
                [owner],
                "source_owner_overlap_too_short",
            )
    return result


def _residual_is_bleed(evidence: SakhaResidualEvidence | None, config: SakhaAymakhConfig) -> bool:
    return (
        evidence is not None
        and evidence.available
        and evidence.correlation >= _source_owner_min_corr(config)
        and evidence.residual_to_target_db <= _source_owner_bleed_residual_db(config)
    )


def _source_owner_min_corr(config: SakhaAymakhConfig) -> float:
    if config.audio_clean_mode == "balanced":
        return max(0.20, config.source_owner_min_corr - 0.08)
    return config.source_owner_min_corr


def _source_owner_bleed_residual_db(config: SakhaAymakhConfig) -> float:
    if config.audio_clean_mode == "balanced":
        return min(-4.0, config.source_owner_bleed_residual_db + 3.0)
    return config.source_owner_bleed_residual_db


def _source_owner_residual_voice_snr_db(config: SakhaAymakhConfig) -> float:
    if config.audio_clean_mode == "balanced":
        return max(10.0, config.source_owner_residual_voice_snr_db - 4.0)
    return config.source_owner_residual_voice_snr_db


def _source_owner_overlap_residual_db(config: SakhaAymakhConfig) -> float:
    return -5.0 if config.audio_clean_mode == "balanced" else -3.0


def _source_owner_overlap_corr_threshold(config: SakhaAymakhConfig) -> float:
    return min(0.72, _waveform_independent_threshold(config) + 0.18)


def _strict_active_keys_for_frame(
    participants: list[SakhaParticipantSpec],
    activities: dict[str, SpeakerActivity],
    idx: int,
    config: SakhaAymakhConfig,
    profiles: dict[str, SakhaDebleedProfile],
    waveform_gate: SakhaWaveformGateContext | None,
    current_focus: str | None,
) -> tuple[list[str], str | None, dict[str, SakhaDebleedFrameScore]]:
    candidates = [part.key for part in participants if activities[part.key].frames[idx].is_active]
    if len(candidates) <= 1:
        return candidates, None, _debleed_frame_scores(candidates, activities, idx, profiles, config)

    scores = _debleed_frame_scores(candidates, activities, idx, profiles, config)
    if not scores:
        return [], "strict_no_scores", scores
    ranked = sorted(scores, key=lambda key: scores[key].score_db, reverse=True)
    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    if second is None:
        return [top], "strict_single", scores

    time_s = activities[candidates[0]].frames[idx].time_s
    pair_metrics = _waveform_pair_metrics(candidates, waveform_gate, time_s)
    cluster_source = _pick_correlated_cluster_source(pair_metrics, scores, current_focus, config)
    if cluster_source is not None:
        return [cluster_source], "correlated_bleed_suppressed", scores

    metric = None if waveform_gate is None else waveform_gate.similarity(top, second, time_s)
    if metric is not None and metric.available:
        if metric.correlation >= _waveform_corr_bleed_threshold(config):
            source = _pick_correlated_source(top, second, metric, scores, current_focus, config)
            return [source], "correlated_bleed_suppressed", scores
        if (
            metric.correlation <= _waveform_independent_threshold(config)
            and scores[top].snr_db >= _waveform_overlap_snr_db(config)
            and scores[second].snr_db >= _waveform_overlap_snr_db(config)
        ):
            return sorted([top, second]), "true_overlap", scores
        return [_pick_strict_single(top, second, scores, current_focus, config)], "strict_single_ambiguous", scores

    gap_db = scores[top].score_db - scores[second].score_db
    fallback_overlap_margin = config.debleed_overlap_margin_db if config.audio_clean_mode == "balanced" else 1.5
    if (
        gap_db <= fallback_overlap_margin
        and scores[top].snr_db >= _waveform_overlap_snr_db(config)
        and scores[second].snr_db >= _waveform_overlap_snr_db(config)
    ):
        return sorted([top, second]), "envelope_true_overlap_fallback", scores
    return [_pick_strict_single(top, second, scores, current_focus, config)], "strict_single_fallback", scores


def _waveform_pair_metrics(
    candidates: list[str],
    waveform_gate: SakhaWaveformGateContext | None,
    time_s: float,
) -> dict[tuple[str, str], SakhaWaveformSimilarity]:
    if waveform_gate is None:
        return {}
    result: dict[tuple[str, str], SakhaWaveformSimilarity] = {}
    for left_idx, key_a in enumerate(candidates):
        for key_b in candidates[left_idx + 1:]:
            metric = waveform_gate.similarity(key_a, key_b, time_s)
            if metric is not None and metric.available:
                result[_pair_key(key_a, key_b)] = metric
    return result


def _pick_correlated_cluster_source(
    pair_metrics: dict[tuple[str, str], SakhaWaveformSimilarity],
    scores: dict[str, SakhaDebleedFrameScore],
    current_focus: str | None,
    config: SakhaAymakhConfig,
) -> str | None:
    if not pair_metrics:
        return None
    correlated = [
        metric
        for metric in pair_metrics.values()
        if metric.available and metric.correlation >= _waveform_corr_bleed_threshold(config)
    ]
    if len(correlated) < 2:
        return None

    votes: dict[str, int] = {}
    for metric in correlated:
        leader = _pick_correlated_source(
            metric.key_a,
            metric.key_b,
            metric,
            scores,
            current_focus,
            config,
        )
        votes[leader] = votes.get(leader, 0) + 1
    max_votes = max(votes.values(), default=0)
    leaders = [key for key, value in votes.items() if value == max_votes]
    if len(leaders) == 1 and max_votes >= 2:
        return leaders[0]
    if current_focus in leaders:
        return current_focus
    return max(leaders, key=lambda key: scores[key].score_db) if leaders else None


def _pair_key(key_a: str, key_b: str) -> tuple[str, str]:
    return tuple(sorted((key_a, key_b)))


def _pick_correlated_source(
    key_a: str,
    key_b: str,
    metric: SakhaWaveformSimilarity,
    scores: dict[str, SakhaDebleedFrameScore],
    current_focus: str | None,
    config: SakhaAymakhConfig,
) -> str:
    if metric.lag_s > config.waveform_lead_tolerance_s:
        return key_a
    if metric.lag_s < -config.waveform_lead_tolerance_s:
        return key_b
    return _pick_strict_single(key_a, key_b, scores, current_focus, config)


def _pick_strict_single(
    key_a: str,
    key_b: str,
    scores: dict[str, SakhaDebleedFrameScore],
    current_focus: str | None,
    config: SakhaAymakhConfig,
) -> str:
    if current_focus in {key_a, key_b}:
        other = key_b if current_focus == key_a else key_a
        if scores[other].score_db - scores[current_focus].score_db < config.strict_switch_margin_db:
            return current_focus
    return key_a if scores[key_a].score_db >= scores[key_b].score_db else key_b


def _apply_strict_sticky_focus(
    active_keys: list[str],
    reason: str | None,
    scores: dict[str, SakhaDebleedFrameScore],
    current_focus: str | None,
    current_focus_start_idx: int,
    idx: int,
    hop_s: float,
    config: SakhaAymakhConfig,
) -> tuple[list[str], str | None]:
    if len(active_keys) != 1 or current_focus is None or active_keys[0] == current_focus:
        return active_keys, reason
    if current_focus not in scores:
        return active_keys, reason
    proposed = active_keys[0]
    held_s = (idx - current_focus_start_idx) * hop_s
    score_gap = scores[proposed].score_db - scores[current_focus].score_db
    if held_s < config.strict_min_speaker_hold_s or score_gap < config.strict_switch_margin_db:
        return [current_focus], "sticky_focus_hold"
    return active_keys, reason


def _active_keys_for_frame(
    participants: list[SakhaParticipantSpec],
    activities: dict[str, SpeakerActivity],
    idx: int,
    config: SakhaAymakhConfig,
    debleed_profiles: dict[str, SakhaDebleedProfile] | None = None,
) -> list[str]:
    active_keys, _reason = _active_keys_for_frame_with_reason(
        participants,
        activities,
        idx,
        config,
        debleed_profiles,
    )
    return active_keys


def _active_keys_for_frame_with_reason(
    participants: list[SakhaParticipantSpec],
    activities: dict[str, SpeakerActivity],
    idx: int,
    config: SakhaAymakhConfig,
    debleed_profiles: dict[str, SakhaDebleedProfile] | None = None,
) -> tuple[list[str], str | None]:
    candidates = [part.key for part in participants if activities[part.key].frames[idx].is_active]
    if len(candidates) <= 1:
        return candidates, None

    if config.debleed_enabled and debleed_profiles:
        scores = _debleed_frame_scores(candidates, activities, idx, debleed_profiles, config)
        if scores:
            strongest = max(scores, key=lambda key: scores[key].score_db)
            strongest_score = scores[strongest].score_db
            if strongest_score < config.debleed_min_leader_score_db:
                keep: list[str] = []
                reason: str | None = "low_energy_suppressed"
            else:
                keep = [strongest]
                reason = None
                for key in sorted(scores, key=lambda item: scores[item].score_db, reverse=True):
                    if key == strongest:
                        continue
                    if strongest_score - scores[key].score_db <= config.debleed_overlap_margin_db:
                        keep.append(key)
                keep = sorted(set(keep))
            keep, reason = _apply_cohost_guest_rescue(
                participants,
                candidates,
                keep,
                scores,
                config,
                reason,
            )
            return sorted(set(keep)), reason

    levels = {
        key: activities[key].frames[idx].envelope_db
        for key in candidates
    }
    strongest = max(candidates, key=lambda key: levels[key])
    keep = [strongest]
    for key in sorted(candidates, key=lambda item: levels[item], reverse=True):
        if key == strongest:
            continue
        if levels[strongest] - levels[key] < config.dominance_delta_db:
            keep.append(key)
    return sorted(set(keep)), None


def _debleed_frame_scores(
    candidates: list[str],
    activities: dict[str, SpeakerActivity],
    idx: int,
    profiles: dict[str, SakhaDebleedProfile],
    config: SakhaAymakhConfig,
) -> dict[str, SakhaDebleedFrameScore]:
    scores: dict[str, SakhaDebleedFrameScore] = {}
    for key in candidates:
        profile = profiles.get(key)
        if profile is None:
            continue
        envelope_db = activities[key].frames[idx].envelope_db
        normalized_db = envelope_db - profile.reference_db
        snr_db = envelope_db - profile.floor_db
        snr_bonus_db = max(0.0, min(snr_db, 60.0) - config.debleed_guest_rescue_snr_db)
        scores[key] = SakhaDebleedFrameScore(
            key=key,
            envelope_db=envelope_db,
            normalized_db=normalized_db,
            snr_db=snr_db,
            score_db=normalized_db + config.debleed_score_snr_weight * snr_bonus_db,
        )
    return scores


def _apply_cohost_guest_rescue(
    participants: list[SakhaParticipantSpec],
    candidates: list[str],
    keep: list[str],
    scores: dict[str, SakhaDebleedFrameScore],
    config: SakhaAymakhConfig,
    reason: str | None,
) -> tuple[list[str], str | None]:
    cohost_key = _key_for_role(participants, "cohost")
    guest_key = _key_for_role(participants, "guest")
    if cohost_key not in candidates or guest_key not in candidates:
        return keep, reason
    cohost_score = scores.get(cohost_key)
    guest_score = scores.get(guest_key)
    if cohost_score is None or guest_score is None:
        return keep, reason

    keep_set = set(keep)
    guest_snr = guest_score.snr_db
    cohost_snr = cohost_score.snr_db
    cohost_minus_guest_snr = cohost_snr - guest_snr
    guest_has_usable_signal = guest_snr >= config.debleed_guest_rescue_snr_db
    cohost_has_usable_signal = cohost_snr >= config.debleed_guest_rescue_snr_db

    if not keep_set and guest_has_usable_signal:
        if cohost_has_usable_signal and cohost_minus_guest_snr < config.debleed_guest_ambiguous_snr_margin_db:
            return sorted({cohost_key, guest_key}), "cohost_guest_ambiguous_open_both"
        return [guest_key], "guest_rescued"

    if cohost_key in keep_set and guest_key not in keep_set:
        if guest_has_usable_signal and cohost_minus_guest_snr < config.debleed_guest_ambiguous_snr_margin_db:
            keep_set.add(guest_key)
            return sorted(keep_set), "cohost_guest_ambiguous_open_both"
        if guest_has_usable_signal:
            return sorted(keep_set), "guest_suppressed_by_cohost"
        return sorted(keep_set), reason

    if guest_key in keep_set and cohost_key not in keep_set:
        guest_minus_cohost_snr = guest_snr - cohost_snr
        if cohost_has_usable_signal and guest_minus_cohost_snr < config.debleed_guest_ambiguous_snr_margin_db:
            keep_set.add(cohost_key)
            return sorted(keep_set), "cohost_guest_ambiguous_open_both"
        return sorted(keep_set), reason or "guest_rescued"

    if guest_key in keep_set and cohost_key in keep_set:
        if (
            guest_has_usable_signal
            and cohost_has_usable_signal
            and abs(cohost_minus_guest_snr) < config.debleed_guest_ambiguous_snr_margin_db
        ):
            return sorted(keep_set), reason or "cohost_guest_ambiguous_open_both"

    return sorted(keep_set), reason


def _make_frame_state(
    participants: list[SakhaParticipantSpec],
    activities: dict[str, SpeakerActivity],
    idx: int,
    active_keys: list[str],
    debleed_reason: str | None = None,
) -> SakhaFrameState:
    active_keys = sorted(set(active_keys))
    active_roles = {_role_for_key(participants, key) for key in active_keys}
    focus_key = _dominant_key(active_keys, activities, idx)

    if not active_keys:
        state = SakhaAymakhState.SILENCE
        focus_key = None
    elif active_roles == {"main_host"}:
        state = SakhaAymakhState.MAIN_HOST_ONLY
        focus_key = _key_for_role(participants, "main_host")
    elif active_roles == {"cohost"}:
        state = SakhaAymakhState.COHOST_ONLY
        focus_key = _key_for_role(participants, "cohost")
    elif active_roles == {"guest"}:
        state = SakhaAymakhState.GUEST_ONLY
        focus_key = _key_for_role(participants, "guest")
    elif active_roles == {"main_host", "cohost"}:
        state = SakhaAymakhState.MAIN_HOST_COHOST
    elif active_roles == {"main_host", "guest"}:
        state = SakhaAymakhState.MAIN_HOST_GUEST
    elif active_roles == {"cohost", "guest"}:
        state = SakhaAymakhState.COHOST_GUEST
        focus_key = _key_for_role(participants, "guest")
    else:
        state = SakhaAymakhState.ALL_OVERLAP

    return SakhaFrameState(
        time_s=activities[participants[0].key].frames[idx].time_s,
        state=state,
        active_keys=tuple(active_keys),
        focus_key=focus_key,
        debleed_reason=debleed_reason,
    )


def _apply_cohost_guest_hysteresis(
    frame_states: list[SakhaFrameState],
    participants: list[SakhaParticipantSpec],
    activities: dict[str, SpeakerActivity],
    profiles: dict[str, SakhaDebleedProfile],
    hop_s: float,
    config: SakhaAymakhConfig,
) -> list[SakhaFrameState]:
    if not config.debleed_enabled or not frame_states or not profiles:
        return frame_states

    cohost_key = _key_for_role(participants, "cohost")
    guest_key = _key_for_role(participants, "guest")
    result = list(frame_states)
    idx = 0
    while idx < len(result):
        if not _is_cohost_only_with_guest_raw(result, activities, idx, cohost_key, guest_key):
            idx += 1
            continue

        start_idx = idx
        while idx < len(result) and _is_cohost_only_with_guest_raw(result, activities, idx, cohost_key, guest_key):
            idx += 1
        end_idx = idx

        if _cohost_only_run_is_confident(
            start_idx,
            end_idx,
            activities,
            profiles,
            cohost_key,
            guest_key,
            hop_s,
            config,
        ):
            continue

        for pos in range(start_idx, end_idx):
            result[pos] = _make_frame_state(
                participants,
                activities,
                pos,
                [cohost_key, guest_key],
                "guest_hysteresis_rescue",
            )

    return result


def _is_cohost_only_with_guest_raw(
    frame_states: list[SakhaFrameState],
    activities: dict[str, SpeakerActivity],
    idx: int,
    cohost_key: str,
    guest_key: str,
) -> bool:
    frame = frame_states[idx]
    if frame.state != SakhaAymakhState.COHOST_ONLY or frame.active_keys != (cohost_key,):
        return False
    return (
        idx < len(activities[cohost_key].frames)
        and idx < len(activities[guest_key].frames)
        and activities[cohost_key].frames[idx].is_active
        and activities[guest_key].frames[idx].is_active
    )


def _cohost_only_run_is_confident(
    start_idx: int,
    end_idx: int,
    activities: dict[str, SpeakerActivity],
    profiles: dict[str, SakhaDebleedProfile],
    cohost_key: str,
    guest_key: str,
    hop_s: float,
    config: SakhaAymakhConfig,
) -> bool:
    guest_profile = profiles.get(guest_key)
    cohost_profile = profiles.get(cohost_key)
    if guest_profile is None or cohost_profile is None:
        return True

    guest_snrs = [
        activities[guest_key].frames[idx].envelope_db - guest_profile.floor_db
        for idx in range(start_idx, end_idx)
    ]
    cohost_snrs = [
        activities[cohost_key].frames[idx].envelope_db - cohost_profile.floor_db
        for idx in range(start_idx, end_idx)
    ]
    if not guest_snrs or not cohost_snrs:
        return True

    if max(guest_snrs) < config.debleed_guest_rescue_snr_db:
        return True

    duration_s = (end_idx - start_idx) * hop_s
    avg_guest_snr = sum(guest_snrs) / len(guest_snrs)
    avg_cohost_snr = sum(cohost_snrs) / len(cohost_snrs)
    return (
        duration_s >= config.debleed_cohost_confidence_s
        and avg_cohost_snr - avg_guest_snr >= config.debleed_guest_ambiguous_snr_margin_db
    )


def _build_debleed_profiles(
    participants: list[SakhaParticipantSpec],
    activities: dict[str, SpeakerActivity],
    config: SakhaAymakhConfig,
) -> dict[str, SakhaDebleedProfile]:
    if not config.debleed_enabled:
        return {}
    profiles: dict[str, SakhaDebleedProfile] = {}
    for part in participants:
        frames = activities.get(part.key, SpeakerActivity(part.label)).frames
        if not frames:
            continue
        all_levels = [frame.envelope_db for frame in frames]
        active_levels = [frame.envelope_db for frame in frames if frame.is_active] or all_levels
        profiles[part.key] = SakhaDebleedProfile(
            key=part.key,
            floor_db=_percentile(all_levels, 10.0),
            reference_db=_percentile(active_levels, 90.0),
        )
    return profiles


def _build_debleed_diagnostics(
    participants: list[SakhaParticipantSpec],
    activities: dict[str, SpeakerActivity],
    frame_states: list[SakhaFrameState],
    debleed_profiles: dict[str, SakhaDebleedProfile],
    audio_open_intervals_s: dict[int, list[tuple[float, float]]],
    hop_s: float,
    config: SakhaAymakhConfig,
) -> dict:
    active_after_by_key = {part.key: 0 for part in participants}
    for frame in frame_states:
        for key in frame.active_keys:
            active_after_by_key[key] = active_after_by_key.get(key, 0) + 1

    participants_diag: dict[str, dict] = {}
    for part in participants:
        frames = activities[part.key].frames
        before_frames = sum(1 for frame in frames if frame.is_active)
        after_frames = active_after_by_key.get(part.key, 0)
        profile = debleed_profiles.get(part.key)
        open_seconds = sum(
            end_s - start_s
            for start_s, end_s in audio_open_intervals_s.get(part.audio_track_index, [])
        )
        participants_diag[part.key] = {
            "label": part.label,
            "track": part.audio_track_index,
            "active_seconds_before": round(before_frames * hop_s, 3),
            "active_seconds_after": round(after_frames * hop_s, 3),
            "suppressed_frames": max(0, before_frames - after_frames),
            "stable_open_seconds": round(open_seconds, 3),
            "floor_db": None if profile is None else round(profile.floor_db, 3),
            "reference_db": None if profile is None else round(profile.reference_db, 3),
            "typical_snr_db": None if profile is None else round(profile.reference_db - profile.floor_db, 3),
        }

    reason_counts: dict[str, int] = {}
    for frame in frame_states:
        if frame.debleed_reason:
            reason_counts[frame.debleed_reason] = reason_counts.get(frame.debleed_reason, 0) + 1
    suppressed_frames = _count_guest_suppressed_by_cohost_frames(
        participants,
        activities,
        frame_states,
        debleed_profiles,
        config,
    )

    return {
        "enabled": config.debleed_enabled,
        "overlap_margin_db": config.debleed_overlap_margin_db,
        "min_leader_score_db": config.debleed_min_leader_score_db,
        "score_snr_weight": config.debleed_score_snr_weight,
        "guest_rescue_snr_db": config.debleed_guest_rescue_snr_db,
        "guest_ambiguous_snr_margin_db": config.debleed_guest_ambiguous_snr_margin_db,
        "cohost_confidence_s": config.debleed_cohost_confidence_s,
        "audio_silence_policy": config.audio_silence_policy,
        "guest_rescued_seconds": round(
            (
                reason_counts.get("guest_rescued", 0)
                + reason_counts.get("guest_hysteresis_rescue", 0)
            )
            * hop_s,
            3,
        ),
        "cohost_guest_ambiguous_open_both_seconds": round(
            reason_counts.get("cohost_guest_ambiguous_open_both", 0) * hop_s,
            3,
        ),
        "guest_suppressed_by_cohost_seconds": round(suppressed_frames * hop_s, 3),
        "decision_counts": reason_counts,
        "top_cohost_only_guest_raw_segments": _cohost_only_guest_raw_segments(
            participants,
            activities,
            frame_states,
            debleed_profiles,
            hop_s,
            config,
            limit=10,
        ),
        "participants": participants_diag,
    }


def _build_waveform_gate_diagnostics(
    frame_states: list[SakhaFrameState],
    hop_s: float,
    config: SakhaAymakhConfig,
    waveform_gate: SakhaWaveformGateContext | None,
) -> dict:
    reason_seconds: dict[str, float] = {}
    single_seconds = {
        "single_main_host": 0.0,
        "single_cohost": 0.0,
        "single_guest": 0.0,
    }
    true_overlap_seconds = 0.0
    correlated_bleed_suppressed_seconds = 0.0
    suspicious: list[dict] = []
    run_start_idx: int | None = None
    run_reason: str | None = None
    run_keys: tuple[str, ...] = ()

    for idx, frame in enumerate(frame_states):
        if frame.debleed_reason:
            reason_seconds[frame.debleed_reason] = reason_seconds.get(frame.debleed_reason, 0.0) + hop_s
        if len(frame.active_keys) == 1:
            key = frame.active_keys[0]
            single_key = f"single_{key}"
            if single_key in single_seconds:
                single_seconds[single_key] += hop_s
        elif len(frame.active_keys) > 1 and frame.debleed_reason in {
            "true_overlap",
            "envelope_true_overlap_fallback",
            "source_owner_true_overlap",
        }:
            true_overlap_seconds += hop_s
        if frame.debleed_reason in {"correlated_bleed_suppressed", "source_owner_bleed_suppressed"}:
            correlated_bleed_suppressed_seconds += hop_s

        suspicious_reason = (
            frame.debleed_reason
            if frame.debleed_reason in {
                "correlated_bleed_suppressed",
                "source_owner_bleed_suppressed",
                "source_owner_overlap_too_short",
                "strict_single_ambiguous",
                "strict_single_fallback",
            }
            else None
        )
        if suspicious_reason is None:
            if run_start_idx is not None:
                suspicious.append(
                    _diagnostic_interval(run_start_idx, idx, hop_s, run_reason or "", run_keys)
                )
                run_start_idx = None
            continue
        if run_start_idx is None:
            run_start_idx = idx
            run_reason = suspicious_reason
            run_keys = frame.active_keys
        elif suspicious_reason != run_reason or frame.active_keys != run_keys:
            suspicious.append(
                _diagnostic_interval(run_start_idx, idx, hop_s, run_reason or "", run_keys)
            )
            run_start_idx = idx
            run_reason = suspicious_reason
            run_keys = frame.active_keys

    if run_start_idx is not None:
        suspicious.append(
            _diagnostic_interval(run_start_idx, len(frame_states), hop_s, run_reason or "", run_keys)
        )

    return {
        "mode": config.audio_clean_mode,
        "waveform_available": waveform_gate is not None and waveform_gate.available,
        "window_s": config.waveform_window_s,
        "step_s": config.waveform_step_s,
        "max_lag_s": config.waveform_max_lag_s,
        "corr_bleed_threshold": _waveform_corr_bleed_threshold(config),
        "independent_threshold": _waveform_independent_threshold(config),
        "overlap_snr_db": _waveform_overlap_snr_db(config),
        "strict_min_speaker_hold_s": config.strict_min_speaker_hold_s,
        "strict_switch_margin_db": config.strict_switch_margin_db,
        "decision_seconds": {
            **{key: round(value, 3) for key, value in single_seconds.items()},
            "true_overlap": round(true_overlap_seconds, 3),
            "correlated_bleed_suppressed": round(correlated_bleed_suppressed_seconds, 3),
        },
        "reason_seconds": {
            key: round(value, 3)
            for key, value in sorted(reason_seconds.items())
        },
        "top_suspicious_single_mic_intervals": sorted(
            suspicious,
            key=lambda item: item["duration"],
            reverse=True,
        )[:10],
    }


def _build_source_owner_diagnostics(
    participants: list[SakhaParticipantSpec],
    activities: dict[str, SpeakerActivity],
    frame_states: list[SakhaFrameState],
    hop_s: float,
    config: SakhaAymakhConfig,
    waveform_gate: SakhaWaveformGateContext | None,
) -> dict:
    owner_seconds = {
        "silence": 0.0,
        "single_main_host": 0.0,
        "single_cohost": 0.0,
        "single_guest": 0.0,
        "true_overlap": 0.0,
    }
    bleed_by_owner = {part.key: 0.0 for part in participants}
    raw_suppressed_by_track = {part.key: 0.0 for part in participants}
    reason_seconds: dict[str, float] = {}
    suspicious: list[dict] = []
    run_start_idx: int | None = None
    run_owner: tuple[str, ...] = ()
    run_suppressed: tuple[str, ...] = ()

    for idx, frame in enumerate(frame_states):
        if frame.debleed_reason:
            reason_seconds[frame.debleed_reason] = reason_seconds.get(frame.debleed_reason, 0.0) + hop_s
        if not frame.active_keys:
            owner_seconds["silence"] += hop_s
        elif len(frame.active_keys) == 1:
            key = frame.active_keys[0]
            label = f"single_{key}"
            if label in owner_seconds:
                owner_seconds[label] += hop_s
        else:
            owner_seconds["true_overlap"] += hop_s

        raw_active = {
            part.key
            for part in participants
            if idx < len(activities[part.key].frames) and activities[part.key].frames[idx].is_active
        }
        final_active = set(frame.active_keys)
        suppressed = tuple(sorted(raw_active - final_active))
        if final_active and suppressed:
            owner = frame.focus_key or sorted(final_active)[0]
            bleed_by_owner[owner] = bleed_by_owner.get(owner, 0.0) + hop_s
            for key in suppressed:
                raw_suppressed_by_track[key] = raw_suppressed_by_track.get(key, 0.0) + hop_s

        if suppressed and final_active:
            owner_tuple = tuple(sorted(final_active))
            if run_start_idx is None:
                run_start_idx = idx
                run_owner = owner_tuple
                run_suppressed = suppressed
            elif owner_tuple != run_owner or suppressed != run_suppressed:
                suspicious.append(
                    _source_owner_interval(run_start_idx, idx, hop_s, run_owner, run_suppressed)
                )
                run_start_idx = idx
                run_owner = owner_tuple
                run_suppressed = suppressed
        elif run_start_idx is not None:
            suspicious.append(
                _source_owner_interval(run_start_idx, idx, hop_s, run_owner, run_suppressed)
            )
            run_start_idx = None

    if run_start_idx is not None:
        suspicious.append(
            _source_owner_interval(run_start_idx, len(frame_states), hop_s, run_owner, run_suppressed)
        )

    return {
        "mode": config.audio_clean_mode,
        "waveform_available": waveform_gate is not None and waveform_gate.available,
        "source_owner_seconds": {
            key: round(value, 3)
            for key, value in owner_seconds.items()
        },
        "bleed_suppressed_seconds_by_owner": {
            key: round(value, 3)
            for key, value in sorted(bleed_by_owner.items())
        },
        "main_host_bleed_suppressed_seconds": round(raw_suppressed_by_track.get("main_host", 0.0), 3),
        "cohost_bleed_suppressed_seconds": round(raw_suppressed_by_track.get("cohost", 0.0), 3),
        "guest_bleed_suppressed_seconds": round(raw_suppressed_by_track.get("guest", 0.0), 3),
        "reason_seconds": {
            key: round(value, 3)
            for key, value in sorted(reason_seconds.items())
        },
        "top_raw_active_muted_as_bleed": sorted(
            suspicious,
            key=lambda item: item["duration"],
            reverse=True,
        )[:10],
        "source_owner_bleed_residual_db": _source_owner_bleed_residual_db(config),
        "source_owner_residual_voice_snr_db": _source_owner_residual_voice_snr_db(config),
        "source_owner_min_corr": _source_owner_min_corr(config),
    }


def _source_owner_interval(
    start_idx: int,
    end_idx: int,
    hop_s: float,
    owner_keys: tuple[str, ...],
    suppressed_keys: tuple[str, ...],
) -> dict:
    return {
        "start": round(start_idx * hop_s, 3),
        "end": round(end_idx * hop_s, 3),
        "duration": round((end_idx - start_idx) * hop_s, 3),
        "owner": list(owner_keys),
        "muted_raw_active": list(suppressed_keys),
    }


def _waveform_corr_bleed_threshold(config: SakhaAymakhConfig) -> float:
    if config.audio_clean_mode == "balanced":
        return min(0.90, config.waveform_corr_bleed_threshold + 0.10)
    return config.waveform_corr_bleed_threshold


def _waveform_independent_threshold(config: SakhaAymakhConfig) -> float:
    if config.audio_clean_mode == "balanced":
        return min(_waveform_corr_bleed_threshold(config) - 0.02, config.waveform_independent_threshold + 0.15)
    return config.waveform_independent_threshold


def _waveform_overlap_snr_db(config: SakhaAymakhConfig) -> float:
    if config.audio_clean_mode == "balanced":
        return max(12.0, config.waveform_overlap_snr_db - 4.0)
    return config.waveform_overlap_snr_db


def _diagnostic_interval(
    start_idx: int,
    end_idx: int,
    hop_s: float,
    reason: str,
    active_keys: tuple[str, ...],
) -> dict:
    return {
        "start": round(start_idx * hop_s, 3),
        "end": round(end_idx * hop_s, 3),
        "duration": round((end_idx - start_idx) * hop_s, 3),
        "reason": reason,
        "active": list(active_keys),
    }


def _count_guest_suppressed_by_cohost_frames(
    participants: list[SakhaParticipantSpec],
    activities: dict[str, SpeakerActivity],
    frame_states: list[SakhaFrameState],
    profiles: dict[str, SakhaDebleedProfile],
    config: SakhaAymakhConfig,
) -> int:
    cohost_key = _key_for_role(participants, "cohost")
    guest_key = _key_for_role(participants, "guest")
    guest_profile = profiles.get(guest_key)
    if guest_profile is None:
        return 0
    count = 0
    for idx, frame in enumerate(frame_states):
        if frame.state != SakhaAymakhState.COHOST_ONLY or guest_key in frame.active_keys:
            continue
        if idx >= len(activities[guest_key].frames) or idx >= len(activities[cohost_key].frames):
            continue
        if not activities[guest_key].frames[idx].is_active or not activities[cohost_key].frames[idx].is_active:
            continue
        guest_snr = activities[guest_key].frames[idx].envelope_db - guest_profile.floor_db
        if guest_snr >= config.debleed_guest_rescue_snr_db:
            count += 1
    return count


def _cohost_only_guest_raw_segments(
    participants: list[SakhaParticipantSpec],
    activities: dict[str, SpeakerActivity],
    frame_states: list[SakhaFrameState],
    profiles: dict[str, SakhaDebleedProfile],
    hop_s: float,
    config: SakhaAymakhConfig,
    *,
    limit: int,
) -> list[dict]:
    cohost_key = _key_for_role(participants, "cohost")
    guest_key = _key_for_role(participants, "guest")
    cohost_profile = profiles.get(cohost_key)
    guest_profile = profiles.get(guest_key)
    if cohost_profile is None or guest_profile is None:
        return []

    segments: list[dict] = []
    idx = 0
    while idx < len(frame_states):
        if not _cohost_only_guest_raw_problem_frame(
            idx,
            frame_states,
            activities,
            cohost_key,
            guest_key,
            guest_profile,
            config,
        ):
            idx += 1
            continue

        start_idx = idx
        cohost_snrs: list[float] = []
        guest_snrs: list[float] = []
        while idx < len(frame_states) and _cohost_only_guest_raw_problem_frame(
            idx,
            frame_states,
            activities,
            cohost_key,
            guest_key,
            guest_profile,
            config,
        ):
            cohost_snrs.append(activities[cohost_key].frames[idx].envelope_db - cohost_profile.floor_db)
            guest_snrs.append(activities[guest_key].frames[idx].envelope_db - guest_profile.floor_db)
            idx += 1

        duration_s = (idx - start_idx) * hop_s
        avg_cohost_snr = sum(cohost_snrs) / len(cohost_snrs)
        avg_guest_snr = sum(guest_snrs) / len(guest_snrs)
        segments.append(
            {
                "start": round(start_idx * hop_s, 3),
                "end": round(idx * hop_s, 3),
                "duration": round(duration_s, 3),
                "avg_cohost_snr_db": round(avg_cohost_snr, 3),
                "avg_guest_snr_db": round(avg_guest_snr, 3),
                "avg_snr_gap_db": round(avg_cohost_snr - avg_guest_snr, 3),
                "max_guest_snr_db": round(max(guest_snrs), 3),
            }
        )

    return sorted(segments, key=lambda item: item["duration"], reverse=True)[:limit]


def _cohost_only_guest_raw_problem_frame(
    idx: int,
    frame_states: list[SakhaFrameState],
    activities: dict[str, SpeakerActivity],
    cohost_key: str,
    guest_key: str,
    guest_profile: SakhaDebleedProfile,
    config: SakhaAymakhConfig,
) -> bool:
    frame = frame_states[idx]
    if frame.state != SakhaAymakhState.COHOST_ONLY or guest_key in frame.active_keys:
        return False
    if idx >= len(activities[guest_key].frames) or idx >= len(activities[cohost_key].frames):
        return False
    guest_frame = activities[guest_key].frames[idx]
    return (
        activities[cohost_key].frames[idx].is_active
        and guest_frame.is_active
        and guest_frame.envelope_db - guest_profile.floor_db >= config.debleed_guest_rescue_snr_db
    )


def _role_for_key(participants: list[SakhaParticipantSpec], key: str) -> str:
    for part in participants:
        if part.key == key:
            return part.role
    raise KeyError(key)


def _key_for_role(participants: list[SakhaParticipantSpec], role: str) -> str:
    for part in participants:
        if part.role == role:
            return part.key
    raise KeyError(role)


def _dominant_key(
    keys: list[str],
    activities: dict[str, SpeakerActivity],
    idx: int,
) -> str | None:
    if not keys:
        return None
    return max(keys, key=lambda key: activities[key].frames[idx].envelope_db)


def _cam_segment(
    start_s: float,
    end_s: float,
    camera_index: int,
    reason: str,
    focus_key: str | None = None,
) -> SakhaCameraSegment:
    return SakhaCameraSegment(
        start_s=start_s,
        end_s=end_s,
        camera_index=camera_index,
        reason=reason,
        focus_key=focus_key,
    )


def _merge_camera_segments(segments: list[SakhaCameraSegment]) -> list[SakhaCameraSegment]:
    if not segments:
        return []
    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        if (
            seg.camera_index == prev.camera_index
            and seg.reason == prev.reason
            and seg.focus_key == prev.focus_key
            and abs(seg.start_s - prev.end_s) < EPS
        ):
            merged[-1] = SakhaCameraSegment(
                start_s=prev.start_s,
                end_s=seg.end_s,
                camera_index=prev.camera_index,
                reason=prev.reason,
                focus_key=prev.focus_key,
            )
        else:
            merged.append(seg)
    return merged


def _insert_reestablishing_wide(
    segments: list[SakhaCameraSegment],
    config: SakhaAymakhConfig,
) -> list[SakhaCameraSegment]:
    if len(segments) < 2:
        return segments

    result = [segments[0]]
    last_all_wide_end = segments[0].end_s if segments[0].camera_index == config.camera_all_wide else 0.0
    turns_since_wide = 0
    prev_focus = segments[0].focus_key

    for seg in segments[1:]:
        if seg.camera_index == config.camera_all_wide:
            last_all_wide_end = seg.end_s
            turns_since_wide = 0
            prev_focus = None
            result.append(seg)
            continue

        if seg.focus_key and seg.focus_key != prev_focus:
            turns_since_wide += 1
        prev_focus = seg.focus_key or prev_focus

        enough_time = seg.start_s - last_all_wide_end >= config.reestablish_all_wide_interval_s
        enough_turns = turns_since_wide >= config.reestablish_min_turns
        enough_room = seg.duration_s >= config.reestablish_all_wide_duration_s + config.shot_hold_time_s
        if enough_time and enough_turns and enough_room:
            wide_end = seg.start_s + config.reestablish_all_wide_duration_s
            result.append(
                _cam_segment(seg.start_s, wide_end, config.camera_all_wide, "reestablish_wide")
            )
            result.append(replace(seg, start_s=wide_end))
            last_all_wide_end = wide_end
            turns_since_wide = 0
        else:
            result.append(seg)

    return result


def _segment_min_duration(seg: SakhaCameraSegment, config: SakhaAymakhConfig) -> float:
    if seg.reason in {"silence", "reestablish_wide"}:
        return min(config.silence_timeout_s, config.overlap_min_hold_s)
    if "cutaway" in seg.reason:
        return min(config.cutaway_duration_s, config.overlap_min_hold_s)
    if "overlap" in seg.reason or "pair_wide" in seg.reason:
        return config.overlap_min_hold_s
    return config.shot_hold_time_s


def _enforce_min_camera_duration(
    segments: list[SakhaCameraSegment],
    config: SakhaAymakhConfig,
) -> list[SakhaCameraSegment]:
    if len(segments) <= 1:
        return segments
    result = [segments[0]]
    mutable = list(segments)
    for idx, seg in enumerate(mutable[1:], start=1):
        if seg.duration_s >= _segment_min_duration(seg, config):
            result.append(seg)
            continue
        prev = result[-1] if result else None
        nxt = mutable[idx + 1] if idx + 1 < len(mutable) else None
        if prev is not None and (nxt is None or prev.duration_s >= nxt.duration_s):
            result[-1] = replace(prev, end_s=seg.end_s)
        elif nxt is not None:
            mutable[idx + 1] = replace(nxt, start_s=seg.start_s)
        else:
            result.append(seg)
    return result


def _enforce_max_visible_hold(
    segments: list[SakhaCameraSegment],
    config: SakhaAymakhConfig,
) -> list[SakhaCameraSegment]:
    if not segments:
        return []

    result: list[SakhaCameraSegment] = []
    run_camera: int | None = None
    run_start_s = 0.0

    for seg in segments:
        current = seg
        if run_camera != current.camera_index:
            run_camera = current.camera_index
            run_start_s = current.start_s

        while current.end_s - run_start_s > config.max_solo_hold_s + EPS:
            cutaway_start_s = max(current.start_s, run_start_s + config.max_solo_hold_s)
            if cutaway_start_s - current.start_s > EPS:
                result.append(replace(current, end_s=cutaway_start_s))

            cutaway_end_s = min(current.end_s, cutaway_start_s + config.cutaway_duration_s)
            if cutaway_end_s - cutaway_start_s <= EPS:
                break

            fallback_camera = _max_hold_cutaway_camera(current.camera_index, config)
            result.append(
                SakhaCameraSegment(
                    start_s=cutaway_start_s,
                    end_s=cutaway_end_s,
                    camera_index=fallback_camera,
                    reason="max_visible_hold_cutaway",
                    focus_key=current.focus_key,
                )
            )

            current = replace(current, start_s=cutaway_end_s)
            run_camera = current.camera_index
            run_start_s = cutaway_end_s
            if current.duration_s <= EPS:
                break

        if current.duration_s > EPS:
            result.append(current)

    return result


def _max_hold_cutaway_camera(camera_index: int, config: SakhaAymakhConfig) -> int:
    if camera_index == config.camera_guest_close:
        preferred = (config.camera_pair_wide, config.camera_all_wide, config.camera_main_host_close)
    elif camera_index in {config.camera_main_host_close, config.camera_pair_wide}:
        preferred = (config.camera_all_wide, config.camera_pair_wide, config.camera_guest_close)
    else:
        preferred = (config.camera_pair_wide, config.camera_main_host_close, config.camera_guest_close)

    for fallback in preferred:
        if fallback != camera_index:
            return fallback
    return camera_index


def _apply_motion_guard(
    segments: list[SakhaCameraSegment],
    config: SakhaAymakhConfig,
    motion_plans: dict[int, CameraMotionPlan],
) -> tuple[list[SakhaCameraSegment], list[SakhaMotionEvent]]:
    result: list[SakhaCameraSegment] = []
    events: list[SakhaMotionEvent] = []
    current_camera: int | None = None

    for seg in segments:
        chosen_camera = seg.camera_index
        chosen_reason = seg.reason
        if _is_camera_blocked(seg.camera_index, seg.start_s, seg.start_s + SAKHA_MOTION_ENTRY_LOOKAHEAD_S, motion_plans):
            fallback = _pick_fallback_camera(
                seg.start_s,
                seg.end_s,
                config,
                motion_plans,
                preferred=(current_camera, config.camera_all_wide, config.camera_pair_wide),
                exclude={seg.camera_index},
            )
            if fallback is not None:
                interval = motion_plans[seg.camera_index].first_interval_between(
                    seg.start_s,
                    seg.start_s + SAKHA_MOTION_ENTRY_LOOKAHEAD_S,
                )
                events.append(
                    SakhaMotionEvent(
                        reason="planned_camera_blocked",
                        time_s=seg.start_s,
                        from_camera_index=current_camera,
                        to_camera_index=fallback,
                        desired_camera_index=seg.camera_index,
                        moving_from_s=None if interval is None else interval.start_s,
                        moving_to_s=None if interval is None else interval.end_s,
                    )
                )
                chosen_camera = fallback
                chosen_reason = "motion_fallback"

        guarded = replace(seg, camera_index=chosen_camera, reason=chosen_reason)
        cursor = guarded.start_s
        while cursor < guarded.end_s - EPS:
            interval = _find_camera_motion_interval_between(
                guarded.camera_index,
                cursor,
                guarded.end_s,
                motion_plans,
            )
            if interval is None:
                result.append(replace(guarded, start_s=cursor))
                current_camera = guarded.camera_index
                break

            escape_s = max(cursor, interval.start_s)
            if escape_s > cursor + EPS:
                result.append(replace(guarded, start_s=cursor, end_s=escape_s))
            fallback = _pick_fallback_camera(
                escape_s,
                guarded.end_s,
                config,
                motion_plans,
                preferred=(config.camera_all_wide, config.camera_pair_wide),
                exclude={guarded.camera_index},
            )
            if fallback is None:
                result.append(replace(guarded, start_s=escape_s, reason="motion_forced_hold"))
                events.append(
                    SakhaMotionEvent(
                        reason="no_static_camera_available",
                        time_s=escape_s,
                        from_camera_index=guarded.camera_index,
                        desired_camera_index=guarded.camera_index,
                        moving_from_s=interval.start_s,
                        moving_to_s=interval.end_s,
                    )
                )
                current_camera = guarded.camera_index
                break

            motion_end = min(interval.end_s, guarded.end_s)
            result.append(
                SakhaCameraSegment(
                    start_s=escape_s,
                    end_s=motion_end,
                    camera_index=fallback,
                    reason="motion_escape",
                    focus_key=guarded.focus_key,
                )
            )
            events.append(
                SakhaMotionEvent(
                    reason="escape_from_motion",
                    time_s=escape_s,
                    from_camera_index=guarded.camera_index,
                    to_camera_index=fallback,
                    desired_camera_index=guarded.camera_index,
                    moving_from_s=interval.start_s,
                    moving_to_s=interval.end_s,
                )
            )
            cursor = motion_end
            current_camera = fallback

    return _merge_camera_segments(result), events


def _is_camera_blocked(
    camera_index: int,
    start_s: float,
    end_s: float,
    motion_plans: dict[int, CameraMotionPlan],
) -> bool:
    plan = motion_plans.get(camera_index)
    return plan is not None and plan.first_interval_between(start_s, end_s) is not None


def _find_camera_motion_interval_between(
    camera_index: int,
    start_s: float,
    end_s: float,
    motion_plans: dict[int, CameraMotionPlan],
):
    plan = motion_plans.get(camera_index)
    if plan is None:
        return None
    return plan.first_interval_between(start_s + EPS, end_s)


def _pick_fallback_camera(
    time_s: float,
    window_end_s: float,
    config: SakhaAymakhConfig,
    motion_plans: dict[int, CameraMotionPlan],
    *,
    preferred: tuple[int | None, ...],
    exclude: set[int] | None = None,
) -> int | None:
    exclude = exclude or set()
    seen: set[int] = set()
    for camera_index in preferred + (config.camera_main_host_close, config.camera_guest_close):
        if camera_index is None or camera_index in seen or camera_index in exclude:
            continue
        seen.add(camera_index)
        if not _is_camera_blocked(camera_index, time_s, window_end_s, motion_plans):
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
        if end - start <= max_gap_frames and flags[start - 1] and flags[end] and not any(competing_flags[start:end]):
            for pos in range(start, end):
                flags[pos] = True


def _remove_short_true_runs(flags: list[bool], min_frames: int) -> None:
    for start, end in _true_runs(flags):
        if end - start < min_frames:
            for pos in range(start, end):
                flags[pos] = False


def _extend_true_runs_to_min(flags: list[bool], min_frames: int, competing_flags: list[bool]) -> None:
    for start, end in _true_runs(flags):
        if end - start >= min_frames:
            continue
        target_end = _cap_target_end_at_competing(min(len(flags), start + min_frames), end, competing_flags)
        for pos in range(end, target_end):
            flags[pos] = True


def _extend_true_runs_tail(flags: list[bool], add_frames: int, competing_flags: list[bool]) -> None:
    for _start, end in _true_runs(flags):
        target_end = _cap_target_end_at_competing(min(len(flags), end + add_frames), end, competing_flags)
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


def _apply_hold_last_audio(
    stable_flags: dict[int, list[bool]],
    frame_states: list[SakhaFrameState],
    participants: list[SakhaParticipantSpec],
) -> None:
    if not stable_flags:
        return
    frame_count = max((len(flags) for flags in stable_flags.values()), default=0)
    any_open = [
        any(idx < len(flags) and flags[idx] for flags in stable_flags.values())
        for idx in range(frame_count)
    ]
    if not any(any_open):
        return

    first_open_idx = next(idx for idx, is_open in enumerate(any_open) if is_open)
    last_open_idx = len(any_open) - 1 - next(
        idx for idx, is_open in enumerate(reversed(any_open)) if is_open
    )
    key_to_track = {part.key: part.audio_track_index for part in participants}
    last_track: int | None = None

    for idx in range(first_open_idx, last_open_idx + 1):
        open_tracks = [
            track_idx
            for track_idx, flags in stable_flags.items()
            if idx < len(flags) and flags[idx]
        ]
        if open_tracks:
            focus_key = frame_states[idx].focus_key if idx < len(frame_states) else None
            focus_track = key_to_track.get(focus_key or "")
            last_track = focus_track if focus_track in open_tracks else open_tracks[0]
            continue
        if last_track is not None and last_track in stable_flags and idx < len(stable_flags[last_track]):
            stable_flags[last_track][idx] = True


def _build_audio_level_segments(
    stable_flags: dict[int, list[bool]],
    hop_s: float,
    config: SakhaAymakhConfig,
) -> list[SakhaAudioLevelSegment]:
    if not stable_flags:
        return []
    frame_count = max((len(flags) for flags in stable_flags.values()), default=0)
    current_levels: dict[int, float] | None = None
    start_idx = 0
    segments: list[SakhaAudioLevelSegment] = []

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
                SakhaAudioLevelSegment(
                    start_s=start_idx * hop_s,
                    end_s=idx * hop_s,
                    levels_db=current_levels,
                )
            )
            current_levels = levels
            start_idx = idx

    if current_levels is not None:
        segments.append(
            SakhaAudioLevelSegment(
                start_s=start_idx * hop_s,
                end_s=frame_count * hop_s,
                levels_db=current_levels,
            )
        )
    return segments


def _bool_runs_to_intervals(flags: list[bool], hop_s: float) -> list[tuple[float, float]]:
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


def _count_states(frame_states: list[SakhaFrameState]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for frame in frame_states:
        counts[frame.state.value] = counts.get(frame.state.value, 0) + 1
    return counts


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _lerp(low: float, high: float, amount: float) -> float:
    return low + (high - low) * _clamp01(amount)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return -100.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = _clamp01(percentile / 100.0) * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac
