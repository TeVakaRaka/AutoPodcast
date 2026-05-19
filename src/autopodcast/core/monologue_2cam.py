"""Offline camera scheduler for a single narrator and two cameras."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field, replace

from autopodcast.core.camera_motion import CameraMotionPlan, MotionInterval
from autopodcast.core.segmenter import (
    collapse_short_neutral_segments,
    merge_adjacent,
    run_length_encode,
)
from autopodcast.models.domain import Segment, SpeakerActivity, SpeakerState

EPS = 1e-9


@dataclass(frozen=True)
class Monologue2CamConfig:
    camera_main: int
    camera_accent: int
    switch_interval_s: float = 30.0
    camera_main_share: float = 0.5
    pause_window_before_s: float = 6.0
    pause_window_after_s: float = 10.0
    min_pause_s: float = 0.35
    micro_pause_merge_s: float = 0.25
    min_camera_hold_s: float = 12.0
    pause_edge_padding_s: float = 0.10
    forced_min_drop_db: float = 3.0
    motion_wait_max_s: float = 8.0
    motion_escape_min_hold_s: float = 2.0
    entry_stability_lookahead_s: float = 1.5

    def validate(self) -> None:
        if self.camera_main < 0:
            raise ValueError(f"camera_main must be >= 0, got {self.camera_main}")
        if self.camera_accent < 0:
            raise ValueError(f"camera_accent must be >= 0, got {self.camera_accent}")
        if self.camera_main == self.camera_accent:
            raise ValueError("camera_main and camera_accent must be different")
        if not 0.0 < self.camera_main_share < 1.0:
            raise ValueError(
                "camera_main_share must be between 0 and 1 "
                f"(exclusive), got {self.camera_main_share}"
            )

        for name, value in [
            ("switch_interval_s", self.switch_interval_s),
            ("pause_window_after_s", self.pause_window_after_s),
            ("min_pause_s", self.min_pause_s),
            ("min_camera_hold_s", self.min_camera_hold_s),
            ("motion_wait_max_s", self.motion_wait_max_s),
            ("motion_escape_min_hold_s", self.motion_escape_min_hold_s),
            ("entry_stability_lookahead_s", self.entry_stability_lookahead_s),
        ]:
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        for name, value in [
            ("pause_window_before_s", self.pause_window_before_s),
            ("micro_pause_merge_s", self.micro_pause_merge_s),
            ("pause_edge_padding_s", self.pause_edge_padding_s),
            ("forced_min_drop_db", self.forced_min_drop_db),
        ]:
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")

    @property
    def camera_accent_share(self) -> float:
        return 1.0 - self.camera_main_share

    @property
    def weighted_cycle_s(self) -> float:
        return max(
            self.switch_interval_s * 2.0,
            self.min_camera_hold_s / self.camera_main_share,
            self.min_camera_hold_s / self.camera_accent_share,
        )

    def target_hold_s(self, camera_index: int) -> float:
        share = (
            self.camera_main_share
            if camera_index == self.camera_main
            else self.camera_accent_share
        )
        return share * self.weighted_cycle_s


@dataclass(frozen=True)
class MonologueCut:
    time_s: float
    camera_index: int
    reason: str
    target_s: float
    delta_s: float
    pause_start_s: float | None = None
    pause_end_s: float | None = None
    pause_duration_s: float | None = None
    min_envelope_db: float | None = None
    fallback_deadline_s: float | None = None
    planned_reason: str | None = None
    planned_time_s: float | None = None
    stable_since_s: float | None = None
    moving_from_s: float | None = None
    moving_to_s: float | None = None


@dataclass(frozen=True)
class MonologueCameraSegment:
    start_s: float
    end_s: float
    camera_index: int
    reason: str

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class MonologueMotionEvent:
    reason: str
    from_camera_index: int
    to_camera_index: int | None = None
    planned_time_s: float | None = None
    actual_time_s: float | None = None
    delay_s: float | None = None
    stable_since_s: float | None = None
    moving_from_s: float | None = None
    moving_to_s: float | None = None


@dataclass
class Monologue2CamPlan:
    speech_segments: list[Segment] = field(default_factory=list)
    pause_segments: list[Segment] = field(default_factory=list)
    camera_segments: list[MonologueCameraSegment] = field(default_factory=list)
    cuts: list[MonologueCut] = field(default_factory=list)
    motion_events: list[MonologueMotionEvent] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def build_monologue_plan(
    activity: SpeakerActivity,
    total_duration_s: float,
    hop_s: float,
    config: Monologue2CamConfig,
    motion_plans: dict[int, CameraMotionPlan] | None = None,
) -> Monologue2CamPlan:
    """Build a two-camera switching plan for a single narrator."""
    config.validate()
    motion_plans = motion_plans or {}

    speech_segments = _build_speech_segments(activity, total_duration_s, hop_s, config)
    pause_segments = [
        seg for seg in speech_segments
        if seg.speaker_state == SpeakerState.SILENCE and seg.duration_s >= config.min_pause_s
    ]

    pause_cut_times = [_pause_cut_time(seg, config.pause_edge_padding_s) for seg in pause_segments]
    frame_times = [frame.time_s for frame in activity.frames]
    normal_deadline_s = max(0.0, total_duration_s - config.min_camera_hold_s)
    emergency_deadline_s = max(0.0, total_duration_s - config.motion_escape_min_hold_s)

    cuts: list[MonologueCut] = []
    motion_events: list[MonologueMotionEvent] = []
    current_camera = config.camera_main
    last_cut_s = 0.0
    retry_after_s = 0.0

    while True:
        next_camera = _alternate_camera(current_camera, config)
        target_s = last_cut_s + config.target_hold_s(current_camera)
        eligible_start_s = max(last_cut_s + config.min_camera_hold_s, retry_after_s)
        if eligible_start_s > total_duration_s + EPS:
            break
        target_s = max(target_s, eligible_start_s)

        cut = _find_base_cut(
            activity,
            pause_segments,
            pause_cut_times,
            frame_times,
            target_s,
            eligible_start_s,
            normal_deadline_s,
            next_camera,
            config,
        )

        resolution_end_s = emergency_deadline_s
        target_motion_interval = None
        target_motion_reason = None
        if cut is not None:
            target_plan = motion_plans.get(next_camera)
            if target_plan is not None:
                target_motion_interval = target_plan.find_interval_at(cut.time_s)
                if target_motion_interval is not None:
                    target_motion_reason = "current"
                else:
                    lookahead_end_s = min(
                        cut.time_s + config.entry_stability_lookahead_s,
                        emergency_deadline_s,
                    )
                    target_motion_interval = target_plan.first_interval_between(
                        cut.time_s + EPS,
                        lookahead_end_s,
                    )
                    if target_motion_interval is not None:
                        target_motion_reason = "future"
            if target_motion_interval is None:
                resolution_end_s = min(cut.time_s, emergency_deadline_s)
            else:
                resolution_end_s = min(
                    cut.time_s + config.motion_wait_max_s,
                    emergency_deadline_s,
                )

        emergency_cut, emergency_event = _find_emergency_escape(
            current_camera=current_camera,
            next_camera=next_camera,
            last_cut_s=last_cut_s,
            upper_bound_s=resolution_end_s,
            emergency_deadline_s=emergency_deadline_s,
            motion_plans=motion_plans,
            config=config,
        )
        if emergency_event is not None:
            motion_events.append(emergency_event)
        if emergency_cut is not None:
            cuts.append(emergency_cut)
            current_camera = emergency_cut.camera_index
            last_cut_s = emergency_cut.time_s
            retry_after_s = last_cut_s
            continue

        if cut is None:
            break
        if cut.time_s <= last_cut_s + EPS:
            break

        if target_motion_interval is not None:
            target_plan = motion_plans.get(next_camera)
            stable_search_start_s = (
                cut.time_s
                if target_motion_reason != "future"
                else target_motion_interval.start_s
            )
            stable_time_s = (
                target_plan.find_first_stable_time(stable_search_start_s, resolution_end_s)
                if target_plan is not None
                else cut.time_s
            )
            if stable_time_s is None:
                blocked_reason = (
                    "target_camera_future_motion"
                    if target_motion_reason == "future"
                    else "target_camera_never_stabilized"
                )
                motion_events.append(
                    MonologueMotionEvent(
                        reason=blocked_reason,
                        from_camera_index=current_camera,
                        to_camera_index=next_camera,
                        planned_time_s=cut.time_s,
                        moving_from_s=target_motion_interval.start_s,
                        moving_to_s=target_motion_interval.end_s,
                    )
                )
                retry_after_s = max(retry_after_s, resolution_end_s + EPS)
                if resolution_end_s <= eligible_start_s + EPS:
                    break
                continue

            stable_since_s = target_plan.stable_since(stable_time_s) if target_plan else 0.0
            delayed_reason = (
                "delayed_for_future_motion"
                if target_motion_reason == "future"
                else "delayed_for_stability"
            )
            motion_events.append(
                MonologueMotionEvent(
                    reason=delayed_reason,
                    from_camera_index=current_camera,
                    to_camera_index=next_camera,
                    planned_time_s=cut.time_s,
                    actual_time_s=stable_time_s,
                    delay_s=stable_time_s - cut.time_s,
                    stable_since_s=stable_since_s,
                    moving_from_s=target_motion_interval.start_s,
                    moving_to_s=target_motion_interval.end_s,
                )
            )
            cut = replace(
                cut,
                time_s=stable_time_s,
                reason=delayed_reason,
                delta_s=stable_time_s - cut.target_s,
                planned_reason=cut.reason,
                planned_time_s=cut.time_s,
                stable_since_s=stable_since_s,
                moving_from_s=target_motion_interval.start_s,
                moving_to_s=target_motion_interval.end_s,
            )

        cuts.append(cut)
        current_camera = cut.camera_index
        last_cut_s = cut.time_s
        retry_after_s = last_cut_s

    camera_segments = _build_camera_segments(cuts, total_duration_s, config)
    forced_cuts = sum(
        1
        for cut in cuts
        if cut.reason == "forced_low_energy" or cut.planned_reason == "forced_low_energy"
    )
    main_duration_s = sum(
        seg.duration_s
        for seg in camera_segments
        if seg.camera_index == config.camera_main
    )
    accent_duration_s = sum(
        seg.duration_s
        for seg in camera_segments
        if seg.camera_index == config.camera_accent
    )
    actual_total_s = main_duration_s + accent_duration_s

    return Monologue2CamPlan(
        speech_segments=speech_segments,
        pause_segments=pause_segments,
        camera_segments=camera_segments,
        cuts=cuts,
        motion_events=motion_events,
        diagnostics={
            "pause_candidates": len(pause_segments),
            "camera_segments": len(camera_segments),
            "cuts": len(cuts),
            "forced_cuts": forced_cuts,
            "motion_enabled": bool(motion_plans),
            "motion_events": len(motion_events),
            "camera_main_target_share": config.camera_main_share,
            "camera_accent_target_share": config.camera_accent_share,
            "camera_main_target_hold_s": config.target_hold_s(config.camera_main),
            "camera_accent_target_hold_s": config.target_hold_s(config.camera_accent),
            "camera_main_actual_share": (
                None if actual_total_s <= EPS else main_duration_s / actual_total_s
            ),
            "camera_accent_actual_share": (
                None if actual_total_s <= EPS else accent_duration_s / actual_total_s
            ),
        },
    )


def _build_speech_segments(
    activity: SpeakerActivity,
    total_duration_s: float,
    hop_s: float,
    config: Monologue2CamConfig,
) -> list[Segment]:
    if total_duration_s <= 0:
        return []

    if not activity.frames:
        return [
            Segment(
                start_s=0.0,
                end_s=total_duration_s,
                camera_index=config.camera_main,
                speaker_state=SpeakerState.SILENCE,
            )
        ]

    states = [
        SpeakerState.SPEAKER_A if frame.is_active else SpeakerState.SILENCE
        for frame in activity.frames
    ]
    segments = run_length_encode(states, hop_s)
    segments[-1] = Segment(
        start_s=segments[-1].start_s,
        end_s=total_duration_s,
        camera_index=segments[-1].camera_index,
        speaker_state=segments[-1].speaker_state,
        speaker_label=segments[-1].speaker_label,
    )
    if config.micro_pause_merge_s > 0:
        segments = collapse_short_neutral_segments(
            segments,
            config.micro_pause_merge_s,
            {SpeakerState.SILENCE},
        )
    segments = merge_adjacent(segments)

    result: list[Segment] = []
    for seg in segments:
        result.append(
            Segment(
                start_s=seg.start_s,
                end_s=seg.end_s,
                camera_index=config.camera_main,
                speaker_state=seg.speaker_state,
                speaker_label=(
                    activity.speaker_label
                    if seg.speaker_state == SpeakerState.SPEAKER_A
                    else None
                ),
            )
        )
    return result


def _pause_cut_time(segment: Segment, padding_s: float) -> float:
    middle_s = (segment.start_s + segment.end_s) / 2.0
    earliest_s = segment.start_s + padding_s
    latest_s = segment.end_s - padding_s
    if earliest_s <= latest_s:
        return min(max(middle_s, earliest_s), latest_s)
    return middle_s


def _find_base_cut(
    activity: SpeakerActivity,
    pause_segments: list[Segment],
    pause_cut_times: list[float],
    frame_times: list[float],
    target_s: float,
    eligible_start_s: float,
    deadline_limit_s: float,
    next_camera: int,
    config: Monologue2CamConfig,
) -> MonologueCut | None:
    if target_s > deadline_limit_s + EPS and eligible_start_s > deadline_limit_s + EPS:
        return None

    cut = _find_pause_after_target(
        pause_segments,
        pause_cut_times,
        target_s,
        eligible_start_s,
        deadline_limit_s,
        next_camera,
        config,
    )
    if cut is None:
        cut = _find_pause_before_target(
            pause_segments,
            pause_cut_times,
            target_s,
            eligible_start_s,
            deadline_limit_s,
            next_camera,
            config,
        )
    if cut is None:
        cut = _find_forced_low_energy_cut(
            activity,
            frame_times,
            target_s,
            eligible_start_s,
            deadline_limit_s,
            next_camera,
            config,
        )
    return cut


def _find_pause_after_target(
    pause_segments: list[Segment],
    pause_cut_times: list[float],
    target_s: float,
    eligible_start_s: float,
    deadline_limit_s: float,
    next_camera: int,
    config: Monologue2CamConfig,
) -> MonologueCut | None:
    after_start_s = max(target_s, eligible_start_s)
    after_end_s = min(target_s + config.pause_window_after_s, deadline_limit_s)
    if after_start_s > after_end_s + EPS:
        return None

    idx = bisect_left(pause_cut_times, after_start_s)
    if idx >= len(pause_cut_times) or pause_cut_times[idx] > after_end_s + EPS:
        return None

    pause = pause_segments[idx]
    cut_time_s = pause_cut_times[idx]
    return MonologueCut(
        time_s=cut_time_s,
        camera_index=next_camera,
        reason="pause_after_target",
        target_s=target_s,
        delta_s=cut_time_s - target_s,
        pause_start_s=pause.start_s,
        pause_end_s=pause.end_s,
        pause_duration_s=pause.duration_s,
    )


def _find_pause_before_target(
    pause_segments: list[Segment],
    pause_cut_times: list[float],
    target_s: float,
    eligible_start_s: float,
    deadline_limit_s: float,
    next_camera: int,
    config: Monologue2CamConfig,
) -> MonologueCut | None:
    before_start_s = max(target_s - config.pause_window_before_s, eligible_start_s)
    before_end_s = min(target_s, deadline_limit_s)
    if before_start_s > before_end_s + EPS:
        return None

    left_idx = bisect_left(pause_cut_times, before_start_s)
    right_idx = bisect_left(pause_cut_times, before_end_s) - 1
    if left_idx > right_idx or right_idx < 0:
        return None

    pause = pause_segments[right_idx]
    cut_time_s = pause_cut_times[right_idx]
    return MonologueCut(
        time_s=cut_time_s,
        camera_index=next_camera,
        reason="pause_before_target",
        target_s=target_s,
        delta_s=cut_time_s - target_s,
        pause_start_s=pause.start_s,
        pause_end_s=pause.end_s,
        pause_duration_s=pause.duration_s,
    )


def _find_forced_low_energy_cut(
    activity: SpeakerActivity,
    frame_times: list[float],
    target_s: float,
    eligible_start_s: float,
    deadline_limit_s: float,
    next_camera: int,
    config: Monologue2CamConfig,
) -> MonologueCut | None:
    window_start_s = max(target_s, eligible_start_s)
    deadline_s = min(target_s + config.pause_window_after_s, deadline_limit_s)
    if window_start_s > deadline_s + EPS:
        return None

    start_idx = bisect_left(frame_times, window_start_s)
    end_idx = bisect_right(frame_times, deadline_s)
    if start_idx >= end_idx:
        cut_time_s = deadline_s
        min_envelope_db = None
    else:
        window_frames = activity.frames[start_idx:end_idx]
        min_frame = min(window_frames, key=lambda frame: frame.envelope_db)
        window_peak_db = max(frame.envelope_db for frame in window_frames)
        min_envelope_db = min_frame.envelope_db
        cut_time_s = (
            min_frame.time_s
            if window_peak_db - min_envelope_db >= config.forced_min_drop_db
            else deadline_s
        )

    return MonologueCut(
        time_s=cut_time_s,
        camera_index=next_camera,
        reason="forced_low_energy",
        target_s=target_s,
        delta_s=cut_time_s - target_s,
        min_envelope_db=min_envelope_db,
        fallback_deadline_s=deadline_s,
    )


def _find_emergency_escape(
    *,
    current_camera: int,
    next_camera: int,
    last_cut_s: float,
    upper_bound_s: float,
    emergency_deadline_s: float,
    motion_plans: dict[int, CameraMotionPlan],
    config: Monologue2CamConfig,
) -> tuple[MonologueCut | None, MonologueMotionEvent | None]:
    current_plan = motion_plans.get(current_camera)
    if current_plan is None:
        return None, None

    earliest_escape_s = last_cut_s + config.motion_escape_min_hold_s
    search_end_s = min(upper_bound_s, emergency_deadline_s)
    if earliest_escape_s > search_end_s + EPS:
        return None, None

    current_motion = current_plan.first_interval_between(earliest_escape_s, search_end_s)
    if current_motion is None:
        return None, None

    trigger_s = max(earliest_escape_s, current_motion.start_s)
    interval_deadline_s = min(current_motion.end_s, emergency_deadline_s)
    search_until_s = min(search_end_s, interval_deadline_s)
    if trigger_s > search_until_s + EPS:
        return None, None

    escape_plan = motion_plans.get(next_camera)
    if escape_plan is None:
        stable_time_s = trigger_s
        stable_since_s = 0.0
    else:
        stable_time_s = escape_plan.find_first_stable_time(trigger_s, search_until_s)
        stable_since_s = (
            escape_plan.stable_since(stable_time_s)
            if stable_time_s is not None
            else None
        )

    if stable_time_s is None:
        if search_until_s >= interval_deadline_s - EPS:
            return None, MonologueMotionEvent(
                reason="no_static_camera_available",
                from_camera_index=current_camera,
                to_camera_index=next_camera,
                planned_time_s=trigger_s,
                moving_from_s=current_motion.start_s,
                moving_to_s=current_motion.end_s,
            )
        return None, None

    cut = MonologueCut(
        time_s=stable_time_s,
        camera_index=next_camera,
        reason="escape_from_motion",
        target_s=trigger_s,
        delta_s=stable_time_s - trigger_s,
        planned_time_s=trigger_s,
        stable_since_s=stable_since_s,
        moving_from_s=current_motion.start_s,
        moving_to_s=current_motion.end_s,
    )
    event = MonologueMotionEvent(
        reason="escape_from_motion",
        from_camera_index=current_camera,
        to_camera_index=next_camera,
        planned_time_s=trigger_s,
        actual_time_s=stable_time_s,
        delay_s=stable_time_s - trigger_s,
        stable_since_s=stable_since_s,
        moving_from_s=current_motion.start_s,
        moving_to_s=current_motion.end_s,
    )
    return cut, event


def _alternate_camera(current_camera: int, config: Monologue2CamConfig) -> int:
    return (
        config.camera_accent
        if current_camera == config.camera_main
        else config.camera_main
    )


def _build_camera_segments(
    cuts: list[MonologueCut],
    total_duration_s: float,
    config: Monologue2CamConfig,
) -> list[MonologueCameraSegment]:
    if total_duration_s <= 0:
        return []

    segments: list[MonologueCameraSegment] = []
    current_camera = config.camera_main
    segment_start_s = 0.0
    segment_reason = "initial"

    for cut in cuts:
        segments.append(
            MonologueCameraSegment(
                start_s=segment_start_s,
                end_s=cut.time_s,
                camera_index=current_camera,
                reason=segment_reason,
            )
        )
        segment_start_s = cut.time_s
        current_camera = cut.camera_index
        segment_reason = cut.reason

    if segment_start_s < total_duration_s - EPS:
        segments.append(
            MonologueCameraSegment(
                start_s=segment_start_s,
                end_s=total_duration_s,
                camera_index=current_camera,
                reason=segment_reason,
            )
        )
    return segments
