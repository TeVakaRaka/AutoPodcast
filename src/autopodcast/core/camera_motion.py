"""Video motion analysis for monologue camera switching."""

from __future__ import annotations

import json
import math
import os
import subprocess
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np

from autopodcast.utils.ffmpeg import _find_tool, _subprocess_kwargs

EPS = 1e-9


@dataclass(frozen=True)
class CameraMotionExecutionConfig:
    speed: str = "balanced"
    hwaccel: str = "auto"

    def validate(self) -> None:
        if self.speed not in {"balanced", "turbo"}:
            raise ValueError(f"Unsupported motion speed '{self.speed}'")
        if self.hwaccel not in {"auto", "cpu", "hybrid"}:
            raise ValueError(f"Unsupported motion hwaccel mode '{self.hwaccel}'")

    def worker_count(
        self,
        clip_count: int,
        *,
        logical_cores: int | None = None,
    ) -> int:
        if clip_count <= 0:
            return 0
        cores = logical_cores or os.cpu_count() or 1
        if self.speed == "balanced":
            requested = min(max(2, cores // 2), 6)
        else:
            requested = min(max(2, cores - 1), 10)
        return max(1, min(requested, clip_count))


@dataclass(frozen=True)
class CameraMotionConfig:
    sample_fps: float = 2.0
    resize_width: int = 192
    border_ratio: float = 0.08
    ring_ratio: float = 0.18
    bottom_ignore_ratio: float = 0.30
    refine_sample_fps: float = 6.0
    refine_resize_width: int = 320
    candidate_score_threshold: float = 0.01
    candidate_padding_s: float = 1.0
    candidate_merge_gap_s: float = 1.0
    smoothing_window_s: float = 1.0
    motion_pre_roll_s: float = 0.4
    moving_on_threshold: float = 0.02
    moving_off_threshold: float = 0.01
    moving_confirm_s: float = 0.5
    stable_confirm_s: float = 0.8

    def validate(self) -> None:
        if self.sample_fps <= 0:
            raise ValueError(f"sample_fps must be positive, got {self.sample_fps}")
        if self.resize_width <= 0:
            raise ValueError(f"resize_width must be positive, got {self.resize_width}")
        if self.refine_sample_fps <= 0:
            raise ValueError(
                f"refine_sample_fps must be positive, got {self.refine_sample_fps}"
            )
        if self.refine_resize_width <= 0:
            raise ValueError(
                f"refine_resize_width must be positive, got {self.refine_resize_width}"
            )
        if not 0 < self.border_ratio < 0.5:
            raise ValueError(f"border_ratio must be between 0 and 0.5, got {self.border_ratio}")
        if not 0 < self.ring_ratio < 0.5:
            raise ValueError(f"ring_ratio must be between 0 and 0.5, got {self.ring_ratio}")
        if self.ring_ratio < self.border_ratio:
            raise ValueError(
                "ring_ratio must be >= border_ratio "
                f"({self.ring_ratio} < {self.border_ratio})"
            )
        if not 0 <= self.bottom_ignore_ratio < 1.0:
            raise ValueError(
                "bottom_ignore_ratio must be between 0 and 1 "
                f"(exclusive upper bound), got {self.bottom_ignore_ratio}"
            )
        if self.candidate_score_threshold < 0:
            raise ValueError(
                "candidate_score_threshold must be non-negative, "
                f"got {self.candidate_score_threshold}"
            )
        if self.candidate_padding_s < 0:
            raise ValueError(
                f"candidate_padding_s must be non-negative, got {self.candidate_padding_s}"
            )
        if self.candidate_merge_gap_s < 0:
            raise ValueError(
                f"candidate_merge_gap_s must be non-negative, got {self.candidate_merge_gap_s}"
            )
        if self.smoothing_window_s <= 0:
            raise ValueError(
                f"smoothing_window_s must be positive, got {self.smoothing_window_s}"
            )
        if self.motion_pre_roll_s < 0:
            raise ValueError(
                f"motion_pre_roll_s must be non-negative, got {self.motion_pre_roll_s}"
            )
        if self.moving_on_threshold < 0:
            raise ValueError(
                f"moving_on_threshold must be non-negative, got {self.moving_on_threshold}"
            )
        if self.moving_off_threshold < 0:
            raise ValueError(
                f"moving_off_threshold must be non-negative, got {self.moving_off_threshold}"
            )
        if self.moving_off_threshold > self.moving_on_threshold:
            raise ValueError(
                "moving_off_threshold must be <= moving_on_threshold "
                f"({self.moving_off_threshold} > {self.moving_on_threshold})"
            )
        if self.moving_confirm_s <= 0:
            raise ValueError(
                f"moving_confirm_s must be positive, got {self.moving_confirm_s}"
            )
        if self.stable_confirm_s <= 0:
            raise ValueError(
                f"stable_confirm_s must be positive, got {self.stable_confirm_s}"
            )


@dataclass(frozen=True)
class MotionInterval:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class CameraSourceClip:
    path: Path
    timeline_start_s: float = 0.0
    timeline_end_s: float | None = None
    source_start_s: float = 0.0
    source_end_s: float | None = None

    @property
    def timeline_duration_s(self) -> float | None:
        if self.timeline_end_s is None:
            return None
        return self.timeline_end_s - self.timeline_start_s

    @property
    def source_duration_s(self) -> float | None:
        if self.source_end_s is None:
            return None
        return self.source_end_s - self.source_start_s


@dataclass(frozen=True)
class CameraMotionPlan:
    source_path: Path | None = None
    sample_fps: float = 4.0
    score_times_s: tuple[float, ...] = field(default_factory=tuple)
    raw_scores: tuple[float, ...] = field(default_factory=tuple)
    smoothed_scores: tuple[float, ...] = field(default_factory=tuple)
    moving_intervals: tuple[MotionInterval, ...] = field(default_factory=tuple)
    diagnostics: dict = field(default_factory=dict)
    _interval_starts: tuple[float, ...] = field(init=False, repr=False, default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_interval_starts",
            tuple(interval.start_s for interval in self.moving_intervals),
        )

    def find_interval_at(self, time_s: float) -> MotionInterval | None:
        idx = bisect_right(self._interval_starts, time_s) - 1
        if idx < 0:
            return None
        interval = self.moving_intervals[idx]
        if interval.start_s - EPS <= time_s < interval.end_s - EPS:
            return interval
        return None

    def is_moving(self, time_s: float) -> bool:
        return self.find_interval_at(time_s) is not None

    def first_interval_between(self, start_s: float, end_s: float) -> MotionInterval | None:
        if start_s > end_s + EPS:
            return None
        idx = bisect_right(self._interval_starts, start_s) - 1
        if idx >= 0:
            interval = self.moving_intervals[idx]
            if interval.end_s > start_s + EPS:
                return interval
        idx = max(idx + 1, 0)
        if idx < len(self.moving_intervals):
            interval = self.moving_intervals[idx]
            if interval.start_s <= end_s + EPS:
                return interval
        return None

    def find_first_stable_time(self, start_s: float, end_s: float) -> float | None:
        if start_s > end_s + EPS:
            return None
        interval = self.find_interval_at(start_s)
        if interval is None:
            return start_s
        if interval.end_s <= end_s + EPS:
            return interval.end_s
        return None

    def stable_since(self, time_s: float) -> float | None:
        if self.is_moving(time_s):
            return None
        idx = bisect_right(self._interval_starts, time_s) - 1
        if idx >= 0:
            interval = self.moving_intervals[idx]
            if interval.end_s <= time_s + EPS:
                return interval.end_s
        return 0.0


@dataclass(frozen=True)
class CameraMotionRuntime:
    speed: str
    requested_hwaccel: str
    active_hwaccel: str
    worker_count: int
    clip_count: int
    cpu_worker_count: int = 0
    gpu_worker_count: int = 0
    cpu_clip_count: int = 0
    gpu_clip_count: int = 0
    probe_clip_path: str | None = None
    hwaccel_probe_used: bool = False


def compute_edge_motion_scores(
    frames: np.ndarray,
    border_ratio: float = 0.08,
    *,
    ring_ratio: float = 0.18,
    bottom_ignore_ratio: float = 0.30,
) -> np.ndarray:
    """Compute top/upper-side motion scores in the range 0..1."""
    return compute_motion_zone_scores(
        frames,
        border_ratio=border_ratio,
        ring_ratio=ring_ratio,
        bottom_ignore_ratio=bottom_ignore_ratio,
    )["score"]


def compute_motion_zone_scores(
    frames: np.ndarray,
    *,
    border_ratio: float = 0.08,
    ring_ratio: float = 0.18,
    bottom_ignore_ratio: float = 0.30,
) -> dict[str, np.ndarray]:
    """Compute motion scores from top and upper-side zones, excluding the lower frame."""
    if frames.ndim != 3:
        raise ValueError(
            f"Expected frames with shape (n, h, w), got {tuple(frames.shape)}"
        )
    if len(frames) < 2:
        empty = np.array([], dtype=float)
        return {
            "score": empty,
            "top": empty,
            "left": empty,
            "right": empty,
            "ring": empty,
        }

    diff = np.abs(np.diff(frames.astype(np.float32), axis=0)) / 255.0
    frame_count, height, width = diff.shape
    border_h = max(1, int(round(height * border_ratio)))
    border_w = max(1, int(round(width * border_ratio)))
    ring_h = max(border_h, int(round(height * ring_ratio)))
    ring_w = max(border_w, int(round(width * ring_ratio)))
    usable_height = max(border_h, int(round(height * (1.0 - bottom_ignore_ratio))))

    top = diff[:, :border_h, :].mean(axis=(1, 2))
    left = diff[:, :usable_height, :border_w].mean(axis=(1, 2))
    right = diff[:, :usable_height, width - border_w:].mean(axis=(1, 2))

    top_ring = diff[:, :ring_h, :].reshape(frame_count, -1)
    left_ring = diff[:, :usable_height, :ring_w].reshape(frame_count, -1)
    right_ring = diff[:, :usable_height, width - ring_w:].reshape(frame_count, -1)
    ring = np.concatenate([top_ring, left_ring, right_ring], axis=1).mean(axis=1)

    directional = np.maximum.reduce([top, left, right])
    score = np.maximum(directional, ring)
    return {
        "score": score,
        "top": top,
        "left": left,
        "right": right,
        "ring": ring,
    }


def smooth_motion_scores(
    scores: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """Apply a trailing rolling mean to motion scores."""
    if len(scores) == 0 or window_size <= 1:
        return scores.astype(float, copy=True)

    result = np.empty(len(scores), dtype=float)
    cumsum = np.cumsum(np.insert(scores.astype(float), 0, 0.0))
    for idx in range(len(scores)):
        start_idx = max(0, idx - window_size + 1)
        total = cumsum[idx + 1] - cumsum[start_idx]
        result[idx] = total / (idx - start_idx + 1)
    return result


def build_motion_plan_from_scores(
    scores: list[float] | np.ndarray,
    sample_fps: float,
    config: CameraMotionConfig,
    *,
    source_path: Path | None = None,
    total_duration_s: float | None = None,
) -> CameraMotionPlan:
    """Build motion intervals from precomputed frame-difference scores."""
    config.validate()
    if sample_fps <= 0:
        raise ValueError(f"sample_fps must be positive, got {sample_fps}")

    scores_arr = np.asarray(scores, dtype=float)
    times = (
        np.arange(1, len(scores_arr) + 1, dtype=float) / sample_fps
        if len(scores_arr)
        else np.array([], dtype=float)
    )
    smoothing_frames = max(1, int(math.ceil(config.smoothing_window_s * sample_fps)))
    moving_frames = max(1, int(math.ceil(config.moving_confirm_s * sample_fps)))
    stable_frames = max(1, int(math.ceil(config.stable_confirm_s * sample_fps)))
    smoothed = smooth_motion_scores(scores_arr, smoothing_frames)

    intervals: list[MotionInterval] = []
    moving = False
    current_start_s: float | None = None
    above_start_idx: int | None = None
    below_start_idx: int | None = None

    for idx, score in enumerate(smoothed):
        if moving:
            if score <= config.moving_off_threshold + EPS:
                if below_start_idx is None:
                    below_start_idx = idx
                if idx - below_start_idx + 1 >= stable_frames:
                    end_idx = below_start_idx + stable_frames - 1
                    end_s = float(times[end_idx])
                    intervals.append(
                        MotionInterval(
                            start_s=(
                                current_start_s
                                if current_start_s is not None
                                else end_s
                            ),
                            end_s=end_s,
                        )
                    )
                    moving = False
                    current_start_s = None
                    above_start_idx = None
                    below_start_idx = None
            else:
                below_start_idx = None
        else:
            if score >= config.moving_on_threshold - EPS:
                if above_start_idx is None:
                    above_start_idx = idx
                if idx - above_start_idx + 1 >= moving_frames:
                    current_start_s = max(
                        0.0,
                        float(times[above_start_idx]) - config.motion_pre_roll_s,
                    )
                    moving = True
                    above_start_idx = None
                    below_start_idx = None
            else:
                above_start_idx = None

    if moving and current_start_s is not None:
        if total_duration_s is None:
            end_s = float(times[-1]) if len(times) else current_start_s
        else:
            end_s = max(current_start_s, float(total_duration_s))
        intervals.append(MotionInterval(start_s=current_start_s, end_s=end_s))

    return CameraMotionPlan(
        source_path=source_path,
        sample_fps=sample_fps,
        score_times_s=tuple(float(value) for value in times.tolist()),
        raw_scores=tuple(float(value) for value in scores_arr.tolist()),
        smoothed_scores=tuple(float(value) for value in smoothed.tolist()),
        moving_intervals=tuple(intervals),
        diagnostics={
            "score_count": int(len(scores_arr)),
            "moving_interval_count": len(intervals),
            "max_raw_score": None if len(scores_arr) == 0 else float(scores_arr.max()),
            "max_smoothed_score": None if len(smoothed) == 0 else float(smoothed.max()),
            "smoothing_window_frames": smoothing_frames,
            "moving_confirm_frames": moving_frames,
            "stable_confirm_frames": stable_frames,
        },
    )


def _build_candidate_windows(
    *,
    score_times_s: tuple[float, ...],
    raw_scores: tuple[float, ...],
    smoothed_scores: tuple[float, ...],
    coarse_intervals: tuple[MotionInterval, ...],
    total_duration_s: float | None,
    config: CameraMotionConfig,
) -> tuple[MotionInterval, ...]:
    if not score_times_s:
        return tuple(coarse_intervals)

    candidate_intervals: list[MotionInterval] = list(coarse_intervals)
    start_idx: int | None = None
    threshold = config.candidate_score_threshold
    frame_step_s = 1.0 / config.sample_fps

    for idx, (raw_score, smoothed_score) in enumerate(zip(raw_scores, smoothed_scores)):
        above = (
            raw_score >= threshold - EPS
            or smoothed_score >= threshold - EPS
        )
        if above and start_idx is None:
            start_idx = idx
        elif not above and start_idx is not None:
            candidate_intervals.append(
                MotionInterval(
                    start_s=max(
                        0.0,
                        score_times_s[start_idx] - frame_step_s - config.candidate_padding_s,
                    ),
                    end_s=score_times_s[idx - 1] + config.candidate_padding_s,
                )
            )
            start_idx = None

    if start_idx is not None:
        interval_end_s = score_times_s[-1] + config.candidate_padding_s
        if total_duration_s is not None:
            interval_end_s = min(interval_end_s, total_duration_s)
        candidate_intervals.append(
            MotionInterval(
                start_s=max(
                    0.0,
                    score_times_s[start_idx] - frame_step_s - config.candidate_padding_s,
                ),
                end_s=interval_end_s,
            )
        )

    merged = _merge_intervals_with_gap(candidate_intervals, config.candidate_merge_gap_s)
    if total_duration_s is not None:
        merged = [
            MotionInterval(
                start_s=max(0.0, interval.start_s),
                end_s=min(float(total_duration_s), interval.end_s),
            )
            for interval in merged
            if interval.end_s > interval.start_s + EPS
        ]
    return tuple(merged)


class CameraMotionAnalyzer:
    """Extract frames with ffmpeg and build a motion plan."""

    def __init__(
        self,
        config: CameraMotionConfig | None = None,
        execution_config: CameraMotionExecutionConfig | None = None,
    ):
        self.config = config or CameraMotionConfig()
        self.config.validate()
        self.execution_config = execution_config or CameraMotionExecutionConfig()
        self.execution_config.validate()
        self.last_runtime = CameraMotionRuntime(
            speed=self.execution_config.speed,
            requested_hwaccel=self.execution_config.hwaccel,
            active_hwaccel="cpu" if self.execution_config.hwaccel == "cpu" else self.execution_config.hwaccel,
            worker_count=1,
            clip_count=0,
            cpu_worker_count=1,
        )

    def analyze(
        self,
        video_path: Path,
        *,
        total_duration_s: float | None = None,
        progress_callback: Callable[[CameraSourceClip], None] | None = None,
    ) -> CameraMotionPlan:
        return self.analyze_clips(
            [CameraSourceClip(path=video_path)],
            total_duration_s=total_duration_s,
            progress_callback=progress_callback,
        )

    def prepare_runtime(
        self,
        clips: list[CameraSourceClip] | tuple[CameraSourceClip, ...],
    ) -> CameraMotionRuntime:
        if not clips:
            raise ValueError("No clips provided for motion analysis")
        worker_count = self.execution_config.worker_count(len(clips))
        active_hwaccel = self._resolve_hwaccel_mode(clips[0])
        cpu_worker_count, gpu_worker_count = self._resolve_worker_split(
            worker_count,
            active_hwaccel=active_hwaccel,
        )
        cpu_clip_count = len(clips)
        gpu_clip_count = 0
        if active_hwaccel == "hybrid":
            backend_modes = self._assign_hybrid_backends(
                list(clips),
                cpu_worker_count=cpu_worker_count,
                gpu_worker_count=gpu_worker_count,
            )
            cpu_clip_count = sum(1 for mode in backend_modes if mode == "cpu")
            gpu_clip_count = sum(1 for mode in backend_modes if mode == "auto")
        runtime = CameraMotionRuntime(
            speed=self.execution_config.speed,
            requested_hwaccel=self.execution_config.hwaccel,
            active_hwaccel=active_hwaccel,
            worker_count=worker_count,
            clip_count=len(clips),
            cpu_worker_count=cpu_worker_count,
            gpu_worker_count=gpu_worker_count,
            cpu_clip_count=cpu_clip_count,
            gpu_clip_count=gpu_clip_count,
            probe_clip_path=str(clips[0].path),
            hwaccel_probe_used=self.execution_config.hwaccel in {"auto", "hybrid"},
        )
        self.last_runtime = runtime
        return runtime

    def analyze_clips(
        self,
        clips: list[CameraSourceClip] | tuple[CameraSourceClip, ...],
        *,
        total_duration_s: float | None = None,
        progress_callback: Callable[[CameraSourceClip], None] | None = None,
        runtime: CameraMotionRuntime | None = None,
    ) -> CameraMotionPlan:
        """Analyze one camera assembled from one or more timeline clips."""
        if not clips:
            raise ValueError("No clips provided for motion analysis")
        runtime = runtime or self.prepare_runtime(clips)
        self.last_runtime = runtime
        clip_plans: list[CameraMotionPlan | None] = [None] * len(clips)
        jobs = [(idx, clip) for idx, clip in enumerate(clips)]
        for idx, clip, plan in self._run_clip_jobs(jobs, runtime=runtime):
            clip_plans[idx] = plan
            if progress_callback is not None:
                progress_callback(clip)

        return self._assemble_clips_plan(
            list(clips),
            [plan for plan in clip_plans if plan is not None],
            total_duration_s=total_duration_s,
            runtime=runtime,
        )

    def analyze_camera_groups(
        self,
        clip_groups: dict[object, list[CameraSourceClip] | tuple[CameraSourceClip, ...]],
        *,
        total_duration_s: float | None = None,
        progress_callback: Callable[[CameraSourceClip], None] | None = None,
        runtime: CameraMotionRuntime | None = None,
    ) -> dict[object, CameraMotionPlan]:
        if not clip_groups:
            raise ValueError("No clip groups provided for motion analysis")
        normalized_groups = {
            key: list(clips)
            for key, clips in clip_groups.items()
            if clips
        }
        if not normalized_groups:
            raise ValueError("No clips provided for motion analysis")

        ordered_jobs = [
            (key, idx, clip)
            for key, clips in normalized_groups.items()
            for idx, clip in enumerate(clips)
        ]
        runtime = runtime or self.prepare_runtime(
            [clip for _key, _idx, clip in ordered_jobs]
        )
        self.last_runtime = runtime
        grouped_plans: dict[object, list[CameraMotionPlan | None]] = {
            key: [None] * len(clips)
            for key, clips in normalized_groups.items()
        }
        jobs = [(job_idx, key, idx, clip) for job_idx, (key, idx, clip) in enumerate(ordered_jobs)]
        for _job_idx, key, idx, clip, plan in self._run_group_jobs(jobs, runtime=runtime):
            grouped_plans[key][idx] = plan
            if progress_callback is not None:
                progress_callback(clip)

        return {
            key: self._assemble_clips_plan(
                normalized_groups[key],
                [plan for plan in plans if plan is not None],
                total_duration_s=total_duration_s,
                runtime=runtime,
            )
            for key, plans in grouped_plans.items()
        }

    def _assemble_clips_plan(
        self,
        clips: list[CameraSourceClip],
        clip_plans: list[CameraMotionPlan],
        *,
        total_duration_s: float | None,
        runtime: CameraMotionRuntime,
    ) -> CameraMotionPlan:
        intervals: list[MotionInterval] = []
        diagnostics_clips: list[dict] = []
        source_paths: list[str] = []

        for clip, clip_plan in zip(clips, clip_plans):
            shift_s = clip.timeline_start_s - clip.source_start_s
            for interval in clip_plan.moving_intervals:
                shifted_start_s = interval.start_s + shift_s
                shifted_end_s = interval.end_s + shift_s
                if clip.timeline_end_s is not None:
                    shifted_start_s = max(shifted_start_s, clip.timeline_start_s)
                    shifted_end_s = min(shifted_end_s, clip.timeline_end_s)
                if shifted_end_s <= shifted_start_s + EPS:
                    continue
                intervals.append(
                    MotionInterval(
                        start_s=shifted_start_s,
                        end_s=shifted_end_s,
                    )
                )
            clip_diag = dict(clip_plan.diagnostics)
            clip_diag.update(
                {
                    "timeline_start_s": clip.timeline_start_s,
                    "timeline_end_s": clip.timeline_end_s,
                    "source_start_s": clip.source_start_s,
                    "source_end_s": clip.source_end_s,
                }
            )
            diagnostics_clips.append(clip_diag)
            source_paths.append(str(clip.path))

        merged = _merge_intervals(intervals)
        overall_end_s = total_duration_s
        if overall_end_s is None:
            ends = [
                clip.timeline_end_s
                for clip in clips
                if clip.timeline_end_s is not None
            ]
            overall_end_s = max(ends) if ends else None
        if overall_end_s is not None:
            merged = tuple(
                MotionInterval(
                    start_s=max(0.0, interval.start_s),
                    end_s=min(float(overall_end_s), interval.end_s),
                )
                for interval in merged
                if interval.end_s > interval.start_s + EPS
            )
        else:
            merged = tuple(merged)

        return CameraMotionPlan(
            source_path=clips[0].path,
            sample_fps=self.config.sample_fps,
            moving_intervals=merged,
            diagnostics={
                "clip_count": len(clips),
                "source_paths": source_paths,
                "moving_interval_count": len(merged),
                "candidate_window_count": sum(
                    int(diag.get("candidate_window_count", 0))
                    for diag in diagnostics_clips
                ),
                "refined_window_count": sum(
                    int(diag.get("refined_window_count", 0))
                    for diag in diagnostics_clips
                ),
                "clips": diagnostics_clips,
                "motion_speed": runtime.speed,
                "motion_hwaccel_requested": runtime.requested_hwaccel,
                "motion_hwaccel_active": runtime.active_hwaccel,
                "motion_worker_count": runtime.worker_count,
                "motion_cpu_worker_count": runtime.cpu_worker_count,
                "motion_gpu_worker_count": runtime.gpu_worker_count,
                "motion_cpu_clip_count": runtime.cpu_clip_count,
                "motion_gpu_clip_count": runtime.gpu_clip_count,
            },
        )

    def _run_clip_jobs(
        self,
        jobs: list[tuple[int, CameraSourceClip]],
        *,
        runtime: CameraMotionRuntime,
    ):
        if runtime.active_hwaccel != "hybrid":
            with ThreadPoolExecutor(max_workers=runtime.worker_count) as executor:
                futures = {
                    executor.submit(self._analyze_clip, clip, runtime=runtime): (idx, clip)
                    for idx, clip in jobs
                }
                for future in as_completed(futures):
                    idx, clip = futures[future]
                    yield idx, clip, future.result()
            return

        backend_modes = self._assign_hybrid_backends(
            [clip for _idx, clip in jobs],
            cpu_worker_count=runtime.cpu_worker_count,
            gpu_worker_count=runtime.gpu_worker_count,
        )
        cpu_jobs = []
        gpu_jobs = []
        for (idx, clip), backend_mode in zip(jobs, backend_modes):
            if backend_mode == "auto":
                gpu_jobs.append((idx, clip))
            else:
                cpu_jobs.append((idx, clip))
        yield from self._run_backend_split_clip_jobs(cpu_jobs, gpu_jobs, runtime=runtime)

    def _run_group_jobs(
        self,
        jobs: list[tuple[int, object, int, CameraSourceClip]],
        *,
        runtime: CameraMotionRuntime,
    ):
        if runtime.active_hwaccel != "hybrid":
            with ThreadPoolExecutor(max_workers=runtime.worker_count) as executor:
                futures = {
                    executor.submit(self._analyze_clip, clip, runtime=runtime): (job_idx, key, idx, clip)
                    for job_idx, key, idx, clip in jobs
                }
                for future in as_completed(futures):
                    job_idx, key, idx, clip = futures[future]
                    yield job_idx, key, idx, clip, future.result()
            return

        backend_modes = self._assign_hybrid_backends(
            [clip for _job_idx, _key, _idx, clip in jobs],
            cpu_worker_count=runtime.cpu_worker_count,
            gpu_worker_count=runtime.gpu_worker_count,
        )
        cpu_jobs = []
        gpu_jobs = []
        for (job_idx, key, idx, clip), backend_mode in zip(jobs, backend_modes):
            item = (job_idx, key, idx, clip)
            if backend_mode == "auto":
                gpu_jobs.append(item)
            else:
                cpu_jobs.append(item)
        yield from self._run_backend_split_group_jobs(cpu_jobs, gpu_jobs, runtime=runtime)

    def _run_backend_split_clip_jobs(
        self,
        cpu_jobs: list[tuple[int, CameraSourceClip]],
        gpu_jobs: list[tuple[int, CameraSourceClip]],
        *,
        runtime: CameraMotionRuntime,
    ):
        with self._backend_executor_bundle(runtime) as executors:
            futures: dict = {}
            cpu_runtime = replace(runtime, active_hwaccel="cpu")
            gpu_runtime = replace(runtime, active_hwaccel="auto")
            if executors["cpu"] is not None:
                futures.update(
                    {
                        executors["cpu"].submit(self._analyze_clip, clip, runtime=cpu_runtime): (idx, clip)
                        for idx, clip in cpu_jobs
                    }
                )
            if executors["gpu"] is not None:
                futures.update(
                    {
                        executors["gpu"].submit(self._analyze_clip, clip, runtime=gpu_runtime): (idx, clip)
                        for idx, clip in gpu_jobs
                    }
                )
            for future in as_completed(futures):
                idx, clip = futures[future]
                yield idx, clip, future.result()

    def _run_backend_split_group_jobs(
        self,
        cpu_jobs: list[tuple[int, object, int, CameraSourceClip]],
        gpu_jobs: list[tuple[int, object, int, CameraSourceClip]],
        *,
        runtime: CameraMotionRuntime,
    ):
        with self._backend_executor_bundle(runtime) as executors:
            futures: dict = {}
            cpu_runtime = replace(runtime, active_hwaccel="cpu")
            gpu_runtime = replace(runtime, active_hwaccel="auto")
            if executors["cpu"] is not None:
                futures.update(
                    {
                        executors["cpu"].submit(self._analyze_clip, clip, runtime=cpu_runtime): (job_idx, key, idx, clip)
                        for job_idx, key, idx, clip in cpu_jobs
                    }
                )
            if executors["gpu"] is not None:
                futures.update(
                    {
                        executors["gpu"].submit(self._analyze_clip, clip, runtime=gpu_runtime): (job_idx, key, idx, clip)
                        for job_idx, key, idx, clip in gpu_jobs
                    }
                )
            for future in as_completed(futures):
                job_idx, key, idx, clip = futures[future]
                yield job_idx, key, idx, clip, future.result()

    @staticmethod
    def _backend_executor_bundle(runtime: CameraMotionRuntime):
        class _ExecutorBundle:
            def __init__(self, runtime_obj: CameraMotionRuntime):
                self.runtime = runtime_obj
                self.executors = {"cpu": None, "gpu": None}

            def __enter__(self):
                if self.runtime.cpu_worker_count > 0:
                    self.executors["cpu"] = ThreadPoolExecutor(max_workers=self.runtime.cpu_worker_count)
                if self.runtime.gpu_worker_count > 0:
                    self.executors["gpu"] = ThreadPoolExecutor(max_workers=self.runtime.gpu_worker_count)
                return self.executors

            def __exit__(self, exc_type, exc, tb):
                for executor in self.executors.values():
                    if executor is not None:
                        executor.shutdown(wait=True)
                return False

        return _ExecutorBundle(runtime)

    def _resolve_worker_split(
        self,
        worker_count: int,
        *,
        active_hwaccel: str,
    ) -> tuple[int, int]:
        if active_hwaccel == "hybrid":
            gpu_worker_count = 1 if worker_count >= 2 else 0
            cpu_worker_count = max(1, worker_count - gpu_worker_count)
            return cpu_worker_count, gpu_worker_count
        if active_hwaccel == "cpu":
            return worker_count, 0
        return 0, worker_count

    def _assign_hybrid_backends(
        self,
        clips: list[CameraSourceClip],
        *,
        cpu_worker_count: int,
        gpu_worker_count: int,
    ) -> list[str]:
        if not clips:
            return []
        if gpu_worker_count <= 0:
            return ["cpu"] * len(clips)
        if cpu_worker_count <= 0:
            return ["auto"] * len(clips)

        backend_loads = {
            "cpu": 0.0,
            "auto": 0.0,
        }
        backend_workers = {
            "cpu": max(1, cpu_worker_count),
            "auto": max(1, gpu_worker_count),
        }
        assignments = ["cpu"] * len(clips)
        order = sorted(
            range(len(clips)),
            key=lambda idx: self._estimate_clip_duration_s(clips[idx]),
            reverse=True,
        )
        for idx in order:
            duration_s = self._estimate_clip_duration_s(clips[idx])
            backend = min(
                ("cpu", "auto"),
                key=lambda mode: backend_loads[mode] / backend_workers[mode],
            )
            assignments[idx] = backend
            backend_loads[backend] += duration_s
        return assignments

    @staticmethod
    def _estimate_clip_duration_s(clip: CameraSourceClip) -> float:
        if clip.source_duration_s is not None and clip.source_duration_s > EPS:
            return float(clip.source_duration_s)
        if clip.timeline_duration_s is not None and clip.timeline_duration_s > EPS:
            return float(clip.timeline_duration_s)
        return 1.0

    def _resolve_hwaccel_mode(self, probe_clip: CameraSourceClip) -> str:
        if self.execution_config.hwaccel == "cpu":
            return "cpu"
        if self.execution_config.hwaccel == "hybrid":
            if _probe_hwaccel_auto(
                probe_clip.path,
                start_s=probe_clip.source_start_s,
                duration_s=probe_clip.source_duration_s,
            ):
                return "hybrid"
            return "cpu"
        if _probe_hwaccel_auto(
            probe_clip.path,
            start_s=probe_clip.source_start_s,
            duration_s=probe_clip.source_duration_s,
        ):
            return "auto"
        return "cpu"

    def _analyze_clip(
        self,
        clip: CameraSourceClip,
        *,
        runtime: CameraMotionRuntime | None = None,
    ) -> CameraMotionPlan:
        width, height = _probe_video_dimensions(clip.path)
        coarse_width = self.config.resize_width
        coarse_height = max(1, int(round(height * coarse_width / width)))
        refine_width = self.config.refine_resize_width
        refine_height = max(1, int(round(height * refine_width / width)))
        clip_duration_s = clip.source_duration_s
        runtime = runtime or self.last_runtime
        coarse_frames = _extract_gray_frames(
            clip.path,
            sample_fps=self.config.sample_fps,
            resize_width=coarse_width,
            resize_height=coarse_height,
            start_s=clip.source_start_s,
            duration_s=clip_duration_s,
            hwaccel_mode=runtime.active_hwaccel,
        )
        coarse_scores = compute_motion_zone_scores(
            coarse_frames,
            border_ratio=self.config.border_ratio,
            ring_ratio=self.config.ring_ratio,
            bottom_ignore_ratio=self.config.bottom_ignore_ratio,
        )
        coarse_plan = build_motion_plan_from_scores(
            coarse_scores["score"],
            self.config.sample_fps,
            self.config,
            source_path=clip.path,
            total_duration_s=clip_duration_s,
        )
        candidate_windows = _build_candidate_windows(
            score_times_s=coarse_plan.score_times_s,
            raw_scores=coarse_plan.raw_scores,
            smoothed_scores=coarse_plan.smoothed_scores,
            coarse_intervals=coarse_plan.moving_intervals,
            total_duration_s=clip_duration_s,
            config=self.config,
        )

        refined_intervals: list[MotionInterval] = []
        for window in candidate_windows:
            window_duration_s = window.end_s - window.start_s
            if window_duration_s <= EPS:
                continue
            refine_frames = _extract_gray_frames(
                clip.path,
                sample_fps=self.config.refine_sample_fps,
                resize_width=refine_width,
                resize_height=refine_height,
                start_s=clip.source_start_s + window.start_s,
                duration_s=window_duration_s,
                hwaccel_mode=runtime.active_hwaccel,
            )
            refine_scores = compute_motion_zone_scores(
                refine_frames,
                border_ratio=self.config.border_ratio,
                ring_ratio=self.config.ring_ratio,
                bottom_ignore_ratio=self.config.bottom_ignore_ratio,
            )
            refine_plan = build_motion_plan_from_scores(
                refine_scores["score"],
                self.config.refine_sample_fps,
                self.config,
                source_path=clip.path,
                total_duration_s=window_duration_s,
            )
            for interval in refine_plan.moving_intervals:
                refined_intervals.append(
                    MotionInterval(
                        start_s=window.start_s + interval.start_s,
                        end_s=window.start_s + interval.end_s,
                    )
                )

        merged_intervals = tuple(_merge_intervals(refined_intervals))
        diagnostics = dict(coarse_plan.diagnostics)
        diagnostics.update(
            {
                "moving_interval_count": len(merged_intervals),
                "source_path": str(clip.path),
                "frame_count": int(coarse_frames.shape[0]),
                "frame_width": coarse_width,
                "frame_height": coarse_height,
                "coarse_sample_fps": self.config.sample_fps,
                "refine_sample_fps": self.config.refine_sample_fps,
                "refine_frame_width": refine_width,
                "refine_frame_height": refine_height,
                "candidate_window_count": len(candidate_windows),
                "refined_window_count": len(candidate_windows),
                "candidate_windows": [
                    _serialize_motion_window(interval)
                    for interval in candidate_windows
                ],
                "refined_windows": [
                    _serialize_motion_window(interval)
                    for interval in candidate_windows
                ],
                "top_max_score": (
                    None
                    if len(coarse_scores["top"]) == 0
                    else float(np.max(coarse_scores["top"]))
                ),
                "left_max_score": (
                    None
                    if len(coarse_scores["left"]) == 0
                    else float(np.max(coarse_scores["left"]))
                ),
                "right_max_score": (
                    None
                    if len(coarse_scores["right"]) == 0
                    else float(np.max(coarse_scores["right"]))
                ),
                "ring_max_score": (
                    None
                    if len(coarse_scores["ring"]) == 0
                    else float(np.max(coarse_scores["ring"]))
                ),
                "motion_hwaccel_active": runtime.active_hwaccel,
                "motion_cpu_worker_count": runtime.cpu_worker_count,
                "motion_gpu_worker_count": runtime.gpu_worker_count,
            }
        )
        return CameraMotionPlan(
            source_path=coarse_plan.source_path,
            sample_fps=coarse_plan.sample_fps,
            score_times_s=coarse_plan.score_times_s,
            raw_scores=coarse_plan.raw_scores,
            smoothed_scores=coarse_plan.smoothed_scores,
            moving_intervals=merged_intervals,
            diagnostics=diagnostics,
        )


def _probe_video_dimensions(path: Path) -> tuple[int, int]:
    ffprobe = _find_tool("ffprobe")
    cmd = [
        ffprobe,
        "-v",
        "quiet",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
        **_subprocess_kwargs(),
    )
    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        raise ValueError(f"No video stream found in {path}")
    width = int(streams[0]["width"])
    height = int(streams[0]["height"])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid video dimensions in {path}: {width}x{height}")
    return width, height


def _extract_gray_frames(
    path: Path,
    *,
    sample_fps: float,
    resize_width: int,
    resize_height: int,
    start_s: float = 0.0,
    duration_s: float | None = None,
    hwaccel_mode: str = "cpu",
) -> np.ndarray:
    ffmpeg = _find_tool("ffmpeg")
    filter_chain = f"fps={sample_fps},scale={resize_width}:{resize_height},format=gray"
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-nostdin",
    ]
    if hwaccel_mode == "auto":
        cmd.extend(["-hwaccel", "auto"])
    if start_s > 0:
        cmd.extend(["-ss", f"{start_s:.6f}"])
    cmd.extend([
        "-i",
        str(path),
    ])
    cmd.extend(["-an", "-sn", "-dn"])
    if hwaccel_mode == "cpu":
        cmd.extend(["-threads", "1"])
    if duration_s is not None and duration_s > 0:
        cmd.extend(["-t", f"{duration_s:.6f}"])
    cmd.extend([
        "-vf",
        filter_chain,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ])
    result = subprocess.run(
        cmd,
        capture_output=True,
        check=True,
        **_subprocess_kwargs(),
    )
    frame_size = resize_width * resize_height
    if frame_size <= 0:
        raise ValueError("Invalid frame size for motion analysis")
    raw = result.stdout
    if len(raw) == 0:
        return np.zeros((0, resize_height, resize_width), dtype=np.uint8)
    remainder = len(raw) % frame_size
    if remainder:
        raw = raw[: len(raw) - remainder]
    if not raw:
        return np.zeros((0, resize_height, resize_width), dtype=np.uint8)
    frames = np.frombuffer(raw, dtype=np.uint8)
    return frames.reshape((-1, resize_height, resize_width))


def _probe_hwaccel_auto(
    path: Path,
    *,
    start_s: float = 0.0,
    duration_s: float | None = None,
) -> bool:
    ffmpeg = _find_tool("ffmpeg")
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-nostdin",
        "-hwaccel",
        "auto",
    ]
    if start_s > 0:
        cmd.extend(["-ss", f"{start_s:.6f}"])
    cmd.extend(["-i", str(path), "-an", "-sn", "-dn", "-frames:v", "1"])
    if duration_s is not None and duration_s > 0:
        cmd.extend(["-t", f"{min(duration_s, 1.0):.6f}"])
    cmd.extend(["-f", "null", "-"])
    result = subprocess.run(
        cmd,
        capture_output=True,
        check=False,
        **_subprocess_kwargs(),
    )
    return result.returncode == 0


def _serialize_motion_window(interval: MotionInterval) -> dict[str, float]:
    return {
        "start_s": round(interval.start_s, 3),
        "end_s": round(interval.end_s, 3),
        "duration_s": round(interval.duration_s, 3),
    }


def _merge_intervals_with_gap(
    intervals: list[MotionInterval],
    max_gap_s: float,
) -> list[MotionInterval]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda interval: (interval.start_s, interval.end_s))
    merged = [ordered[0]]
    for interval in ordered[1:]:
        prev = merged[-1]
        if interval.start_s <= prev.end_s + max_gap_s + EPS:
            merged[-1] = MotionInterval(
                start_s=prev.start_s,
                end_s=max(prev.end_s, interval.end_s),
            )
        else:
            merged.append(interval)
    return merged


def _merge_intervals(intervals: list[MotionInterval]) -> list[MotionInterval]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda interval: (interval.start_s, interval.end_s))
    merged = [ordered[0]]
    for interval in ordered[1:]:
        prev = merged[-1]
        if interval.start_s <= prev.end_s + EPS:
            merged[-1] = MotionInterval(
                start_s=prev.start_s,
                end_s=max(prev.end_s, interval.end_s),
            )
        else:
            merged.append(interval)
    return merged
