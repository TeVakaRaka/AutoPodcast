"""Small JSON cache for camera motion plans."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from autopodcast.core.camera_motion import CameraMotionConfig, CameraMotionPlan, CameraSourceClip, MotionInterval


def motion_cache_key(
    clips: list[CameraSourceClip] | tuple[CameraSourceClip, ...],
    config: CameraMotionConfig,
) -> str:
    """Build a stable cache key for a camera group and motion config."""
    payload = {
        "version": 1,
        "config": _motion_config_payload(config),
        "clips": [_clip_payload(clip) for clip in clips],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_motion_plan_from_cache(cache_dir: Path, key: str) -> CameraMotionPlan | None:
    path = _cache_path(cache_dir, key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return motion_plan_from_dict(data)
    except Exception:
        return None


def save_motion_plan_to_cache(cache_dir: Path, key: str, plan: CameraMotionPlan) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, key)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(motion_plan_to_dict(plan), ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(path)
    return path


def motion_plan_to_dict(plan: CameraMotionPlan) -> dict[str, Any]:
    return {
        "source_path": None if plan.source_path is None else str(plan.source_path),
        "sample_fps": plan.sample_fps,
        "score_times_s": list(plan.score_times_s),
        "raw_scores": list(plan.raw_scores),
        "smoothed_scores": list(plan.smoothed_scores),
        "moving_intervals": [
            {"start_s": interval.start_s, "end_s": interval.end_s}
            for interval in plan.moving_intervals
        ],
        "diagnostics": plan.diagnostics,
    }


def motion_plan_from_dict(data: dict[str, Any]) -> CameraMotionPlan:
    return CameraMotionPlan(
        source_path=None if data.get("source_path") is None else Path(data["source_path"]),
        sample_fps=float(data.get("sample_fps", 4.0)),
        score_times_s=tuple(float(value) for value in data.get("score_times_s", [])),
        raw_scores=tuple(float(value) for value in data.get("raw_scores", [])),
        smoothed_scores=tuple(float(value) for value in data.get("smoothed_scores", [])),
        moving_intervals=tuple(
            MotionInterval(float(item["start_s"]), float(item["end_s"]))
            for item in data.get("moving_intervals", [])
        ),
        diagnostics=dict(data.get("diagnostics", {})),
    )


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.json"


def _motion_config_payload(config: CameraMotionConfig) -> dict[str, Any]:
    return {
        "sample_fps": config.sample_fps,
        "resize_width": config.resize_width,
        "border_ratio": config.border_ratio,
        "ring_ratio": config.ring_ratio,
        "bottom_ignore_ratio": config.bottom_ignore_ratio,
        "refine_sample_fps": config.refine_sample_fps,
        "refine_resize_width": config.refine_resize_width,
        "candidate_score_threshold": config.candidate_score_threshold,
        "candidate_padding_s": config.candidate_padding_s,
        "candidate_merge_gap_s": config.candidate_merge_gap_s,
        "smoothing_window_s": config.smoothing_window_s,
        "motion_pre_roll_s": config.motion_pre_roll_s,
        "moving_on_threshold": config.moving_on_threshold,
        "moving_off_threshold": config.moving_off_threshold,
        "moving_confirm_s": config.moving_confirm_s,
        "stable_confirm_s": config.stable_confirm_s,
    }


def _clip_payload(clip: CameraSourceClip) -> dict[str, Any]:
    path = Path(clip.path)
    try:
        stat = path.stat()
        size = stat.st_size
        mtime_ns = stat.st_mtime_ns
    except OSError:
        size = None
        mtime_ns = None
    return {
        "path": str(path.resolve(strict=False)),
        "size": size,
        "mtime_ns": mtime_ns,
        "timeline_start_s": clip.timeline_start_s,
        "timeline_end_s": clip.timeline_end_s,
        "source_start_s": clip.source_start_s,
        "source_end_s": clip.source_end_s,
    }
