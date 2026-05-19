"""Tests for camera motion plan caching."""

from __future__ import annotations

from pathlib import Path

from autopodcast.core.camera_motion import CameraMotionConfig, CameraMotionPlan, CameraSourceClip, MotionInterval
from autopodcast.core.motion_cache import (
    load_motion_plan_from_cache,
    motion_cache_key,
    save_motion_plan_to_cache,
)


def test_motion_cache_round_trip(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    plan = CameraMotionPlan(
        source_path=tmp_path / "cam.mp4",
        sample_fps=2.0,
        moving_intervals=(MotionInterval(1.0, 2.5),),
        diagnostics={"clip_count": 1},
    )

    save_motion_plan_to_cache(cache_dir, "abc", plan)
    loaded = load_motion_plan_from_cache(cache_dir, "abc")

    assert loaded == plan


def test_motion_cache_key_changes_when_file_changes(tmp_path: Path):
    video = tmp_path / "cam.mp4"
    video.write_bytes(b"one")
    clip = CameraSourceClip(path=video, timeline_start_s=0.0, timeline_end_s=10.0)
    config = CameraMotionConfig()

    key_before = motion_cache_key([clip], config)
    video.write_bytes(b"two plus size")
    key_after = motion_cache_key([clip], config)

    assert key_after != key_before


def test_motion_cache_key_changes_when_config_changes(tmp_path: Path):
    video = tmp_path / "cam.mp4"
    video.write_bytes(b"same")
    clip = CameraSourceClip(path=video)

    key_before = motion_cache_key([clip], CameraMotionConfig(sample_fps=2.0))
    key_after = motion_cache_key([clip], CameraMotionConfig(sample_fps=3.0))

    assert key_after != key_before


def test_missing_or_invalid_cache_is_a_miss(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "bad.json").write_text("{", encoding="utf-8")

    assert load_motion_plan_from_cache(cache_dir, "missing") is None
    assert load_motion_plan_from_cache(cache_dir, "bad") is None
