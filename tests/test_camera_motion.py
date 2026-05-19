"""Tests for motion interval detection and camera source resolution."""

from __future__ import annotations

import gzip
import uuid
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from autopodcast.core.camera_motion import (
    CameraMotionAnalyzer,
    CameraMotionConfig,
    CameraMotionExecutionConfig,
    CameraMotionPlan,
    CameraMotionRuntime,
    CameraSourceClip,
    MotionInterval,
    build_motion_plan_from_scores,
    compute_edge_motion_scores,
)
from autopodcast.core.monologue_sources import (
    _build_resolved_sources,
    resolve_monologue_camera_sources,
)
from autopodcast.prproj_patcher import TICKS_PER_SECOND
from tests.test_prproj_patcher import _build_synthetic_prproj


def _motion_config(**overrides) -> CameraMotionConfig:
    config = CameraMotionConfig(smoothing_window_s=0.25)
    data = {**config.__dict__, **overrides}
    return CameraMotionConfig(**data)


class TestCameraMotionIntervals:
    def test_stable_scores_produce_no_moving_intervals(self):
        plan = build_motion_plan_from_scores(
            [0.001] * 20,
            4.0,
            _motion_config(),
            total_duration_s=5.0,
        )

        assert plan.moving_intervals == ()

    def test_long_motion_creates_interval(self):
        plan = build_motion_plan_from_scores(
            [0.001] * 4 + [0.05] * 8 + [0.0] * 6,
            4.0,
            _motion_config(),
            total_duration_s=6.0,
        )

        assert len(plan.moving_intervals) == 1
        interval = plan.moving_intervals[0]
        assert 0.8 <= interval.start_s <= 0.9
        assert interval.end_s > interval.start_s

    def test_short_burst_below_confirm_window_is_ignored(self):
        plan = build_motion_plan_from_scores(
            [0.001] * 4 + [0.05] + [0.001] * 8,
            4.0,
            _motion_config(),
            total_duration_s=4.0,
        )

        assert plan.moving_intervals == ()

    def test_camera_becomes_stable_only_after_quiet_window(self):
        plan = build_motion_plan_from_scores(
            [0.05] * 4 + [0.0] * 5,
            4.0,
            _motion_config(),
            total_duration_s=3.0,
        )

        assert len(plan.moving_intervals) == 1
        interval = plan.moving_intervals[0]
        assert 0.0 <= interval.start_s <= 0.1
        assert 1.9 <= interval.end_s <= 2.1

    def test_motion_pre_roll_moves_interval_start_before_confirmation(self):
        plan = build_motion_plan_from_scores(
            [0.001] * 4 + [0.05] * 3 + [0.001] * 8,
            4.0,
            _motion_config(motion_pre_roll_s=0.25),
            total_duration_s=4.0,
        )

        assert len(plan.moving_intervals) == 1
        interval = plan.moving_intervals[0]
        assert interval.start_s == pytest.approx(1.0)

    def test_analyze_clips_shifts_clip_intervals_to_timeline(self, tmp_path: Path, monkeypatch):
        clip_a = CameraSourceClip(
            path=tmp_path / "cam_a.mp4",
            timeline_start_s=0.0,
            timeline_end_s=10.0,
            source_start_s=5.0,
            source_end_s=15.0,
        )
        clip_b = CameraSourceClip(
            path=tmp_path / "cam_b.mp4",
            timeline_start_s=10.0,
            timeline_end_s=20.0,
            source_start_s=0.0,
            source_end_s=10.0,
        )
        analyzer = CameraMotionAnalyzer(_motion_config())

        def _fake_analyze_clip(self, clip, *, runtime=None):
            if clip.path == clip_a.path:
                return CameraMotionPlan(
                    source_path=clip.path,
                    moving_intervals=(MotionInterval(6.0, 8.0),),
                    diagnostics={"clip": "a"},
                )
            return CameraMotionPlan(
                source_path=clip.path,
                moving_intervals=(MotionInterval(0.5, 1.5),),
                diagnostics={"clip": "b"},
            )

        monkeypatch.setattr(
            CameraMotionAnalyzer,
            "_analyze_clip",
            _fake_analyze_clip,
        )

        plan = analyzer.analyze_clips([clip_a, clip_b], total_duration_s=20.0)

        assert plan.moving_intervals == (
            MotionInterval(start_s=1.0, end_s=3.0),
            MotionInterval(start_s=10.5, end_s=11.5),
        )
        assert plan.diagnostics["clip_count"] == 2

    def test_analyze_clips_reports_progress_per_clip(self, tmp_path: Path, monkeypatch):
        clips = [
            CameraSourceClip(path=tmp_path / "cam_a.mp4"),
            CameraSourceClip(path=tmp_path / "cam_b.mp4"),
        ]
        analyzer = CameraMotionAnalyzer(_motion_config())
        seen = []

        def _fake_analyze_clip(self, clip, *, runtime=None):
            return CameraMotionPlan(
                source_path=clip.path,
                moving_intervals=(),
                diagnostics={},
            )

        monkeypatch.setattr(
            CameraMotionAnalyzer,
            "_analyze_clip",
            _fake_analyze_clip,
        )

        analyzer.analyze_clips(clips, progress_callback=lambda clip: seen.append(clip.path.name))

        assert sorted(seen) == ["cam_a.mp4", "cam_b.mp4"]

    def test_bottom_motion_is_ignored_by_zone_scores(self):
        frames = np.zeros((5, 100, 100), dtype=np.uint8)
        for idx in range(1, 5):
            frames[idx, 75:, 10 * idx:10 * idx + 10] = 255

        scores = compute_edge_motion_scores(
            frames,
            border_ratio=0.08,
            bottom_ignore_ratio=0.30,
        )

        assert float(scores.max()) == pytest.approx(0.0)

    def test_top_motion_produces_zone_scores(self):
        frames = np.zeros((5, 100, 100), dtype=np.uint8)
        for idx in range(1, 5):
            frames[idx, :15, 10 * idx:10 * idx + 10] = 255

        scores = compute_edge_motion_scores(
            frames,
            border_ratio=0.08,
            bottom_ignore_ratio=0.30,
        )

        assert float(scores.max()) > 0.02

    def test_analyze_clip_refines_candidate_windows(self, tmp_path: Path, monkeypatch):
        clip = CameraSourceClip(
            path=tmp_path / "cam.mp4",
            source_start_s=0.0,
            source_end_s=3.0,
        )
        clip.path.write_bytes(b"fake")
        analyzer = CameraMotionAnalyzer(_motion_config())

        coarse_frames = np.zeros((5, 100, 100), dtype=np.uint8)
        for idx in range(1, 5):
            coarse_frames[idx, :15, 10 * idx:10 * idx + 10] = 255
        refine_frames = np.zeros((9, 100, 100), dtype=np.uint8)
        for idx in range(1, 8):
            refine_frames[idx, :15, 6 * idx:6 * idx + 12] = 255

        monkeypatch.setattr(
            "autopodcast.core.camera_motion._probe_video_dimensions",
            lambda _path: (100, 100),
        )

        def _fake_extract_frames(
            path,
            *,
            sample_fps,
            resize_width,
            resize_height,
            start_s=0.0,
            duration_s=None,
            hwaccel_mode="cpu",
        ):
            return coarse_frames if sample_fps == analyzer.config.sample_fps else refine_frames

        monkeypatch.setattr(
            "autopodcast.core.camera_motion._extract_gray_frames",
            _fake_extract_frames,
        )

        plan = analyzer.analyze_clips([clip], total_duration_s=3.0)

        assert len(plan.moving_intervals) == 1
        assert plan.diagnostics["candidate_window_count"] >= 1
        assert plan.diagnostics["refined_window_count"] >= 1


class TestCameraMotionExecution:
    @pytest.mark.parametrize(
        ("speed", "logical_cores", "clip_count", "expected"),
        [
            ("balanced", 1, 10, 2),
            ("balanced", 8, 10, 4),
            ("balanced", 32, 10, 6),
            ("balanced", 32, 3, 3),
            ("turbo", 2, 10, 2),
            ("turbo", 8, 10, 7),
            ("turbo", 32, 20, 10),
            ("turbo", 32, 3, 3),
        ],
    )
    def test_worker_count_policy(self, speed, logical_cores, clip_count, expected):
        config = CameraMotionExecutionConfig(speed=speed)

        assert (
            config.worker_count(clip_count, logical_cores=logical_cores)
            == expected
        )

    def test_prepare_runtime_uses_hwaccel_probe_success(self, tmp_path: Path, monkeypatch):
        clip = CameraSourceClip(path=tmp_path / "cam.mp4")
        clip.path.write_bytes(b"fake")
        analyzer = CameraMotionAnalyzer(
            _motion_config(),
            CameraMotionExecutionConfig(speed="balanced", hwaccel="auto"),
        )

        monkeypatch.setattr(
            "autopodcast.core.camera_motion._probe_hwaccel_auto",
            lambda *args, **kwargs: True,
        )

        runtime = analyzer.prepare_runtime([clip])

        assert runtime.active_hwaccel == "auto"
        assert runtime.requested_hwaccel == "auto"
        assert runtime.hwaccel_probe_used is True

    def test_prepare_runtime_falls_back_to_cpu_when_probe_fails(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        clip = CameraSourceClip(path=tmp_path / "cam.mp4")
        clip.path.write_bytes(b"fake")
        analyzer = CameraMotionAnalyzer(
            _motion_config(),
            CameraMotionExecutionConfig(speed="balanced", hwaccel="auto"),
        )

        monkeypatch.setattr(
            "autopodcast.core.camera_motion._probe_hwaccel_auto",
            lambda *args, **kwargs: False,
        )

        runtime = analyzer.prepare_runtime([clip])

        assert runtime.active_hwaccel == "cpu"
        assert runtime.requested_hwaccel == "auto"

    def test_prepare_runtime_cpu_mode_skips_probe(self, tmp_path: Path, monkeypatch):
        clip = CameraSourceClip(path=tmp_path / "cam.mp4")
        clip.path.write_bytes(b"fake")
        analyzer = CameraMotionAnalyzer(
            _motion_config(),
            CameraMotionExecutionConfig(speed="turbo", hwaccel="cpu"),
        )

        def _boom(*args, **kwargs):
            raise AssertionError("probe should not run in cpu mode")

        monkeypatch.setattr(
            "autopodcast.core.camera_motion._probe_hwaccel_auto",
            _boom,
        )

        runtime = analyzer.prepare_runtime([clip])

        assert runtime.active_hwaccel == "cpu"
        assert runtime.requested_hwaccel == "cpu"
        assert runtime.hwaccel_probe_used is False

    def test_prepare_runtime_hybrid_mode_uses_split_workers(self, tmp_path: Path, monkeypatch):
        clips = [
            CameraSourceClip(path=tmp_path / f"cam_{idx}.mp4")
            for idx in range(4)
        ]
        for clip in clips:
            clip.path.write_bytes(b"fake")
        analyzer = CameraMotionAnalyzer(
            _motion_config(),
            CameraMotionExecutionConfig(speed="balanced", hwaccel="hybrid"),
        )

        monkeypatch.setattr(
            "autopodcast.core.camera_motion._probe_hwaccel_auto",
            lambda *args, **kwargs: True,
        )
        monkeypatch.setattr(
            "autopodcast.core.camera_motion.os.cpu_count",
            lambda: 8,
        )

        runtime = analyzer.prepare_runtime(clips)

        assert runtime.active_hwaccel == "hybrid"
        assert runtime.requested_hwaccel == "hybrid"
        assert runtime.worker_count == 4
        assert runtime.cpu_worker_count == 3
        assert runtime.gpu_worker_count == 1
        assert runtime.cpu_clip_count + runtime.gpu_clip_count == 4
        assert runtime.gpu_clip_count >= 1

    def test_analyze_camera_groups_uses_shared_runtime(self, tmp_path: Path, monkeypatch):
        main_clip = CameraSourceClip(path=tmp_path / "main.mp4")
        accent_clip = CameraSourceClip(path=tmp_path / "accent.mp4")
        analyzer = CameraMotionAnalyzer(
            _motion_config(),
            CameraMotionExecutionConfig(speed="turbo", hwaccel="cpu"),
        )
        seen = []
        runtime = CameraMotionRuntime(
            speed="turbo",
            requested_hwaccel="cpu",
            active_hwaccel="cpu",
            worker_count=2,
            clip_count=2,
        )

        def _fake_analyze_clip(self, clip, *, runtime=None):
            seen.append((clip.path.name, runtime.active_hwaccel, runtime.worker_count))
            return CameraMotionPlan(
                source_path=clip.path,
                moving_intervals=(),
                diagnostics={},
            )

        monkeypatch.setattr(
            CameraMotionAnalyzer,
            "_analyze_clip",
            _fake_analyze_clip,
        )

        plans = analyzer.analyze_camera_groups(
            {"main": [main_clip], "accent": [accent_clip]},
            runtime=runtime,
        )

        assert sorted(seen) == [
            ("accent.mp4", "cpu", 2),
            ("main.mp4", "cpu", 2),
        ]
        assert plans["main"].diagnostics["motion_worker_count"] == 2
        assert plans["accent"].diagnostics["motion_hwaccel_active"] == "cpu"

    def test_analyze_camera_groups_hybrid_uses_cpu_and_gpu_backends(self, tmp_path: Path, monkeypatch):
        clips = {
            "main": [
                CameraSourceClip(path=tmp_path / "main_a.mp4", source_end_s=30.0),
                CameraSourceClip(path=tmp_path / "main_b.mp4", source_end_s=12.0),
            ],
            "accent": [
                CameraSourceClip(path=tmp_path / "accent_a.mp4", source_end_s=18.0),
                CameraSourceClip(path=tmp_path / "accent_b.mp4", source_end_s=8.0),
            ],
        }
        analyzer = CameraMotionAnalyzer(
            _motion_config(),
            CameraMotionExecutionConfig(speed="balanced", hwaccel="hybrid"),
        )
        seen = []
        runtime = CameraMotionRuntime(
            speed="balanced",
            requested_hwaccel="hybrid",
            active_hwaccel="hybrid",
            worker_count=4,
            clip_count=4,
            cpu_worker_count=3,
            gpu_worker_count=1,
            cpu_clip_count=3,
            gpu_clip_count=1,
        )

        def _fake_analyze_clip(self, clip, *, runtime=None):
            seen.append((clip.path.name, runtime.active_hwaccel))
            return CameraMotionPlan(
                source_path=clip.path,
                moving_intervals=(),
                diagnostics={},
            )

        monkeypatch.setattr(
            CameraMotionAnalyzer,
            "_analyze_clip",
            _fake_analyze_clip,
        )

        plans = analyzer.analyze_camera_groups(clips, runtime=runtime)

        backends = {backend for _name, backend in seen}
        assert backends == {"auto", "cpu"}
        assert plans["main"].diagnostics["motion_hwaccel_active"] == "hybrid"
        assert plans["main"].diagnostics["motion_cpu_worker_count"] == 3
        assert plans["main"].diagnostics["motion_gpu_worker_count"] == 1


class TestMonologueSourceResolution:
    def test_manual_paths_win_over_xml_and_project(self, tmp_path: Path):
        prproj_path = tmp_path / "input.prproj"
        prproj_path.write_bytes(b"placeholder")
        xml_path = tmp_path / "project.xml"
        xml_path.write_text("<xmeml></xmeml>", encoding="utf-8")
        main = tmp_path / "manual_main.mp4"
        accent = tmp_path / "manual_accent.mp4"
        main.write_bytes(b"main")
        accent.write_bytes(b"accent")

        resolved = resolve_monologue_camera_sources(
            prproj_path,
            "Seq",
            0,
            1,
            camera_main_file=str(main),
            camera_accent_file=str(accent),
            xml_path=str(xml_path),
        )

        assert resolved.method == "manual"
        assert resolved.main_path == main
        assert resolved.accent_path == accent
        assert resolved.main_clips[0].path == main
        assert resolved.accent_clips[0].path == accent

    def test_xml_resolution_maps_requested_angles(self, tmp_path: Path):
        main = tmp_path / "cam_main.mp4"
        accent = tmp_path / "cam_accent.mp4"
        main.write_bytes(b"main")
        accent.write_bytes(b"accent")
        xml_path = tmp_path / "sequence.xml"
        xml_path.write_text(
            f"""<?xml version="1.0" encoding="UTF-8"?>
<xmeml version="5">
  <sequence>
    <name>Seq</name>
    <rate><timebase>25</timebase><ntsc>FALSE</ntsc></rate>
    <duration>100</duration>
    <media>
      <video>
        <track>
          <clipitem>
            <file>
              <name>main</name>
              <pathurl>file://localhost{main.as_posix()}</pathurl>
              <rate><timebase>25</timebase><ntsc>FALSE</ntsc></rate>
            </file>
            <start>0</start>
            <end>100</end>
            <in>0</in>
            <out>100</out>
          </clipitem>
        </track>
        <track>
          <clipitem>
            <file>
              <name>accent</name>
              <pathurl>file://localhost{accent.as_posix()}</pathurl>
              <rate><timebase>25</timebase><ntsc>FALSE</ntsc></rate>
            </file>
            <start>0</start>
            <end>100</end>
            <in>0</in>
            <out>100</out>
          </clipitem>
        </track>
      </video>
      <audio />
    </media>
  </sequence>
</xmeml>
""",
            encoding="utf-8",
        )
        prproj_path = tmp_path / "unused.prproj"
        prproj_path.write_bytes(b"unused")

        resolved = resolve_monologue_camera_sources(
            prproj_path,
            "Seq",
            0,
            1,
            xml_path=str(xml_path),
        )

        assert resolved.method == "xml"
        assert resolved.main_path == main
        assert resolved.accent_path == accent
        assert len(resolved.main_clips) == 1
        assert len(resolved.accent_clips) == 1

    def test_prproj_resolution_collects_multicam_track_clips(self, tmp_path: Path):
        prproj_path = tmp_path / "multicam.prproj"
        prproj_path.write_bytes(_build_multicam_source_prproj())

        resolved = resolve_monologue_camera_sources(
            prproj_path,
            "Seq",
            0,
            1,
        )

        assert resolved.method == "prproj"
        assert len(resolved.main_clips) == 2
        assert len(resolved.accent_clips) == 2
        assert resolved.main_clips[0].path == tmp_path / "cams" / "A001.MP4"
        assert resolved.main_clips[1].timeline_start_s == pytest.approx(10.0)
        assert resolved.accent_clips[0].path == tmp_path / "cams" / "B001.MP4"
        assert resolved.accent_clips[1].timeline_start_s == pytest.approx(10.0)

    def test_resolution_failure_mentions_manual_override(self, tmp_path: Path):
        prproj_path = tmp_path / "input.prproj"
        prproj_path.write_bytes(_build_synthetic_prproj())

        with pytest.raises(ValueError) as exc_info:
            resolve_monologue_camera_sources(prproj_path, "TestSeq", 0, 1)

        message = str(exc_info.value)
        assert "--camera-main-file" in message
        assert "--camera-accent-file" in message

    def test_relative_and_absolute_duplicates_collapse_to_one_source(self, tmp_path: Path):
        rel_path = Path("Исходники") / "cam.mp4"
        abs_path = tmp_path / "Исходники" / "cam.mp4"
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_bytes(b"cam")

        with pytest.raises(ValueError) as exc_info:
            _build_resolved_sources(
                [rel_path, abs_path],
                0,
                1,
                method="prproj",
                base_dir=tmp_path,
            )

        assert "Found only 1 video source track(s)" in str(exc_info.value)


def _build_multicam_source_prproj() -> bytes:
    root = ET.Element("PremiereData")

    seq = ET.SubElement(root, "Sequence", ObjectID="1")
    ET.SubElement(seq, "Name").text = "Seq"
    seq_track_groups = ET.SubElement(seq, "TrackGroups")
    seq_group = ET.SubElement(seq_track_groups, "TrackGroupRef")
    ET.SubElement(seq_group, "Second", ObjectRef="2")

    outer_vtg = ET.SubElement(root, "VideoTrackGroup", ObjectID="2")
    outer_tg = ET.SubElement(outer_vtg, "TrackGroup")
    outer_tracks = ET.SubElement(outer_tg, "Tracks")
    ET.SubElement(outer_tracks, "Track", Index="0", ObjectURef="outer-track")

    outer_track = ET.SubElement(root, "VideoClipTrack", ObjectID="10", ObjectUID="outer-track")
    outer_ci = ET.SubElement(outer_track, "ClipItems")
    outer_tis = ET.SubElement(outer_ci, "TrackItems")
    ET.SubElement(outer_tis, "TrackItem", Index="0", ObjectRef="20")

    source = ET.SubElement(root, "VideoSequenceSource", ObjectID="50")
    source_track_groups = ET.SubElement(source, "TrackGroups")
    source_group = ET.SubElement(source_track_groups, "TrackGroupRef")
    ET.SubElement(source_group, "Second", ObjectRef="52")

    source_vtg = ET.SubElement(root, "VideoTrackGroup", ObjectID="52")
    source_tg = ET.SubElement(source_vtg, "TrackGroup")
    source_tracks = ET.SubElement(source_tg, "Tracks")
    ET.SubElement(source_tracks, "Track", Index="0", ObjectURef="cam-a")
    ET.SubElement(source_tracks, "Track", Index="1", ObjectURef="cam-b")

    cam_a_track = ET.SubElement(root, "VideoClipTrack", ObjectID="60", ObjectUID="cam-a")
    cam_a_ci = ET.SubElement(cam_a_track, "ClipItems")
    cam_a_tis = ET.SubElement(cam_a_ci, "TrackItems")
    ET.SubElement(cam_a_tis, "TrackItem", Index="0", ObjectRef="70")
    ET.SubElement(cam_a_tis, "TrackItem", Index="1", ObjectRef="71")

    cam_b_track = ET.SubElement(root, "VideoClipTrack", ObjectID="61", ObjectUID="cam-b")
    cam_b_ci = ET.SubElement(cam_b_track, "ClipItems")
    cam_b_tis = ET.SubElement(cam_b_ci, "TrackItems")
    ET.SubElement(cam_b_tis, "TrackItem", Index="0", ObjectRef="80")
    ET.SubElement(cam_b_tis, "TrackItem", Index="1", ObjectRef="81")

    _make_multicam_item(root, "20", "25", "30", "50", 0.0, 20.0, is_multicam=True)
    _make_multicam_item(root, "70", "170", "270", "370", 0.0, 10.0, path="cams/A001.MP4")
    _make_multicam_item(root, "71", "171", "271", "371", 10.0, 20.0, path="cams/A002.MP4")
    _make_multicam_item(root, "80", "180", "280", "380", 0.0, 10.0, path="cams/B001.MP4")
    _make_multicam_item(root, "81", "181", "281", "381", 10.0, 20.0, path="cams/B002.MP4")

    xml_bytes = ET.tostring(root, encoding="utf-8")
    return gzip.compress(xml_bytes)


def _make_multicam_item(
    root: ET.Element,
    item_oid: str,
    subclip_oid: str,
    clip_oid: str,
    source_oid: str,
    start_s: float,
    end_s: float,
    *,
    path: str | None = None,
    is_multicam: bool = False,
) -> None:
    item = ET.SubElement(root, "VideoClipTrackItem", ObjectID=item_oid)
    clip_track_item = ET.SubElement(item, "ClipTrackItem")
    track_item = ET.SubElement(clip_track_item, "TrackItem")
    ET.SubElement(track_item, "Start").text = str(round(start_s * TICKS_PER_SECOND))
    ET.SubElement(track_item, "End").text = str(round(end_s * TICKS_PER_SECOND))
    ET.SubElement(clip_track_item, "SubClip", ObjectRef=subclip_oid)

    subclip = ET.SubElement(root, "SubClip", ObjectID=subclip_oid)
    ET.SubElement(subclip, "Clip", ObjectRef=clip_oid)
    ET.SubElement(subclip, "MasterClip", ObjectURef=str(uuid.uuid4()))

    clip_root = ET.SubElement(root, "VideoClip", ObjectID=clip_oid)
    clip = ET.SubElement(clip_root, "Clip")
    ET.SubElement(clip, "Source", ObjectRef=source_oid)
    ET.SubElement(clip, "InPoint").text = "0"
    ET.SubElement(clip, "OutPoint").text = str(round((end_s - start_s) * TICKS_PER_SECOND))
    if is_multicam:
        ET.SubElement(clip, "IsMulticam").text = "true"
        ET.SubElement(clip, "SelectedTrackIndex").text = "0"

    if path is not None:
        media = ET.SubElement(root, "VideoMediaSource", ObjectID=source_oid)
        ET.SubElement(media, "ActualMediaFilePath").text = path
