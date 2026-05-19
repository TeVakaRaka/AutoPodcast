"""Tests for single-narrator two-camera switching."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import soundfile as sf
from click.testing import CliRunner

from autopodcast.cli import cli
from autopodcast.core.camera_motion import CameraMotionPlan, MotionInterval
from autopodcast.core.monologue_2cam import Monologue2CamConfig, build_monologue_plan
from autopodcast.models.domain import AnalysisFrame, SpeakerActivity
from tests.conftest import make_speech_pattern
from tests.test_prproj_patcher import _build_synthetic_prproj


def _make_activity(
    total_duration_s: float,
    hop_s: float,
    speech_intervals: list[tuple[float, float]],
    dips: list[tuple[float, float, float]] | None = None,
) -> SpeakerActivity:
    frames: list[AnalysisFrame] = []
    n_frames = int(total_duration_s / hop_s)
    dips = dips or []

    for idx in range(n_frames):
        time_s = idx * hop_s
        is_active = any(start_s <= time_s < end_s for start_s, end_s in speech_intervals)
        envelope_db = -12.0 if is_active else -60.0
        for center_s, width_s, dip_db in dips:
            if center_s - width_s / 2.0 <= time_s < center_s + width_s / 2.0:
                envelope_db = dip_db
                break
        frames.append(
            AnalysisFrame(
                time_s=time_s,
                rms_db=envelope_db,
                envelope_db=envelope_db,
                is_active=is_active,
                peak_db=envelope_db,
            )
        )

    return SpeakerActivity(speaker_label="narrator", frames=frames)


def _default_config(**overrides) -> Monologue2CamConfig:
    config = Monologue2CamConfig(camera_main=0, camera_accent=1)
    data = {**config.__dict__, **overrides}
    return Monologue2CamConfig(**data)


def _motion_plan(*intervals: tuple[float, float]) -> CameraMotionPlan:
    return CameraMotionPlan(
        moving_intervals=tuple(
            MotionInterval(start_s=start_s, end_s=end_s)
            for start_s, end_s in intervals
        )
    )


class TestMonologuePlanner:
    def test_pause_after_target_wins(self):
        activity = _make_activity(
            total_duration_s=70.0,
            hop_s=0.1,
            speech_intervals=[(0.0, 30.0), (30.8, 70.0)],
        )

        plan = build_monologue_plan(activity, 70.0, 0.1, _default_config())

        assert plan.cuts
        first_cut = plan.cuts[0]
        assert first_cut.reason == "pause_after_target"
        assert 30.3 <= first_cut.time_s <= 30.5
        assert first_cut.camera_index == 1

    def test_pause_before_target_used_when_after_window_empty(self):
        activity = _make_activity(
            total_duration_s=70.0,
            hop_s=0.1,
            speech_intervals=[(0.0, 29.0), (29.8, 70.0)],
        )

        plan = build_monologue_plan(activity, 70.0, 0.1, _default_config())

        assert plan.cuts
        first_cut = plan.cuts[0]
        assert first_cut.reason == "pause_before_target"
        assert 29.3 <= first_cut.time_s <= 29.5
        assert first_cut.time_s < 30.0

    def test_forced_low_energy_cut_when_no_pause_exists(self):
        activity = _make_activity(
            total_duration_s=70.0,
            hop_s=0.1,
            speech_intervals=[(0.0, 70.0)],
            dips=[(34.0, 0.2, -28.0)],
        )

        plan = build_monologue_plan(activity, 70.0, 0.1, _default_config())

        assert plan.cuts
        first_cut = plan.cuts[0]
        assert first_cut.reason == "forced_low_energy"
        assert 33.8 <= first_cut.time_s <= 34.1
        assert first_cut.min_envelope_db == -28.0

    def test_micro_pauses_are_not_treated_as_switch_candidates(self):
        activity = _make_activity(
            total_duration_s=90.0,
            hop_s=0.05,
            speech_intervals=[
                (0.0, 29.95),
                (30.05, 44.95),
                (45.05, 59.95),
                (60.05, 90.0),
            ],
        )

        plan = build_monologue_plan(activity, 90.0, 0.05, _default_config())

        assert plan.pause_segments == []
        assert all(cut.reason == "forced_low_energy" for cut in plan.cuts)
        assert len(plan.cuts) == 2

    def test_min_camera_hold_is_respected(self):
        activity = _make_activity(
            total_duration_s=80.0,
            hop_s=0.1,
            speech_intervals=[
                (0.0, 20.2),
                (20.8, 40.2),
                (40.8, 60.2),
                (60.8, 80.0),
            ],
        )

        plan = build_monologue_plan(
            activity,
            80.0,
            0.1,
            _default_config(switch_interval_s=15.0, min_camera_hold_s=20.0),
        )

        assert len(plan.cuts) >= 2
        for prev_cut, next_cut in zip(plan.cuts, plan.cuts[1:]):
            assert next_cut.time_s - prev_cut.time_s >= 20.0 - 1e-6

    def test_five_hour_synthetic_case_stays_reasonable(self):
        total_duration_s = 5 * 60 * 60
        hop_s = 0.5
        speech_intervals = []
        t = 0.0
        while t < total_duration_s:
            speech_intervals.append((t, min(t + 29.2, total_duration_s)))
            t += 30.0

        activity = _make_activity(total_duration_s, hop_s, speech_intervals)
        plan = build_monologue_plan(activity, total_duration_s, hop_s, _default_config())

        assert 550 <= len(plan.cuts) <= 600
        assert len(plan.camera_segments) == len(plan.cuts) + 1
        assert all(cut.reason in {"pause_after_target", "pause_before_target"} for cut in plan.cuts[:10])
        assert all(cut.reason != "forced_low_energy" for cut in plan.cuts[:10])

    def test_camera_share_biases_screen_time(self):
        activity = _make_activity(
            total_duration_s=240.0,
            hop_s=0.1,
            speech_intervals=[
                (0.0, 48.0),
                (48.8, 60.0),
                (60.8, 108.0),
                (108.8, 120.0),
                (120.8, 168.0),
                (168.8, 180.0),
                (180.8, 228.0),
                (228.8, 240.0),
            ],
        )

        plan = build_monologue_plan(
            activity,
            240.0,
            0.1,
            _default_config(camera_main_share=0.8),
        )

        assert len(plan.camera_segments) >= 4
        assert 47.5 <= plan.camera_segments[0].duration_s <= 48.5
        assert 11.5 <= plan.camera_segments[1].duration_s <= 12.5
        assert plan.diagnostics["camera_main_target_hold_s"] == pytest.approx(48.0)
        assert plan.diagnostics["camera_accent_target_hold_s"] == pytest.approx(12.0)
        assert 0.75 <= plan.diagnostics["camera_main_actual_share"] <= 0.85

    def test_planned_cut_waits_for_target_stability(self):
        activity = _make_activity(
            total_duration_s=80.0,
            hop_s=0.1,
            speech_intervals=[(0.0, 30.0), (30.8, 80.0)],
        )

        plan = build_monologue_plan(
            activity,
            80.0,
            0.1,
            _default_config(),
            motion_plans={
                0: _motion_plan(),
                1: _motion_plan((30.0, 34.0)),
            },
        )

        assert plan.cuts
        first_cut = plan.cuts[0]
        assert first_cut.reason == "delayed_for_stability"
        assert first_cut.planned_reason == "pause_after_target"
        assert first_cut.planned_time_s is not None
        assert 34.0 <= first_cut.time_s <= 34.1
        assert any(event.reason == "delayed_for_stability" for event in plan.motion_events)

    def test_target_camera_never_stabilized_skips_cut(self):
        activity = _make_activity(
            total_duration_s=75.0,
            hop_s=0.1,
            speech_intervals=[(0.0, 75.0)],
            dips=[(34.0, 0.2, -28.0)],
        )

        plan = build_monologue_plan(
            activity,
            75.0,
            0.1,
            _default_config(),
            motion_plans={
                0: _motion_plan(),
                1: _motion_plan((30.0, 75.0)),
            },
        )

        assert plan.cuts == []
        assert any(
            event.reason == "target_camera_never_stabilized"
            for event in plan.motion_events
        )

    def test_planned_cut_delays_for_future_motion(self):
        activity = _make_activity(
            total_duration_s=80.0,
            hop_s=0.1,
            speech_intervals=[(0.0, 30.0), (30.8, 80.0)],
        )

        plan = build_monologue_plan(
            activity,
            80.0,
            0.1,
            _default_config(),
            motion_plans={
                0: _motion_plan(),
                1: _motion_plan((30.8, 32.0)),
            },
        )

        assert plan.cuts
        first_cut = plan.cuts[0]
        assert first_cut.reason == "delayed_for_future_motion"
        assert first_cut.planned_reason == "pause_after_target"
        assert first_cut.planned_time_s is not None
        assert 32.0 <= first_cut.time_s <= 32.1
        assert any(
            event.reason == "delayed_for_future_motion"
            for event in plan.motion_events
        )

    def test_future_motion_can_skip_target_camera_entry(self):
        activity = _make_activity(
            total_duration_s=42.5,
            hop_s=0.1,
            speech_intervals=[(0.0, 30.0), (30.8, 42.5)],
        )

        plan = build_monologue_plan(
            activity,
            42.5,
            0.1,
            _default_config(),
            motion_plans={
                0: _motion_plan(),
                1: _motion_plan((30.8, 42.5)),
            },
        )

        assert plan.cuts == []
        assert any(
            event.reason == "target_camera_future_motion"
            for event in plan.motion_events
        )

    def test_escape_from_motion_preempts_normal_switch(self):
        activity = _make_activity(
            total_duration_s=80.0,
            hop_s=0.1,
            speech_intervals=[(0.0, 30.0), (30.8, 80.0)],
        )

        plan = build_monologue_plan(
            activity,
            80.0,
            0.1,
            _default_config(),
            motion_plans={
                0: _motion_plan((20.0, 28.0)),
                1: _motion_plan(),
            },
        )

        assert plan.cuts
        first_cut = plan.cuts[0]
        assert first_cut.reason == "escape_from_motion"
        assert 20.0 <= first_cut.time_s <= 20.1
        assert any(event.reason == "escape_from_motion" for event in plan.motion_events)

    def test_no_static_camera_available_does_not_emergency_cut(self):
        activity = _make_activity(
            total_duration_s=60.0,
            hop_s=0.1,
            speech_intervals=[(0.0, 60.0)],
            dips=[(34.0, 0.2, -28.0)],
        )

        plan = build_monologue_plan(
            activity,
            60.0,
            0.1,
            _default_config(),
            motion_plans={
                0: _motion_plan((20.0, 32.0)),
                1: _motion_plan((19.0, 34.0)),
            },
        )

        assert not any(cut.reason == "escape_from_motion" for cut in plan.cuts)
        assert any(
            event.reason == "no_static_camera_available"
            for event in plan.motion_events
        )

    def test_scheduler_recovers_after_target_camera_never_stabilized(self):
        activity = _make_activity(
            total_duration_s=120.0,
            hop_s=0.1,
            speech_intervals=[
                (0.0, 44.6),
                (45.4, 69.6),
                (70.4, 120.0),
            ],
        )

        plan = build_monologue_plan(
            activity,
            120.0,
            0.1,
            _default_config(camera_main_share=0.75),
            motion_plans={
                0: _motion_plan((55.0, 80.0)),
                1: _motion_plan(),
            },
        )

        assert any(
            event.reason == "target_camera_never_stabilized"
            for event in plan.motion_events
        )
        assert len(plan.cuts) >= 2
        assert plan.cuts[0].camera_index == 1
        assert plan.cuts[1].camera_index == 0
        assert 80.0 <= plan.cuts[1].time_s <= 90.0
        assert plan.camera_segments[-1].start_s > 70.0


class TestMonologueCLI:
    def test_auto_switch_monologue_smoke(self, tmp_path: Path):
        sr = 16000
        audio = make_speech_pattern(
            [
                (0.0, 29.6, -10.0),
                (30.6, 59.6, -10.0),
                (60.6, 75.0, -10.0),
            ],
            total_duration_s=75.0,
            sr=sr,
        )
        mic_path = tmp_path / "narrator.wav"
        sf.write(str(mic_path), audio, sr)

        in_path = tmp_path / "input.prproj"
        in_path.write_bytes(_build_synthetic_prproj())
        out_path = tmp_path / "output.prproj"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "auto-switch-monologue",
                "--in", str(in_path),
                "--mic", str(mic_path),
                "--seq", "TestSeq",
                "--out", str(out_path),
                "--camera-main", "1",
                "--camera-accent", "2",
                "--camera-main-share", "80",
                "--no-motion-check",
            ],
        )

        assert result.exit_code == 0, result.output
        assert out_path.exists()

        log_path = out_path.with_suffix(out_path.suffix + ".log.jsonl")
        assert log_path.exists()

        with log_path.open(encoding="utf-8") as f:
            entries = [json.loads(line) for line in f if line.strip()]

        cuts_entry = next(entry for entry in entries if entry.get("event") == "camera_cuts")
        timing_entry = next(entry for entry in entries if entry.get("event") == "timing_summary")
        assert cuts_entry["count"] >= 1
        assert cuts_entry["first_angle"] == 0
        assert {cut["angle"] for cut in cuts_entry["cuts"]}.issubset({0, 1})
        assert timing_entry["status"] == "success"
        assert timing_entry["motion_enabled"] is False
        assert timing_entry["motion_analysis_s"] == 0.0
        assert timing_entry["premiere_generation_s"] >= 0.0
        assert timing_entry["prproj_patch_s"] >= 0.0
        assert timing_entry["total_runtime_s"] >= timing_entry["prproj_patch_s"]
        assert "motion check: disabled" in result.output
        assert "target share: main=80.0%" in result.output
        assert "Timings: motion=" in result.output

    def test_auto_switch_monologue_motion_guard_smoke(self, tmp_path: Path, monkeypatch):
        sr = 16000
        audio = make_speech_pattern(
            [
                (0.0, 29.6, -10.0),
                (30.6, 59.6, -10.0),
                (60.6, 75.0, -10.0),
            ],
            total_duration_s=75.0,
            sr=sr,
        )
        mic_path = tmp_path / "narrator.wav"
        sf.write(str(mic_path), audio, sr)

        in_path = tmp_path / "input.prproj"
        in_path.write_bytes(_build_synthetic_prproj())
        out_path = tmp_path / "output.prproj"
        main_video = tmp_path / "cam_main.mp4"
        accent_video = tmp_path / "cam_accent.mp4"
        main_video.write_bytes(b"main")
        accent_video.write_bytes(b"accent")

        def _fake_prepare_runtime(self, clips):
            return self.last_runtime

        def _fake_analyze_camera_groups(
            self,
            clip_groups,
            *,
            total_duration_s=None,
            progress_callback=None,
            runtime=None,
        ):
            plans = {}
            for angle, clips in clip_groups.items():
                if progress_callback is not None:
                    for clip in clips:
                        progress_callback(clip)
                if Path(clips[0].path) == accent_video:
                    plans[angle] = _motion_plan((30.0, 34.0))
                else:
                    plans[angle] = _motion_plan()
            return plans

        monkeypatch.setattr(
            "autopodcast.cli.CameraMotionAnalyzer.prepare_runtime",
            _fake_prepare_runtime,
        )
        monkeypatch.setattr(
            "autopodcast.cli.CameraMotionAnalyzer.analyze_camera_groups",
            _fake_analyze_camera_groups,
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "auto-switch-monologue",
                "--in", str(in_path),
                "--mic", str(mic_path),
                "--seq", "TestSeq",
                "--out", str(out_path),
                "--camera-main", "1",
                "--camera-accent", "2",
                "--camera-main-file", str(main_video),
                "--camera-accent-file", str(accent_video),
            ],
        )

        assert result.exit_code == 0, result.output
        log_path = out_path.with_suffix(out_path.suffix + ".log.jsonl")
        assert log_path.exists()

        with log_path.open(encoding="utf-8") as f:
            entries = [json.loads(line) for line in f if line.strip()]

        motion_entry = next(entry for entry in entries if entry.get("event") == "monologue_motion_events")
        assert any(event["reason"] == "delayed_for_stability" for event in motion_entry["events"])
        sources_entry = next(entry for entry in entries if entry.get("event") == "motion_sources")
        assert sources_entry["method"] == "manual"

    def test_auto_switch_monologue_motion_runtime_default_hybrid_is_logged(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        sr = 16000
        audio = make_speech_pattern(
            [(0.0, 29.6, -10.0), (30.6, 45.0, -10.0)],
            total_duration_s=45.0,
            sr=sr,
        )
        mic_path = tmp_path / "narrator.wav"
        sf.write(str(mic_path), audio, sr)

        in_path = tmp_path / "input.prproj"
        in_path.write_bytes(_build_synthetic_prproj())
        out_path = tmp_path / "output.prproj"
        main_video = tmp_path / "main.mp4"
        accent_video = tmp_path / "accent.mp4"
        main_video.write_bytes(b"main")
        accent_video.write_bytes(b"accent")

        def _fake_prepare_runtime(self, clips):
            from autopodcast.core.camera_motion import CameraMotionRuntime

            runtime = CameraMotionRuntime(
                speed="balanced",
                requested_hwaccel="hybrid",
                active_hwaccel="hybrid",
                worker_count=4,
                clip_count=len(clips),
                cpu_worker_count=3,
                gpu_worker_count=1,
                cpu_clip_count=3,
                gpu_clip_count=1,
                probe_clip_path=str(clips[0].path),
                hwaccel_probe_used=True,
            )
            self.last_runtime = runtime
            return runtime

        def _fake_analyze_camera_groups(
            self,
            clip_groups,
            *,
            total_duration_s=None,
            progress_callback=None,
            runtime=None,
        ):
            if progress_callback is not None:
                for clips in clip_groups.values():
                    for clip in clips:
                        progress_callback(clip)
            return {angle: _motion_plan() for angle in clip_groups}

        monkeypatch.setattr(
            "autopodcast.cli.CameraMotionAnalyzer.prepare_runtime",
            _fake_prepare_runtime,
        )
        monkeypatch.setattr(
            "autopodcast.cli.CameraMotionAnalyzer.analyze_camera_groups",
            _fake_analyze_camera_groups,
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "auto-switch-monologue",
                "--in", str(in_path),
                "--mic", str(mic_path),
                "--seq", "TestSeq",
                "--out", str(out_path),
                "--camera-main-file", str(main_video),
                "--camera-accent-file", str(accent_video),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "motion execution: workers=4, cpu_workers=3, gpu_workers=1, hwaccel=hybrid" in result.output

        log_path = out_path.with_suffix(out_path.suffix + ".log.jsonl")
        with log_path.open(encoding="utf-8") as f:
            entries = [json.loads(line) for line in f if line.strip()]
        motion_config = next(entry for entry in entries if entry.get("event") == "motion_config")
        timing_entry = next(entry for entry in entries if entry.get("event") == "timing_summary")
        assert motion_config["motion_speed"] == "balanced"
        assert motion_config["motion_hwaccel_requested"] == "hybrid"
        assert motion_config["motion_hwaccel_active"] == "hybrid"
        assert motion_config["motion_worker_count"] == 4
        assert motion_config["motion_cpu_worker_count"] == 3
        assert motion_config["motion_gpu_worker_count"] == 1
        assert timing_entry["status"] == "success"
        assert timing_entry["motion_enabled"] is True
        assert timing_entry["motion_speed"] == "balanced"
        assert timing_entry["motion_hwaccel_requested"] == "hybrid"
        assert timing_entry["motion_hwaccel_active"] == "hybrid"
        assert timing_entry["motion_cpu_worker_count"] == 3
        assert timing_entry["motion_gpu_worker_count"] == 1
        assert timing_entry["motion_analysis_s"] >= 0.0
        assert timing_entry["premiere_generation_s"] >= 0.0
        assert timing_entry["total_runtime_s"] >= timing_entry["motion_analysis_s"]

    def test_auto_switch_monologue_hybrid_runtime_is_logged(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        sr = 16000
        audio = make_speech_pattern(
            [(0.0, 29.6, -10.0), (30.6, 45.0, -10.0)],
            total_duration_s=45.0,
            sr=sr,
        )
        mic_path = tmp_path / "narrator.wav"
        sf.write(str(mic_path), audio, sr)

        in_path = tmp_path / "input.prproj"
        in_path.write_bytes(_build_synthetic_prproj())
        out_path = tmp_path / "output.prproj"
        main_video = tmp_path / "main.mp4"
        accent_video = tmp_path / "accent.mp4"
        main_video.write_bytes(b"main")
        accent_video.write_bytes(b"accent")

        def _fake_prepare_runtime(self, clips):
            from autopodcast.core.camera_motion import CameraMotionRuntime

            runtime = CameraMotionRuntime(
                speed="balanced",
                requested_hwaccel="hybrid",
                active_hwaccel="hybrid",
                worker_count=4,
                clip_count=len(clips),
                cpu_worker_count=3,
                gpu_worker_count=1,
                cpu_clip_count=3,
                gpu_clip_count=1,
                probe_clip_path=str(clips[0].path),
                hwaccel_probe_used=True,
            )
            self.last_runtime = runtime
            return runtime

        def _fake_analyze_camera_groups(
            self,
            clip_groups,
            *,
            total_duration_s=None,
            progress_callback=None,
            runtime=None,
        ):
            if progress_callback is not None:
                for clips in clip_groups.values():
                    for clip in clips:
                        progress_callback(clip)
            return {angle: _motion_plan() for angle in clip_groups}

        monkeypatch.setattr(
            "autopodcast.cli.CameraMotionAnalyzer.prepare_runtime",
            _fake_prepare_runtime,
        )
        monkeypatch.setattr(
            "autopodcast.cli.CameraMotionAnalyzer.analyze_camera_groups",
            _fake_analyze_camera_groups,
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "auto-switch-monologue",
                "--in", str(in_path),
                "--mic", str(mic_path),
                "--seq", "TestSeq",
                "--out", str(out_path),
                "--camera-main-file", str(main_video),
                "--camera-accent-file", str(accent_video),
                "--motion-hwaccel", "hybrid",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "motion execution: workers=4, cpu_workers=3, gpu_workers=1, hwaccel=hybrid" in result.output

        log_path = out_path.with_suffix(out_path.suffix + ".log.jsonl")
        with log_path.open(encoding="utf-8") as f:
            entries = [json.loads(line) for line in f if line.strip()]
        motion_config = next(entry for entry in entries if entry.get("event") == "motion_config")
        timing_entry = next(entry for entry in entries if entry.get("event") == "timing_summary")
        assert motion_config["motion_hwaccel_requested"] == "hybrid"
        assert motion_config["motion_hwaccel_active"] == "hybrid"
        assert motion_config["motion_cpu_worker_count"] == 3
        assert motion_config["motion_gpu_worker_count"] == 1
        assert timing_entry["motion_hwaccel_requested"] == "hybrid"
        assert timing_entry["motion_hwaccel_active"] == "hybrid"
        assert timing_entry["motion_cpu_clip_count"] == 3
        assert timing_entry["motion_gpu_clip_count"] == 1
