"""Tests for the SAKHA AYMAKH 2-host + 1-guest planner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from click.testing import CliRunner

from autopodcast.cli import cli
from autopodcast.core.sakha_aimakh import (
    SakhaAymakhConfig,
    SakhaAymakhState,
    SakhaParticipantSpec,
    build_sakha_aimakh_plan,
    config_from_controls,
)
from autopodcast.models.domain import AnalysisFrame, SpeakerActivity
from tests.conftest import make_speech_pattern
from tests.test_prproj_patcher import _build_synthetic_prproj


def _activity(
    label: str,
    active_runs: list[tuple[float, float]],
    total_s: float,
    hop_s: float = 0.1,
    active_db: float = -18.0,
    inactive_db: float = -60.0,
) -> SpeakerActivity:
    frames = []
    n_frames = int(total_s / hop_s)
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


def _level_activity(
    label: str,
    level_runs: list[tuple[float, float, float]],
    total_s: float,
    hop_s: float = 0.1,
    default_db: float = -42.0,
    force_active: bool = False,
) -> SpeakerActivity:
    frames = []
    n_frames = int(total_s / hop_s)
    for idx in range(n_frames):
        time_s = idx * hop_s
        level = default_db
        for start, end, run_level in level_runs:
            if start <= time_s < end:
                level = run_level
                break
        frames.append(
            AnalysisFrame(
                time_s=time_s,
                rms_db=level,
                envelope_db=level,
                is_active=force_active or level > default_db,
                peak_db=level + 2.0,
            )
        )
    return SpeakerActivity(speaker_label=label, frames=frames)


def _participants() -> list[SakhaParticipantSpec]:
    return [
        SakhaParticipantSpec("main_host", "main_host", "main_host", 0),
        SakhaParticipantSpec("cohost", "cohost", "cohost", 1),
        SakhaParticipantSpec("guest", "guest", "guest", 2),
    ]


def _config(**overrides) -> SakhaAymakhConfig:
    base = SakhaAymakhConfig(
        camera_main_host_close=0,
        camera_guest_close=1,
        camera_pair_wide=2,
        camera_all_wide=3,
        reestablish_all_wide_interval_s=1000.0,
        shot_hold_time_s=0.5,
        overlap_min_hold_s=0.4,
        silence_timeout_s=0.4,
    )
    return SakhaAymakhConfig(**(base.__dict__ | overrides))


def _plan(
    activities: dict[str, SpeakerActivity],
    total_s: float,
    config: SakhaAymakhConfig | None = None,
    audio_arrays_by_key: dict[str, np.ndarray] | None = None,
    audio_sample_rate: int | None = None,
):
    cfg = config or _config()
    return build_sakha_aimakh_plan(
        _participants(),
        activities,
        0.1,
        cfg,
        audio_arrays_by_key=audio_arrays_by_key,
        audio_sample_rate=audio_sample_rate,
    )


def _speech_noise(total_s: float, sample_rate: int, runs: list[tuple[float, float]], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    audio = np.zeros(int(total_s * sample_rate), dtype=np.float64)
    for start_s, end_s in runs:
        start = int(start_s * sample_rate)
        end = int(end_s * sample_rate)
        audio[start:end] = rng.normal(0.0, 0.35, end - start)
    return audio


def _delay(audio: np.ndarray, samples: int) -> np.ndarray:
    result = np.zeros_like(audio)
    if samples > 0:
        result[samples:] = audio[:-samples]
    elif samples < 0:
        result[:samples] = audio[-samples:]
    else:
        result[:] = audio
    return result


def _max_visible_run_s(camera_segments) -> float:
    if not camera_segments:
        return 0.0
    max_run = 0.0
    run_camera = camera_segments[0].camera_index
    run_start = camera_segments[0].start_s
    run_end = camera_segments[0].end_s
    for seg in camera_segments[1:]:
        if seg.camera_index == run_camera:
            run_end = seg.end_s
        else:
            max_run = max(max_run, run_end - run_start)
            run_camera = seg.camera_index
            run_start = seg.start_s
            run_end = seg.end_s
    return max(max_run, run_end - run_start)


class TestSakhaAymakhPlanner:
    def test_solo_main_host_uses_main_close(self):
        plan = _plan(
            {
                "main_host": _activity("main_host", [(0.0, 2.0)], 3.0),
                "cohost": _activity("cohost", [], 3.0),
                "guest": _activity("guest", [], 3.0),
            },
            3.0,
        )

        assert plan.speech_segments[0].state == SakhaAymakhState.MAIN_HOST_ONLY
        assert plan.camera_segments[0].camera_index == 0
        assert plan.camera_segments[0].reason == "main_host_close"

    def test_solo_cohost_uses_pair_wide(self):
        plan = _plan(
            {
                "main_host": _activity("main_host", [], 3.0),
                "cohost": _activity("cohost", [(0.0, 2.0)], 3.0),
                "guest": _activity("guest", [], 3.0),
            },
            3.0,
        )

        assert plan.speech_segments[0].state == SakhaAymakhState.COHOST_ONLY
        assert plan.camera_segments[0].camera_index == 2

    def test_solo_guest_uses_guest_close_then_wide_cutaway(self):
        cfg = _config(guest_cutaway_interval_s=2.0, cutaway_duration_s=0.6)
        plan = _plan(
            {
                "main_host": _activity("main_host", [], 6.0),
                "cohost": _activity("cohost", [], 6.0),
                "guest": _activity("guest", [(0.0, 5.5)], 6.0),
            },
            6.0,
            cfg,
        )

        assert any(seg.camera_index == 1 and seg.reason == "guest_close" for seg in plan.camera_segments)
        # Guest re-establish cuts to the студийный общий (camera_all_wide=3), not the co-host
        # pair shot — the co-host is usually silent so the pair shot is dead weight.
        assert any(seg.camera_index == 3 and seg.reason == "guest_wide_cutaway" for seg in plan.camera_segments)

    def test_cohost_guest_overlap_uses_pair_wide(self):
        plan = _plan(
            {
                "main_host": _activity("main_host", [], 3.0),
                "cohost": _activity("cohost", [(0.0, 2.0)], 3.0),
                "guest": _activity("guest", [(0.0, 2.0)], 3.0),
            },
            3.0,
        )

        assert plan.speech_segments[0].state == SakhaAymakhState.COHOST_GUEST
        assert plan.camera_segments[0].camera_index == 2

    def test_main_host_guest_overlap_uses_all_wide(self):
        plan = _plan(
            {
                "main_host": _activity("main_host", [(0.0, 2.0)], 3.0),
                "cohost": _activity("cohost", [], 3.0),
                "guest": _activity("guest", [(0.0, 2.0)], 3.0),
            },
            3.0,
        )

        assert plan.speech_segments[0].state == SakhaAymakhState.MAIN_HOST_GUEST
        assert plan.camera_segments[0].camera_index == 3

    def test_all_three_overlap_uses_all_wide(self):
        plan = _plan(
            {
                "main_host": _activity("main_host", [(0.0, 2.0)], 3.0),
                "cohost": _activity("cohost", [(0.0, 2.0)], 3.0),
                "guest": _activity("guest", [(0.0, 2.0)], 3.0),
            },
            3.0,
            _config(audio_clean_mode="legacy"),
        )

        assert plan.speech_segments[0].state == SakhaAymakhState.ALL_OVERLAP
        assert plan.camera_segments[0].camera_index == 3

    def test_max_solo_hold_inserts_cutaways(self):
        cfg = _config(max_solo_hold_s=6.0, solo_cutaway_interval_s=6.0, cutaway_duration_s=0.8)
        plan = _plan(
            {
                "main_host": _activity("main_host", [(0.0, 14.0)], 14.0),
                "cohost": _activity("cohost", [], 14.0),
                "guest": _activity("guest", [], 14.0),
            },
            14.0,
            cfg,
        )

        main_segments = [seg for seg in plan.camera_segments if seg.reason == "main_host_close"]
        assert main_segments
        assert max(seg.duration_s for seg in main_segments) <= 6.0 + 1e-6
        assert any(seg.reason == "main_host_cutaway" for seg in plan.camera_segments)

    def test_fragmented_guest_solo_still_respects_visible_hold(self):
        cfg = _config(
            audio_clean_mode="legacy",
            max_solo_hold_s=20.0,
            guest_cutaway_interval_s=1000.0,
            cutaway_duration_s=1.0,
        )
        cohost_blips = [(start, start + 0.1) for start in range(10, 70, 10)]
        plan = _plan(
            {
                "main_host": _activity("main_host", [], 80.0),
                "cohost": _activity("cohost", cohost_blips, 80.0),
                "guest": _activity("guest", [(0.0, 80.0)], 80.0),
            },
            80.0,
            cfg,
        )

        assert _max_visible_run_s(plan.camera_segments) <= 20.0 + 1e-6
        assert any(seg.reason == "max_visible_hold_cutaway" for seg in plan.camera_segments)

    def test_debleed_suppresses_hot_guest_leak_but_keeps_real_overlap(self):
        plan = _plan(
            {
                "main_host": _level_activity(
                    "main_host",
                    [(0.0, 2.0, -12.0)],
                    6.0,
                    default_db=-30.0,
                    force_active=True,
                ),
                "cohost": _level_activity(
                    "cohost",
                    [(2.0, 4.0, -13.0)],
                    6.0,
                    default_db=-32.0,
                    force_active=True,
                ),
                "guest": _level_activity(
                    "guest",
                    [(0.0, 2.0, -20.0), (2.0, 6.0, -12.0)],
                    6.0,
                    default_db=-28.0,
                    force_active=True,
                ),
            },
            6.0,
            _config(audio_clean_mode="legacy"),
        )

        assert plan.frame_states[5].state == SakhaAymakhState.MAIN_HOST_ONLY
        assert plan.frame_states[5].active_keys == ("main_host",)
        assert plan.frame_states[25].state == SakhaAymakhState.COHOST_GUEST
        assert set(plan.frame_states[25].active_keys) == {"cohost", "guest"}
        assert plan.frame_states[45].state == SakhaAymakhState.GUEST_ONLY
        assert plan.diagnostics["debleed"]["participants"]["guest"]["suppressed_frames"] > 0

    def test_source_owner_keeps_guest_when_cohost_mic_catches_guest_bleed(self):
        sr = 1000
        total_s = 5.0
        guest_voice = _speech_noise(total_s, sr, [(0.0, 2.0)], seed=1)
        audio = {
            "main_host": np.zeros_like(guest_voice),
            "cohost": 1.5 * _delay(guest_voice, 4),
            "guest": guest_voice,
        }
        plan = _plan(
            {
                "main_host": _level_activity("main_host", [], 5.0, default_db=-70.0),
                "cohost": _level_activity(
                    "cohost",
                    [(0.0, 2.0, -30.0)],
                    5.0,
                    default_db=-65.0,
                ),
                "guest": _level_activity(
                    "guest",
                    [(0.0, 2.0, -34.0), (2.0, 3.0, -18.0)],
                    5.0,
                    default_db=-62.0,
                ),
            },
            5.0,
            audio_arrays_by_key=audio,
            audio_sample_rate=sr,
        )

        assert plan.frame_states[5].active_keys == ("guest",)
        assert plan.frame_states[5].state == SakhaAymakhState.GUEST_ONLY
        assert plan.frame_states[5].debleed_reason == "source_owner_bleed_suppressed"
        assert plan.audio_open_intervals_s[2]
        assert plan.diagnostics["waveform_gate"]["decision_seconds"]["correlated_bleed_suppressed"] > 0
        assert plan.diagnostics["source_owner"]["guest_bleed_suppressed_seconds"] == 0.0
        assert plan.diagnostics["source_owner"]["cohost_bleed_suppressed_seconds"] > 0

    def test_source_owner_keeps_cohost_when_guest_mic_catches_cohost_bleed(self):
        sr = 1000
        total_s = 4.0
        cohost_voice = _speech_noise(total_s, sr, [(0.0, 1.2)], seed=2)
        audio = {
            "main_host": np.zeros_like(cohost_voice),
            "cohost": cohost_voice,
            "guest": 1.4 * _delay(cohost_voice, 5),
        }
        plan = _plan(
            {
                "main_host": _level_activity("main_host", [], 4.0, default_db=-70.0),
                "cohost": _level_activity(
                    "cohost",
                    [(0.0, 1.2, -18.0)],
                    4.0,
                    default_db=-65.0,
                ),
                "guest": _level_activity(
                    "guest",
                    [(0.0, 1.2, -46.0), (2.0, 4.0, -18.0)],
                    4.0,
                    default_db=-62.0,
                ),
            },
            4.0,
            audio_arrays_by_key=audio,
            audio_sample_rate=sr,
        )

        assert plan.frame_states[5].state == SakhaAymakhState.COHOST_ONLY
        assert plan.frame_states[5].active_keys == ("cohost",)
        assert "guest" not in plan.frame_states[5].active_keys

    def test_source_owner_independent_cohost_guest_overlap_opens_both_mics(self):
        sr = 1000
        total_s = 4.0
        audio = {
            "main_host": np.zeros(int(total_s * sr), dtype=np.float64),
            "cohost": _speech_noise(total_s, sr, [(0.0, 1.5)], seed=3),
            "guest": _speech_noise(total_s, sr, [(0.0, 1.5)], seed=4),
        }
        plan = _plan(
            {
                "main_host": _level_activity("main_host", [], 4.0, default_db=-70.0),
                "cohost": _level_activity(
                    "cohost",
                    [(0.0, 1.5, -31.0)],
                    4.0,
                    default_db=-65.0,
                ),
                "guest": _level_activity(
                    "guest",
                    [(0.0, 1.5, -35.0), (1.5, 2.0, -18.0)],
                    4.0,
                    default_db=-62.0,
                ),
            },
            4.0,
            audio_arrays_by_key=audio,
            audio_sample_rate=sr,
        )

        assert set(plan.frame_states[8].active_keys) == {"cohost", "guest"}
        assert plan.frame_states[8].state == SakhaAymakhState.COHOST_GUEST
        assert plan.frame_states[8].debleed_reason == "source_owner_true_overlap"

    def test_source_owner_all_three_main_host_bleed_chooses_single_source(self):
        sr = 1000
        total_s = 3.0
        main_voice = _speech_noise(total_s, sr, [(0.0, 2.0)], seed=5)
        audio = {
            "main_host": main_voice,
            "cohost": 1.3 * _delay(main_voice, 3),
            "guest": 1.1 * _delay(main_voice, 6),
        }
        plan = _plan(
            {
                "main_host": _activity("main_host", [(0.0, 2.0)], 3.0, active_db=-24.0, inactive_db=-70.0),
                "cohost": _activity("cohost", [(0.0, 2.0)], 3.0, active_db=-18.0, inactive_db=-70.0),
                "guest": _activity("guest", [(0.0, 2.0)], 3.0, active_db=-20.0, inactive_db=-70.0),
            },
            3.0,
            audio_arrays_by_key=audio,
            audio_sample_rate=sr,
        )

        assert plan.frame_states[5].state == SakhaAymakhState.MAIN_HOST_ONLY
        assert plan.frame_states[5].active_keys == ("main_host",)
        assert plan.frame_states[5].debleed_reason == "source_owner_bleed_suppressed"
        assert plan.diagnostics["source_owner"]["cohost_bleed_suppressed_seconds"] > 0
        assert plan.diagnostics["source_owner"]["guest_bleed_suppressed_seconds"] > 0

    def test_source_owner_short_false_cohost_win_inside_guest_turn_stays_guest(self):
        sr = 1000
        total_s = 5.0
        guest_voice = _speech_noise(total_s, sr, [(0.0, 2.0)], seed=6)
        audio = {
            "main_host": np.zeros_like(guest_voice),
            "cohost": 1.8 * _delay(guest_voice, 4),
            "guest": guest_voice,
        }
        plan = _plan(
            {
                "main_host": _level_activity("main_host", [], 5.0, default_db=-70.0),
                "cohost": _level_activity(
                    "cohost",
                    [(0.8, 1.1, -24.0)],
                    5.0,
                    default_db=-65.0,
                ),
                "guest": _level_activity(
                    "guest",
                    [(0.0, 2.0, -34.0), (2.0, 3.0, -18.0)],
                    5.0,
                    default_db=-62.0,
                ),
            },
            5.0,
            audio_arrays_by_key=audio,
            audio_sample_rate=sr,
        )

        assert plan.frame_states[9].active_keys == ("guest",)
        assert plan.frame_states[9].focus_key == "guest"
        assert plan.diagnostics["waveform_gate"]["decision_seconds"]["correlated_bleed_suppressed"] > 0
        assert plan.diagnostics["source_owner"]["source_owner_seconds"]["single_guest"] > 0

    def test_full_silence_mutes_all_tracks(self):
        plan = _plan(
            {
                "main_host": _activity("main_host", [], 3.0),
                "cohost": _activity("cohost", [], 3.0),
                "guest": _activity("guest", [], 3.0),
            },
            3.0,
        )

        assert plan.audio_open_intervals_s == {0: [], 1: [], 2: []}
        assert plan.camera_segments[0].camera_index == 3

    def test_internal_silence_holds_last_active_mic(self):
        plan = _plan(
            {
                "main_host": _activity("main_host", [(0.0, 1.0), (2.0, 3.0)], 4.0),
                "cohost": _activity("cohost", [], 4.0),
                "guest": _activity("guest", [], 4.0),
            },
            4.0,
        )

        assert plan.audio_open_intervals_s[0][0][0] == 0.0
        assert plan.audio_open_intervals_s[0][0][1] >= 3.0

    def test_final_silence_can_remain_muted(self):
        plan = _plan(
            {
                "main_host": _activity("main_host", [(0.0, 1.0)], 4.0),
                "cohost": _activity("cohost", [], 4.0),
                "guest": _activity("guest", [], 4.0),
            },
            4.0,
        )

        assert plan.audio_open_intervals_s[0][-1][1] < 4.0

    def test_overlap_keeps_active_mics_open_and_inactive_closed(self):
        plan = _plan(
            {
                "main_host": _activity("main_host", [], 3.0),
                "cohost": _activity("cohost", [(0.5, 1.5)], 3.0),
                "guest": _activity("guest", [(0.5, 1.5)], 3.0),
            },
            3.0,
        )

        assert plan.audio_open_intervals_s[0] == []
        assert plan.audio_open_intervals_s[1]
        assert plan.audio_open_intervals_s[2]

    def test_short_false_audio_open_is_removed(self):
        plan = _plan(
            {
                "main_host": _activity("main_host", [(0.0, 1.5)], 2.0),
                "cohost": _activity("cohost", [(0.4, 0.5)], 2.0),
                "guest": _activity("guest", [], 2.0),
            },
            2.0,
        )

        assert plan.audio_open_intervals_s[1] == []
        assert plan.audio_open_intervals_raw_s[1] == [(0.4, 0.5)]

    def test_temperature_and_intensity_change_tuning(self):
        calm = config_from_controls(
            camera_main_host_close=0,
            camera_guest_close=1,
            camera_pair_wide=2,
            camera_all_wide=3,
            temperature=0,
            cut_intensity=0,
        )
        active = config_from_controls(
            camera_main_host_close=0,
            camera_guest_close=1,
            camera_pair_wide=2,
            camera_all_wide=3,
            temperature=100,
            cut_intensity=100,
        )

        assert active.shot_hold_time_s < calm.shot_hold_time_s
        assert active.guest_cutaway_interval_s < calm.guest_cutaway_interval_s
        assert active.dominance_delta_db < calm.dominance_delta_db
        assert active.cut_search_window_s > calm.cut_search_window_s
        assert calm.debleed_guest_ambiguous_snr_margin_db > active.debleed_guest_ambiguous_snr_margin_db
        assert calm.debleed_guest_rescue_snr_db < active.debleed_guest_rescue_snr_db


class TestSakhaAymakhCli:
    def test_help_is_available(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["auto-switch-sakha-aimakh", "--help"])

        assert result.exit_code == 0, result.output
        assert "--mic-main-host" in result.output
        assert "--camera-pair-wide" in result.output
        assert "default: 3" in result.output
        assert "--reaction-sensitivity" in result.output
        assert "--audio-clean-mode" in result.output
        assert "--motion-cache / --no-motion-cache" in result.output

    def test_smoke_writes_log_and_audio_intervals(self, tmp_path: Path):
        sr = 16000
        main = make_speech_pattern([(0.0, 1.0, -10.0)], 4.0, sr)
        cohost = make_speech_pattern([(1.2, 2.0, -10.0)], 4.0, sr)
        guest = make_speech_pattern([(2.2, 3.5, -10.0)], 4.0, sr)
        main_path = tmp_path / "main.wav"
        cohost_path = tmp_path / "cohost.wav"
        guest_path = tmp_path / "guest.wav"
        sf.write(str(main_path), main, sr)
        sf.write(str(cohost_path), cohost, sr)
        sf.write(str(guest_path), guest, sr)

        in_path = tmp_path / "input.prproj"
        out_path = tmp_path / "output.prproj"
        in_path.write_bytes(_build_synthetic_prproj())

        result = CliRunner().invoke(
            cli,
            [
                "auto-switch-sakha-aimakh",
                "--in", str(in_path),
                "--seq", "TestSeq",
                "--out", str(out_path),
                "--mic-main-host", str(main_path),
                "--mic-cohost", str(cohost_path),
                "--mic-guest", str(guest_path),
                "--camera-main-host", "1",
                "--camera-guest-close", "2",
                "--camera-pair-wide", "3",
                "--camera-all-wide", "4",
                "--detector-backend", "rms",
            ],
        )

        assert result.exit_code == 0, result.output
        assert out_path.exists()
        log_path = out_path.with_suffix(out_path.suffix + ".log.jsonl")
        assert log_path.exists()
        entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        assert any(entry.get("event") == "sakha_camera_segments" for entry in entries)
        assert any(entry.get("event") == "waveform_gate_diagnostics" for entry in entries)
        assert any(entry.get("event") == "source_owner_diagnostics" for entry in entries)
        assert any(entry.get("event") == "sakha_leak_matrix" for entry in entries)
        assert any(entry.get("event") == "audio_track_intervals" for entry in entries)

    def test_cli_auto_resolves_audio_sources_from_xml(self, tmp_path: Path):
        sr = 16000
        main = make_speech_pattern([(0.0, 1.0, -10.0)], 4.0, sr)
        cohost = make_speech_pattern([(1.2, 2.0, -10.0)], 4.0, sr)
        guest = make_speech_pattern([(2.2, 3.5, -10.0)], 4.0, sr)
        main_path = tmp_path / "main.wav"
        cohost_path = tmp_path / "cohost.wav"
        guest_path = tmp_path / "guest.wav"
        sf.write(str(main_path), main, sr)
        sf.write(str(cohost_path), cohost, sr)
        sf.write(str(guest_path), guest, sr)

        xml_path = tmp_path / "episode.xml"
        xml_path.write_text(
            f"""<?xml version="1.0" encoding="UTF-8"?>
<xmeml version="5">
  <sequence>
    <name>TestSeq</name>
    <media>
      <audio>
        <track>
          <clipitem><file><name>main</name><pathurl>file://localhost{main_path.as_posix()}</pathurl></file></clipitem>
        </track>
        <track>
          <clipitem><file><name>cohost</name><pathurl>file://localhost{cohost_path.as_posix()}</pathurl></file></clipitem>
        </track>
        <track>
          <clipitem><file><name>guest</name><pathurl>file://localhost{guest_path.as_posix()}</pathurl></file></clipitem>
        </track>
      </audio>
    </media>
  </sequence>
</xmeml>
""",
            encoding="utf-8",
        )

        in_path = tmp_path / "input.prproj"
        out_path = tmp_path / "output.prproj"
        in_path.write_bytes(_build_synthetic_prproj())

        result = CliRunner().invoke(
            cli,
            [
                "auto-switch-sakha-aimakh",
                "--in", str(in_path),
                "--seq", "TestSeq",
                "--out", str(out_path),
                "--xml", str(xml_path),
                "--camera-main-host", "1",
                "--camera-guest-close", "2",
                "--camera-pair-wide", "3",
                "--camera-all-wide", "4",
                "--detector-backend", "rms",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Audio sources: xml" in result.output
        log_path = out_path.with_suffix(out_path.suffix + ".log.jsonl")
        entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        audio_source_entry = next(entry for entry in entries if entry.get("event") == "audio_sources")
        assert audio_source_entry["method"] == "xml"
        assert audio_source_entry["tracks"]["0"]["path"] == str(main_path)
        assert audio_source_entry["tracks"]["1"]["path"] == str(cohost_path)
        assert audio_source_entry["tracks"]["2"]["path"] == str(guest_path)

    def test_cli_runs_with_default_camera_mapping(self, tmp_path: Path):
        sr = 16000
        main_path = tmp_path / "main.wav"
        cohost_path = tmp_path / "cohost.wav"
        guest_path = tmp_path / "guest.wav"
        sf.write(str(main_path), make_speech_pattern([(0.0, 0.8, -10.0)], 2.0, sr), sr)
        sf.write(str(cohost_path), make_speech_pattern([(0.9, 1.2, -10.0)], 2.0, sr), sr)
        sf.write(str(guest_path), make_speech_pattern([(1.3, 1.8, -10.0)], 2.0, sr), sr)

        in_path = tmp_path / "input.prproj"
        out_path = tmp_path / "output.prproj"
        in_path.write_bytes(_build_synthetic_prproj())

        result = CliRunner().invoke(
            cli,
            [
                "auto-switch-sakha-aimakh",
                "--in", str(in_path),
                "--seq", "TestSeq",
                "--out", str(out_path),
                "--mic-main-host", str(main_path),
                "--mic-cohost", str(cohost_path),
                "--mic-guest", str(guest_path),
                "--detector-backend", "rms",
            ],
        )

        assert result.exit_code == 0, result.output
        entries = [
            json.loads(line)
            for line in out_path.with_suffix(out_path.suffix + ".log.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        mapping = next(entry for entry in entries if entry.get("event") == "camera_mapping")
        assert mapping["camera_main_host_premiere"] == 1
        assert mapping["camera_guest_close_premiere"] == 2
        assert mapping["camera_pair_wide_premiere"] == 3
        assert mapping["camera_all_wide_premiere"] == 4
        assert any(entry.get("event") == "debleed_diagnostics" for entry in entries)

    def test_reaction_sensitivity_matches_legacy_temperature(self, tmp_path: Path):
        sr = 16000
        main_path = tmp_path / "main.wav"
        cohost_path = tmp_path / "cohost.wav"
        guest_path = tmp_path / "guest.wav"
        sf.write(str(main_path), make_speech_pattern([(0.0, 0.6, -10.0)], 1.5, sr), sr)
        sf.write(str(cohost_path), make_speech_pattern([(0.6, 1.0, -10.0)], 1.5, sr), sr)
        sf.write(str(guest_path), make_speech_pattern([(1.0, 1.4, -10.0)], 1.5, sr), sr)

        configs = []
        for idx, flag in enumerate(["--temperature", "--reaction-sensitivity"]):
            in_path = tmp_path / f"input_{idx}.prproj"
            out_path = tmp_path / f"output_{idx}.prproj"
            in_path.write_bytes(_build_synthetic_prproj())
            result = CliRunner().invoke(
                cli,
                [
                    "auto-switch-sakha-aimakh",
                    "--in", str(in_path),
                    "--seq", "TestSeq",
                    "--out", str(out_path),
                    "--mic-main-host", str(main_path),
                    "--mic-cohost", str(cohost_path),
                    "--mic-guest", str(guest_path),
                    "--detector-backend", "rms",
                    flag, "25",
                ],
            )
            assert result.exit_code == 0, result.output
            entries = [
                json.loads(line)
                for line in out_path.with_suffix(out_path.suffix + ".log.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            configs.append(next(entry for entry in entries if entry.get("event") == "sakha_aimakh_config"))

        assert configs[0]["reaction_sensitivity"] == configs[1]["reaction_sensitivity"] == 25.0
        assert configs[0]["dominance_delta_db"] == configs[1]["dominance_delta_db"]
        assert configs[0]["debleed_overlap_margin_db"] == configs[1]["debleed_overlap_margin_db"]


def test_calibrated_mode_picks_real_speaker_over_overgained_bleed():
    """calibrated mode normalises every channel to its own reference level, so
    an over-gained microphone whose bleed is louder in raw dB still loses to
    the real (quieter) speaker — and the previous owner's hysteresis is
    correctly overridden once the rival is decisively louder."""
    cfg = _config(audio_clean_mode="calibrated")
    plan = _plan(
        {
            "main_host": _level_activity("main_host", [], 6.0, default_db=-70.0),
            # Hot cohost mic: own speech at -8 dB, bleed at -22 dB — still
            # louder in raw dB than the guest's -30 dB direct voice.
            "cohost": _level_activity(
                "cohost",
                [(0.0, 3.0, -8.0), (3.0, 6.0, -22.0)],
                6.0,
                default_db=-50.0,
            ),
            "guest": _level_activity(
                "guest",
                [(3.0, 6.0, -30.0)],
                6.0,
                default_db=-70.0,
            ),
        },
        6.0,
        cfg,
    )

    # [0,3]: only the cohost talks -> it becomes the held owner.
    assert plan.frame_states[10].active_keys == ("cohost",)
    # [3,6]: the guest talks, the cohost mic only catches bleed. Although the
    # cohost is louder in raw dB, normalised loudness hands the frame to the
    # guest and overrides the held cohost.
    assert plan.frame_states[40].active_keys == ("guest",)
    assert plan.frame_states[40].state == SakhaAymakhState.GUEST_ONLY


def test_calibrated_mode_opens_both_mics_on_genuine_overlap():
    """When two channels are both close to their own reference level they are
    both really talking, so calibrated mode opens both."""
    cfg = _config(audio_clean_mode="calibrated")
    plan = _plan(
        {
            "main_host": _level_activity("main_host", [], 4.0, default_db=-70.0),
            "cohost": _level_activity("cohost", [(0.0, 4.0, -20.0)], 4.0, default_db=-60.0),
            "guest": _level_activity("guest", [(0.0, 4.0, -25.0)], 4.0, default_db=-70.0),
        },
        4.0,
        cfg,
    )

    assert set(plan.frame_states[20].active_keys) == {"cohost", "guest"}
    assert plan.frame_states[20].state == SakhaAymakhState.COHOST_GUEST


def test_studio_mode_guest_much_louder_picks_guest():
    """Guest clearly loudest (guest -20, cohost only -34 leak, main silent): leak-matrix
    unmixing opens the guest mic and keeps the cohost (bleed) mic closed."""
    cfg = _config(audio_clean_mode="studio")
    plan = _plan(
        {
            "main_host": _level_activity("main_host", [], 4.0, default_db=-60.0),
            "cohost": _level_activity("cohost", [(0.5, 3.5, -34.0)], 4.0, default_db=-60.0),
            "guest": _level_activity("guest", [(0.5, 3.5, -20.0)], 4.0, default_db=-60.0),
        },
        4.0,
        cfg,
    )
    frame = plan.frame_states[20]
    assert "guest" in frame.active_keys
    assert "cohost" not in frame.active_keys
    assert frame.state == SakhaAymakhState.GUEST_ONLY


def test_studio_mode_leak_louder_than_direct_picks_source():
    """The cohost mic catches the guest's voice LOUDER (-26) than the guest's own mic (-34), but
    it is a lag-aligned scaled copy: the waveform cross-check keeps the guest (true source) open
    and the cohost (bleed) muted."""
    sr = 1000
    guest_voice = _speech_noise(5.0, sr, [(0.0, 2.0)], seed=1)
    audio = {
        "main_host": np.zeros_like(guest_voice),
        "cohost": 1.8 * _delay(guest_voice, 4),
        "guest": guest_voice,
    }
    cfg = _config(audio_clean_mode="studio")
    plan = _plan(
        {
            "main_host": _level_activity("main_host", [], 5.0, default_db=-70.0),
            "cohost": _level_activity("cohost", [(0.0, 2.0, -26.0)], 5.0, default_db=-65.0),
            "guest": _level_activity("guest", [(0.0, 2.0, -34.0), (2.0, 3.0, -18.0)], 5.0, default_db=-62.0),
        },
        5.0,
        cfg,
        audio_arrays_by_key=audio,
        audio_sample_rate=sr,
    )
    for idx in (5, 10):
        frame = plan.frame_states[idx]
        assert frame.active_keys == ("guest",), (idx, frame.active_keys)
        assert "cohost" not in frame.active_keys
    assert plan.audio_open_intervals_s[1] == [], "cohost bleed must stay muted"


def test_studio_mode_independent_overlap_opens_both():
    """Two genuinely independent speakers (uncorrelated waveforms) at solid levels both open."""
    sr = 1000
    audio = {
        "main_host": np.zeros(int(4.0 * sr), dtype=np.float64),
        "cohost": _speech_noise(4.0, sr, [(0.0, 2.5)], seed=3),
        "guest": _speech_noise(4.0, sr, [(0.0, 2.5)], seed=4),
    }
    cfg = _config(audio_clean_mode="studio")
    plan = _plan(
        {
            "main_host": _level_activity("main_host", [], 4.0, default_db=-70.0),
            "cohost": _level_activity("cohost", [(0.0, 2.5, -22.0)], 4.0, default_db=-65.0),
            "guest": _level_activity("guest", [(0.0, 2.5, -20.0)], 4.0, default_db=-62.0),
        },
        4.0,
        cfg,
        audio_arrays_by_key=audio,
        audio_sample_rate=sr,
    )
    assert set(plan.frame_states[12].active_keys) == {"cohost", "guest"}


def test_studio_mode_no_audio_fallback_is_deterministic():
    """Without per-track audio the leak-matrix fallback still picks the clearly-loudest source."""
    cfg = _config(audio_clean_mode="studio")
    plan = _plan(
        {
            "main_host": _level_activity("main_host", [(0.5, 3.5, -18.0)], 4.0, default_db=-60.0),
            "cohost": _level_activity("cohost", [(0.5, 3.5, -40.0)], 4.0, default_db=-60.0),
            "guest": _level_activity("guest", [], 4.0, default_db=-60.0),
        },
        4.0,
        cfg,
    )
    frame = plan.frame_states[20]
    assert frame.active_keys == ("main_host",), frame.active_keys
    assert frame.state == SakhaAymakhState.MAIN_HOST_ONLY
