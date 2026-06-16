"""CLI interface for AutoPodcast."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from threading import Lock

import click
import numpy as np

from autopodcast import __version__
from autopodcast.config import build_config
from autopodcast.core.analyzer import analyze_speaker
from autopodcast.core.audio_loader import load_and_align
from autopodcast.core.audio_sources import resolve_sequence_audio_sources
from autopodcast.core.cross_cancel import CrossCancelConfig, cross_cancel
from autopodcast.core.auto_switch_4cams import (
    ParticipantSpec,
    Roundtable4CamConfig,
    build_roundtable_plan,
)
from autopodcast.core.auto_switch_custom import (
    CustomPerson,
    CustomSwitchConfig,
    build_custom_plan,
)
from autopodcast.core.camera_motion import (
    CameraMotionAnalyzer,
    CameraMotionConfig,
    CameraMotionExecutionConfig,
)
from autopodcast.core.monologue_2cam import (
    Monologue2CamConfig,
    build_monologue_plan,
)
from autopodcast.core.monologue_sources import (
    resolve_monologue_camera_sources,
    resolve_sequence_camera_sources,
)
from autopodcast.core.motion_cache import (
    load_motion_plan_from_cache,
    motion_cache_key,
    save_motion_plan_to_cache,
)
from autopodcast.core.sakha_aimakh import (
    SakhaAymakhConfig,
    SakhaParticipantSpec,
    build_sakha_aimakh_plan,
    config_from_controls as build_sakha_config_from_controls,
)
from autopodcast.core.detector import apply_cross_gate, combine_speakers, detect_activity
from autopodcast.core.ducking import generate_ducking_events
from autopodcast.core.segmenter import build_audio_mute_segments, build_camera_segments, run_length_encode
from autopodcast.core.camera_scheduler import camera_events_to_segments, schedule_camera_events
from autopodcast.export.json_export import load_timeline, save_timeline
from autopodcast.import_xml import parse_premiere_xml
from autopodcast.models.domain import Timeline
from autopodcast.models.project import AudioInput, ProjectConfig


def _maybe_apply_cross_cancel(
    audio_arrays: list[np.ndarray],
    sample_rate: int,
    enabled: bool,
    fir_taps: int,
) -> list[np.ndarray]:
    """Pre-process raw mic tracks with Wiener-Hopf cross-channel cancellation.

    When ``enabled`` is False or only one track is present, returns the input
    unchanged. Otherwise fits a per-pair FIR filter on solo-speaker segments
    and subtracts the predicted bleed from each channel before the RMS / VAD
    detector sees it. Goal: suppress fake speaker activations caused by
    cross-talk between mics during long silences on one side.
    """
    if not enabled or len(audio_arrays) < 2:
        return audio_arrays
    return cross_cancel(
        audio_arrays,
        sample_rate,
        CrossCancelConfig(fir_taps=fir_taps),
    )


def _cross_cancel_options(default_enabled: bool = False):
    """Decorator factory: add --cross-cancel / --cross-cancel-fir-taps options.

    ``default_enabled`` controls whether cross-channel cancellation runs by
    default. The 1-host-1-guest ``auto-multicam`` command enables it, because
    mic bleed there routinely makes the detector report 'both speakers' and
    pins the camera on the wide shot. Other commands keep it opt-in.
    """
    def deco(f):
        f = click.option(
            "--cross-cancel-fir-taps",
            default=256,
            type=int,
            show_default=True,
            help="FIR filter length (samples) for cross-channel cancellation. 256 ~ 16 ms at 16 kHz.",
        )(f)
        f = click.option(
            "--cross-cancel/--no-cross-cancel",
            "enable_cross_cancel",
            default=default_enabled,
            show_default=True,
            help="Subtract per-pair mic bleed estimates before speech detection (recommended for multi-mic setups with audible cross-talk).",
        )(f)
        return f
    return deco


def _parse_role_overrides(overrides: tuple[str, ...]) -> dict[str, str]:
    """Parse --role-override 'label=role' args into a dict."""
    result = {}
    for item in overrides:
        if "=" not in item:
            raise click.BadParameter(
                f"Expected format 'label=role', got '{item}'",
                param_hint="--role-override",
            )
        label, role = item.split("=", 1)
        result[label.strip()] = role.strip()
    return result


CAMERA_SEGMENTER_SIGNATURE = "denseoverlap_v2"
CAMERA_PLANNER_SIGNATURE = "takeover_v1_dialoguewide_v1_stickywide_v1_dialoguecluster_v1"
MONOLOGUE_PLANNER_SIGNATURE = "monologue_pause_target_motion_guard_v3"
SAKHA_AYMAKH_PLANNER_SIGNATURE = "sakha_aimakh_v5_source_owner_detector"


def _serialize_segment(seg) -> dict:
    return {
        "start": round(seg.start_s, 3),
        "end": round(seg.end_s, 3),
        "state": seg.speaker_state.value,
        "label": seg.speaker_label,
    }


def _serialize_camera_event(ev) -> dict:
    return {
        "start": round(ev.start_s, 3),
        "end": round(ev.end_s, 3),
        "angle": ev.camera_index,
        "state": ev.speaker_state.value,
        "label": ev.speaker_label,
    }


def _serialize_roundtable_speech_segment(seg) -> dict:
    return {
        "start": round(seg.start_s, 3),
        "end": round(seg.end_s, 3),
        "state": seg.state.value,
        "focus": seg.focus_key,
        "active": list(seg.active_keys),
    }


def _serialize_roundtable_conversation_phase(phase) -> dict:
    return {
        "start": round(phase.start_s, 3),
        "end": round(phase.end_s, 3),
        "phase": phase.phase.value,
        "focus": phase.focus_key,
        "active": list(phase.active_keys),
        "segment_start_idx": phase.segment_start_idx,
        "segment_end_idx": phase.segment_end_idx,
    }


def _serialize_roundtable_camera_segment(seg) -> dict:
    return {
        "start": round(seg.start_s, 3),
        "end": round(seg.end_s, 3),
        "angle": seg.camera_index,
        "reason": seg.reason,
        "focus": seg.focus_key,
    }


def _serialize_roundtable_motion_event(event) -> dict:
    return {
        "reason": event.reason,
        "time": round(event.time_s, 3),
        "from_angle": event.from_camera_index,
        "to_angle": event.to_camera_index,
        "desired_angle": event.desired_camera_index,
        "moving_from": None if event.moving_from_s is None else round(event.moving_from_s, 3),
        "moving_to": None if event.moving_to_s is None else round(event.moving_to_s, 3),
    }


def _serialize_sakha_speech_segment(seg) -> dict:
    return {
        "start": round(seg.start_s, 3),
        "end": round(seg.end_s, 3),
        "state": seg.state.value,
        "focus": seg.focus_key,
        "active": list(seg.active_keys),
    }


def _serialize_sakha_camera_segment(seg) -> dict:
    return {
        "start": round(seg.start_s, 3),
        "end": round(seg.end_s, 3),
        "angle": seg.camera_index,
        "reason": seg.reason,
        "focus": seg.focus_key,
    }


def _serialize_sakha_motion_event(event) -> dict:
    return {
        "reason": event.reason,
        "time": round(event.time_s, 3),
        "from_angle": event.from_camera_index,
        "to_angle": event.to_camera_index,
        "desired_angle": event.desired_camera_index,
        "moving_from": None if event.moving_from_s is None else round(event.moving_from_s, 3),
        "moving_to": None if event.moving_to_s is None else round(event.moving_to_s, 3),
    }


def _serialize_audio_level_segment(seg) -> dict:
    return {
        "start": round(seg.start_s, 3),
        "end": round(seg.end_s, 3),
        "levels_db": {str(track_idx): level for track_idx, level in sorted(seg.levels_db.items())},
    }


def _serialize_interval_map(interval_map: dict[int, list[tuple[float, float]]]) -> dict[str, list[dict[str, float]]]:
    return {
        str(track_idx): [
            {"start": round(start_s, 3), "end": round(end_s, 3)}
            for start_s, end_s in intervals
        ]
        for track_idx, intervals in sorted(interval_map.items())
    }


def _serialize_monologue_camera_segment(seg) -> dict:
    return {
        "start": round(seg.start_s, 3),
        "end": round(seg.end_s, 3),
        "angle": seg.camera_index,
        "reason": seg.reason,
    }


def _serialize_monologue_cut(cut) -> dict:
    return {
        "time": round(cut.time_s, 3),
        "angle": cut.camera_index,
        "reason": cut.reason,
        "target": round(cut.target_s, 3),
        "delta": round(cut.delta_s, 3),
        "planned_reason": cut.planned_reason,
        "planned_time": None if cut.planned_time_s is None else round(cut.planned_time_s, 3),
        "pause_start": None if cut.pause_start_s is None else round(cut.pause_start_s, 3),
        "pause_end": None if cut.pause_end_s is None else round(cut.pause_end_s, 3),
        "pause_duration": None if cut.pause_duration_s is None else round(cut.pause_duration_s, 3),
        "min_envelope_db": None if cut.min_envelope_db is None else round(cut.min_envelope_db, 3),
        "fallback_deadline": None if cut.fallback_deadline_s is None else round(cut.fallback_deadline_s, 3),
        "stable_since": None if cut.stable_since_s is None else round(cut.stable_since_s, 3),
        "moving_from": None if cut.moving_from_s is None else round(cut.moving_from_s, 3),
        "moving_to": None if cut.moving_to_s is None else round(cut.moving_to_s, 3),
    }


def _serialize_motion_interval(interval) -> dict:
    return {
        "start": round(interval.start_s, 3),
        "end": round(interval.end_s, 3),
        "duration": round(interval.duration_s, 3),
    }


def _serialize_monologue_motion_event(event) -> dict:
    return {
        "reason": event.reason,
        "from_angle": event.from_camera_index,
        "to_angle": event.to_camera_index,
        "planned_time": None if event.planned_time_s is None else round(event.planned_time_s, 3),
        "actual_time": None if event.actual_time_s is None else round(event.actual_time_s, 3),
        "delay": None if event.delay_s is None else round(event.delay_s, 3),
        "stable_since": None if event.stable_since_s is None else round(event.stable_since_s, 3),
        "moving_from": None if event.moving_from_s is None else round(event.moving_from_s, 3),
        "moving_to": None if event.moving_to_s is None else round(event.moving_to_s, 3),
    }


def _append_jsonl_entry(path: Path, entry: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@click.group()
def cli():
    """AutoPodcast - automatic rough-cut podcast editing."""
    pass


@cli.command()
@click.option("--audio-a", required=True, type=click.Path(exists=True), help="Path to speaker A audio/video")
@click.option("--label-a", default="host", help="Label for speaker A")
@click.option("--camera-a", default=1, type=int, help="Premiere angle for speaker A (1-based, default: 1)")
@click.option("--audio-b", required=True, type=click.Path(exists=True), help="Path to speaker B audio/video")
@click.option("--label-b", default="guest", help="Label for speaker B")
@click.option("--camera-b", default=2, type=int, help="Premiere angle for speaker B (1-based, default: 2)")
@click.option("--camera-wide", default=3, type=int, help="Premiere angle for wide/default shot (1-based, default: 3)")
@click.option("--output", "-o", default="timeline.json", type=click.Path(), help="Output JSON path")
@click.option("--speech-threshold", default=-24.0, type=float, help="Speech onset threshold in dB")
@click.option("--release-threshold", default=-30.0, type=float, help="Speech release threshold in dB")
@click.option("--hangover", default=1000.0, type=float, help="Hangover duration in ms")
@click.option("--min-segment", default=2000.0, type=float, help="Minimum segment length in ms")
@click.option("--debounce", default=300.0, type=float, help="Debounce duration in ms")
@click.option("--ducking/--no-ducking", default=True, help="Enable audio ducking")
@click.option("--ducking-db", default=-96.0, type=float, help="Ducking level in dB")
@click.option("--long-talk-threshold", default=15.0, type=float, help="Long talk threshold in seconds")
@click.option("--wide-duration", default=5.0, type=float, help="Wide cutaway duration in seconds")
@click.option("--fps", default=29.97, type=float, help="Timeline frame rate")
@click.option("--input-gain", default=0.0, type=float, help="Input gain in dB (applied before analysis)")
@click.option("--max-segment", default=20.0, type=float, help="Maximum segment length in seconds")
@click.option("--gate-fade", default=0.15, type=float, help="Gate fade duration in seconds")
@click.option("--long-talk-mode", default="repeat", type=click.Choice(["once", "repeat"]), help="Wide cutaway mode for long talks")
@click.option("--wide-cooldown", default=20.0, type=float, help="Minimum seconds between wide cutaways")
@click.option("--dialogue-wide-interval", default=24.0, type=float, help="Minimum seconds between re-establishing wide shots during active dialogue")
@click.option("--dialogue-wide-duration", default=2.0, type=float, help="Duration of each re-establishing dialogue wide shot in seconds")
@click.option("--dialogue-wide-min-turns", default=3, type=int, help="Minimum speaker exchanges before inserting a dialogue wide shot")
@click.option("--role-override", multiple=True, help="Role override in format 'label=role' (repeatable)")
@click.option("--detector-backend", default="auto", type=click.Choice(["auto", "rms", "silero"]), help="Speech detector backend")
@click.option("--vad-threshold", default=0.5, type=float, help="Silero VAD speech probability threshold")
@click.option("--vad-min-speech-ms", default=120.0, type=float, help="Silero VAD minimum speech duration in ms")
@click.option("--vad-min-silence-ms", default=80.0, type=float, help="Silero VAD minimum silence duration in ms")
@click.option("--vad-speech-pad-ms", default=30.0, type=float, help="Silero VAD padding around detected speech in ms")
@_cross_cancel_options()
def analyze(
    audio_a, label_a, camera_a,
    audio_b, label_b, camera_b,
    camera_wide, output,
    speech_threshold, release_threshold,
    hangover, min_segment, debounce,
    ducking, ducking_db, long_talk_threshold, wide_duration, fps,
    input_gain, max_segment,
    gate_fade, long_talk_mode, wide_cooldown,
    dialogue_wide_interval, dialogue_wide_duration, dialogue_wide_min_turns, role_override,
    detector_backend, vad_threshold, vad_min_speech_ms, vad_min_silence_ms, vad_speech_pad_ms,
    enable_cross_cancel, cross_cancel_fir_taps,
):
    """Analyze audio and generate a timeline."""
    # Validate 1-based angles
    for label, val in [("camera-a", camera_a), ("camera-b", camera_b), ("camera-wide", camera_wide)]:
        if val < 1:
            raise click.BadParameter(
                f"Angles are 1-based (Premiere Angle 1,2,3...)",
                param_hint=f"--{label}",
            )

    # Convert 1-based (Premiere UI) → 0-based (internal)
    camera_a -= 1
    camera_b -= 1
    camera_wide -= 1

    overrides = _parse_role_overrides(role_override)
    config = build_config(
        audio_a=audio_a, label_a=label_a, camera_a=camera_a,
        audio_b=audio_b, label_b=label_b, camera_b=camera_b,
        camera_wide=camera_wide, output_dir=str(Path(output).parent),
        speech_threshold_db=speech_threshold,
        release_threshold_db=release_threshold,
        hangover_ms=hangover,
        min_segment_ms=min_segment,
        debounce_ms=debounce,
        ducking_enabled=ducking,
        ducking_db=ducking_db,
        long_talk_threshold_sec=long_talk_threshold,
        wide_duration_sec=wide_duration,
        input_gain_db=input_gain,
        max_segment_sec=max_segment,
        gate_fade_s=gate_fade,
        long_talk_mode=long_talk_mode,
        wide_cooldown_sec=wide_cooldown,
        dialogue_wide_interval_sec=dialogue_wide_interval,
        dialogue_wide_duration_sec=dialogue_wide_duration,
        dialogue_wide_min_turns=dialogue_wide_min_turns,
        speaker_role_overrides=overrides,
        detector_backend=detector_backend,
        vad_threshold=vad_threshold,
        vad_min_speech_ms=vad_min_speech_ms,
        vad_min_silence_ms=vad_min_silence_ms,
        vad_speech_pad_ms=vad_speech_pad_ms,
    )

    click.echo(f"Loading audio: {audio_a}, {audio_b}")
    paths = [config.audio_inputs[0].path, config.audio_inputs[1].path]
    audio_arrays = load_and_align(paths, config.sample_rate)
    duration_s = len(audio_arrays[0]) / config.sample_rate

    click.echo(f"Duration: {duration_s:.1f}s")
    audio_arrays = list(
        _maybe_apply_cross_cancel(
            audio_arrays,
            config.sample_rate,
            enabled=enable_cross_cancel,
            fir_taps=cross_cancel_fir_taps,
        )
    )
    if enable_cross_cancel:
        click.echo(f"Cross-channel cancellation: fir_taps={cross_cancel_fir_taps}, channels={len(audio_arrays)}")
    click.echo("Analyzing speaker activity...")

    gain_a = config.input_gain_a_db or config.input_gain_db
    gain_b = config.input_gain_b_db or config.input_gain_db
    activity_a = analyze_speaker(audio_arrays[0], label_a, config, gain_db=gain_a)
    activity_b = analyze_speaker(audio_arrays[1], label_b, config, gain_db=gain_b)

    click.echo("Detecting speech...")
    detect_activity(activity_a, config, audio=audio_arrays[0])
    detect_activity(activity_b, config, audio=audio_arrays[1])

    click.echo("Combining speakers...")
    states = combine_speakers(activity_a, activity_b)
    if config.cross_gate_db > 0:
        states = apply_cross_gate(states, activity_a, activity_b, config.cross_gate_db)

    click.echo("Segmenting timeline...")
    segments = build_camera_segments(states, config)

    click.echo("Scheduling cameras...")
    camera_events = schedule_camera_events(segments, config)
    segments = camera_events_to_segments(camera_events)

    ducking_events = generate_ducking_events(segments, config)

    timeline = Timeline(
        segments=segments,
        ducking_events=ducking_events,
        total_duration_s=duration_s,
        fps=fps,
    )

    output_path = Path(output)
    save_timeline(timeline, output_path)
    click.echo(f"Timeline saved: {output_path} ({len(segments)} segments)")


@cli.command()
@click.option("--timeline", "-t", required=True, type=click.Path(exists=True), help="Timeline JSON file")
def preview(timeline):
    """Preview timeline in terminal."""
    tl = load_timeline(Path(timeline))

    click.echo(f"Duration: {tl.total_duration_s:.1f}s | FPS: {tl.fps} | Segments: {len(tl.segments)}")
    click.echo("-" * 70)

    for i, seg in enumerate(tl.segments):
        label = seg.speaker_label or seg.speaker_state.value
        bar_width = max(1, int(seg.duration_s / tl.total_duration_s * 50))
        bar = "#" * bar_width

        click.echo(
            f"  {i+1:3d}  "
            f"{seg.start_s:7.2f}s - {seg.end_s:7.2f}s  "
            f"[cam{seg.camera_index}]  "
            f"{label:<12s}  "
            f"{bar}"
        )

    if tl.ducking_events:
        click.echo(f"\nDucking events: {len(tl.ducking_events)}")


@cli.command()
@click.option("--timeline", "-t", required=True, type=click.Path(exists=True), help="Timeline JSON file")
@click.option("--cam0", required=True, type=click.Path(), help="Wide camera file path")
@click.option("--cam1", required=True, type=click.Path(), help="Camera 1 file path")
@click.option("--cam2", required=True, type=click.Path(), help="Camera 2 file path")
@click.option("--mic0", type=click.Path(), default=None, help="Mic 0 (speaker A) file path")
@click.option("--mic1", type=click.Path(), default=None, help="Mic 1 (speaker B) file path")
@click.option("--output", "-o", default="roughcut.xml", type=click.Path(), help="Output XML path")
@click.option("--name", default="AutoPodcast Rough Cut", help="Sequence name")
@click.option("--jsx/--no-jsx", default=False, help="Generate ExtendScript for multicam setup")
def export(timeline, cam0, cam1, cam2, mic0, mic1, output, name, jsx):
    """Export timeline as FCP 7 XML for Premiere Pro."""
    from autopodcast.export.fcp7xml import save_fcp7xml

    tl = load_timeline(Path(timeline))

    camera_paths = [
        str(Path(cam0).resolve()),
        str(Path(cam1).resolve()),
        str(Path(cam2).resolve()),
    ]

    audio_paths = None
    if mic0 and mic1:
        audio_paths = [
            str(Path(mic0).resolve()),
            str(Path(mic1).resolve()),
        ]

    output_path = Path(output)
    save_fcp7xml(tl, camera_paths, output_path, audio_paths, name)
    click.echo(f"FCP 7 XML exported: {output_path}")

    if jsx:
        from autopodcast.export.multicam_jsx import generate_multicam_jsx
        jsx_content = generate_multicam_jsx(str(output_path.resolve()), tl, name)
        jsx_path = output_path.with_suffix(".jsx")
        jsx_path.write_text(jsx_content, encoding="utf-8")
        click.echo(f"ExtendScript: {jsx_path}")


@cli.command("from-xml")
@click.argument("xml_file", type=click.Path(exists=True))
@click.option("--audio-a", default=1, type=int, help="Audio track number for host (1-based)")
@click.option("--audio-b", default=2, type=int, help="Audio track number for guest (1-based)")
@click.option("--label-a", default="host", help="Label for speaker A")
@click.option("--label-b", default="guest", help="Label for speaker B")
@click.option("--cam-wide", default=1, type=int, help="Video track number for wide camera (1-based)")
@click.option("--cam-host", default=None, type=int, help="Video track number for host close-up (1-based)")
@click.option("--cam-guest", default=None, type=int, help="Video track number for guest close-up (1-based)")
@click.option("--output", "-o", default="roughcut.xml", type=click.Path(), help="Output XML path")
@click.option("--speech-threshold", default=-24.0, type=float, help="Speech onset threshold in dB")
@click.option("--release-threshold", default=-30.0, type=float, help="Speech release threshold in dB")
@click.option("--hangover", default=1000.0, type=float, help="Hangover duration in ms")
@click.option("--min-segment", default=2000.0, type=float, help="Minimum segment length in ms")
@click.option("--debounce", default=300.0, type=float, help="Debounce duration in ms")
@click.option("--ducking/--no-ducking", default=True, help="Enable audio ducking")
@click.option("--ducking-db", default=-96.0, type=float, help="Ducking level in dB")
@click.option("--long-talk-threshold", default=15.0, type=float, help="Long talk threshold in seconds")
@click.option("--wide-duration", default=5.0, type=float, help="Wide cutaway duration in seconds")
@click.option("--input-gain", default=0.0, type=float, help="Input gain in dB (applied before analysis)")
@click.option("--max-segment", default=20.0, type=float, help="Maximum segment length in seconds")
@click.option("--gate-fade", default=0.15, type=float, help="Gate fade duration in seconds")
@click.option("--long-talk-mode", default="repeat", type=click.Choice(["once", "repeat"]), help="Wide cutaway mode for long talks")
@click.option("--wide-cooldown", default=20.0, type=float, help="Minimum seconds between wide cutaways")
@click.option("--dialogue-wide-interval", default=24.0, type=float, help="Minimum seconds between re-establishing wide shots during active dialogue")
@click.option("--dialogue-wide-duration", default=2.0, type=float, help="Duration of each re-establishing dialogue wide shot in seconds")
@click.option("--dialogue-wide-min-turns", default=3, type=int, help="Minimum speaker exchanges before inserting a dialogue wide shot")
@click.option("--role-override", multiple=True, help="Role override in format 'label=role' (repeatable)")
@click.option("--detector-backend", default="auto", type=click.Choice(["auto", "rms", "silero"]), help="Speech detector backend")
@click.option("--vad-threshold", default=0.5, type=float, help="Silero VAD speech probability threshold")
@click.option("--vad-min-speech-ms", default=120.0, type=float, help="Silero VAD minimum speech duration in ms")
@click.option("--vad-min-silence-ms", default=80.0, type=float, help="Silero VAD minimum silence duration in ms")
@click.option("--vad-speech-pad-ms", default=30.0, type=float, help="Silero VAD padding around detected speech in ms")
@click.option("--jsx/--no-jsx", default=False, help="Generate ExtendScript for multicam setup")
@click.option("--media-dir", default=None, type=click.Path(exists=True, file_okay=False),
              help="Directory with media files (if paths in XML are invalid)")
@_cross_cancel_options()
def from_xml(
    xml_file, audio_a, audio_b, label_a, label_b, cam_wide, cam_host, cam_guest, output,
    speech_threshold, release_threshold,
    hangover, min_segment, debounce,
    ducking, ducking_db, long_talk_threshold, wide_duration,
    input_gain, max_segment,
    gate_fade, long_talk_mode, wide_cooldown,
    dialogue_wide_interval, dialogue_wide_duration, dialogue_wide_min_turns, role_override,
    detector_backend, vad_threshold, vad_min_speech_ms, vad_min_silence_ms, vad_speech_pad_ms,
    jsx,
    media_dir,
    enable_cross_cancel, cross_cancel_fir_taps,
):
    """Analyze Premiere XML and generate rough-cut XML.

    Reads a Premiere Pro FCP 7 XML export, extracts audio/video file paths,
    analyzes audio for speaker activity, and generates a new XML with camera switches.
    """
    from autopodcast.export.fcp7xml import save_fcp7xml

    xml_path = Path(xml_file)
    click.echo(f"Parsing: {xml_path.name}")
    seq = parse_premiere_xml(xml_path)

    click.echo(f"Sequence: {seq.name} ({seq.width}x{seq.height}, {seq.fps}fps)")
    click.echo(f"Video tracks: {len(seq.video_tracks)}")
    for i, t in enumerate(seq.video_tracks, 1):
        click.echo(f"  [{i}] {t.name}")
    click.echo(f"Audio tracks: {len(seq.audio_tracks)}")
    for i, t in enumerate(seq.audio_tracks, 1):
        click.echo(f"  [{i}] {t.name}")

    # Validate track indices
    if audio_a < 1 or audio_a > len(seq.audio_tracks):
        raise click.BadParameter(
            f"audio-a={audio_a}, but only {len(seq.audio_tracks)} audio tracks found",
            param_hint="--audio-a",
        )
    if audio_b < 1 or audio_b > len(seq.audio_tracks):
        raise click.BadParameter(
            f"audio-b={audio_b}, but only {len(seq.audio_tracks)} audio tracks found",
            param_hint="--audio-b",
        )
    if cam_wide < 1 or cam_wide > len(seq.video_tracks):
        raise click.BadParameter(
            f"cam-wide={cam_wide}, but only {len(seq.video_tracks)} video tracks found",
            param_hint="--cam-wide",
        )

    # Audio paths (0-based index)
    track_a = seq.audio_tracks[audio_a - 1]
    track_b = seq.audio_tracks[audio_b - 1]

    click.echo(f"\nHost audio:  [{audio_a}] {track_a.name}")
    click.echo(f"Guest audio: [{audio_b}] {track_b.name}")

    # Camera mapping:
    # cam_wide_idx (0-based) -> camera_index 0 (wide/default)
    # cam_host/cam_guest -> camera_index 1 (host), 2 (guest)
    cam_wide_idx = cam_wide - 1
    wide_track = seq.video_tracks[cam_wide_idx]

    if cam_host is not None and cam_guest is not None:
        # Explicit mapping
        if cam_host < 1 or cam_host > len(seq.video_tracks):
            raise click.BadParameter(
                f"cam-host={cam_host}, but only {len(seq.video_tracks)} video tracks found",
                param_hint="--cam-host",
            )
        if cam_guest < 1 or cam_guest > len(seq.video_tracks):
            raise click.BadParameter(
                f"cam-guest={cam_guest}, but only {len(seq.video_tracks)} video tracks found",
                param_hint="--cam-guest",
            )
        host_track = seq.video_tracks[cam_host - 1]
        guest_track = seq.video_tracks[cam_guest - 1]
        other_tracks = [host_track, guest_track]
    else:
        # Auto mapping: remaining tracks in order
        other_tracks = [t for i, t in enumerate(seq.video_tracks) if i != cam_wide_idx]

    click.echo(f"Wide camera: [{cam_wide}] {wide_track.name}")
    for i, t in enumerate(other_tracks):
        role = "host" if i == 0 else "guest"
        click.echo(f"  cam {role}: {t.name}")

    # Build camera_paths list: index 0=wide, 1=host, 2=guest
    camera_paths = [wide_track.file_path] + [t.file_path for t in other_tracks]
    camera_pathurls = [wide_track.pathurl] + [t.pathurl for t in other_tracks]
    camera_fps_list = [wide_track.fps] + [t.fps for t in other_tracks]

    # Audio paths for export
    audio_paths = [track_a.file_path, track_b.file_path]
    audio_pathurls = [track_a.pathurl, track_b.pathurl]
    audio_fps_list = [track_a.fps, track_b.fps]

    fps = seq.fps

    click.echo(f"Sequence fps: {fps}, Clip fps: {camera_fps_list}")

    # Build config
    overrides = _parse_role_overrides(role_override)
    config = build_config(
        audio_a=track_a.file_path, label_a=label_a, camera_a=1,
        audio_b=track_b.file_path, label_b=label_b, camera_b=2,
        camera_wide=0,
        output_dir=str(Path(output).parent),
        speech_threshold_db=speech_threshold,
        release_threshold_db=release_threshold,
        hangover_ms=hangover,
        min_segment_ms=min_segment,
        debounce_ms=debounce,
        ducking_enabled=ducking,
        ducking_db=ducking_db,
        long_talk_threshold_sec=long_talk_threshold,
        wide_duration_sec=wide_duration,
        input_gain_db=input_gain,
        max_segment_sec=max_segment,
        gate_fade_s=gate_fade,
        long_talk_mode=long_talk_mode,
        wide_cooldown_sec=wide_cooldown,
        dialogue_wide_interval_sec=dialogue_wide_interval,
        dialogue_wide_duration_sec=dialogue_wide_duration,
        dialogue_wide_min_turns=dialogue_wide_min_turns,
        speaker_role_overrides=overrides,
        detector_backend=detector_backend,
        vad_threshold=vad_threshold,
        vad_min_speech_ms=vad_min_speech_ms,
        vad_min_silence_ms=vad_min_silence_ms,
        vad_speech_pad_ms=vad_speech_pad_ms,
        sequence_width=seq.width,
        sequence_height=seq.height,
    )

    # Load and analyze audio
    search_dir = Path(media_dir) if media_dir else None
    click.echo("\nLoading audio...")
    audio_arrays = load_and_align(
        [Path(track_a.file_path), Path(track_b.file_path)],
        config.sample_rate,
        search_dir=search_dir,
    )
    duration_s = len(audio_arrays[0]) / config.sample_rate
    click.echo(f"Duration: {duration_s:.1f}s")
    audio_arrays = list(
        _maybe_apply_cross_cancel(
            audio_arrays,
            config.sample_rate,
            enabled=enable_cross_cancel,
            fir_taps=cross_cancel_fir_taps,
        )
    )
    if enable_cross_cancel:
        click.echo(f"Cross-channel cancellation: fir_taps={cross_cancel_fir_taps}, channels={len(audio_arrays)}")

    click.echo("Analyzing speaker activity...")
    gain_a = config.input_gain_a_db or config.input_gain_db
    gain_b = config.input_gain_b_db or config.input_gain_db
    activity_a = analyze_speaker(audio_arrays[0], "host", config, gain_db=gain_a)
    activity_b = analyze_speaker(audio_arrays[1], "guest", config, gain_db=gain_b)

    click.echo("Detecting speech...")
    detect_activity(activity_a, config, audio=audio_arrays[0])
    detect_activity(activity_b, config, audio=audio_arrays[1])

    click.echo("Combining speakers...")
    states = combine_speakers(activity_a, activity_b)
    if config.cross_gate_db > 0:
        states = apply_cross_gate(states, activity_a, activity_b, config.cross_gate_db)

    click.echo("Segmenting timeline...")
    segments = build_camera_segments(states, config)

    click.echo("Scheduling cameras...")
    camera_events = schedule_camera_events(segments, config)
    segments = camera_events_to_segments(camera_events)

    ducking_events = generate_ducking_events(segments, config)

    timeline = Timeline(
        segments=segments,
        ducking_events=ducking_events,
        total_duration_s=duration_s,
        fps=fps,
    )

    # Convert source offsets from input XML (sequence frames → seconds)
    cam_source_in_s = [
        wide_track.source_in_frames / fps,
        *(t.source_in_frames / fps for t in other_tracks),
    ]
    audio_source_in_s = [
        track_a.source_in_frames / fps,
        track_b.source_in_frames / fps,
    ]

    # Export XML with original pathurls and source offsets
    output_path = Path(output)
    save_fcp7xml(
        timeline, camera_paths, output_path, audio_paths,
        sequence_name=f"{seq.name} - Rough Cut",
        sequence_width=seq.width,
        sequence_height=seq.height,
        camera_pathurls=camera_pathurls,
        audio_pathurls=audio_pathurls,
        camera_fps=camera_fps_list,
        audio_fps=audio_fps_list,
        camera_source_in_s=cam_source_in_s,
        audio_source_in_s=audio_source_in_s,
    )
    click.echo(f"\nRough cut XML: {output_path} ({len(segments)} segments)")

    if jsx:
        from autopodcast.export.multicam_jsx import generate_multicam_jsx
        jsx_content = generate_multicam_jsx(
            str(output_path.resolve()), timeline,
            sequence_name=f"{seq.name} - Rough Cut",
        )
        jsx_path = output_path.with_suffix(".jsx")
        jsx_path.write_text(jsx_content, encoding="utf-8")
        click.echo(f"ExtendScript: {jsx_path}")

    click.echo("Import into Premiere Pro: File > Import")


@cli.command()
@click.option("--mic-a", required=True, type=click.Path(exists=True), help="Path to speaker A audio file")
@click.option("--mic-b", required=True, type=click.Path(exists=True), help="Path to speaker B audio file")
@click.option("--label-a", default="host", help="Label for speaker A")
@click.option("--label-b", default="guest", help="Label for speaker B")
@click.option("--window", default=3.0, type=float, help="Noise estimation window in seconds")
@click.option("--percentile", default=95.0, type=float, help="RMS percentile for noise floor")
@click.option("--margin", default=12.0, type=float, help="Margin above noise floor in dB")
def calibrate(mic_a, mic_b, label_a, label_b, window, percentile, margin):
    """Estimate noise floor and suggest speech thresholds."""
    from autopodcast.core.audio_loader import load_audio
    from autopodcast.core.calibrator import calibrate as do_calibrate

    results = []
    for path, label in [(mic_a, label_a), (mic_b, label_b)]:
        audio = load_audio(Path(path), target_sr=16000)
        result = do_calibrate(audio, 16000, label, window_s=window, percentile=percentile, margin_db=margin)
        results.append(result)

    for r in results:
        click.echo(f"\n[{r.label}]")
        click.echo(f"  Noise floor:       {r.noise_floor_db:.1f} dB")
        click.echo(f"  Speech threshold:  {r.speech_threshold_db:.1f} dB")
        click.echo(f"  Release threshold: {r.release_threshold_db:.1f} dB")

    click.echo("\nSuggested flags:")
    click.echo(
        f"  --speech-threshold {results[0].speech_threshold_db:.1f} "
        f"--release-threshold {results[0].release_threshold_db:.1f}"
    )


def _resolve_speaker_mics(in_file, seq, specs, xml_file):
    """Auto-resolve any missing speaker microphone from the input .prproj/XML sequence
    audio tracks, mirroring auto-switch-sakha-aimakh. ``specs`` is a list of
    ``(audio_track_index_0based, label, mic_path_or_None)`` in speaker order. The project is
    only parsed when at least one mic is missing; manually supplied --mic paths pass through.

    Returns ``(resolved_paths, audio_sources_log_or_None, mute_audio_ok)``.
    """
    resolved: list[Path | None] = [Path(p) if p is not None else None for (_ti, _lbl, p) in specs]
    track_indices = [ti for (ti, _lbl, _p) in specs]
    labels = [lbl for (_ti, lbl, _p) in specs]
    audio_sources_log = None
    if any(p is None for p in resolved):
        try:
            sources = resolve_sequence_audio_sources(
                Path(in_file), seq, track_indices, xml_path=xml_file
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"Audio sources: {sources.method}")
        audio_sources_log = {"event": "audio_sources", "method": sources.method, "tracks": {}}
        for i, ti in enumerate(track_indices):
            source = sources.sources_by_track[ti]
            if resolved[i] is None:
                resolved[i] = source.representative_path
            click.echo(f"  {labels[i]}: track {ti + 1} -> {resolved[i]}")
            audio_sources_log["tracks"][str(ti)] = {
                "label": labels[i],
                "name": source.name,
                "path": str(source.representative_path),
                "paths": [str(pp) for pp in source.paths],
            }
    if any(p is None for p in resolved):
        raise click.ClickException(
            "Missing microphone paths. Provide the --mic option(s) manually or let the command "
            "resolve them from audio tracks in the .prproj/XML."
        )
    mute_audio_ok = True
    unique = {str(p.resolve()) for p in resolved}
    if len(unique) < len(resolved):
        click.echo(
            "Warning: несколько дорожек указывают на один и тот же аудиофайл — "
            "раздельный анализ по спикерам невозможен. Укажите отдельные файлы через --mic; "
            "заглушение дорожек отключено.",
            err=True,
        )
        mute_audio_ok = False
    return resolved, audio_sources_log, mute_audio_ok


@cli.command("auto-switch-4cams")
@click.option("--in", "in_file", required=True, type=click.Path(exists=True), help="Input .prproj file")
@click.option("--seq", required=True, help="Sequence name in .prproj")
@click.option("--out", "out_file", required=True, type=click.Path(), help="Output .prproj file")
@click.option("--mic-host", type=click.Path(exists=True), help="Host microphone audio file (empty = auto-resolve from project)")
@click.option("--mic-guest-1", type=click.Path(exists=True), help="Guest 1 microphone audio file (empty = auto-resolve from project)")
@click.option("--mic-guest-2", type=click.Path(exists=True), help="Guest 2 microphone audio file (empty = auto-resolve from project)")
@click.option("--mic-guest-3", type=click.Path(exists=True), help="Guest 3 microphone audio file (empty = auto-resolve from project)")
@click.option("--label-host", default="host", help="Label for host track")
@click.option("--label-guest-1", default="guest_1", help="Label for guest 1 track")
@click.option("--label-guest-2", default="guest_2", help="Label for guest 2 track")
@click.option("--label-guest-3", default="guest_3", help="Label for guest 3 track")
@click.option("--camera-all-wide", default=1, type=int, help="Premiere angle for full wide (CAM_1, 1-based)")
@click.option("--camera-guests-wide", default=2, type=int, help="Premiere angle for guest wide (CAM_2, 1-based)")
@click.option("--camera-guest-close", default=3, type=int, help="Premiere angle for moving guest close-up (CAM_3, 1-based)")
@click.option("--camera-host-close", default=4, type=int, help="Premiere angle for host close-up (CAM_4, 1-based)")
@click.option("--audio-track-host", default=1, type=int, help="Audio track number for host mic (1-based)")
@click.option("--audio-track-guest-1", default=2, type=int, help="Audio track number for guest 1 mic (1-based)")
@click.option("--audio-track-guest-2", default=3, type=int, help="Audio track number for guest 2 mic (1-based)")
@click.option("--audio-track-guest-3", default=4, type=int, help="Audio track number for guest 3 mic (1-based)")
@click.option("--speech-threshold", default=-27.0, type=float, help="Speech onset threshold in dBFS")
@click.option("--release-threshold", default=-31.0, type=float, help="Speech release threshold in dBFS")
@click.option("--input-gain", default=0.0, type=float, help="Input gain in dB for all mics")
@click.option("--input-gain-host", default=0.0, type=float, help="Input gain in dB for host mic")
@click.option("--input-gain-guest-1", default=0.0, type=float, help="Input gain in dB for guest 1 mic")
@click.option("--input-gain-guest-2", default=0.0, type=float, help="Input gain in dB for guest 2 mic")
@click.option("--input-gain-guest-3", default=0.0, type=float, help="Input gain in dB for guest 3 mic")
@click.option("--detector-backend", default="auto", type=click.Choice(["auto", "rms", "silero"]), help="Speech detector backend")
@click.option("--vad-threshold", default=0.65, type=float, help="Silero VAD speech probability threshold")
@click.option("--vad-min-speech-ms", default=120.0, type=float, help="Silero VAD minimum speech duration in ms")
@click.option("--vad-min-silence-ms", default=80.0, type=float, help="Silero VAD minimum silence duration in ms")
@click.option("--vad-speech-pad-ms", default=30.0, type=float, help="Silero VAD padding around detected speech in ms")
@click.option("--min-speech-duration-ms", default=120.0, type=float, help="Minimum stable speech duration in ms")
@click.option("--release-ms", default=220.0, type=float, help="Speech release debounce in ms")
@click.option("--hold-ms", default=350.0, type=float, help="Hold time after speech in ms")
@click.option("--dominance-delta-db", default=6.0, type=float, help="dB delta to treat a quieter channel as bleed")
@click.option("--shot-hold", default=1.4, type=float, help="Minimum steady shot duration in seconds")
@click.option("--cooldown", default=0.9, type=float, help="Minimum time between video cuts in seconds")
@click.option("--silence-timeout", default=0.9, type=float, help="Silence timeout before switching to full wide")
@click.option("--cam3-wait-timeout", default=0.0, type=float, help="Camera 3 reposition delay in seconds (0 = switch immediately)")
@click.option("--cam3-cut-in-grace", default=0.15, type=float, help="Grace window to cut into ready camera 3")
@click.option("--attenuation-level", default=-18.0, type=float, help="Target dB for inactive but not yet hard-muted speakers")
@click.option("--standby-level", default=-9.0, type=float, help="Target dB for recently active speakers")
@click.option("--hard-mute-timeout", default=2.0, type=float, help="Seconds before inactive speaker is fully muted")
@click.option("--audio-pre-roll", default=0.24, type=float, help="Open audio this many seconds before detected onset")
@click.option("--audio-post-roll", default=0.12, type=float, help="Keep audio open this many seconds after detected end")
@click.option("--reestablish-wide-interval", default=25.0, type=float, help="Minimum seconds between full-width re-establishing shots")
@click.option("--reestablish-wide-duration", default=1.8, type=float, help="Duration of each re-establishing full wide shot")
@click.option("--reestablish-min-turns", default=3, type=int, help="Minimum speaker turns before re-establishing full wide")
@click.option("--motion-check/--no-motion-check", default=False, help="Avoid switching to moving cameras in 4-cam mode")
@click.option("--motion-hwaccel", default="hybrid", type=click.Choice(["cpu", "hybrid"]), help="Motion analysis acceleration policy: hybrid = split clips between CPU and GPU decode")
@click.option("--xml", "xml_file", type=click.Path(exists=True), help="Premiere FCP7 XML export for more reliable camera source resolution")
@click.option("--mute-audio/--no-mute-audio", default=True, help="Mute inactive speaker mics")
@click.option("--log/--no-log", default=True, help="Write JSONL log file next to output")
@click.option("--fps", default=0.0, type=float, help="Sequence frame rate for frame-aligned cuts (0 = auto-detect)")
@_cross_cancel_options()
def auto_switch_4cams_cmd(
    in_file, seq, out_file,
    mic_host, mic_guest_1, mic_guest_2, mic_guest_3,
    label_host, label_guest_1, label_guest_2, label_guest_3,
    camera_all_wide, camera_guests_wide, camera_guest_close, camera_host_close,
    audio_track_host, audio_track_guest_1, audio_track_guest_2, audio_track_guest_3,
    speech_threshold, release_threshold,
    input_gain, input_gain_host, input_gain_guest_1, input_gain_guest_2, input_gain_guest_3,
    detector_backend, vad_threshold, vad_min_speech_ms, vad_min_silence_ms, vad_speech_pad_ms,
    min_speech_duration_ms, release_ms, hold_ms, dominance_delta_db,
    shot_hold, cooldown, silence_timeout, cam3_wait_timeout, cam3_cut_in_grace,
    attenuation_level, standby_level, hard_mute_timeout,
    audio_pre_roll, audio_post_roll,
    reestablish_wide_interval, reestablish_wide_duration, reestablish_min_turns,
    motion_check, motion_hwaccel, xml_file,
    mute_audio, log, fps,
    enable_cross_cancel, cross_cancel_fir_taps,
):
    """Full pipeline for 1 host + 3 guests + 4 cameras."""
    from .prproj_patcher import patch_prproj, read_audio_offsets, segments_to_cuts
    from .core.audio_loader import apply_offset

    for label, value in [
        ("camera-all-wide", camera_all_wide),
        ("camera-guests-wide", camera_guests_wide),
        ("camera-guest-close", camera_guest_close),
        ("camera-host-close", camera_host_close),
        ("audio-track-host", audio_track_host),
        ("audio-track-guest-1", audio_track_guest_1),
        ("audio-track-guest-2", audio_track_guest_2),
        ("audio-track-guest-3", audio_track_guest_3),
    ]:
        if value < 1:
            raise click.BadParameter("Values are 1-based and must be >= 1", param_hint=f"--{label}")

    camera_all_wide -= 1
    camera_guests_wide -= 1
    camera_guest_close -= 1
    camera_host_close -= 1

    _resolved_mics, _audio_sources_log, _mute_ok = _resolve_speaker_mics(
        in_file, seq,
        [
            (audio_track_host - 1, label_host, mic_host),
            (audio_track_guest_1 - 1, label_guest_1, mic_guest_1),
            (audio_track_guest_2 - 1, label_guest_2, mic_guest_2),
            (audio_track_guest_3 - 1, label_guest_3, mic_guest_3),
        ],
        xml_file,
    )
    mic_host, mic_guest_1, mic_guest_2, mic_guest_3 = _resolved_mics
    mute_audio = mute_audio and _mute_ok

    participant_specs = [
        ParticipantSpec(key="host", label=label_host, role="host", audio_track_index=audio_track_host - 1),
        ParticipantSpec(key="guest_1", label=label_guest_1, role="guest", audio_track_index=audio_track_guest_1 - 1),
        ParticipantSpec(key="guest_2", label=label_guest_2, role="guest", audio_track_index=audio_track_guest_2 - 1),
        ParticipantSpec(key="guest_3", label=label_guest_3, role="guest", audio_track_index=audio_track_guest_3 - 1),
    ]

    analysis_config = ProjectConfig(
        audio_inputs=[
            AudioInput(path=Path(mic_host), speaker_label=label_host, camera_index=camera_host_close),
            AudioInput(path=Path(mic_guest_1), speaker_label=label_guest_1, camera_index=camera_guest_close),
            AudioInput(path=Path(mic_guest_2), speaker_label=label_guest_2, camera_index=camera_guest_close),
            AudioInput(path=Path(mic_guest_3), speaker_label=label_guest_3, camera_index=camera_guest_close),
        ],
        output_dir=Path(out_file).parent,
        default_camera=camera_all_wide,
        both_speaking_camera=camera_all_wide,
        speech_threshold_db=speech_threshold,
        release_threshold_db=release_threshold,
        detector_backend=detector_backend,
        vad_threshold=vad_threshold,
        vad_min_speech_ms=vad_min_speech_ms,
        vad_min_silence_ms=vad_min_silence_ms,
        vad_speech_pad_ms=vad_speech_pad_ms,
        input_gain_db=input_gain,
        detection_min_on_s=min_speech_duration_ms / 1000.0,
        detection_min_off_s=release_ms / 1000.0,
        detection_hangover_s=hold_ms / 1000.0,
        hangover_ms=hold_ms,
        audio_pre_roll_s=audio_pre_roll,
        audio_post_roll_s=audio_post_roll,
    )
    analysis_config.validate()

    roundtable_config = Roundtable4CamConfig(
        camera_all_wide=camera_all_wide,
        camera_guests_wide=camera_guests_wide,
        camera_guest_close=camera_guest_close,
        camera_host_close=camera_host_close,
        dominance_delta_db=dominance_delta_db,
        shot_hold_time_s=shot_hold,
        cooldown_s=cooldown,
        silence_timeout_s=silence_timeout,
        cam3_wait_timeout_s=cam3_wait_timeout,
        cam3_cut_in_grace_s=cam3_cut_in_grace,
        audio_hard_mute_timeout_s=hard_mute_timeout,
        audio_pre_roll_s=audio_pre_roll,
        audio_post_roll_s=audio_post_roll,
        audio_standby_db=standby_level,
        audio_inactive_db=attenuation_level,
        reestablish_all_wide_interval_s=reestablish_wide_interval,
        reestablish_all_wide_duration_s=reestablish_wide_duration,
        reestablish_min_turns=reestablish_min_turns,
        audio_recent_hold_s=hold_ms / 1000.0,
    )
    motion_config = CameraMotionConfig()
    motion_execution_config = CameraMotionExecutionConfig(
        speed="balanced",
        hwaccel=motion_hwaccel,
    )

    mic_paths = [Path(mic_host), Path(mic_guest_1), Path(mic_guest_2), Path(mic_guest_3)]
    gains = {
        "host": input_gain_host or input_gain,
        "guest_1": input_gain_guest_1 or input_gain,
        "guest_2": input_gain_guest_2 or input_gain,
        "guest_3": input_gain_guest_3 or input_gain,
    }

    try:
        offsets = read_audio_offsets(Path(in_file), seq)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:
        click.echo(
            f"Warning: could not read audio offsets for sequence '{seq}': {exc}. "
            "Continuing with zero offsets."
        )
        offsets = {}

    click.echo(f"Loading audio: {', '.join(str(path) for path in mic_paths)}")
    audio_arrays = load_and_align(mic_paths, analysis_config.sample_rate)
    for idx, participant in enumerate(participant_specs):
        offset_s = offsets.get(participant.audio_track_index, 0.0)
        if offset_s > 0:
            audio_arrays[idx] = apply_offset(audio_arrays[idx], offset_s, analysis_config.sample_rate)

    max_len = max(len(arr) for arr in audio_arrays)
    for idx, arr in enumerate(audio_arrays):
        if len(arr) < max_len:
            padded = np.zeros(max_len, dtype=np.float64)
            padded[:len(arr)] = arr
            audio_arrays[idx] = padded

    duration_s = max_len / analysis_config.sample_rate
    click.echo(f"Duration: {duration_s:.1f}s")

    if enable_cross_cancel:
        click.echo(
            f"Cross-channel cancellation: fir_taps={cross_cancel_fir_taps}, "
            f"channels={len(audio_arrays)}"
        )
        audio_arrays = list(
            _maybe_apply_cross_cancel(
                audio_arrays,
                analysis_config.sample_rate,
                enabled=True,
                fir_taps=cross_cancel_fir_taps,
            )
        )

    activities = {}
    detector_diags = {}
    click.echo("Analyzing participant activity...")
    for idx, participant in enumerate(participant_specs):
        activity = analyze_speaker(
            audio_arrays[idx],
            participant.label,
            analysis_config,
            gain_db=gains[participant.key],
        )
        diag = {}
        detect_activity(activity, analysis_config, diagnostics=diag, audio=audio_arrays[idx])
        activities[participant.key] = activity
        detector_diags[participant.key] = diag
        click.echo(
            f"  {participant.label}: {diag.get('active_seconds', 0.0)}s active / "
            f"{diag.get('total_seconds', 0.0)}s total, backend={diag.get('detector_backend')}"
        )
        if diag.get("detector_warning"):
            click.echo(f"  {participant.label}: {diag['detector_warning']}")

    hop_s = analysis_config.hop_ms / 1000.0
    motion_plans = None
    resolved_camera_sources = None
    motion_runtime = None
    motion_analysis_s = 0.0
    if motion_check:
        requested_angles = sorted({
            camera_all_wide,
            camera_guests_wide,
            camera_guest_close,
            camera_host_close,
        })
        try:
            resolved_camera_sources = resolve_sequence_camera_sources(
                Path(in_file),
                seq,
                requested_angles,
                xml_path=xml_file,
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

        click.echo(f"Motion sources: {resolved_camera_sources.method}")
        for angle in requested_angles:
            source = resolved_camera_sources.sources_by_angle[angle]
            click.echo(
                f"  angle {angle + 1}: {source.representative_path} "
                f"({len(source.clips)} clip{'s' if len(source.clips) != 1 else ''})"
            )

        analyzer = CameraMotionAnalyzer(motion_config, motion_execution_config)
        clip_groups = {
            angle: list(source.clips)
            for angle, source in sorted(resolved_camera_sources.sources_by_angle.items())
        }
        total_motion_clips = sum(len(clips) for clips in clip_groups.values())
        all_motion_clips = [
            clip
            for clips in clip_groups.values()
            for clip in clips
        ]
        motion_runtime = analyzer.prepare_runtime(all_motion_clips)
        click.echo(
            "  motion execution: "
            f"workers={motion_runtime.worker_count}, "
            f"cpu_workers={motion_runtime.cpu_worker_count}, "
            f"gpu_workers={motion_runtime.gpu_worker_count}, "
            f"hwaccel={motion_runtime.active_hwaccel}"
        )
        click.echo("Analyzing camera motion...")
        motion_started_at = time.perf_counter()
        with click.progressbar(
            length=total_motion_clips,
            label="  motion clips",
            show_eta=True,
            show_percent=True,
        ) as motion_bar:
            progress_lock = Lock()

            def _on_motion_clip_done(_clip):
                with progress_lock:
                    motion_bar.update(1)

            motion_plans = analyzer.analyze_camera_groups(
                clip_groups,
                total_duration_s=duration_s,
                progress_callback=_on_motion_clip_done,
                runtime=motion_runtime,
            )
        motion_analysis_s = time.perf_counter() - motion_started_at
        click.echo(
            "  motion intervals: "
            + ", ".join(
                f"angle {angle + 1}={len(plan_motion.moving_intervals)}"
                for angle, plan_motion in sorted(motion_plans.items())
            )
        )
    else:
        click.echo("  motion check: disabled")

    plan = build_roundtable_plan(
        participant_specs,
        activities,
        hop_s,
        roundtable_config,
        motion_plans=motion_plans,
    )

    click.echo(
        f"Planning complete: {len(plan.speech_segments)} speech segments, "
        f"{len(plan.camera_segments)} camera segments"
    )

    first_angle, cuts = segments_to_cuts(plan.camera_segments)
    click.echo(f"Camera switches: {len(cuts)} cuts (first angle: {first_angle})")

    out = Path(out_file)
    log_path = out.with_suffix(out.suffix + ".log.jsonl") if log else None

    log_entries = [
        {
            "event": "build_info",
            "package_version": __version__,
            "segmenter_signature": "roundtable4cams_v2_conversation_audio_stabilized",
            "planner_signature": "roundtable4cams_v2_conversation_audio_stabilized",
        },
        {
            "event": "camera_mapping",
            "camera_all_wide_internal": camera_all_wide,
            "camera_guests_wide_internal": camera_guests_wide,
            "camera_guest_close_internal": camera_guest_close,
            "camera_host_close_internal": camera_host_close,
            "camera_all_wide_premiere": camera_all_wide + 1,
            "camera_guests_wide_premiere": camera_guests_wide + 1,
            "camera_guest_close_premiere": camera_guest_close + 1,
            "camera_host_close_premiere": camera_host_close + 1,
            "participants": [
                {
                    "key": part.key,
                    "label": part.label,
                    "role": part.role,
                    "audio_track_index_internal": part.audio_track_index,
                    "audio_track_index_premiere": part.audio_track_index + 1,
                }
                for part in participant_specs
            ],
        },
        {
            "event": "roundtable_config",
            "speech_threshold_db": speech_threshold,
            "release_threshold_db": release_threshold,
            "dominance_delta_db": roundtable_config.dominance_delta_db,
            "shot_hold_time_s": roundtable_config.shot_hold_time_s,
            "cooldown_s": roundtable_config.cooldown_s,
            "host_return_min_s": roundtable_config.host_return_min_s,
            "silence_timeout_s": roundtable_config.silence_timeout_s,
            "cam3_wait_timeout_s": roundtable_config.cam3_wait_timeout_s,
            "cam3_cut_in_grace_s": roundtable_config.cam3_cut_in_grace_s,
            "guest_close_min_domination_s": roundtable_config.guest_close_min_domination_s,
            "guest_close_min_duration_s": roundtable_config.guest_close_min_duration_s,
            "guest_close_max_continuous_s": roundtable_config.guest_close_max_continuous_s,
            "guest_close_cooldown_s": roundtable_config.guest_close_cooldown_s,
            "guest_close_overlap_clear_s": roundtable_config.guest_close_overlap_clear_s,
            "guest_close_side_settle_s": roundtable_config.guest_close_side_settle_s,
            "guest_cluster_bridge_host_s": roundtable_config.guest_cluster_bridge_host_s,
            "guest_cluster_window_s": roundtable_config.guest_cluster_window_s,
            "host_screen_share_cap_window_s": roundtable_config.host_screen_share_cap_window_s,
            "host_screen_share_cap_ratio": roundtable_config.host_screen_share_cap_ratio,
            "host_screen_share_cap_min_guest_s": roundtable_config.host_screen_share_cap_min_guest_s,
            "all_overlap_trigger_s": roundtable_config.all_overlap_trigger_s,
            "all_overlap_sticky_s": roundtable_config.all_overlap_sticky_s,
            "all_overlap_window_s": roundtable_config.all_overlap_window_s,
            "all_overlap_min_islands": roundtable_config.all_overlap_min_islands,
            "all_overlap_min_turns": roundtable_config.all_overlap_min_turns,
            "audio_pre_roll_s": roundtable_config.audio_pre_roll_s,
            "audio_post_roll_s": roundtable_config.audio_post_roll_s,
            "audio_recent_hold_s": roundtable_config.audio_recent_hold_s,
            "audio_hard_mute_timeout_s": roundtable_config.audio_hard_mute_timeout_s,
            "audio_min_on_s": roundtable_config.audio_min_on_s,
            "audio_merge_gap_s": roundtable_config.audio_merge_gap_s,
            "audio_min_open_after_trigger_s": roundtable_config.audio_min_open_after_trigger_s,
            "audio_release_hold_s": roundtable_config.audio_release_hold_s,
            "audio_min_closed_s": roundtable_config.audio_min_closed_s,
            "audio_standby_db": roundtable_config.audio_standby_db,
            "audio_inactive_db": roundtable_config.audio_inactive_db,
            "reestablish_all_wide_interval_s": roundtable_config.reestablish_all_wide_interval_s,
            "reestablish_all_wide_duration_s": roundtable_config.reestablish_all_wide_duration_s,
            "reestablish_min_turns": roundtable_config.reestablish_min_turns,
            "motion_entry_lookahead_s": plan.diagnostics.get("motion_entry_lookahead_s"),
        },
        {
            "event": "motion_config",
            "enabled": motion_check,
            "motion_speed": motion_execution_config.speed,
            "motion_hwaccel_requested": motion_execution_config.hwaccel,
            "motion_hwaccel_active": None if motion_runtime is None else motion_runtime.active_hwaccel,
            "motion_worker_count": 0 if motion_runtime is None else motion_runtime.worker_count,
            "motion_cpu_worker_count": 0 if motion_runtime is None else motion_runtime.cpu_worker_count,
            "motion_gpu_worker_count": 0 if motion_runtime is None else motion_runtime.gpu_worker_count,
            "motion_cpu_clip_count": 0 if motion_runtime is None else motion_runtime.cpu_clip_count,
            "motion_gpu_clip_count": 0 if motion_runtime is None else motion_runtime.gpu_clip_count,
            "sample_fps": motion_config.sample_fps,
            "resize_width": motion_config.resize_width,
            "border_ratio": motion_config.border_ratio,
            "ring_ratio": motion_config.ring_ratio,
            "bottom_zone_disabled": True,
            "bottom_ignore_ratio": motion_config.bottom_ignore_ratio,
            "refine_sample_fps": motion_config.refine_sample_fps,
            "refine_resize_width": motion_config.refine_resize_width,
            "candidate_score_threshold": motion_config.candidate_score_threshold,
            "candidate_padding_s": motion_config.candidate_padding_s,
            "candidate_merge_gap_s": motion_config.candidate_merge_gap_s,
            "smoothing_window_s": motion_config.smoothing_window_s,
            "motion_pre_roll_s": motion_config.motion_pre_roll_s,
            "moving_on_threshold": motion_config.moving_on_threshold,
            "moving_off_threshold": motion_config.moving_off_threshold,
            "moving_confirm_s": motion_config.moving_confirm_s,
            "stable_confirm_s": motion_config.stable_confirm_s,
            "motion_analysis_s": round(motion_analysis_s, 3),
        },
        {
            "event": "detector_diagnostics",
            "participants": {
                part.key: {
                    "label": part.label,
                    "backend": detector_diags[part.key].get("detector_backend"),
                    "warning": detector_diags[part.key].get("detector_warning"),
                    "active_seconds": detector_diags[part.key].get("active_seconds"),
                    "total_seconds": detector_diags[part.key].get("total_seconds"),
                    "safety_forced_frames": detector_diags[part.key].get("safety_forced_frames"),
                    "mask_filter_removed": detector_diags[part.key].get("mask_filter_removed"),
                    "mask_filter_gap_filled": detector_diags[part.key].get("mask_filter_gap_filled"),
                }
                for part in participant_specs
            },
        },
        {
            "event": "roundtable_speech_segments",
            "count": len(plan.speech_segments),
            "segments": [_serialize_roundtable_speech_segment(seg) for seg in plan.speech_segments],
        },
        {
            "event": "roundtable_conversation_phases",
            "count": len(plan.conversation_phases),
            "phases": [_serialize_roundtable_conversation_phase(phase) for phase in plan.conversation_phases],
        },
        {
            "event": "roundtable_camera_segments",
            "count": len(plan.camera_segments),
            "segments": [_serialize_roundtable_camera_segment(seg) for seg in plan.camera_segments],
        },
        {
            "event": "roundtable_motion_events",
            "count": len(plan.motion_events),
            "events": [_serialize_roundtable_motion_event(event) for event in plan.motion_events],
        },
        {
            "event": "roundtable_camera_balance_stats",
            **plan.camera_balance_stats,
        },
        {
            "event": "audio_track_intervals_raw",
            "tracks": _serialize_interval_map(plan.audio_open_intervals_raw_s),
        },
        {
            "event": "roundtable_audio_levels",
            "count": len(plan.audio_level_segments),
            "segments": [_serialize_audio_level_segment(seg) for seg in plan.audio_level_segments],
        },
        {
            "event": "camera_cuts",
            "first_angle": first_angle,
            "count": len(cuts),
            "cuts": [
                {"time": round(cut["time"], 3), "angle": cut["angle"]}
                for cut in cuts
            ],
        },
    ]
    if resolved_camera_sources is not None:
        log_entries.append(
            {
                "event": "motion_sources",
                "method": resolved_camera_sources.method,
                "angles": {
                    str(angle): {
                        "path": str(source.representative_path),
                        "clips": [
                            {
                                "path": str(clip.path),
                                "timeline_start": round(clip.timeline_start_s, 3),
                                "timeline_end": None if clip.timeline_end_s is None else round(clip.timeline_end_s, 3),
                                "source_start": round(clip.source_start_s, 3),
                                "source_end": None if clip.source_end_s is None else round(clip.source_end_s, 3),
                            }
                            for clip in source.clips
                        ],
                    }
                    for angle, source in sorted(resolved_camera_sources.sources_by_angle.items())
                },
                "discovered_paths": [str(path) for path in resolved_camera_sources.discovered_paths],
            }
        )
    if motion_plans is not None:
        for angle, plan_motion in sorted(motion_plans.items()):
            log_entries.append(
                {
                    "event": "camera_motion_intervals",
                    "angle": angle,
                    "source_path": None if plan_motion.source_path is None else str(plan_motion.source_path),
                    "count": len(plan_motion.moving_intervals),
                    "intervals": [
                        _serialize_motion_interval(interval)
                        for interval in plan_motion.moving_intervals
                    ],
                    "diagnostics": plan_motion.diagnostics,
                }
            )

    patch_prproj(
        Path(in_file),
        cuts,
        seq,
        out,
        first_angle=first_angle,
        audio_track_intervals_s=plan.audio_open_intervals_s if mute_audio else None,
        log_path=log_path,
        source_offset_s=0.0,
        audio_source_offset_s=0.0,
        audio_overlap_s=0.0,
        audio_pre_roll_s=roundtable_config.audio_pre_roll_s,
        audio_post_roll_s=roundtable_config.audio_post_roll_s,
        prelude_log_entries=log_entries if log else None,
        fps=fps,
    )
    click.echo(f"Done: {out_file}")
    if log_path and log_path.exists():
        click.echo(f"Log: {log_path}")


def _parse_person_spec(spec_str: str) -> tuple[str, int, int]:
    """Parse a --person 'label:audio_track:camera_angle' value (1-based ints)."""
    parts = spec_str.split(":")
    if len(parts) != 3:
        raise click.BadParameter(
            f"Expected 'label:audio_track:camera_angle', got '{spec_str}'",
            param_hint="--person",
        )
    label = parts[0].strip()
    if not label:
        raise click.BadParameter("Person label must not be empty", param_hint="--person")
    try:
        audio_track = int(parts[1])
        camera_angle = int(parts[2])
    except ValueError:
        raise click.BadParameter(
            f"audio_track and camera_angle must be integers, got '{spec_str}'",
            param_hint="--person",
        )
    if audio_track < 1 or camera_angle < 1:
        raise click.BadParameter(
            "audio_track and camera_angle are 1-based and must be >= 1",
            param_hint="--person",
        )
    return label, audio_track, camera_angle


@cli.command("auto-switch-custom")
@click.option("--in", "in_file", required=True, type=click.Path(exists=True), help="Input .prproj file")
@click.option("--seq", required=True, help="Sequence name in .prproj")
@click.option("--out", "out_file", required=True, type=click.Path(), help="Output .prproj file")
@click.option(
    "--person", "person_specs", multiple=True, required=True,
    help="Repeatable. 'label:audio_track:camera_angle' (1-based audio track & camera angle). "
         "People who share a camera_angle share that shot (e.g. two guests on one medium). "
         "Do not use ':' inside the label.",
)
@click.option("--wide-camera", required=True, type=int, help="Premiere angle of the wide (obshchak) shot (1-based)")
@click.option(
    "--mic", "mic_files", multiple=True, type=click.Path(exists=True),
    help="Optional explicit mic file per person, in --person order. Omit to auto-resolve from the project.",
)
@click.option("--xml", "xml_file", type=click.Path(exists=True), help="Premiere FCP7 XML for reliable audio source resolution")
@click.option("--speech-threshold", default=-27.0, type=float, help="Speech onset threshold in dBFS")
@click.option("--release-threshold", default=-31.0, type=float, help="Speech release threshold in dBFS")
@click.option("--input-gain", default=0.0, type=float, help="Input gain in dB for all mics")
@click.option("--detector-backend", default="auto", type=click.Choice(["auto", "rms", "silero"]), help="Speech detector backend")
@click.option("--vad-threshold", default=0.65, type=float, help="Silero VAD speech probability threshold")
@click.option("--vad-min-speech-ms", default=120.0, type=float, help="Silero VAD minimum speech duration in ms")
@click.option("--vad-min-silence-ms", default=80.0, type=float, help="Silero VAD minimum silence duration in ms")
@click.option("--vad-speech-pad-ms", default=30.0, type=float, help="Silero VAD padding around detected speech in ms")
@click.option("--min-speech-duration-ms", default=120.0, type=float, help="Minimum stable speech duration in ms")
@click.option("--release-ms", default=220.0, type=float, help="Speech release debounce in ms")
@click.option("--hold-ms", default=350.0, type=float, help="Hold time after speech in ms")
@click.option("--dominance-delta-db", default=6.0, type=float, help="dB delta to treat a quieter channel as bleed")
@click.option("--shot-hold", default=1.4, type=float, help="Minimum steady shot duration in seconds")
@click.option("--reestablish-interval", default=0.0, type=float, help="Cut to wide after this many seconds on one close shot (0 = off)")
@click.option("--reestablish-hold", default=1.5, type=float, help="Duration of each re-establishing wide shot")
@click.option("--audio-pre-roll", default=0.24, type=float, help="Open audio this many seconds before detected onset")
@click.option("--audio-post-roll", default=0.12, type=float, help="Keep audio open this many seconds after detected end")
@click.option("--mute-audio/--no-mute-audio", default=True, help="Mute inactive speaker mics")
@click.option("--log/--no-log", default=True, help="Write JSONL log file next to output")
@click.option("--fps", default=0.0, type=float, help="Sequence frame rate for frame-aligned cuts (0 = auto-detect)")
@_cross_cancel_options()
def auto_switch_custom_cmd(
    in_file, seq, out_file,
    person_specs, wide_camera, mic_files, xml_file,
    speech_threshold, release_threshold, input_gain,
    detector_backend, vad_threshold, vad_min_speech_ms, vad_min_silence_ms, vad_speech_pad_ms,
    min_speech_duration_ms, release_ms, hold_ms, dominance_delta_db,
    shot_hold, reestablish_interval, reestablish_hold,
    audio_pre_roll, audio_post_roll,
    mute_audio, log, fps,
    enable_cross_cancel, cross_cancel_fir_taps,
):
    """Configurable pipeline: any number of people and cameras, custom person->camera mapping.

    The camera rule: a person speaking alone -> their camera; people who share a
    camera both speaking -> that shared shot; speakers on different cameras
    overlapping (or silence) -> the wide (obshchak) shot.
    """
    from .prproj_patcher import patch_prproj, read_audio_offsets, segments_to_cuts
    from .core.audio_loader import apply_offset

    if wide_camera < 1:
        raise click.BadParameter("Values are 1-based and must be >= 1", param_hint="--wide-camera")

    parsed = [_parse_person_spec(s) for s in person_specs]

    if mic_files and len(mic_files) != len(parsed):
        raise click.BadParameter(
            f"--mic given {len(mic_files)} time(s) but there are {len(parsed)} --person entries; "
            "provide one --mic per person (in order) or none at all.",
            param_hint="--mic",
        )
    mic_by_index = list(mic_files) if mic_files else [None] * len(parsed)

    tracks = [track for (_lbl, track, _ang) in parsed]
    if len(set(tracks)) != len(tracks):
        raise click.BadParameter(
            f"Audio tracks must be unique across people, got {tracks}",
            param_hint="--person",
        )

    wide0 = wide_camera - 1

    specs_for_resolve = [
        (track - 1, label, mic_by_index[i])
        for i, (label, track, _ang) in enumerate(parsed)
    ]
    resolved_mics, audio_sources_log, mute_ok = _resolve_speaker_mics(
        in_file, seq, specs_for_resolve, xml_file
    )
    mute_audio = mute_audio and mute_ok

    people = [
        CustomPerson(
            key=f"person_{i}",
            label=label,
            audio_track_index=track - 1,
            camera_angle=angle - 1,
        )
        for i, (label, track, angle) in enumerate(parsed)
    ]

    analysis_config = ProjectConfig(
        audio_inputs=[
            AudioInput(
                path=Path(resolved_mics[i]),
                speaker_label=people[i].label,
                camera_index=people[i].camera_angle,
            )
            for i in range(len(people))
        ],
        output_dir=Path(out_file).parent,
        default_camera=wide0,
        both_speaking_camera=wide0,
        speech_threshold_db=speech_threshold,
        release_threshold_db=release_threshold,
        detector_backend=detector_backend,
        vad_threshold=vad_threshold,
        vad_min_speech_ms=vad_min_speech_ms,
        vad_min_silence_ms=vad_min_silence_ms,
        vad_speech_pad_ms=vad_speech_pad_ms,
        input_gain_db=input_gain,
        detection_min_on_s=min_speech_duration_ms / 1000.0,
        detection_min_off_s=release_ms / 1000.0,
        detection_hangover_s=hold_ms / 1000.0,
        hangover_ms=hold_ms,
        audio_pre_roll_s=audio_pre_roll,
        audio_post_roll_s=audio_post_roll,
    )
    analysis_config.validate()

    custom_config = CustomSwitchConfig(
        wide_camera=wide0,
        dominance_delta_db=dominance_delta_db,
        shot_hold_s=shot_hold,
        reestablish_interval_s=reestablish_interval,
        reestablish_hold_s=reestablish_hold,
        audio_pre_roll_s=audio_pre_roll,
        audio_post_roll_s=audio_post_roll,
        audio_recent_hold_s=hold_ms / 1000.0,
    )

    try:
        offsets = read_audio_offsets(Path(in_file), seq)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - offsets are best-effort
        click.echo(
            f"Warning: could not read audio offsets for sequence '{seq}': {exc}. "
            "Continuing with zero offsets."
        )
        offsets = {}

    mic_paths = [Path(resolved_mics[i]) for i in range(len(people))]
    click.echo(f"Loading audio: {', '.join(str(p) for p in mic_paths)}")
    audio_arrays = load_and_align(mic_paths, analysis_config.sample_rate)
    for idx, person in enumerate(people):
        offset_s = offsets.get(person.audio_track_index, 0.0)
        if offset_s > 0:
            audio_arrays[idx] = apply_offset(audio_arrays[idx], offset_s, analysis_config.sample_rate)

    max_len = max(len(arr) for arr in audio_arrays)
    for idx, arr in enumerate(audio_arrays):
        if len(arr) < max_len:
            padded = np.zeros(max_len, dtype=np.float64)
            padded[:len(arr)] = arr
            audio_arrays[idx] = padded
    duration_s = max_len / analysis_config.sample_rate
    click.echo(f"Duration: {duration_s:.1f}s")

    if enable_cross_cancel:
        click.echo(
            f"Cross-channel cancellation: fir_taps={cross_cancel_fir_taps}, "
            f"channels={len(audio_arrays)}"
        )
        audio_arrays = list(
            _maybe_apply_cross_cancel(
                audio_arrays, analysis_config.sample_rate,
                enabled=True, fir_taps=cross_cancel_fir_taps,
            )
        )

    activities = {}
    click.echo("Analyzing participant activity...")
    for idx, person in enumerate(people):
        activity = analyze_speaker(audio_arrays[idx], person.label, analysis_config, gain_db=input_gain)
        diag = {}
        detect_activity(activity, analysis_config, diagnostics=diag, audio=audio_arrays[idx])
        activities[person.key] = activity
        click.echo(
            f"  {person.label}: {diag.get('active_seconds', 0.0)}s active / "
            f"{diag.get('total_seconds', 0.0)}s total, backend={diag.get('detector_backend')}"
        )

    hop_s = analysis_config.hop_ms / 1000.0
    plan = build_custom_plan(people, activities, hop_s, custom_config)
    click.echo(f"Planning complete: {len(plan.camera_segments)} camera segments")

    first_angle, cuts = segments_to_cuts(plan.camera_segments)
    click.echo(f"Camera switches: {len(cuts)} cuts (first angle: {first_angle + 1})")

    out = Path(out_file)
    log_path = out.with_suffix(out.suffix + ".log.jsonl") if log else None
    log_entries = [
        {
            "event": "build_info",
            "package_version": __version__,
            "planner_signature": "custom_v1_static_mapping",
        },
        {
            "event": "custom_mapping",
            "wide_camera_internal": wide0,
            "people": plan.diagnostics.get("people", []),
        },
    ]
    if audio_sources_log:
        log_entries.append(audio_sources_log)

    click.echo("Patching project...")
    patch_prproj(
        Path(in_file),
        cuts,
        seq,
        out,
        first_angle=first_angle,
        audio_track_intervals_s=plan.audio_open_intervals_s if mute_audio else None,
        log_path=log_path,
        source_offset_s=0.0,
        audio_source_offset_s=0.0,
        audio_overlap_s=0.0,
        audio_pre_roll_s=custom_config.audio_pre_roll_s,
        audio_post_roll_s=custom_config.audio_post_roll_s,
        prelude_log_entries=log_entries if log else None,
        fps=fps,
    )
    click.echo(f"Done: {out_file}")
    if log_path and log_path.exists():
        click.echo(f"Log: {log_path}")


@cli.command("auto-switch-sakha-aimakh")
@click.option("--in", "in_file", required=True, type=click.Path(exists=True), help="Input .prproj file")
@click.option("--seq", required=True, help="Sequence name in .prproj")
@click.option("--out", "out_file", required=True, type=click.Path(), help="Output .prproj file")
@click.option("--mic-main-host", type=click.Path(exists=True), help="Main host microphone audio file (auto-resolved from project when omitted)")
@click.option("--mic-cohost", type=click.Path(exists=True), help="Second host microphone audio file (auto-resolved from project when omitted)")
@click.option("--mic-guest", type=click.Path(exists=True), help="Guest microphone audio file (auto-resolved from project when omitted)")
@click.option("--label-main-host", default="main_host", help="Label for main host track")
@click.option("--label-cohost", default="cohost", help="Label for second host track")
@click.option("--label-guest", default="guest", help="Label for guest track")
@click.option("--camera-main-host", default=1, show_default=True, type=int, help="Premiere angle for main host close-up (1-based)")
@click.option("--camera-guest-close", default=2, show_default=True, type=int, help="Premiere angle for guest close-up (1-based)")
@click.option("--camera-pair-wide", default=3, show_default=True, type=int, help="Premiere angle for second host + guest wide (1-based)")
@click.option("--camera-all-wide", default=4, show_default=True, type=int, help="Premiere angle for full wide (1-based)")
@click.option("--audio-track-main-host", default=1, type=int, help="Audio track number for main host mic (1-based)")
@click.option("--audio-track-cohost", default=2, type=int, help="Audio track number for second host mic (1-based)")
@click.option("--audio-track-guest", default=3, type=int, help="Audio track number for guest mic (1-based)")
@click.option("--reaction-sensitivity", default=50.0, type=float, help="Reaction sensitivity 0..100% (lower = stickier/suppresses leak reactions)")
@click.option("--temperature", default=None, type=float, help="Legacy alias / override for --reaction-sensitivity")
@click.option("--cut-intensity", default=50.0, type=float, help="Cut intensity 0..100%")
@click.option("--max-solo-hold", default=60.0, type=float, help="Maximum seconds to hold any solo speaker shot")
@click.option("--dominance-delta-db", default=None, type=float, help="Override dB delta to treat quieter channel as bleed")
@click.option("--shot-hold", default=None, type=float, help="Override minimum steady shot duration in seconds")
@click.option("--silence-timeout", default=None, type=float, help="Override silence timeout before full wide")
@click.option("--cutaway-duration", default=None, type=float, help="Override cutaway duration in seconds")
@click.option("--reestablish-wide-interval", default=None, type=float, help="Override seconds between full wide re-establishing shots")
@click.option("--reestablish-wide-duration", default=None, type=float, help="Override duration of full wide re-establishing shots")
@click.option("--reestablish-min-turns", default=None, type=int, help="Override minimum turns before re-establishing full wide")
@click.option("--speech-threshold", default=-27.0, type=float, help="Speech onset threshold in dBFS")
@click.option("--release-threshold", default=-31.0, type=float, help="Speech release threshold in dBFS")
@click.option("--input-gain", default=0.0, type=float, help="Input gain in dB for all mics")
@click.option("--input-gain-main-host", default=0.0, type=float, help="Input gain in dB for main host mic")
@click.option("--input-gain-cohost", default=0.0, type=float, help="Input gain in dB for second host mic")
@click.option("--input-gain-guest", default=0.0, type=float, help="Input gain in dB for guest mic")
@click.option("--detector-backend", default="auto", type=click.Choice(["auto", "rms", "silero"]), help="Speech detector backend")
@click.option("--vad-threshold", default=0.65, type=float, help="Silero VAD speech probability threshold")
@click.option("--vad-min-speech-ms", default=120.0, type=float, help="Silero VAD minimum speech duration in ms")
@click.option("--vad-min-silence-ms", default=80.0, type=float, help="Silero VAD minimum silence duration in ms")
@click.option("--vad-speech-pad-ms", default=30.0, type=float, help="Silero VAD padding around detected speech in ms")
@click.option("--min-speech-duration-ms", default=120.0, type=float, help="Minimum stable speech duration in ms")
@click.option("--release-ms", default=220.0, type=float, help="Speech release debounce in ms")
@click.option("--hold-ms", default=350.0, type=float, help="Hold time after speech in ms")
@click.option("--audio-pre-roll", default=0.24, type=float, help="Open audio this many seconds before detected onset")
@click.option("--audio-post-roll", default=0.12, type=float, help="Keep audio open this many seconds after detected end")
@click.option("--speaker-momentum", default=2.0, type=float, help="studio: how much a long-talking speaker sticks (1.0=off, 2.0=2x harder to interrupt)")
@click.option("--priority-main-host", default=1.0, type=float, help="studio: mic priority for main host (1.0=neutral, higher=stickier/preferred)")
@click.option("--priority-cohost", default=1.0, type=float, help="studio: mic priority for cohost (1.0=neutral)")
@click.option("--priority-guest", default=1.0, type=float, help="studio: mic priority for guest (1.0=neutral)")
@click.option("--audio-clean-mode", default="studio", type=click.Choice(["studio", "calibrated", "strict", "balanced", "legacy"]), show_default=True, help="SAKHA audio leak cleanup mode")
@click.option("--motion-check/--no-motion-check", default=False, help="Avoid switching to moving cameras in SAKHA AYMAKH mode")
@click.option("--motion-hwaccel", default="hybrid", type=click.Choice(["cpu", "hybrid"]), help="Motion analysis acceleration policy")
@click.option("--motion-speed", default="balanced", type=click.Choice(["balanced", "turbo"]), help="Motion analysis speed policy")
@click.option("--motion-cache/--no-motion-cache", default=True, help="Cache motion analysis for this mode")
@click.option("--motion-cache-dir", default=None, type=click.Path(file_okay=False), help="Directory for motion cache")
@click.option("--xml", "xml_file", type=click.Path(exists=True), help="Premiere FCP7 XML export for more reliable camera source resolution")
@click.option("--mute-audio/--no-mute-audio", default=True, help="Mute inactive speaker mics")
@click.option("--log/--no-log", default=True, help="Write JSONL log file next to output")
@click.option("--fps", default=0.0, type=float, help="Sequence frame rate for frame-aligned cuts (0 = auto-detect)")
@_cross_cancel_options()
def auto_switch_sakha_aimakh_cmd(
    in_file, seq, out_file,
    mic_main_host, mic_cohost, mic_guest,
    label_main_host, label_cohost, label_guest,
    camera_main_host, camera_guest_close, camera_pair_wide, camera_all_wide,
    audio_track_main_host, audio_track_cohost, audio_track_guest,
    reaction_sensitivity, temperature, cut_intensity, max_solo_hold,
    dominance_delta_db, shot_hold, silence_timeout, cutaway_duration,
    reestablish_wide_interval, reestablish_wide_duration, reestablish_min_turns,
    speech_threshold, release_threshold,
    input_gain, input_gain_main_host, input_gain_cohost, input_gain_guest,
    detector_backend, vad_threshold, vad_min_speech_ms, vad_min_silence_ms, vad_speech_pad_ms,
    min_speech_duration_ms, release_ms, hold_ms,
    audio_pre_roll, audio_post_roll,
    speaker_momentum, priority_main_host, priority_cohost, priority_guest,
    audio_clean_mode,
    motion_check, motion_hwaccel, motion_speed, motion_cache, motion_cache_dir, xml_file,
    mute_audio, log, fps,
    enable_cross_cancel, cross_cancel_fir_taps,
):
    """Full pipeline for SAKHA AYMAKH: 2 hosts + 1 guest + 4 cameras."""
    from .core.audio_loader import apply_offset
    from .prproj_patcher import patch_prproj, read_audio_offsets, segments_to_cuts

    for label_name, value in [
        ("camera-main-host", camera_main_host),
        ("camera-guest-close", camera_guest_close),
        ("camera-pair-wide", camera_pair_wide),
        ("camera-all-wide", camera_all_wide),
        ("audio-track-main-host", audio_track_main_host),
        ("audio-track-cohost", audio_track_cohost),
        ("audio-track-guest", audio_track_guest),
    ]:
        if value < 1:
            raise click.BadParameter("Values are 1-based and must be >= 1", param_hint=f"--{label_name}")
    reaction_sensitivity_value = (
        temperature if temperature is not None else reaction_sensitivity
    )
    for label_name, value in [
        ("reaction-sensitivity", reaction_sensitivity_value),
        ("cut-intensity", cut_intensity),
    ]:
        if not 0.0 <= value <= 100.0:
            raise click.BadParameter("Value must be between 0 and 100", param_hint=f"--{label_name}")
    if max_solo_hold <= 0:
        raise click.BadParameter("max-solo-hold must be positive", param_hint="--max-solo-hold")

    camera_main_host -= 1
    camera_guest_close -= 1
    camera_pair_wide -= 1
    camera_all_wide -= 1

    participant_specs = [
        SakhaParticipantSpec("main_host", label_main_host, "main_host", audio_track_main_host - 1),
        SakhaParticipantSpec("cohost", label_cohost, "cohost", audio_track_cohost - 1),
        SakhaParticipantSpec("guest", label_guest, "guest", audio_track_guest - 1),
    ]
    if len({part.audio_track_index for part in participant_specs}) != len(participant_specs):
        raise click.BadParameter(
            "Audio tracks for main host, cohost, and guest must be different",
            param_hint="--audio-track-*",
        )

    mic_path_by_key: dict[str, Path | None] = {
        "main_host": None if mic_main_host is None else Path(mic_main_host),
        "cohost": None if mic_cohost is None else Path(mic_cohost),
        "guest": None if mic_guest is None else Path(mic_guest),
    }
    audio_sources_log: dict | None = None
    if any(path is None for path in mic_path_by_key.values()):
        try:
            resolved_audio_sources = resolve_sequence_audio_sources(
                Path(in_file),
                seq,
                [part.audio_track_index for part in participant_specs],
                xml_path=xml_file,
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

        click.echo(f"Audio sources: {resolved_audio_sources.method}")
        audio_sources_log = {
            "event": "audio_sources",
            "method": resolved_audio_sources.method,
            "tracks": {},
        }
        for part in participant_specs:
            source = resolved_audio_sources.sources_by_track[part.audio_track_index]
            if mic_path_by_key[part.key] is None:
                mic_path_by_key[part.key] = source.representative_path
            click.echo(
                f"  {part.label}: track {part.audio_track_index + 1} -> "
                f"{mic_path_by_key[part.key]}"
            )
            audio_sources_log["tracks"][str(part.audio_track_index)] = {
                "role": part.role,
                "label": part.label,
                "name": source.name,
                "path": str(source.representative_path),
                "paths": [str(path) for path in source.paths],
            }

    mic_paths = [
        mic_path_by_key["main_host"],
        mic_path_by_key["cohost"],
        mic_path_by_key["guest"],
    ]
    if any(path is None for path in mic_paths):
        raise click.ClickException(
            "Missing microphone paths. Provide --mic-* manually or let the command "
            "resolve them from audio tracks in the .prproj/XML."
        )
    resolved_mic_paths = [path for path in mic_paths if path is not None]

    unique_mic_paths = {str(p.resolve()) for p in resolved_mic_paths}
    if len(unique_mic_paths) < len(resolved_mic_paths):
        click.echo(
            "Warning: несколько дорожек указывают на один и тот же аудиофайл — "
            "раздельный анализ по спикерам невозможен. "
            "Укажите отдельные файлы через --mic-main-host / --mic-cohost / --mic-guest; "
            "заглушение дорожек отключено.",
            err=True,
        )
        mute_audio = False

    analysis_config = ProjectConfig(
        audio_inputs=[
            AudioInput(path=resolved_mic_paths[0], speaker_label=label_main_host, camera_index=camera_main_host),
            AudioInput(path=resolved_mic_paths[1], speaker_label=label_cohost, camera_index=camera_pair_wide),
            AudioInput(path=resolved_mic_paths[2], speaker_label=label_guest, camera_index=camera_guest_close),
        ],
        output_dir=Path(out_file).parent,
        default_camera=camera_all_wide,
        both_speaking_camera=camera_all_wide,
        speech_threshold_db=speech_threshold,
        release_threshold_db=release_threshold,
        detector_backend=detector_backend,
        vad_threshold=vad_threshold,
        vad_min_speech_ms=vad_min_speech_ms,
        vad_min_silence_ms=vad_min_silence_ms,
        vad_speech_pad_ms=vad_speech_pad_ms,
        input_gain_db=input_gain,
        detection_min_on_s=min_speech_duration_ms / 1000.0,
        detection_min_off_s=release_ms / 1000.0,
        detection_hangover_s=hold_ms / 1000.0,
        hangover_ms=hold_ms,
        audio_pre_roll_s=audio_pre_roll,
        audio_post_roll_s=audio_post_roll,
    )
    analysis_config.validate()

    sakha_config = build_sakha_config_from_controls(
        camera_main_host_close=camera_main_host,
        camera_guest_close=camera_guest_close,
        camera_pair_wide=camera_pair_wide,
        camera_all_wide=camera_all_wide,
        temperature=reaction_sensitivity_value,
        cut_intensity=cut_intensity,
        max_solo_hold_s=max_solo_hold,
    )
    overrides = {
        "studio_momentum": speaker_momentum,
        "studio_priority_main_host": priority_main_host,
        "studio_priority_cohost": priority_cohost,
        "studio_priority_guest": priority_guest,
        "dominance_delta_db": dominance_delta_db,
        "shot_hold_time_s": shot_hold,
        "silence_timeout_s": silence_timeout,
        "cutaway_duration_s": cutaway_duration,
        "reestablish_all_wide_interval_s": reestablish_wide_interval,
        "reestablish_all_wide_duration_s": reestablish_wide_duration,
        "reestablish_min_turns": reestablish_min_turns,
    }
    sakha_config = replace(
        sakha_config,
        audio_clean_mode=audio_clean_mode,
        audio_pre_roll_s=audio_pre_roll,
        audio_post_roll_s=audio_post_roll,
        **{name: value for name, value in overrides.items() if value is not None},
    )
    sakha_config.validate()

    motion_config = CameraMotionConfig()
    motion_execution_config = CameraMotionExecutionConfig(
        speed=motion_speed,
        hwaccel=motion_hwaccel,
    )

    gains = {
        "main_host": input_gain_main_host or input_gain,
        "cohost": input_gain_cohost or input_gain,
        "guest": input_gain_guest or input_gain,
    }

    try:
        offsets = read_audio_offsets(Path(in_file), seq)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:
        click.echo(
            f"Warning: could not read audio offsets for sequence '{seq}': {exc}. "
            "Continuing with zero offsets."
        )
        offsets = {}

    click.echo(f"Loading audio: {', '.join(str(path) for path in resolved_mic_paths)}")
    audio_arrays = load_and_align(
        resolved_mic_paths,
        analysis_config.sample_rate,
        search_dir=Path(in_file).parent,
    )
    for idx, participant in enumerate(participant_specs):
        offset_s = offsets.get(participant.audio_track_index, 0.0)
        if offset_s > 0:
            audio_arrays[idx] = apply_offset(audio_arrays[idx], offset_s, analysis_config.sample_rate)

    max_len = max(len(arr) for arr in audio_arrays)
    for idx, arr in enumerate(audio_arrays):
        if len(arr) < max_len:
            padded = np.zeros(max_len, dtype=np.float64)
            padded[:len(arr)] = arr
            audio_arrays[idx] = padded

    duration_s = max_len / analysis_config.sample_rate
    click.echo(f"Duration: {duration_s:.1f}s")

    if enable_cross_cancel:
        click.echo(
            f"Cross-channel cancellation: fir_taps={cross_cancel_fir_taps}, "
            f"channels={len(audio_arrays)}"
        )
        audio_arrays = list(
            _maybe_apply_cross_cancel(
                audio_arrays,
                analysis_config.sample_rate,
                enabled=True,
                fir_taps=cross_cancel_fir_taps,
            )
        )

    activities = {}
    detector_diags = {}
    click.echo("Analyzing participant activity...")
    for idx, participant in enumerate(participant_specs):
        activity = analyze_speaker(
            audio_arrays[idx],
            participant.label,
            analysis_config,
            gain_db=gains[participant.key],
        )
        diag = {}
        detect_activity(activity, analysis_config, diagnostics=diag, audio=audio_arrays[idx])
        activities[participant.key] = activity
        detector_diags[participant.key] = diag
        click.echo(
            f"  {participant.label}: {diag.get('active_seconds', 0.0)}s active / "
            f"{diag.get('total_seconds', 0.0)}s total, backend={diag.get('detector_backend')}"
        )
        if diag.get("detector_warning"):
            click.echo(f"  {participant.label}: {diag['detector_warning']}")

    motion_plans = None
    resolved_camera_sources = None
    motion_runtime = None
    motion_analysis_s = 0.0
    motion_cache_stats = {
        "enabled": bool(motion_cache),
        "cache_dir": None,
        "hits": 0,
        "misses": 0,
        "keys": {},
    }
    if motion_check:
        requested_angles = sorted({
            camera_main_host,
            camera_guest_close,
            camera_pair_wide,
            camera_all_wide,
        })
        try:
            resolved_camera_sources = resolve_sequence_camera_sources(
                Path(in_file),
                seq,
                requested_angles,
                xml_path=xml_file,
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

        click.echo(f"Motion sources: {resolved_camera_sources.method}")
        for angle in requested_angles:
            source = resolved_camera_sources.sources_by_angle[angle]
            click.echo(
                f"  angle {angle + 1}: {source.representative_path} "
                f"({len(source.clips)} clip{'s' if len(source.clips) != 1 else ''})"
            )

        analyzer = CameraMotionAnalyzer(motion_config, motion_execution_config)
        clip_groups = {
            angle: list(source.clips)
            for angle, source in sorted(resolved_camera_sources.sources_by_angle.items())
        }
        cache_dir = Path(motion_cache_dir) if motion_cache_dir else Path(out_file).parent / ".autopodcast_motion_cache"
        motion_cache_stats["cache_dir"] = str(cache_dir)
        motion_plans = {}
        uncached_groups = {}
        if motion_cache:
            for angle, clips in clip_groups.items():
                key = motion_cache_key(clips, motion_config)
                motion_cache_stats["keys"][str(angle)] = key
                cached = load_motion_plan_from_cache(cache_dir, key)
                if cached is None:
                    motion_cache_stats["misses"] += 1
                    uncached_groups[angle] = clips
                else:
                    motion_cache_stats["hits"] += 1
                    motion_plans[angle] = cached
        else:
            uncached_groups = dict(clip_groups)
            motion_cache_stats["misses"] = len(uncached_groups)

        if uncached_groups:
            all_motion_clips = [
                clip
                for clips in uncached_groups.values()
                for clip in clips
            ]
            motion_runtime = analyzer.prepare_runtime(all_motion_clips)
            click.echo(
                "  motion execution: "
                f"workers={motion_runtime.worker_count}, "
                f"cpu_workers={motion_runtime.cpu_worker_count}, "
                f"gpu_workers={motion_runtime.gpu_worker_count}, "
                f"hwaccel={motion_runtime.active_hwaccel}"
            )
            click.echo(
                f"  motion cache: hits={motion_cache_stats['hits']}, "
                f"misses={motion_cache_stats['misses']}"
            )
            click.echo("Analyzing camera motion...")
            motion_started_at = time.perf_counter()
            with click.progressbar(
                length=len(all_motion_clips),
                label="  motion clips",
                show_eta=True,
                show_percent=True,
            ) as motion_bar:
                progress_lock = Lock()

                def _on_motion_clip_done(_clip):
                    with progress_lock:
                        motion_bar.update(1)

                analyzed_plans = analyzer.analyze_camera_groups(
                    uncached_groups,
                    total_duration_s=duration_s,
                    progress_callback=_on_motion_clip_done,
                    runtime=motion_runtime,
                )
            motion_analysis_s = time.perf_counter() - motion_started_at
            motion_plans.update(analyzed_plans)
            if motion_cache:
                for angle, plan_motion in analyzed_plans.items():
                    key = motion_cache_stats["keys"].get(str(angle)) or motion_cache_key(clip_groups[angle], motion_config)
                    save_motion_plan_to_cache(cache_dir, key, plan_motion)
        else:
            click.echo(
                f"  motion cache: hits={motion_cache_stats['hits']}, "
                f"misses={motion_cache_stats['misses']}"
            )
            click.echo("  motion check: all camera motion plans loaded from cache")

        click.echo(
            "  motion intervals: "
            + ", ".join(
                f"angle {angle + 1}={len(plan_motion.moving_intervals)}"
                for angle, plan_motion in sorted(motion_plans.items())
            )
        )
    else:
        click.echo("  motion check: disabled")

    click.echo("Building SAKHA AYMAKH plan...")
    plan = build_sakha_aimakh_plan(
        participant_specs,
        activities,
        analysis_config.hop_ms / 1000.0,
        sakha_config,
        motion_plans=motion_plans,
        audio_arrays_by_key={
            participant_specs[idx].key: audio_arrays[idx]
            for idx in range(len(participant_specs))
        },
        audio_sample_rate=analysis_config.sample_rate,
    )

    click.echo(
        f"Planning complete: {len(plan.speech_segments)} speech segments, "
        f"{len(plan.camera_segments)} camera segments"
    )
    debleed_diag = plan.diagnostics.get("debleed", {})
    if debleed_diag.get("enabled"):
        click.echo("Debleed:")
        for part in participant_specs:
            part_diag = debleed_diag.get("participants", {}).get(part.key, {})
            click.echo(
                f"  {part.label}: active "
                f"{part_diag.get('active_seconds_before', 0.0)}s -> "
                f"{part_diag.get('active_seconds_after', 0.0)}s, "
                f"open={part_diag.get('stable_open_seconds', 0.0)}s"
            )
        click.echo(
            "  guest rescue: "
            f"rescued={debleed_diag.get('guest_rescued_seconds', 0.0)}s, "
            f"open-both={debleed_diag.get('cohost_guest_ambiguous_open_both_seconds', 0.0)}s, "
            f"suppressed-by-cohost={debleed_diag.get('guest_suppressed_by_cohost_seconds', 0.0)}s"
        )
    waveform_diag = plan.diagnostics.get("waveform_gate", {})
    if waveform_diag:
        decision_seconds = waveform_diag.get("decision_seconds", {})
        click.echo(
            "Waveform gate: "
            f"mode={waveform_diag.get('mode')}, "
            f"available={waveform_diag.get('waveform_available')}, "
            f"true-overlap={decision_seconds.get('true_overlap', 0.0)}s, "
            f"correlated-suppressed={decision_seconds.get('correlated_bleed_suppressed', 0.0)}s"
        )
    source_owner_diag = plan.diagnostics.get("source_owner", {})
    if source_owner_diag:
        owner_seconds = source_owner_diag.get("source_owner_seconds", {})
        click.echo(
            "Source owner: "
            f"mode={source_owner_diag.get('mode')}, "
            f"available={source_owner_diag.get('waveform_available')}, "
            f"main={owner_seconds.get('single_main_host', 0.0)}s, "
            f"cohost={owner_seconds.get('single_cohost', 0.0)}s, "
            f"guest={owner_seconds.get('single_guest', 0.0)}s, "
            f"overlap={owner_seconds.get('true_overlap', 0.0)}s"
        )

    first_angle, cuts = segments_to_cuts(plan.camera_segments)
    click.echo(f"Camera switches: {len(cuts)} cuts (first angle: {first_angle})")

    out = Path(out_file)
    log_path = out.with_suffix(out.suffix + ".log.jsonl") if log else None
    log_entries = [
        {
            "event": "build_info",
            "package_version": __version__,
            "planner_signature": SAKHA_AYMAKH_PLANNER_SIGNATURE,
        },
        {
            "event": "camera_mapping",
            "camera_main_host_internal": camera_main_host,
            "camera_guest_close_internal": camera_guest_close,
            "camera_pair_wide_internal": camera_pair_wide,
            "camera_all_wide_internal": camera_all_wide,
            "camera_main_host_premiere": camera_main_host + 1,
            "camera_guest_close_premiere": camera_guest_close + 1,
            "camera_pair_wide_premiere": camera_pair_wide + 1,
            "camera_all_wide_premiere": camera_all_wide + 1,
            "participants": [
                {
                    "key": part.key,
                    "label": part.label,
                    "role": part.role,
                    "audio_track_index_internal": part.audio_track_index,
                    "audio_track_index_premiere": part.audio_track_index + 1,
                }
                for part in participant_specs
            ],
        },
        {
            "event": "sakha_aimakh_config",
            "temperature": reaction_sensitivity_value,
            "reaction_sensitivity": reaction_sensitivity_value,
            "cut_intensity": cut_intensity,
            "speech_threshold_db": speech_threshold,
            "release_threshold_db": release_threshold,
            "dominance_delta_db": sakha_config.dominance_delta_db,
            "shot_hold_time_s": sakha_config.shot_hold_time_s,
            "silence_timeout_s": sakha_config.silence_timeout_s,
            "max_solo_hold_s": sakha_config.max_solo_hold_s,
            "solo_cutaway_interval_s": sakha_config.solo_cutaway_interval_s,
            "guest_cutaway_interval_s": sakha_config.guest_cutaway_interval_s,
            "cutaway_duration_s": sakha_config.cutaway_duration_s,
            "cut_search_window_s": sakha_config.cut_search_window_s,
            "forced_min_drop_db": sakha_config.forced_min_drop_db,
            "reestablish_all_wide_interval_s": sakha_config.reestablish_all_wide_interval_s,
            "reestablish_all_wide_duration_s": sakha_config.reestablish_all_wide_duration_s,
            "reestablish_min_turns": sakha_config.reestablish_min_turns,
            "audio_pre_roll_s": sakha_config.audio_pre_roll_s,
            "audio_post_roll_s": sakha_config.audio_post_roll_s,
            "audio_min_on_s": sakha_config.audio_min_on_s,
            "audio_merge_gap_s": sakha_config.audio_merge_gap_s,
            "audio_min_open_after_trigger_s": sakha_config.audio_min_open_after_trigger_s,
            "audio_release_hold_s": sakha_config.audio_release_hold_s,
            "audio_min_closed_s": sakha_config.audio_min_closed_s,
            "audio_silence_policy": sakha_config.audio_silence_policy,
            "audio_clean_mode": sakha_config.audio_clean_mode,
            "calibrated_switch_margin_db": sakha_config.calibrated_switch_margin_db,
            "calibrated_overlap_floor_db": sakha_config.calibrated_overlap_floor_db,
            "calibrated_min_hold_s": sakha_config.calibrated_min_hold_s,
            "debleed_enabled": sakha_config.debleed_enabled,
            "debleed_overlap_margin_db": sakha_config.debleed_overlap_margin_db,
            "debleed_min_leader_score_db": sakha_config.debleed_min_leader_score_db,
            "debleed_score_snr_weight": sakha_config.debleed_score_snr_weight,
            "debleed_guest_rescue_snr_db": sakha_config.debleed_guest_rescue_snr_db,
            "debleed_guest_ambiguous_snr_margin_db": sakha_config.debleed_guest_ambiguous_snr_margin_db,
            "debleed_cohost_confidence_s": sakha_config.debleed_cohost_confidence_s,
            "waveform_window_s": sakha_config.waveform_window_s,
            "waveform_step_s": sakha_config.waveform_step_s,
            "waveform_max_lag_s": sakha_config.waveform_max_lag_s,
            "waveform_corr_bleed_threshold": sakha_config.waveform_corr_bleed_threshold,
            "waveform_independent_threshold": sakha_config.waveform_independent_threshold,
            "waveform_overlap_snr_db": sakha_config.waveform_overlap_snr_db,
            "strict_min_speaker_hold_s": sakha_config.strict_min_speaker_hold_s,
            "strict_switch_margin_db": sakha_config.strict_switch_margin_db,
            "source_owner_bleed_residual_db": sakha_config.source_owner_bleed_residual_db,
            "source_owner_residual_voice_snr_db": sakha_config.source_owner_residual_voice_snr_db,
            "source_owner_min_corr": sakha_config.source_owner_min_corr,
            "source_owner_tie_margin_db": sakha_config.source_owner_tie_margin_db,
            "motion_entry_lookahead_s": plan.diagnostics.get("motion_entry_lookahead_s"),
        },
        {
            "event": "motion_config",
            "enabled": motion_check,
            "motion_speed": motion_execution_config.speed,
            "motion_hwaccel_requested": motion_execution_config.hwaccel,
            "motion_hwaccel_active": None if motion_runtime is None else motion_runtime.active_hwaccel,
            "motion_worker_count": 0 if motion_runtime is None else motion_runtime.worker_count,
            "motion_cpu_worker_count": 0 if motion_runtime is None else motion_runtime.cpu_worker_count,
            "motion_gpu_worker_count": 0 if motion_runtime is None else motion_runtime.gpu_worker_count,
            "motion_cpu_clip_count": 0 if motion_runtime is None else motion_runtime.cpu_clip_count,
            "motion_gpu_clip_count": 0 if motion_runtime is None else motion_runtime.gpu_clip_count,
            "sample_fps": motion_config.sample_fps,
            "resize_width": motion_config.resize_width,
            "refine_sample_fps": motion_config.refine_sample_fps,
            "refine_resize_width": motion_config.refine_resize_width,
            "motion_analysis_s": round(motion_analysis_s, 3),
        },
        {
            "event": "motion_cache",
            **motion_cache_stats,
        },
        {
            "event": "detector_diagnostics",
            "participants": {
                part.key: {
                    "label": part.label,
                    "backend": detector_diags[part.key].get("detector_backend"),
                    "warning": detector_diags[part.key].get("detector_warning"),
                    "active_seconds": detector_diags[part.key].get("active_seconds"),
                    "total_seconds": detector_diags[part.key].get("total_seconds"),
                    "safety_forced_frames": detector_diags[part.key].get("safety_forced_frames"),
                    "mask_filter_removed": detector_diags[part.key].get("mask_filter_removed"),
                    "mask_filter_gap_filled": detector_diags[part.key].get("mask_filter_gap_filled"),
                }
                for part in participant_specs
            },
        },
        {
            "event": "debleed_diagnostics",
            **plan.diagnostics.get("debleed", {}),
        },
        {
            "event": "waveform_gate_diagnostics",
            **plan.diagnostics.get("waveform_gate", {}),
        },
        {
            "event": "source_owner_diagnostics",
            **plan.diagnostics.get("source_owner", {}),
        },
        {
            "event": "sakha_leak_matrix",
            **plan.diagnostics.get("leak_matrix", {}),
        },
        {
            "event": "sakha_speech_segments",
            "count": len(plan.speech_segments),
            "segments": [_serialize_sakha_speech_segment(seg) for seg in plan.speech_segments],
        },
        {
            "event": "sakha_camera_segments",
            "count": len(plan.camera_segments),
            "segments": [_serialize_sakha_camera_segment(seg) for seg in plan.camera_segments],
            "diagnostics": plan.diagnostics,
        },
        {
            "event": "sakha_motion_events",
            "count": len(plan.motion_events),
            "events": [_serialize_sakha_motion_event(event) for event in plan.motion_events],
        },
        {
            "event": "audio_track_intervals_raw",
            "tracks": _serialize_interval_map(plan.audio_open_intervals_raw_s),
        },
        {
            "event": "sakha_audio_levels",
            "count": len(plan.audio_level_segments),
            "segments": [_serialize_audio_level_segment(seg) for seg in plan.audio_level_segments],
        },
        {
            "event": "camera_cuts",
            "first_angle": first_angle,
            "count": len(cuts),
            "cuts": [
                {"time": round(cut["time"], 3), "angle": cut["angle"]}
                for cut in cuts
            ],
        },
    ]
    if audio_sources_log is not None:
        log_entries.insert(2, audio_sources_log)
    if resolved_camera_sources is not None:
        log_entries.append(
            {
                "event": "motion_sources",
                "method": resolved_camera_sources.method,
                "angles": {
                    str(angle): {
                        "path": str(source.representative_path),
                        "clips": [
                            {
                                "path": str(clip.path),
                                "timeline_start": round(clip.timeline_start_s, 3),
                                "timeline_end": None if clip.timeline_end_s is None else round(clip.timeline_end_s, 3),
                                "source_start": round(clip.source_start_s, 3),
                                "source_end": None if clip.source_end_s is None else round(clip.source_end_s, 3),
                            }
                            for clip in source.clips
                        ],
                    }
                    for angle, source in sorted(resolved_camera_sources.sources_by_angle.items())
                },
                "discovered_paths": [str(path) for path in resolved_camera_sources.discovered_paths],
            }
        )
    if motion_plans is not None:
        for angle, plan_motion in sorted(motion_plans.items()):
            log_entries.append(
                {
                    "event": "camera_motion_intervals",
                    "angle": angle,
                    "source_path": None if plan_motion.source_path is None else str(plan_motion.source_path),
                    "count": len(plan_motion.moving_intervals),
                    "intervals": [_serialize_motion_interval(interval) for interval in plan_motion.moving_intervals],
                    "diagnostics": plan_motion.diagnostics,
                }
            )

    click.echo("Patching project...")
    patch_prproj(
        Path(in_file),
        cuts,
        seq,
        out,
        first_angle=first_angle,
        audio_track_intervals_s=plan.audio_open_intervals_s if mute_audio else None,
        log_path=log_path,
        source_offset_s=0.0,
        audio_source_offset_s=0.0,
        audio_overlap_s=0.0,
        audio_pre_roll_s=sakha_config.audio_pre_roll_s,
        audio_post_roll_s=sakha_config.audio_post_roll_s,
        prelude_log_entries=log_entries if log else None,
        fps=fps,
    )
    click.echo(f"Done: {out_file}")
    if log_path and log_path.exists():
        click.echo(f"Log: {log_path}")


@cli.command("auto-multicam")
@click.option("--in", "in_file", required=True, type=click.Path(exists=True), help="Input .prproj file")
@click.option("--mic-a", type=click.Path(exists=True), help="Host microphone audio file (empty = auto-resolve from project)")
@click.option("--mic-b", type=click.Path(exists=True), help="Guest microphone audio file (empty = auto-resolve from project)")
@click.option("--seq", required=True, help="Sequence name in .prproj")
@click.option("--out", "out_file", required=True, type=click.Path(), help="Output .prproj file")
@click.option("--label-a", default="host", help="Label for speaker A (default: host)")
@click.option("--label-b", default="guest", help="Label for speaker B (default: guest)")
@click.option("--camera-host", default=1, type=int, help="Premiere angle for host (1-based, default: 1)")
@click.option("--camera-guest", default=2, type=int, help="Premiere angle for guest (1-based, default: 2)")
@click.option("--camera-wide", default=3, type=int, help="Premiere angle for wide/both/silence (1-based, default: 3)")
@click.option("--speech-threshold", default=-27.0, type=float, help="Speech onset threshold in dB")
@click.option("--input-gain", default=0.0, type=float, help="Input gain in dB (applied to both mics unless per-mic gain set)")
@click.option("--input-gain-a", default=0.0, type=float, help="Input gain for mic A (host) in dB, overrides --input-gain")
@click.option("--input-gain-b", default=0.0, type=float, help="Input gain for mic B (guest) in dB, overrides --input-gain")
@click.option("--mute-audio/--no-mute-audio", default=True, help="Mute inactive speaker mics (default: on)")
@click.option("--cross-gate-db", default=6.0, type=float, help="Cross-gate threshold in dB (0 = disabled, default: 6)")
@click.option("--detector-backend", default="auto", type=click.Choice(["auto", "rms", "silero"]), help="Speech detector backend")
@click.option("--vad-threshold", default=0.65, type=float, help="Silero VAD speech probability threshold (0.65 suppresses cross-talk from the other mic)")
@click.option("--vad-min-speech-ms", default=120.0, type=float, help="Silero VAD minimum speech duration in ms")
@click.option("--vad-min-silence-ms", default=80.0, type=float, help="Silero VAD minimum silence duration in ms")
@click.option("--vad-speech-pad-ms", default=30.0, type=float, help="Silero VAD padding around detected speech in ms")
@click.option("--audio-mute-min-segment", default=500.0, type=float, help="Minimum audio mute segment length in ms")
@click.option("--audio-pre-roll", default=0.24, type=float, help="Open speaker audio this many seconds before detected onset")
@click.option("--audio-post-roll", default=0.12, type=float, help="Keep speaker audio open this many seconds after detected end")
@click.option("--dialogue-wide-interval", default=24.0, type=float, help="Minimum seconds between re-establishing wide shots during active dialogue")
@click.option("--dialogue-wide-duration", default=2.0, type=float, help="Duration of each re-establishing dialogue wide shot in seconds")
@click.option("--dialogue-wide-min-turns", default=3, type=int, help="Minimum speaker exchanges before inserting a dialogue wide shot")
@click.option("--log/--no-log", default=True, help="Write JSONL log file next to output (default: on)")
@click.option("--audio-track-host", default=1, type=int, help="Audio track number for host (1-based, default: 1)")
@click.option("--audio-track-guest", default=2, type=int, help="Audio track number for guest (1-based, default: 2)")
@click.option("--xml", "xml_file", type=click.Path(exists=True), help="Premiere FCP7 XML export for more reliable audio/camera source resolution")
@click.option("--fps", default=0.0, type=float, help="Sequence frame rate for frame-aligned cuts (0 = auto-detect from .prproj)")
@_cross_cancel_options(default_enabled=True)
def auto_multicam_cmd(
    in_file, mic_a, mic_b, seq, out_file,
    label_a, label_b,
    camera_host, camera_guest, camera_wide,
    speech_threshold, input_gain, input_gain_a, input_gain_b,
    mute_audio, cross_gate_db,
    detector_backend, vad_threshold, vad_min_speech_ms, vad_min_silence_ms, vad_speech_pad_ms,
    audio_mute_min_segment, audio_pre_roll, audio_post_roll,
    dialogue_wide_interval, dialogue_wide_duration, dialogue_wide_min_turns, log,
    audio_track_host, audio_track_guest,
    fps, xml_file,
    enable_cross_cancel, cross_cancel_fir_taps,
):
    """Full pipeline: analyze audio -> switch cameras -> patch .prproj."""
    from .prproj_patcher import patch_prproj, segments_to_cuts, read_audio_offsets
    from .core.audio_loader import apply_offset

    # Validate 1-based angles
    for label, val in [("camera-host", camera_host), ("camera-guest", camera_guest), ("camera-wide", camera_wide)]:
        if val < 1:
            raise click.BadParameter(
                f"Angles are 1-based (Premiere Angle 1,2,3...)",
                param_hint=f"--{label}",
            )

    # Convert 1-based (Premiere UI) → 0-based (internal / SelectedTrackIndex)
    camera_host -= 1
    camera_guest -= 1
    camera_wide -= 1

    _resolved_mics, _audio_sources_log, _mute_ok = _resolve_speaker_mics(
        in_file, seq,
        [
            (audio_track_host - 1, label_a, mic_a),
            (audio_track_guest - 1, label_b, mic_b),
        ],
        xml_file,
    )
    mic_a, mic_b = _resolved_mics
    mute_audio = mute_audio and _mute_ok

    # 1. Build config and load audio
    config = build_config(
        audio_a=mic_a, label_a=label_a, camera_a=camera_host,
        audio_b=mic_b, label_b=label_b, camera_b=camera_guest,
        camera_wide=camera_wide,
        speech_threshold_db=speech_threshold,
        input_gain_db=input_gain,
        input_gain_a_db=input_gain_a,
        input_gain_b_db=input_gain_b,
        cross_gate_db=cross_gate_db,
        detector_backend=detector_backend,
        vad_threshold=vad_threshold,
        vad_min_speech_ms=vad_min_speech_ms,
        vad_min_silence_ms=vad_min_silence_ms,
        vad_speech_pad_ms=vad_speech_pad_ms,
        audio_mute_min_segment_ms=audio_mute_min_segment,
        audio_pre_roll_s=audio_pre_roll,
        audio_post_roll_s=audio_post_roll,
        dialogue_wide_interval_sec=dialogue_wide_interval,
        dialogue_wide_duration_sec=dialogue_wide_duration,
        dialogue_wide_min_turns=dialogue_wide_min_turns,
    )

    # Read per-mic offsets from .prproj (InPoint of each audio track)
    try:
        offsets = read_audio_offsets(Path(in_file), seq)
        offset_a = offsets.get(audio_track_host - 1, 0.0)
        offset_b = offsets.get(audio_track_guest - 1, 0.0)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:
        click.echo(
            f"Warning: could not read audio offsets for sequence '{seq}': {exc}. "
            "Continuing with zero offsets."
        )
        offset_a, offset_b = 0.0, 0.0

    if offset_a > 0 or offset_b > 0:
        click.echo(f"Audio offsets: host={offset_a:.3f}s, guest={offset_b:.3f}s")

    click.echo(f"Loading audio: {mic_a}, {mic_b}")
    paths = [config.audio_inputs[0].path, config.audio_inputs[1].path]
    audio_arrays = load_and_align(paths, config.sample_rate)

    # Apply per-mic offsets (trim start of each file)
    if offset_a > 0:
        audio_arrays[0] = apply_offset(audio_arrays[0], offset_a, config.sample_rate)
    if offset_b > 0:
        audio_arrays[1] = apply_offset(audio_arrays[1], offset_b, config.sample_rate)
    # Re-align lengths after trimming
    max_len = max(len(a) for a in audio_arrays)
    for i, a in enumerate(audio_arrays):
        if len(a) < max_len:
            padded = np.zeros(max_len, dtype=np.float64)
            padded[:len(a)] = a
            audio_arrays[i] = padded

    duration_s = len(audio_arrays[0]) / config.sample_rate
    click.echo(f"Duration: {duration_s:.1f}s")

    audio_arrays = list(
        _maybe_apply_cross_cancel(
            audio_arrays,
            config.sample_rate,
            enabled=enable_cross_cancel,
            fir_taps=cross_cancel_fir_taps,
        )
    )
    if enable_cross_cancel:
        click.echo(f"Cross-channel cancellation: fir_taps={cross_cancel_fir_taps}, channels={len(audio_arrays)}")

    # 2. Detect speech activity
    click.echo("Analyzing speaker activity...")
    gain_a = config.input_gain_a_db or config.input_gain_db
    gain_b = config.input_gain_b_db or config.input_gain_db
    activity_a = analyze_speaker(audio_arrays[0], label_a, config, gain_db=gain_a)
    activity_b = analyze_speaker(audio_arrays[1], label_b, config, gain_db=gain_b)

    diag_a, diag_b = {}, {}
    detect_activity(activity_a, config, diagnostics=diag_a, audio=audio_arrays[0])
    detect_activity(activity_b, config, diagnostics=diag_b, audio=audio_arrays[1])

    click.echo(
        f"  {label_a}:  {diag_a['active_seconds']}s active / {diag_a['total_seconds']}s total, "
        f"backend={diag_a['detector_backend']}, safety={diag_a['safety_forced_frames']} frames"
    )
    click.echo(
        f"  {label_b}: {diag_b['active_seconds']}s active / {diag_b['total_seconds']}s total, "
        f"backend={diag_b['detector_backend']}, safety={diag_b['safety_forced_frames']} frames"
    )
    for lbl, diag in [(label_a, diag_a), (label_b, diag_b)]:
        mf_removed = diag.get("mask_filter_removed", 0)
        mf_filled = diag.get("mask_filter_gap_filled", 0)
        if mf_removed or mf_filled:
            click.echo(f"  {lbl}:  mask_filter removed {mf_removed} segments, filled {mf_filled} gaps")
        if diag.get("detector_warning"):
            click.echo(f"  {lbl}:  {diag['detector_warning']}")

    states = combine_speakers(activity_a, activity_b)
    if config.cross_gate_db > 0:
        states = apply_cross_gate(states, activity_a, activity_b, config.cross_gate_db)

    # 3. Segment + camera scheduling (filtered for video cuts)
    click.echo("Segmenting timeline...")
    raw_camera_segments = run_length_encode(states, config.hop_ms / 1000.0)
    segments = build_camera_segments(states, config)

    click.echo("Scheduling cameras...")
    camera_events = schedule_camera_events(segments, config)
    camera_segments = camera_events_to_segments(camera_events)

    # Audio mute pipeline: preserve genuine overlap so short interjections are not erased.
    audio_mute_segments = build_audio_mute_segments(states, config)

    click.echo(f"Analysis: {len(camera_segments)} camera segments, {len(audio_mute_segments)} audio mute segments")

    # 4. Convert segments → cuts (video uses camera_segments)
    first_angle, cuts = segments_to_cuts(camera_segments)
    click.echo(f"Camera switches: {len(cuts)} cuts (first angle: {first_angle})")

    # 5. Patch .prproj
    out = Path(out_file)
    log_path = out.with_suffix(out.suffix + ".log.jsonl") if log else None

    # Map audio track index (angle order) → speaker index (0=host, 1=guest)
    audio_track_map = {audio_track_host - 1: 0, audio_track_guest - 1: 1}

    log_prelude_entries = [
        {
            "event": "build_info",
            "package_version": __version__,
            "segmenter_signature": CAMERA_SEGMENTER_SIGNATURE,
            "planner_signature": CAMERA_PLANNER_SIGNATURE,
        },
        {
            "event": "camera_mapping",
            "host_label": label_a,
            "guest_label": label_b,
            "host_angle_internal": camera_host,
            "guest_angle_internal": camera_guest,
            "wide_angle_internal": camera_wide,
            "host_angle_premiere": camera_host + 1,
            "guest_angle_premiere": camera_guest + 1,
            "wide_angle_premiere": camera_wide + 1,
            "audio_track_map": audio_track_map,
        },
        {
            "event": "camera_config",
            "camera_debounce_ms": config.camera_debounce_ms,
            "camera_min_segment_ms": config.camera_min_segment_ms,
            "camera_takeover_min_segment_ms": config.camera_takeover_min_segment_ms,
            "camera_takeover_context_ms": config.camera_takeover_context_ms,
            "camera_both_min_segment_ms": config.camera_both_min_segment_ms,
            "camera_silence_min_segment_ms": config.camera_silence_min_segment_ms,
            "min_camera_event_s": config.min_camera_event_s,
            "long_talk_threshold_sec": config.long_talk_threshold_sec,
            "wide_duration_sec": config.wide_duration_sec,
            "long_talk_mode": config.long_talk_mode,
            "wide_cooldown_sec": config.wide_cooldown_sec,
            "dialogue_wide_interval_sec": config.dialogue_wide_interval_sec,
            "dialogue_wide_duration_sec": config.dialogue_wide_duration_sec,
            "dialogue_wide_min_turns": config.dialogue_wide_min_turns,
            "sticky_wide_max_bridge_sec": config.sticky_wide_max_bridge_sec,
            "sticky_wide_max_turn_sec": config.sticky_wide_max_turn_sec,
            "sticky_wide_min_turns": config.sticky_wide_min_turns,
            "dialogue_cluster_max_span_sec": config.dialogue_cluster_max_span_sec,
            "dialogue_cluster_max_turn_sec": config.dialogue_cluster_max_turn_sec,
            "dialogue_cluster_min_turns": config.dialogue_cluster_min_turns,
        },
        {
            "event": "detector_diagnostics",
            "speaker_a": {
                "label": label_a,
                "backend": diag_a.get("detector_backend"),
                "warning": diag_a.get("detector_warning"),
                "active_seconds": diag_a.get("active_seconds"),
                "total_seconds": diag_a.get("total_seconds"),
                "safety_forced_frames": diag_a.get("safety_forced_frames"),
                "mask_filter_removed": diag_a.get("mask_filter_removed"),
                "mask_filter_gap_filled": diag_a.get("mask_filter_gap_filled"),
            },
            "speaker_b": {
                "label": label_b,
                "backend": diag_b.get("detector_backend"),
                "warning": diag_b.get("detector_warning"),
                "active_seconds": diag_b.get("active_seconds"),
                "total_seconds": diag_b.get("total_seconds"),
                "safety_forced_frames": diag_b.get("safety_forced_frames"),
                "mask_filter_removed": diag_b.get("mask_filter_removed"),
                "mask_filter_gap_filled": diag_b.get("mask_filter_gap_filled"),
            },
        },
        {
            "event": "camera_segments_rle_raw",
            "count": len(raw_camera_segments),
            "segments": [_serialize_segment(seg) for seg in raw_camera_segments],
        },
        {
            "event": "camera_segments_before_schedule",
            "count": len(segments),
            "segments": [_serialize_segment(seg) for seg in segments],
        },
        {
            "event": "camera_events_after_schedule",
            "count": len(camera_events),
            "events": [_serialize_camera_event(ev) for ev in camera_events],
        },
        {
            "event": "camera_cuts",
            "first_angle": first_angle,
            "count": len(cuts),
            "cuts": [
                {"time": round(cut["time"], 3), "angle": cut["angle"]}
                for cut in cuts
            ],
        },
    ]

    n = patch_prproj(
        Path(in_file), cuts, seq, out,
        first_angle=first_angle,
        audio_segments=audio_mute_segments if mute_audio else None,
        log_path=log_path,
        audio_track_map=audio_track_map,
        source_offset_s=0.0,
        audio_source_offset_s=0.0,
        audio_overlap_s=config.audio_overlap_s,
        audio_pre_roll_s=config.audio_pre_roll_s,
        audio_post_roll_s=config.audio_post_roll_s,
        prelude_log_entries=log_prelude_entries if log else None,
        fps=fps,
    )
    click.echo(f"Done: {n} segments -> {out_file}")
    if log_path and log_path.exists():
        click.echo(f"Log: {log_path}")


@cli.command("auto-switch-monologue")
@click.option("--in", "in_file", required=True, type=click.Path(exists=True), help="Input .prproj file")
@click.option("--mic", type=click.Path(exists=True), help="Narrator microphone audio file (empty = auto-resolve from project)")
@click.option("--seq", required=True, help="Sequence name in .prproj")
@click.option("--out", "out_file", required=True, type=click.Path(), help="Output .prproj file")
@click.option("--label", default="narrator", help="Label for narrator track")
@click.option("--camera-main", default=1, type=int, help="Premiere angle for main camera (1-based, default: 1)")
@click.option("--camera-accent", default=2, type=int, help="Premiere angle for accent camera (1-based, default: 2)")
@click.option("--camera-main-file", type=click.Path(exists=True), help="Override source video file for main camera")
@click.option("--camera-accent-file", type=click.Path(exists=True), help="Override source video file for accent camera")
@click.option("--xml", "xml_file", type=click.Path(exists=True), help="Premiere FCP7 XML export for more reliable camera source resolution")
@click.option("--audio-track", default=1, type=int, help="Audio track number for narrator mic (1-based, default: 1)")
@click.option("--switch-interval", default=30.0, type=float, help="Target seconds between camera switches")
@click.option("--camera-main-share", default=50.0, type=float, help="Target share of the main camera in percent (default: 50)")
@click.option("--pause-window-before", default=6.0, type=float, help="How far before the target to look for a natural pause")
@click.option("--pause-window-after", default=10.0, type=float, help="How far after the target to wait for a natural pause")
@click.option("--min-pause", default=0.35, type=float, help="Minimum silence duration to count as a natural switch point")
@click.option("--min-hold", default=12.0, type=float, help="Minimum seconds to hold each camera before switching again")
@click.option("--motion-check/--no-motion-check", default=True, help="Avoid switching to moving cameras (default: on)")
@click.option("--motion-hwaccel", default="hybrid", type=click.Choice(["cpu", "hybrid"]), help="Motion analysis acceleration policy: hybrid = split clips between CPU and GPU decode")
@click.option("--speech-threshold", default=-24.0, type=float, help="Speech onset threshold in dB")
@click.option("--input-gain", default=0.0, type=float, help="Input gain in dB")
@click.option("--detector-backend", default="auto", type=click.Choice(["auto", "rms", "silero"]), help="Speech detector backend")
@click.option("--vad-threshold", default=0.5, type=float, help="Silero VAD speech probability threshold")
@click.option("--vad-min-speech-ms", default=120.0, type=float, help="Silero VAD minimum speech duration in ms")
@click.option("--vad-min-silence-ms", default=80.0, type=float, help="Silero VAD minimum silence duration in ms")
@click.option("--vad-speech-pad-ms", default=30.0, type=float, help="Silero VAD padding around detected speech in ms")
@click.option("--log/--no-log", default=True, help="Write JSONL log file next to output (default: on)")
@click.option("--fps", default=0.0, type=float, help="Sequence frame rate for frame-aligned cuts (0 = auto-detect from .prproj)")
def auto_switch_monologue_cmd(
    in_file, mic, seq, out_file,
    label,
    camera_main, camera_accent, camera_main_file, camera_accent_file, xml_file, audio_track,
    switch_interval, camera_main_share, pause_window_before, pause_window_after, min_pause, min_hold,
    motion_check, motion_hwaccel,
    speech_threshold, input_gain,
    detector_backend, vad_threshold, vad_min_speech_ms, vad_min_silence_ms, vad_speech_pad_ms,
    log,
    fps,
):
    """Full pipeline: analyze one narrator mic -> switch two cameras -> patch .prproj."""
    from .core.audio_loader import apply_offset, load_audio
    from .prproj_patcher import patch_prproj, read_audio_offsets, segments_to_cuts

    command_started_at = time.perf_counter()

    for label_name, value in [("camera-main", camera_main), ("camera-accent", camera_accent)]:
        if value < 1:
            raise click.BadParameter(
                "Angles are 1-based (Premiere Angle 1,2,3...)",
                param_hint=f"--{label_name}",
            )
    if camera_main == camera_accent:
        raise click.BadParameter(
            "Main and accent cameras must be different angles",
            param_hint="--camera-accent",
        )
    if audio_track < 1:
        raise click.BadParameter(
            "Audio tracks are 1-based",
            param_hint="--audio-track",
        )
    if not 0.0 < camera_main_share < 100.0:
        raise click.BadParameter(
            "Main camera share must be between 0 and 100",
            param_hint="--camera-main-share",
        )

    camera_main -= 1
    camera_accent -= 1

    mic = _resolve_speaker_mics(in_file, seq, [(audio_track - 1, label, mic)], xml_file)[0][0]

    analysis_config = ProjectConfig(
        audio_inputs=[
            AudioInput(
                path=Path(mic),
                speaker_label=label,
                camera_index=camera_main,
            )
        ],
        default_camera=camera_main,
        both_speaking_camera=camera_main,
        speech_threshold_db=speech_threshold,
        input_gain_db=input_gain,
        detector_backend=detector_backend,
        vad_threshold=vad_threshold,
        vad_min_speech_ms=vad_min_speech_ms,
        vad_min_silence_ms=vad_min_silence_ms,
        vad_speech_pad_ms=vad_speech_pad_ms,
    )
    analysis_config.validate()

    planner_config = Monologue2CamConfig(
        camera_main=camera_main,
        camera_accent=camera_accent,
        switch_interval_s=switch_interval,
        camera_main_share=camera_main_share / 100.0,
        pause_window_before_s=pause_window_before,
        pause_window_after_s=pause_window_after,
        min_pause_s=min_pause,
        min_camera_hold_s=min_hold,
    )
    motion_config = CameraMotionConfig()
    motion_execution_config = CameraMotionExecutionConfig(
        speed="balanced",
        hwaccel=motion_hwaccel,
    )

    try:
        offsets = read_audio_offsets(Path(in_file), seq)
        audio_offset_s = offsets.get(audio_track - 1, 0.0)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:
        click.echo(
            f"Warning: could not read audio offsets for sequence '{seq}': {exc}. "
            "Continuing with zero offset."
        )
        audio_offset_s = 0.0

    if audio_offset_s > 0:
        click.echo(f"Audio offset: {audio_offset_s:.3f}s")

    click.echo(f"Loading audio: {mic}")
    audio_load_started_at = time.perf_counter()
    audio = load_audio(Path(mic), analysis_config.sample_rate)
    if audio_offset_s > 0:
        audio = apply_offset(audio, audio_offset_s, analysis_config.sample_rate)
    audio_load_s = time.perf_counter() - audio_load_started_at

    duration_s = len(audio) / analysis_config.sample_rate
    click.echo(f"Duration: {duration_s:.1f}s")

    click.echo("Analyzing narrator activity...")
    activity_started_at = time.perf_counter()
    activity = analyze_speaker(audio, label, analysis_config, gain_db=analysis_config.input_gain_db)
    diagnostics = {}
    detect_activity(activity, analysis_config, diagnostics=diagnostics, audio=audio)
    activity_analysis_s = time.perf_counter() - activity_started_at
    click.echo(
        f"  {label}: {diagnostics['active_seconds']}s active / {diagnostics['total_seconds']}s total, "
        f"backend={diagnostics['detector_backend']}, safety={diagnostics['safety_forced_frames']} frames"
    )
    if diagnostics.get("detector_warning"):
        click.echo(f"  {label}: {diagnostics['detector_warning']}")

    click.echo("Planning camera switches...")
    click.echo(
        "  target share: "
        f"main={planner_config.camera_main_share * 100:.1f}% "
        f"(~{planner_config.target_hold_s(camera_main):.1f}s), "
        f"accent={planner_config.camera_accent_share * 100:.1f}% "
        f"(~{planner_config.target_hold_s(camera_accent):.1f}s)"
    )
    motion_plans = None
    resolved_sources = None
    motion_runtime = None
    motion_analysis_s = 0.0
    if motion_check:
        try:
            resolved_sources = resolve_monologue_camera_sources(
                Path(in_file),
                seq,
                camera_main,
                camera_accent,
                camera_main_file=camera_main_file,
                camera_accent_file=camera_accent_file,
                xml_path=xml_file,
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

        click.echo(f"Motion sources: {resolved_sources.method}")
        click.echo(
            f"  main:   {resolved_sources.main_path} "
            f"({len(resolved_sources.main_clips)} clip{'s' if len(resolved_sources.main_clips) != 1 else ''})"
        )
        click.echo(
            f"  accent: {resolved_sources.accent_path} "
            f"({len(resolved_sources.accent_clips)} clip{'s' if len(resolved_sources.accent_clips) != 1 else ''})"
        )

        analyzer = CameraMotionAnalyzer(motion_config, motion_execution_config)
        clip_groups = {
            camera_main: list(resolved_sources.main_clips),
            camera_accent: list(resolved_sources.accent_clips),
        }
        total_motion_clips = sum(len(clips) for clips in clip_groups.values())
        all_motion_clips = [
            clip
            for clips in clip_groups.values()
            for clip in clips
        ]
        motion_runtime = analyzer.prepare_runtime(all_motion_clips)
        click.echo(
            "  motion execution: "
            f"workers={motion_runtime.worker_count}, "
            f"cpu_workers={motion_runtime.cpu_worker_count}, "
            f"gpu_workers={motion_runtime.gpu_worker_count}, "
            f"hwaccel={motion_runtime.active_hwaccel}"
        )
        click.echo("Analyzing camera motion...")
        motion_started_at = time.perf_counter()
        with click.progressbar(
            length=total_motion_clips,
            label="  motion clips",
            show_eta=True,
            show_percent=True,
        ) as motion_bar:
            progress_lock = Lock()

            def _on_motion_clip_done(_clip):
                with progress_lock:
                    motion_bar.update(1)

            motion_plans = analyzer.analyze_camera_groups(
                clip_groups,
                total_duration_s=duration_s,
                progress_callback=_on_motion_clip_done,
                runtime=motion_runtime,
            )
        motion_analysis_s = time.perf_counter() - motion_started_at
        click.echo(
            "  motion intervals: "
            f"main={len(motion_plans[camera_main].moving_intervals)}, "
            f"accent={len(motion_plans[camera_accent].moving_intervals)}"
        )
    else:
        click.echo("  motion check: disabled")

    planning_started_at = time.perf_counter()
    plan = build_monologue_plan(
        activity,
        duration_s,
        analysis_config.hop_ms / 1000.0,
        planner_config,
        motion_plans=motion_plans,
    )
    camera_planning_s = time.perf_counter() - planning_started_at
    click.echo(
        f"Analysis: {len(plan.speech_segments)} speech segments, "
        f"{len(plan.pause_segments)} pause candidates, "
        f"{len(plan.camera_segments)} camera segments"
    )

    first_angle, cuts = segments_to_cuts(plan.camera_segments)
    click.echo(f"Camera switches: {len(cuts)} cuts (first angle: {first_angle})")

    out = Path(out_file)
    log_path = out.with_suffix(out.suffix + ".log.jsonl") if log else None

    log_prelude_entries = [
        {
            "event": "build_info",
            "package_version": __version__,
            "planner_signature": MONOLOGUE_PLANNER_SIGNATURE,
        },
        {
            "event": "camera_mapping",
            "label": label,
            "main_angle_internal": camera_main,
            "accent_angle_internal": camera_accent,
            "main_angle_premiere": camera_main + 1,
            "accent_angle_premiere": camera_accent + 1,
            "audio_track": audio_track,
        },
        {
            "event": "monologue_camera_config",
            "switch_interval_s": planner_config.switch_interval_s,
            "camera_main_share": planner_config.camera_main_share,
            "camera_accent_share": planner_config.camera_accent_share,
            "camera_main_target_hold_s": planner_config.target_hold_s(camera_main),
            "camera_accent_target_hold_s": planner_config.target_hold_s(camera_accent),
            "pause_window_before_s": planner_config.pause_window_before_s,
            "pause_window_after_s": planner_config.pause_window_after_s,
            "min_pause_s": planner_config.min_pause_s,
            "micro_pause_merge_s": planner_config.micro_pause_merge_s,
            "min_camera_hold_s": planner_config.min_camera_hold_s,
            "pause_edge_padding_s": planner_config.pause_edge_padding_s,
            "forced_min_drop_db": planner_config.forced_min_drop_db,
            "motion_wait_max_s": planner_config.motion_wait_max_s,
            "motion_escape_min_hold_s": planner_config.motion_escape_min_hold_s,
            "entry_stability_lookahead_s": planner_config.entry_stability_lookahead_s,
        },
        {
            "event": "motion_config",
            "enabled": motion_check,
            "motion_speed": motion_execution_config.speed,
            "motion_hwaccel_requested": motion_execution_config.hwaccel,
            "motion_hwaccel_active": None if motion_runtime is None else motion_runtime.active_hwaccel,
            "motion_worker_count": 0 if motion_runtime is None else motion_runtime.worker_count,
            "motion_cpu_worker_count": 0 if motion_runtime is None else motion_runtime.cpu_worker_count,
            "motion_gpu_worker_count": 0 if motion_runtime is None else motion_runtime.gpu_worker_count,
            "motion_cpu_clip_count": 0 if motion_runtime is None else motion_runtime.cpu_clip_count,
            "motion_gpu_clip_count": 0 if motion_runtime is None else motion_runtime.gpu_clip_count,
            "entry_stability_lookahead_s": planner_config.entry_stability_lookahead_s,
            "sample_fps": motion_config.sample_fps,
            "resize_width": motion_config.resize_width,
            "border_ratio": motion_config.border_ratio,
            "ring_ratio": motion_config.ring_ratio,
            "bottom_zone_disabled": True,
            "bottom_ignore_ratio": motion_config.bottom_ignore_ratio,
            "refine_sample_fps": motion_config.refine_sample_fps,
            "refine_resize_width": motion_config.refine_resize_width,
            "candidate_score_threshold": motion_config.candidate_score_threshold,
            "candidate_padding_s": motion_config.candidate_padding_s,
            "candidate_merge_gap_s": motion_config.candidate_merge_gap_s,
            "smoothing_window_s": motion_config.smoothing_window_s,
            "motion_pre_roll_s": motion_config.motion_pre_roll_s,
            "moving_on_threshold": motion_config.moving_on_threshold,
            "moving_off_threshold": motion_config.moving_off_threshold,
            "moving_confirm_s": motion_config.moving_confirm_s,
            "stable_confirm_s": motion_config.stable_confirm_s,
        },
        {
            "event": "detector_diagnostics",
            "speaker": {
                "label": label,
                "backend": diagnostics.get("detector_backend"),
                "warning": diagnostics.get("detector_warning"),
                "active_seconds": diagnostics.get("active_seconds"),
                "total_seconds": diagnostics.get("total_seconds"),
                "safety_forced_frames": diagnostics.get("safety_forced_frames"),
                "mask_filter_removed": diagnostics.get("mask_filter_removed"),
                "mask_filter_gap_filled": diagnostics.get("mask_filter_gap_filled"),
            },
        },
        {
            "event": "monologue_speech_segments",
            "count": len(plan.speech_segments),
            "segments": [_serialize_segment(seg) for seg in plan.speech_segments],
        },
        {
            "event": "monologue_pause_candidates",
            "count": len(plan.pause_segments),
            "segments": [_serialize_segment(seg) for seg in plan.pause_segments],
        },
        {
            "event": "monologue_camera_segments",
            "count": len(plan.camera_segments),
            "segments": [_serialize_monologue_camera_segment(seg) for seg in plan.camera_segments],
            "diagnostics": plan.diagnostics,
        },
        {
            "event": "camera_cuts",
            "first_angle": first_angle,
            "count": len(cuts),
            "cuts": [_serialize_monologue_cut(cut) for cut in plan.cuts],
        },
        {
            "event": "monologue_motion_events",
            "count": len(plan.motion_events),
            "events": [_serialize_monologue_motion_event(event) for event in plan.motion_events],
        },
    ]
    if resolved_sources is not None:
        log_prelude_entries.append(
            {
                "event": "motion_sources",
                "method": resolved_sources.method,
                "main_angle": camera_main,
                "accent_angle": camera_accent,
                "main_path": str(resolved_sources.main_path),
                "accent_path": str(resolved_sources.accent_path),
                "discovered_paths": [str(path) for path in resolved_sources.discovered_paths],
                "main_clips": [
                    {
                        "path": str(clip.path),
                        "timeline_start": round(clip.timeline_start_s, 3),
                        "timeline_end": None if clip.timeline_end_s is None else round(clip.timeline_end_s, 3),
                        "source_start": round(clip.source_start_s, 3),
                        "source_end": None if clip.source_end_s is None else round(clip.source_end_s, 3),
                    }
                    for clip in resolved_sources.main_clips
                ],
                "accent_clips": [
                    {
                        "path": str(clip.path),
                        "timeline_start": round(clip.timeline_start_s, 3),
                        "timeline_end": None if clip.timeline_end_s is None else round(clip.timeline_end_s, 3),
                        "source_start": round(clip.source_start_s, 3),
                        "source_end": None if clip.source_end_s is None else round(clip.source_end_s, 3),
                    }
                    for clip in resolved_sources.accent_clips
                ],
            }
        )
    if motion_plans is not None:
        for angle, plan_motion in sorted(motion_plans.items()):
            log_prelude_entries.append(
                {
                    "event": "camera_motion_intervals",
                    "angle": angle,
                    "source_path": None if plan_motion.source_path is None else str(plan_motion.source_path),
                    "count": len(plan_motion.moving_intervals),
                    "intervals": [
                        _serialize_motion_interval(interval)
                        for interval in plan_motion.moving_intervals
                    ],
                    "diagnostics": plan_motion.diagnostics,
                }
            )

    patch_started_at = time.perf_counter()
    patch_status = "success"
    n = 0
    try:
        n = patch_prproj(
            Path(in_file),
            cuts,
            seq,
            out,
            first_angle=first_angle,
            log_path=log_path,
            source_offset_s=0.0,
            audio_source_offset_s=0.0,
            prelude_log_entries=log_prelude_entries if log else None,
            fps=fps,
        )
    except Exception:
        patch_status = "failed"
        raise
    finally:
        prproj_patch_s = time.perf_counter() - patch_started_at
        total_runtime_s = time.perf_counter() - command_started_at
        if log_path and log_path.exists():
            _append_jsonl_entry(
                log_path,
                {
                    "event": "timing_summary",
                    "status": patch_status,
                    "motion_enabled": motion_check,
                    "motion_speed": motion_execution_config.speed,
                    "motion_hwaccel_requested": motion_execution_config.hwaccel,
                    "motion_hwaccel_active": None if motion_runtime is None else motion_runtime.active_hwaccel,
                    "motion_cpu_worker_count": 0 if motion_runtime is None else motion_runtime.cpu_worker_count,
                    "motion_gpu_worker_count": 0 if motion_runtime is None else motion_runtime.gpu_worker_count,
                    "motion_cpu_clip_count": 0 if motion_runtime is None else motion_runtime.cpu_clip_count,
                    "motion_gpu_clip_count": 0 if motion_runtime is None else motion_runtime.gpu_clip_count,
                    "audio_load_s": round(audio_load_s, 3),
                    "activity_analysis_s": round(activity_analysis_s, 3),
                    "motion_analysis_s": round(motion_analysis_s, 3),
                    "camera_planning_s": round(camera_planning_s, 3),
                    "prproj_patch_s": round(prproj_patch_s, 3),
                    "premiere_generation_s": round(prproj_patch_s, 3),
                    "total_runtime_s": round(total_runtime_s, 3),
                },
            )
    click.echo(f"Done: {n} segments -> {out_file}")
    click.echo(
        "Timings: "
        f"motion={motion_analysis_s:.1f}s, "
        f"prproj={prproj_patch_s:.1f}s, "
        f"total={total_runtime_s:.1f}s"
    )
    if log_path and log_path.exists():
        click.echo(f"Log: {log_path}")


@cli.command("patch-prproj")
@click.option("--in", "in_file", required=True, type=click.Path(exists=True), help="Input .prproj file")
@click.option("--cuts", required=True, type=click.Path(exists=True), help="JSON file with cut points")
@click.option("--seq", required=True, help="Sequence name")
@click.option("--out", "out_file", required=True, type=click.Path(), help="Output .prproj file")
@click.option("--fps", default=0.0, type=float, help="Sequence fps for frame-aligned cuts (0 = auto-detect)")
def patch_prproj_cmd(in_file, cuts, seq, out_file, fps):
    """Patch multicam angle switches in a Premiere .prproj file."""
    import json

    from .prproj_patcher import patch_prproj

    with open(cuts) as f:
        cuts_data = json.load(f)

    n = patch_prproj(Path(in_file), cuts_data, seq, Path(out_file), fps=fps)
    click.echo(f"Patched: {n} segments -> {out_file}")


@cli.command("gui")
def gui_cmd():
    """Launch the graphical interface (native window)."""
    from autopodcast.gui.main import main
    main()
