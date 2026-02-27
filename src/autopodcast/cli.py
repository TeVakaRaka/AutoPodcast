"""CLI interface for AutoPodcast."""

from __future__ import annotations

from pathlib import Path

import click

from autopodcast.config import build_config
from autopodcast.core.analyzer import analyze_speaker
from autopodcast.core.audio_loader import load_and_align
from autopodcast.core.detector import combine_speakers, detect_activity
from autopodcast.core.ducking import generate_ducking_events
from autopodcast.core.segmenter import segment_timeline
from autopodcast.core.switcher import assign_cameras
from autopodcast.export.json_export import load_timeline, save_timeline
from autopodcast.models.domain import Timeline


@click.group()
def cli():
    """AutoPodcast — automatic rough-cut podcast editing."""
    pass


@cli.command()
@click.option("--audio-a", required=True, type=click.Path(exists=True), help="Path to speaker A audio/video")
@click.option("--label-a", default="host", help="Label for speaker A")
@click.option("--camera-a", default=1, type=int, help="Camera index for speaker A")
@click.option("--audio-b", required=True, type=click.Path(exists=True), help="Path to speaker B audio/video")
@click.option("--label-b", default="guest", help="Label for speaker B")
@click.option("--camera-b", default=2, type=int, help="Camera index for speaker B")
@click.option("--camera-wide", default=0, type=int, help="Camera index for wide/default shot")
@click.option("--output", "-o", default="timeline.json", type=click.Path(), help="Output JSON path")
@click.option("--speech-threshold", default=-28.0, type=float, help="Speech onset threshold in dB")
@click.option("--release-threshold", default=-33.0, type=float, help="Speech release threshold in dB")
@click.option("--hangover", default=600.0, type=float, help="Hangover duration in ms")
@click.option("--min-segment", default=2000.0, type=float, help="Minimum segment length in ms")
@click.option("--debounce", default=300.0, type=float, help="Debounce duration in ms")
@click.option("--ducking/--no-ducking", default=False, help="Enable audio ducking")
@click.option("--ducking-db", default=-12.0, type=float, help="Ducking level in dB")
@click.option("--fps", default=29.97, type=float, help="Timeline frame rate")
def analyze(
    audio_a, label_a, camera_a,
    audio_b, label_b, camera_b,
    camera_wide, output,
    speech_threshold, release_threshold,
    hangover, min_segment, debounce,
    ducking, ducking_db, fps,
):
    """Analyze audio and generate a timeline."""
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
    )

    click.echo(f"Loading audio: {audio_a}, {audio_b}")
    paths = [config.audio_inputs[0].path, config.audio_inputs[1].path]
    audio_arrays = load_and_align(paths, config.sample_rate)
    duration_s = len(audio_arrays[0]) / config.sample_rate

    click.echo(f"Duration: {duration_s:.1f}s")
    click.echo("Analyzing speaker activity...")

    activity_a = analyze_speaker(audio_arrays[0], label_a, config)
    activity_b = analyze_speaker(audio_arrays[1], label_b, config)

    click.echo("Detecting speech...")
    detect_activity(activity_a, config)
    detect_activity(activity_b, config)

    click.echo("Combining speakers...")
    states = combine_speakers(activity_a, activity_b)

    click.echo("Segmenting timeline...")
    segments = segment_timeline(states, config)

    click.echo("Assigning cameras...")
    segments = assign_cameras(segments, config)

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
def export(timeline, cam0, cam1, cam2, mic0, mic1, output, name):
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
