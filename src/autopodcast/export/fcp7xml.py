"""FCP 7 XML generator for Premiere Pro import."""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
from pathlib import Path

from autopodcast.models.domain import DuckingEvent, Segment, Timeline


def seconds_to_frames(seconds: float, fps: float) -> int:
    """Convert seconds to frame count."""
    return round(seconds * fps)


def _make_rate_element(parent: ET.Element, fps: float) -> ET.Element:
    """Add <rate> element with timebase and ntsc flag."""
    rate = ET.SubElement(parent, "rate")
    # For 29.97, timebase=30, ntsc=TRUE
    if abs(fps - 29.97) < 0.1:
        ET.SubElement(rate, "timebase").text = "30"
        ET.SubElement(rate, "ntsc").text = "TRUE"
    elif abs(fps - 23.976) < 0.1:
        ET.SubElement(rate, "timebase").text = "24"
        ET.SubElement(rate, "ntsc").text = "TRUE"
    else:
        ET.SubElement(rate, "timebase").text = str(int(round(fps)))
        ET.SubElement(rate, "ntsc").text = "FALSE"
    return rate


def _make_timecode(parent: ET.Element, fps: float) -> ET.Element:
    """Add <timecode> element (start at 01:00:00:00, TV standard)."""
    tc = ET.SubElement(parent, "timecode")
    _make_rate_element(tc, fps)
    ET.SubElement(tc, "string").text = "01:00:00:00"
    ET.SubElement(tc, "frame").text = str(seconds_to_frames(3600.0, fps))
    ET.SubElement(tc, "displayformat").text = "NDF"
    return tc


def _make_pathurl(filepath: str) -> str:
    """Create cross-platform pathurl using Path.as_uri()."""
    return Path(filepath).as_uri()


def _make_file_definition(
    parent: ET.Element,
    file_id: str,
    filepath: str,
    duration_frames: int,
    fps: float,
    media_type: str = "video",
) -> ET.Element:
    """Create a full <file> definition element (first occurrence)."""
    file_el = ET.SubElement(parent, "file", id=file_id)
    ET.SubElement(file_el, "name").text = Path(filepath).name
    ET.SubElement(file_el, "pathurl").text = _make_pathurl(filepath)
    _make_rate_element(file_el, fps)
    ET.SubElement(file_el, "duration").text = str(duration_frames)

    media = ET.SubElement(file_el, "media")
    if media_type == "video":
        video = ET.SubElement(media, "video")
        ET.SubElement(video, "duration").text = str(duration_frames)
    else:
        audio = ET.SubElement(media, "audio")
        ET.SubElement(audio, "duration").text = str(duration_frames)

    return file_el


def _make_file_reference(parent: ET.Element, file_id: str) -> ET.Element:
    """Create a <file id="X"/> back-reference (subsequent occurrences)."""
    return ET.SubElement(parent, "file", id=file_id)


def _make_clipitem(
    parent: ET.Element,
    clip_id: str,
    name: str,
    start_frame: int,
    end_frame: int,
    in_frame: int,
    out_frame: int,
    fps: float,
    file_id: str,
    filepath: str,
    duration_frames: int,
    media_type: str,
    defined_files: set[str],
) -> ET.Element:
    """Create a <clipitem> in a track with file definition or back-reference."""
    clip = ET.SubElement(parent, "clipitem", id=clip_id)
    ET.SubElement(clip, "name").text = name
    ET.SubElement(clip, "duration").text = str(out_frame - in_frame)
    _make_rate_element(clip, fps)
    ET.SubElement(clip, "start").text = str(start_frame)
    ET.SubElement(clip, "end").text = str(end_frame)
    ET.SubElement(clip, "in").text = str(in_frame)
    ET.SubElement(clip, "out").text = str(out_frame)

    if file_id not in defined_files:
        _make_file_definition(
            clip, file_id, filepath, duration_frames, fps, media_type
        )
        defined_files.add(file_id)
    else:
        _make_file_reference(clip, file_id)

    return clip


def generate_fcp7xml(
    timeline: Timeline,
    camera_paths: list[str],
    audio_paths: list[str] | None = None,
    sequence_name: str = "AutoPodcast Rough Cut",
    sequence_width: int = 1920,
    sequence_height: int = 1080,
) -> str:
    """Generate FCP 7 XML string from a Timeline.

    Args:
        timeline: The Timeline with segments and ducking events.
        camera_paths: Absolute paths to camera files, indexed by camera_index.
        audio_paths: Optional absolute paths to audio files for audio tracks.
        sequence_name: Name for the sequence.
        sequence_width: Video width in pixels.
        sequence_height: Video height in pixels.

    Returns:
        XML string ready to write to file.
    """
    fps = timeline.fps
    total_frames = seconds_to_frames(timeline.total_duration_s, fps)

    # Track which file IDs have been fully defined
    defined_files: set[str] = set()

    root = ET.Element("xmeml", version="5")

    sequence = ET.SubElement(root, "sequence")
    ET.SubElement(sequence, "name").text = sequence_name
    ET.SubElement(sequence, "duration").text = str(total_frames)
    _make_rate_element(sequence, fps)

    # Timecode (01:00:00:00 TV standard)
    _make_timecode(sequence, fps)

    media = ET.SubElement(sequence, "media")

    # --- Video section ---
    video_section = ET.SubElement(media, "video")

    # Video format
    vid_format = ET.SubElement(video_section, "format")
    vid_sc = ET.SubElement(vid_format, "samplecharacteristics")
    ET.SubElement(vid_sc, "width").text = str(sequence_width)
    ET.SubElement(vid_sc, "height").text = str(sequence_height)
    ET.SubElement(vid_sc, "pixelaspectratio").text = "Square"
    _make_rate_element(vid_sc, fps)

    # Video track
    video_track = ET.SubElement(video_section, "track")
    ET.SubElement(video_track, "enabled").text = "TRUE"
    ET.SubElement(video_track, "locked").text = "FALSE"

    # Map camera index -> file info
    file_ids: dict[int, str] = {}
    for cam_idx, cam_path in enumerate(camera_paths):
        file_ids[cam_idx] = f"file-cam{cam_idx}"

    # Create clip items for each segment
    for seg_idx, seg in enumerate(timeline.segments):
        cam_idx = seg.camera_index
        if cam_idx >= len(camera_paths):
            cam_idx = 0

        file_id = file_ids.get(cam_idx, file_ids.get(0, "file-cam0"))

        start_frame = seconds_to_frames(seg.start_s, fps)
        end_frame = seconds_to_frames(seg.end_s, fps)

        _make_clipitem(
            video_track,
            clip_id=f"clipitem-v{seg_idx}",
            name=f"Cam{cam_idx} - {seg.speaker_state.value}",
            start_frame=start_frame,
            end_frame=end_frame,
            in_frame=start_frame,
            out_frame=end_frame,
            fps=fps,
            file_id=file_id,
            filepath=camera_paths[cam_idx],
            duration_frames=total_frames,
            media_type="video",
            defined_files=defined_files,
        )

    # --- Audio section ---
    if audio_paths:
        audio_section = ET.SubElement(media, "audio")

        # Audio format
        aud_format = ET.SubElement(audio_section, "format")
        aud_sc = ET.SubElement(aud_format, "samplecharacteristics")
        ET.SubElement(aud_sc, "depth").text = "16"
        ET.SubElement(aud_sc, "samplerate").text = "48000"
        outputs = ET.SubElement(audio_section, "outputs")
        group = ET.SubElement(outputs, "group")
        ET.SubElement(group, "index").text = "1"
        ET.SubElement(group, "numchannels").text = "2"
        downmix = ET.SubElement(group, "downmix").text = "0"
        channel = ET.SubElement(group, "channel")
        ET.SubElement(channel, "index").text = "1"

        for track_idx, audio_path in enumerate(audio_paths):
            audio_track = ET.SubElement(audio_section, "track")
            ET.SubElement(audio_track, "enabled").text = "TRUE"
            ET.SubElement(audio_track, "locked").text = "FALSE"

            file_id = f"file-audio{track_idx}"

            # Single clip spanning entire duration
            clip = _make_clipitem(
                audio_track,
                clip_id=f"clipitem-a{track_idx}",
                name=Path(audio_path).stem,
                start_frame=0,
                end_frame=total_frames,
                in_frame=0,
                out_frame=total_frames,
                fps=fps,
                file_id=file_id,
                filepath=audio_path,
                duration_frames=total_frames,
                media_type="audio",
                defined_files=defined_files,
            )

            # Add ducking keyframes if present
            track_events = [
                e for e in timeline.ducking_events if e.track_index == track_idx
            ]
            if track_events:
                _add_volume_filter(clip, track_events, fps, total_frames)

    # Generate XML string
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")

    buf = io.BytesIO()
    tree.write(buf, encoding="utf-8", xml_declaration=True)
    return buf.getvalue().decode("utf-8")


def _add_volume_filter(
    clip: ET.Element,
    events: list[DuckingEvent],
    fps: float,
    total_frames: int,
) -> None:
    """Add a volume filter with keyframes to a clip element."""
    filt = ET.SubElement(clip, "filter")
    effect = ET.SubElement(filt, "effect")
    ET.SubElement(effect, "name").text = "Audio Levels"
    ET.SubElement(effect, "effectid").text = "audiolevels"
    ET.SubElement(effect, "effecttype").text = "audio"

    param = ET.SubElement(effect, "parameter")
    ET.SubElement(param, "parameterid").text = "level"
    ET.SubElement(param, "name").text = "Level"

    for event in events:
        keyframe = ET.SubElement(param, "keyframe")
        ET.SubElement(keyframe, "when").text = str(
            seconds_to_frames(event.time_s, fps)
        )
        # FCP uses linear scale: 0 dB = 1.0, -12 dB ≈ 0.25
        linear = 10.0 ** (event.target_db / 20.0)
        ET.SubElement(keyframe, "value").text = f"{linear:.4f}"


def save_fcp7xml(
    timeline: Timeline,
    camera_paths: list[str],
    output_path: Path,
    audio_paths: list[str] | None = None,
    sequence_name: str = "AutoPodcast Rough Cut",
    sequence_width: int = 1920,
    sequence_height: int = 1080,
) -> None:
    """Generate and save FCP 7 XML to file."""
    xml_str = generate_fcp7xml(
        timeline, camera_paths, audio_paths, sequence_name,
        sequence_width, sequence_height,
    )
    output_path.write_text(xml_str, encoding="utf-8")
