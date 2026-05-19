"""Parser for Premiere Pro FCP 7 XML exports."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse


@dataclass
class TrackClip:
    name: str
    file_path: str       # decoded local path
    pathurl: str          # original pathurl from XML
    fps: float
    source_in_frames: int = 0  # <in> from clipitem — trim offset in sequence frames


@dataclass
class ParsedSequence:
    name: str
    fps: float
    duration_frames: int
    width: int
    height: int
    video_tracks: list[TrackClip]
    audio_tracks: list[TrackClip]


def _parse_rate(rate_el: ET.Element | None) -> float:
    """Parse a <rate> element into float fps, handling NTSC drop-frame rates.

    If <ntsc>TRUE</ntsc>, applies the 1000/1001 factor:
      timebase=24  -> 23.976 (24000/1001)
      timebase=30  -> 29.97  (30000/1001)
      timebase=60  -> 59.94  (60000/1001)
    """
    if rate_el is None:
        return 25.0

    tb_el = rate_el.find("timebase")
    timebase = int(tb_el.text) if tb_el is not None and tb_el.text else 25

    ntsc_el = rate_el.find("ntsc")
    is_ntsc = ntsc_el is not None and ntsc_el.text and ntsc_el.text.upper() == "TRUE"

    if is_ntsc:
        return timebase * 1000.0 / 1001.0
    return float(timebase)


def _pathurl_to_local(pathurl: str) -> str:
    """Convert file:// pathurl to a local filesystem path.

    Mac:     file://localhost/Volumes/T7/... -> /Volumes/T7/...
    Windows: file://localhost/D:/...         -> D:\\...
    """
    parsed = urlparse(unquote(pathurl))
    path = parsed.path

    # Windows: /D:/folder/file.mp4 -> D:\folder\file.mp4
    if len(path) >= 3 and path[0] == "/" and path[2] == ":":
        path = path[1:].replace("/", "\\")

    return path


def _extract_track_clip(track_el: ET.Element) -> TrackClip | None:
    """Extract the first clipitem with a file path from a track element."""
    for clipitem in track_el.findall("clipitem"):
        file_el = clipitem.find("file")
        if file_el is None:
            continue

        pathurl_el = file_el.find("pathurl")
        if pathurl_el is None or not pathurl_el.text:
            continue

        name_el = file_el.find("name")
        name = name_el.text if name_el is not None and name_el.text else ""

        pathurl = pathurl_el.text
        file_path = _pathurl_to_local(pathurl)

        clip_fps = _parse_rate(file_el.find("rate"))

        in_el = clipitem.find("in")
        source_in_frames = int(in_el.text) if in_el is not None and in_el.text else 0

        return TrackClip(
            name=name,
            file_path=file_path,
            pathurl=pathurl,
            fps=clip_fps,
            source_in_frames=source_in_frames,
        )

    return None


def parse_premiere_xml(xml_path: Path) -> ParsedSequence:
    """Parse a Premiere Pro FCP 7 XML export.

    Returns a ParsedSequence with video and audio tracks that contain clips.
    Tracks without clipitems are skipped.
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    # Find the sequence element
    sequence = root.find(".//sequence")
    if sequence is None:
        raise ValueError(f"No <sequence> found in {xml_path}")

    # Sequence metadata
    seq_name_el = sequence.find("name")
    seq_name = seq_name_el.text if seq_name_el is not None and seq_name_el.text else ""

    fps = _parse_rate(sequence.find("rate"))

    dur_el = sequence.find("duration")
    duration_frames = int(dur_el.text) if dur_el is not None and dur_el.text else 0

    # Video format
    width, height = 1920, 1080
    vid_sc = sequence.find(".//media/video/format/samplecharacteristics")
    if vid_sc is not None:
        w_el = vid_sc.find("width")
        h_el = vid_sc.find("height")
        if w_el is not None and w_el.text:
            width = int(w_el.text)
        if h_el is not None and h_el.text:
            height = int(h_el.text)

    # Video tracks
    video_tracks: list[TrackClip] = []
    video_section = sequence.find(".//media/video")
    if video_section is not None:
        for track_el in video_section.findall("track"):
            clip = _extract_track_clip(track_el)
            if clip is not None:
                video_tracks.append(clip)

    # Audio tracks
    audio_tracks: list[TrackClip] = []
    audio_section = sequence.find(".//media/audio")
    if audio_section is not None:
        for track_el in audio_section.findall("track"):
            clip = _extract_track_clip(track_el)
            if clip is not None:
                audio_tracks.append(clip)

    return ParsedSequence(
        name=seq_name,
        fps=fps,
        duration_frames=duration_frames,
        width=width,
        height=height,
        video_tracks=video_tracks,
        audio_tracks=audio_tracks,
    )
