"""Resolve multicam source clips for monologue motion analysis."""

from __future__ import annotations

import gzip
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree as ET

from autopodcast.core.camera_motion import CameraSourceClip
from autopodcast.prproj_patcher import (
    TICKS_PER_SECOND,
    _build_obj_map,
    _build_uid_map,
    _find_multicam_track_item,
    _find_sequence,
)

VIDEO_EXTENSIONS = {
    ".avi",
    ".braw",
    ".m2ts",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".mts",
    ".mxf",
    ".r3d",
    ".ts",
    ".webm",
    ".wmv",
}


@dataclass(frozen=True)
class ResolvedCameraSource:
    representative_path: Path
    clips: tuple[CameraSourceClip, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ResolvedMonologueSources:
    main_path: Path
    accent_path: Path
    method: str
    discovered_paths: tuple[Path, ...] = field(default_factory=tuple)
    main_clips: tuple[CameraSourceClip, ...] = field(default_factory=tuple)
    accent_clips: tuple[CameraSourceClip, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ResolvedSequenceCameraSources:
    method: str
    discovered_paths: tuple[Path, ...] = field(default_factory=tuple)
    sources_by_angle: dict[int, ResolvedCameraSource] = field(default_factory=dict)


def resolve_sequence_camera_sources(
    prproj_path: Path,
    seq_name: str,
    camera_angles: list[int] | tuple[int, ...],
    *,
    xml_path: str | None = None,
) -> ResolvedSequenceCameraSources:
    """Resolve source clips for arbitrary camera angles in a multicam sequence."""
    requested_angles = sorted(set(camera_angles))
    if not requested_angles:
        raise ValueError("No camera angles were requested.")

    errors: list[str] = []

    if xml_path:
        try:
            track_sources = _parse_xml_video_tracks(Path(xml_path))
            return _build_sequence_sources(
                track_sources,
                requested_angles,
                method="xml",
                base_dir=Path(xml_path).parent,
            )
        except Exception as exc:
            errors.append(f"XML: {exc}")

    try:
        track_sources = _resolve_prproj_track_sources(prproj_path, seq_name)
        return _build_sequence_sources(
            track_sources,
            requested_angles,
            method="prproj",
            base_dir=prproj_path.parent,
        )
    except Exception as exc:
        errors.append(f".prproj: {exc}")

    details = ""
    if errors:
        details = " Details: " + " | ".join(errors)
    raise ValueError(
        "Could not resolve camera source files automatically. "
        "Re-run with an XML export, or disable motion-check for this run."
        + details
    )


def resolve_monologue_camera_sources(
    prproj_path: Path,
    seq_name: str,
    camera_main: int,
    camera_accent: int,
    *,
    camera_main_file: str | None = None,
    camera_accent_file: str | None = None,
    xml_path: str | None = None,
) -> ResolvedMonologueSources:
    """Resolve source clips for the requested camera angles."""
    if (camera_main_file is None) != (camera_accent_file is None):
        raise ValueError(
            "Provide both --camera-main-file and --camera-accent-file together."
        )

    if camera_main_file and camera_accent_file:
        main = Path(camera_main_file).expanduser()
        accent = Path(camera_accent_file).expanduser()
        return ResolvedMonologueSources(
            main_path=main,
            accent_path=accent,
            method="manual",
            discovered_paths=(main, accent),
            main_clips=(CameraSourceClip(path=main),),
            accent_clips=(CameraSourceClip(path=accent),),
        )

    errors: list[str] = []

    if xml_path:
        try:
            resolved = resolve_sequence_camera_sources(
                prproj_path,
                seq_name,
                [camera_main, camera_accent],
                xml_path=xml_path,
            )
            return _resolved_sequence_to_monologue(
                resolved,
                camera_main=camera_main,
                camera_accent=camera_accent,
            )
        except Exception as exc:
            errors.append(f"XML: {exc}")

    try:
        resolved = resolve_sequence_camera_sources(
            prproj_path,
            seq_name,
            [camera_main, camera_accent],
        )
        return _resolved_sequence_to_monologue(
            resolved,
            camera_main=camera_main,
            camera_accent=camera_accent,
        )
    except Exception as exc:
        errors.append(f".prproj: {exc}")

    details = ""
    if errors:
        details = " Details: " + " | ".join(errors)
    raise ValueError(
        "Could not resolve camera source files automatically. "
        "Re-run with --camera-main-file and --camera-accent-file, "
        "or provide --xml for a Premiere FCP7 XML export."
        + details
    )


def _resolved_sequence_to_monologue(
    resolved: ResolvedSequenceCameraSources,
    *,
    camera_main: int,
    camera_accent: int,
) -> ResolvedMonologueSources:
    main = resolved.sources_by_angle[camera_main]
    accent = resolved.sources_by_angle[camera_accent]
    if _path_key(main.representative_path) == _path_key(accent.representative_path):
        raise ValueError(
            "Resolved the same source file for both requested camera angles. "
            "Re-run with --camera-main-file and --camera-accent-file, "
            "or provide --xml with distinct camera sources."
        )
    return ResolvedMonologueSources(
        main_path=main.representative_path,
        accent_path=accent.representative_path,
        method=resolved.method,
        discovered_paths=resolved.discovered_paths,
        main_clips=main.clips,
        accent_clips=accent.clips,
    )


def _resolve_from_xml(
    xml_path: Path,
    camera_main: int,
    camera_accent: int,
) -> ResolvedMonologueSources:
    tracks = _parse_xml_video_tracks(xml_path)
    return _build_resolved_sources(
        tracks,
        camera_main,
        camera_accent,
        method="xml",
        base_dir=xml_path.parent,
    )


def _resolve_from_prproj(
    prproj_path: Path,
    seq_name: str,
    camera_main: int,
    camera_accent: int,
) -> ResolvedMonologueSources:
    with gzip.open(prproj_path, "rb") as f:
        xml_bytes = f.read()
    root = ET.fromstring(xml_bytes.decode("utf-8"))

    obj_map = _build_obj_map(root)
    uid_map = _build_uid_map(root)
    seq = _find_sequence(root, seq_name)
    item, _track = _find_multicam_track_item(root, seq, obj_map, uid_map)

    source = _resolve_multicam_source(item, obj_map)
    track_sources = []
    if source is not None:
        track_sources = _resolve_multicam_video_tracks(
            source,
            obj_map,
            uid_map,
            base_dir=prproj_path.parent,
        )

    if not track_sources:
        clip_ref = None
        subclip = None
        subclip_ref = item.find(".//SubClip")
        if subclip_ref is not None and subclip_ref.get("ObjectRef"):
            subclip = obj_map.get(subclip_ref.get("ObjectRef"))
        if subclip is not None:
            clip = subclip.find("Clip")
            if clip is not None:
                clip_ref = clip.get("ObjectRef")
        orig_clip = obj_map.get(clip_ref) if clip_ref else None
        seed_elements = [element for element in [item, subclip, orig_clip, source] if element is not None]
        discovered = _collect_referenced_video_paths(
            seed_elements,
            obj_map,
            uid_map,
            base_dir=prproj_path.parent,
        )
        track_sources = [
            ResolvedCameraSource(
                representative_path=path,
                clips=(CameraSourceClip(path=path),),
            )
            for path in discovered
        ]

    return _build_resolved_sources(
        track_sources,
        camera_main,
        camera_accent,
        method="prproj",
        base_dir=prproj_path.parent,
    )


def _resolve_prproj_track_sources(
    prproj_path: Path,
    seq_name: str,
) -> list[ResolvedCameraSource] | list[Path]:
    with gzip.open(prproj_path, "rb") as f:
        xml_bytes = f.read()
    root = ET.fromstring(xml_bytes.decode("utf-8"))

    obj_map = _build_obj_map(root)
    uid_map = _build_uid_map(root)
    seq = _find_sequence(root, seq_name)
    item, _track = _find_multicam_track_item(root, seq, obj_map, uid_map)

    source = _resolve_multicam_source(item, obj_map)
    track_sources = []
    if source is not None:
        track_sources = _resolve_multicam_video_tracks(
            source,
            obj_map,
            uid_map,
            base_dir=prproj_path.parent,
        )

    if not track_sources:
        clip_ref = None
        subclip = None
        subclip_ref = item.find(".//SubClip")
        if subclip_ref is not None and subclip_ref.get("ObjectRef"):
            subclip = obj_map.get(subclip_ref.get("ObjectRef"))
        if subclip is not None:
            clip = subclip.find("Clip")
            if clip is not None:
                clip_ref = clip.get("ObjectRef")
        orig_clip = obj_map.get(clip_ref) if clip_ref else None
        seed_elements = [element for element in [item, subclip, orig_clip, source] if element is not None]
        discovered = _collect_referenced_video_paths(
            seed_elements,
            obj_map,
            uid_map,
            base_dir=prproj_path.parent,
        )
        track_sources = discovered

    return track_sources


def _build_resolved_sources(
    discovered: list[ResolvedCameraSource] | list[list[CameraSourceClip]] | list[Path],
    camera_main: int,
    camera_accent: int,
    *,
    method: str,
    base_dir: Path | None = None,
) -> ResolvedMonologueSources:
    track_sources = _normalize_track_sources(discovered, base_dir)
    max_idx = max(camera_main, camera_accent)
    if len(track_sources) <= max_idx:
        raise ValueError(
            f"Found only {len(track_sources)} video source track(s), but angle "
            f"{max_idx + 1} was requested."
        )

    main = track_sources[camera_main]
    accent = track_sources[camera_accent]
    if _path_key(main.representative_path) == _path_key(accent.representative_path):
        raise ValueError(
            "Resolved the same source file for both requested camera angles. "
            "Re-run with --camera-main-file and --camera-accent-file, "
            "or provide --xml with distinct camera sources."
        )

    discovered_paths = tuple(track.representative_path for track in track_sources)
    return ResolvedMonologueSources(
        main_path=main.representative_path,
        accent_path=accent.representative_path,
        method=method,
        discovered_paths=discovered_paths,
        main_clips=main.clips,
        accent_clips=accent.clips,
    )


def _build_sequence_sources(
    discovered: list[ResolvedCameraSource] | list[list[CameraSourceClip]] | list[Path],
    camera_angles: list[int] | tuple[int, ...],
    *,
    method: str,
    base_dir: Path | None = None,
) -> ResolvedSequenceCameraSources:
    track_sources = _normalize_track_sources(discovered, base_dir)
    max_idx = max(camera_angles)
    if len(track_sources) <= max_idx:
        raise ValueError(
            f"Found only {len(track_sources)} video source track(s), but angle "
            f"{max_idx + 1} was requested."
        )
    return ResolvedSequenceCameraSources(
        method=method,
        discovered_paths=tuple(track.representative_path for track in track_sources),
        sources_by_angle={
            angle: track_sources[angle]
            for angle in camera_angles
        },
    )


def _normalize_track_sources(
    discovered: list[ResolvedCameraSource] | list[list[CameraSourceClip]] | list[Path],
    base_dir: Path | None,
) -> list[ResolvedCameraSource]:
    normalized: list[ResolvedCameraSource] = []
    seen_tracks: set[tuple[str, ...]] = set()

    for entry in discovered:
        if isinstance(entry, ResolvedCameraSource):
            clips = list(entry.clips)
        elif isinstance(entry, list):
            clips = list(entry)
        else:
            clips = [CameraSourceClip(path=entry)]

        prepared: list[CameraSourceClip] = []
        for clip in clips:
            resolved_path = _resolve_candidate_path(clip.path, base_dir)
            prepared.append(
                CameraSourceClip(
                    path=resolved_path,
                    timeline_start_s=clip.timeline_start_s,
                    timeline_end_s=clip.timeline_end_s,
                    source_start_s=clip.source_start_s,
                    source_end_s=clip.source_end_s,
                )
            )
        prepared.sort(key=lambda clip: (clip.timeline_start_s, clip.source_start_s, str(clip.path)))
        prepared = _dedupe_clips(prepared)
        if not prepared:
            continue
        track_key = tuple(
            (
                f"{_path_key(clip.path)}|{clip.timeline_start_s:.6f}|"
                f"{-1.0 if clip.timeline_end_s is None else clip.timeline_end_s:.6f}|"
                f"{clip.source_start_s:.6f}|"
                f"{-1.0 if clip.source_end_s is None else clip.source_end_s:.6f}"
            )
            for clip in prepared
        )
        if track_key in seen_tracks:
            continue
        seen_tracks.add(track_key)
        normalized.append(
            ResolvedCameraSource(
                representative_path=prepared[0].path,
                clips=tuple(prepared),
            )
        )

    return normalized


def _dedupe_clips(clips: list[CameraSourceClip]) -> list[CameraSourceClip]:
    seen: set[str] = set()
    result: list[CameraSourceClip] = []
    for clip in clips:
        key = (
            f"{_path_key(clip.path)}|{clip.timeline_start_s:.6f}|"
            f"{-1.0 if clip.timeline_end_s is None else clip.timeline_end_s:.6f}|"
            f"{clip.source_start_s:.6f}|"
            f"{-1.0 if clip.source_end_s is None else clip.source_end_s:.6f}"
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(clip)
    return result


def _resolve_multicam_source(
    item: ET.Element,
    obj_map: dict[str, ET.Element],
) -> ET.Element | None:
    subclip_ref = item.find(".//SubClip")
    if subclip_ref is None or not subclip_ref.get("ObjectRef"):
        return None
    subclip = obj_map.get(subclip_ref.get("ObjectRef"))
    if subclip is None:
        return None
    clip_ref = subclip.find("Clip")
    if clip_ref is None or not clip_ref.get("ObjectRef"):
        return None
    clip = obj_map.get(clip_ref.get("ObjectRef"))
    if clip is None:
        return None
    source_ref = clip.find(".//Source")
    if source_ref is None or not source_ref.get("ObjectRef"):
        return None
    return obj_map.get(source_ref.get("ObjectRef"))


def _resolve_multicam_video_tracks(
    source: ET.Element,
    obj_map: dict[str, ET.Element],
    uid_map: dict[str, ET.Element],
    *,
    base_dir: Path | None,
) -> list[ResolvedCameraSource]:
    track_groups = source.find("TrackGroups")
    if track_groups is None:
        track_groups = _find_first_reachable_track_groups(source, obj_map, uid_map)
    if track_groups is None:
        return []

    vtg = None
    for tg in track_groups:
        second = tg.find("Second")
        if second is None:
            continue
        ref = second.get("ObjectRef")
        if not ref:
            continue
        el = obj_map.get(ref)
        if el is not None and el.tag == "VideoTrackGroup":
            vtg = el
            break
    if vtg is None:
        return []

    tg_inner = vtg.find("TrackGroup")
    if tg_inner is None:
        return []
    tracks_el = tg_inner.find("Tracks")
    if tracks_el is None:
        return []

    result: list[ResolvedCameraSource] = []
    for track_ref_el in sorted(tracks_el, key=lambda el: int(el.get("Index", "0"))):
        track = _resolve_track_ref(track_ref_el, obj_map, uid_map)
        if track is None or track.tag != "VideoClipTrack":
            continue
        clips = _resolve_video_track_clips(track, obj_map, uid_map, base_dir=base_dir)
        if not clips:
            continue
        result.append(
            ResolvedCameraSource(
                representative_path=clips[0].path,
                clips=tuple(clips),
            )
        )
    return result


def _find_first_reachable_track_groups(
    source: ET.Element,
    obj_map: dict[str, ET.Element],
    uid_map: dict[str, ET.Element],
    *,
    max_depth: int = 8,
) -> ET.Element | None:
    queue = deque([(source, 0)])
    seen: set[int] = set()
    while queue:
        element, depth = queue.popleft()
        if id(element) in seen or depth > max_depth:
            continue
        seen.add(id(element))
        track_groups = element.find("TrackGroups")
        if track_groups is not None:
            return track_groups
        for node in element.iter():
            obj_ref = node.get("ObjectRef")
            if obj_ref and obj_ref in obj_map:
                queue.append((obj_map[obj_ref], depth + 1))
            uid_ref = node.get("ObjectURef")
            if uid_ref and uid_ref in uid_map:
                queue.append((uid_map[uid_ref], depth + 1))
    return None


def _resolve_track_ref(
    track_ref_el: ET.Element,
    obj_map: dict[str, ET.Element],
    uid_map: dict[str, ET.Element],
) -> ET.Element | None:
    uid_ref = track_ref_el.get("ObjectURef")
    if uid_ref:
        track = uid_map.get(uid_ref)
        if track is not None:
            return track
    obj_ref = track_ref_el.get("ObjectRef")
    if obj_ref:
        return obj_map.get(obj_ref)
    return None


def _resolve_video_track_clips(
    track: ET.Element,
    obj_map: dict[str, ET.Element],
    uid_map: dict[str, ET.Element],
    *,
    base_dir: Path | None,
) -> list[CameraSourceClip]:
    clip_items = track.find(".//ClipItems")
    if clip_items is None:
        return []
    track_items = clip_items.find("TrackItems")
    if track_items is None:
        return []

    clips: list[CameraSourceClip] = []
    items: list[ET.Element] = []
    for ti_el in track_items:
        ti_ref = ti_el.get("ObjectRef")
        if not ti_ref:
            continue
        item = obj_map.get(ti_ref)
        if item is not None and item.tag == "VideoClipTrackItem":
            items.append(item)

    items.sort(key=lambda item: _read_tick(item, ".//Start"))
    for item in items:
        clip = _resolve_video_track_item_clip(item, obj_map, uid_map, base_dir=base_dir)
        if clip is not None:
            clips.append(clip)
    return clips


def _resolve_video_track_item_clip(
    item: ET.Element,
    obj_map: dict[str, ET.Element],
    uid_map: dict[str, ET.Element],
    *,
    base_dir: Path | None,
) -> CameraSourceClip | None:
    start_s = _read_tick(item, ".//Start") / TICKS_PER_SECOND
    end_s = _read_tick(item, ".//End") / TICKS_PER_SECOND
    if end_s <= start_s:
        return None

    subclip = None
    subclip_ref = item.find(".//SubClip")
    if subclip_ref is not None and subclip_ref.get("ObjectRef"):
        subclip = obj_map.get(subclip_ref.get("ObjectRef"))
    if subclip is None:
        return None

    clip = None
    clip_ref = subclip.find("Clip")
    if clip_ref is not None and clip_ref.get("ObjectRef"):
        clip = obj_map.get(clip_ref.get("ObjectRef"))
    if clip is None:
        return None

    source_in_s = _read_tick(clip, ".//InPoint") / TICKS_PER_SECOND
    raw_out_ticks = _read_tick(clip, ".//OutPoint")
    source_out_s = raw_out_ticks / TICKS_PER_SECOND if raw_out_ticks > 0 else None
    if source_out_s is None or source_out_s <= source_in_s:
        source_out_s = source_in_s + (end_s - start_s)

    source = None
    source_ref = clip.find(".//Source")
    if source_ref is not None and source_ref.get("ObjectRef"):
        source = obj_map.get(source_ref.get("ObjectRef"))

    discovered = _collect_referenced_video_paths(
        [element for element in [item, subclip, clip, source] if element is not None],
        obj_map,
        uid_map,
        base_dir=base_dir,
        max_paths=1,
    )
    if not discovered:
        return None

    return CameraSourceClip(
        path=discovered[0],
        timeline_start_s=start_s,
        timeline_end_s=end_s,
        source_start_s=source_in_s,
        source_end_s=source_out_s,
    )


def _parse_xml_video_tracks(xml_path: Path) -> list[ResolvedCameraSource]:
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    sequence = root.find(".//sequence")
    if sequence is None:
        raise ValueError(f"No <sequence> found in {xml_path}")

    sequence_fps = _parse_rate(sequence.find("rate"))
    video_section = sequence.find(".//media/video")
    if video_section is None:
        return []

    result: list[ResolvedCameraSource] = []
    for track_el in video_section.findall("track"):
        clips: list[CameraSourceClip] = []
        for clipitem in track_el.findall("clipitem"):
            file_el = clipitem.find("file")
            if file_el is None:
                continue
            pathurl_el = file_el.find("pathurl")
            if pathurl_el is None or not pathurl_el.text:
                continue
            path = _pathurl_to_local(pathurl_el.text)
            clip_fps = _parse_rate(file_el.find("rate"))
            start_frames = _read_int_text(clipitem.find("start"))
            end_frames = _read_int_text(clipitem.find("end"))
            in_frames = _read_int_text(clipitem.find("in"))
            out_frames = _read_int_text(clipitem.find("out"))
            if end_frames <= start_frames:
                continue
            source_end_s = (
                out_frames / clip_fps
                if out_frames > in_frames
                else in_frames / clip_fps + (end_frames - start_frames) / sequence_fps
            )
            clips.append(
                CameraSourceClip(
                    path=path,
                    timeline_start_s=start_frames / sequence_fps,
                    timeline_end_s=end_frames / sequence_fps,
                    source_start_s=in_frames / clip_fps,
                    source_end_s=source_end_s,
                )
            )
        if clips:
            result.append(
                ResolvedCameraSource(
                    representative_path=clips[0].path,
                    clips=tuple(clips),
                )
            )
    return result


def _parse_rate(rate_el: ET.Element | None) -> float:
    if rate_el is None:
        return 25.0
    tb_el = rate_el.find("timebase")
    timebase = int(tb_el.text) if tb_el is not None and tb_el.text else 25
    ntsc_el = rate_el.find("ntsc")
    is_ntsc = ntsc_el is not None and ntsc_el.text and ntsc_el.text.upper() == "TRUE"
    if is_ntsc:
        return timebase * 1000.0 / 1001.0
    return float(timebase)


def _pathurl_to_local(pathurl: str) -> Path:
    parsed = urlparse(unquote(pathurl))
    path = parsed.path
    if len(path) >= 3 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    return Path(path.replace("\\", os.sep))


def _read_int_text(el: ET.Element | None) -> int:
    if el is None or not el.text:
        return 0
    return int(el.text)


def _read_tick(element: ET.Element, xpath: str) -> int:
    target = element.find(xpath)
    if target is None or not target.text:
        return 0
    try:
        return int(target.text)
    except ValueError:
        return 0


def _collect_referenced_video_paths(
    seed_elements: list[ET.Element],
    obj_map: dict[str, ET.Element],
    uid_map: dict[str, ET.Element],
    *,
    max_depth: int = 8,
    base_dir: Path | None = None,
    max_paths: int | None = None,
) -> list[Path]:
    queue = deque((element, 0) for element in seed_elements)
    seen: set[int] = set()
    results: list[Path] = []
    seen_paths: set[str] = set()

    while queue:
        element, depth = queue.popleft()
        element_id = id(element)
        if element_id in seen or depth > max_depth:
            continue
        seen.add(element_id)

        for candidate in _extract_video_paths_from_element(element):
            candidate = _resolve_candidate_path(candidate, base_dir)
            key = _path_key(candidate)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            results.append(candidate)
            if max_paths is not None and len(results) >= max_paths:
                return results

        for node in element.iter():
            obj_ref = node.get("ObjectRef")
            if obj_ref:
                target = obj_map.get(obj_ref)
                if target is not None:
                    queue.append((target, depth + 1))
            uid_ref = node.get("ObjectURef")
            if uid_ref:
                target = uid_map.get(uid_ref)
                if target is not None:
                    queue.append((target, depth + 1))

    return results


def _extract_video_paths_from_element(element: ET.Element) -> list[Path]:
    results: list[Path] = []
    for node in element.iter():
        text = (node.text or "").strip()
        if not text:
            continue
        candidate = _normalize_candidate_path(text)
        if candidate is None:
            continue
        if candidate.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        results.append(candidate)
    return results


def _normalize_candidate_path(text: str) -> Path | None:
    if text.startswith("file://"):
        return _pathurl_to_local(text)
    if not _looks_like_path(text):
        return None
    normalized = Path(text.replace("\\", os.sep))
    if not normalized.suffix:
        return None
    return normalized.expanduser()


def _looks_like_path(text: str) -> bool:
    lowered = text.casefold()
    if not any(lowered.endswith(ext) for ext in VIDEO_EXTENSIONS):
        return False
    return (
        "/" in text
        or "\\" in text
        or (len(text) >= 3 and text[1] == ":" and text[2] in {"\\", "/"})
    )


def _resolve_candidate_path(path: Path, base_dir: Path | None) -> Path:
    candidate = path.expanduser()
    if base_dir is not None and not candidate.is_absolute():
        candidate = base_dir / candidate
    return Path(os.path.normpath(str(candidate)))


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))
