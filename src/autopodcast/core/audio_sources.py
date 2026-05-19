"""Resolve sequence audio source files from Premiere project/XML files."""

from __future__ import annotations

import gzip
import os
import re
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree as ET

from autopodcast.prproj_patcher import (
    _build_obj_map,
    _build_uid_map,
    _find_audio_track_items,
    _find_sequence,
)

AUDIO_MEDIA_EXTENSIONS = {
    ".aac",
    ".aif",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".wav",
    ".wma",
}

VIDEO_MEDIA_EXTENSIONS = {
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

SUPPORTED_MEDIA_EXTENSIONS = AUDIO_MEDIA_EXTENSIONS | VIDEO_MEDIA_EXTENSIONS


@dataclass(frozen=True)
class ResolvedAudioSource:
    track_index: int
    representative_path: Path
    name: str = ""
    paths: tuple[Path, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ResolvedSequenceAudioSources:
    method: str
    sources_by_track: dict[int, ResolvedAudioSource] = field(default_factory=dict)
    discovered_paths: tuple[Path, ...] = field(default_factory=tuple)


def discover_sequence_audio_sources(
    prproj_path: Path,
    seq_name: str,
    *,
    xml_path: str | None = None,
) -> ResolvedSequenceAudioSources:
    """Discover all audio track source files for a sequence."""
    errors: list[str] = []

    if xml_path:
        try:
            return _discover_audio_from_xml(Path(xml_path), seq_name)
        except Exception as exc:
            errors.append(f"XML: {exc}")

    try:
        return _discover_audio_from_prproj(prproj_path, seq_name)
    except Exception as exc:
        errors.append(f".prproj: {exc}")

    details = ""
    if errors:
        details = " Details: " + " | ".join(errors)
    raise ValueError(
        "Could not resolve audio source files automatically. "
        "Use --mic-* manually, or provide --xml for a Premiere FCP7 XML export."
        + details
    )


def resolve_sequence_audio_sources(
    prproj_path: Path,
    seq_name: str,
    audio_tracks: list[int] | tuple[int, ...],
    *,
    xml_path: str | None = None,
) -> ResolvedSequenceAudioSources:
    """Resolve selected 0-based audio track source files."""
    requested_tracks = sorted(set(audio_tracks))
    if not requested_tracks:
        raise ValueError("No audio tracks were requested.")

    discovered = discover_sequence_audio_sources(
        prproj_path,
        seq_name,
        xml_path=xml_path,
    )
    missing = [
        track_idx
        for track_idx in requested_tracks
        if track_idx not in discovered.sources_by_track
    ]
    if missing:
        available = ", ".join(str(idx + 1) for idx in sorted(discovered.sources_by_track))
        missing_text = ", ".join(str(idx + 1) for idx in missing)
        raise ValueError(
            f"Could not resolve audio source for track(s): {missing_text}. "
            f"Available audio tracks: {available or 'none'}."
        )

    selected = {
        track_idx: discovered.sources_by_track[track_idx]
        for track_idx in requested_tracks
    }
    paths = tuple(
        path
        for source in selected.values()
        for path in source.paths
    )
    return ResolvedSequenceAudioSources(
        method=discovered.method,
        sources_by_track=selected,
        discovered_paths=paths,
    )


def _discover_audio_from_prproj(prproj_path: Path, seq_name: str) -> ResolvedSequenceAudioSources:
    with gzip.open(prproj_path, "rb") as f:
        xml_bytes = f.read()
    root = ET.fromstring(xml_bytes.decode("utf-8"))

    obj_map = _build_obj_map(root)
    uid_map = _build_uid_map(root)
    seq = _find_sequence(root, seq_name)
    audio_items = _find_audio_track_items(root, seq, obj_map, uid_map)

    sources: dict[int, ResolvedAudioSource] = {}
    discovered_paths: list[Path] = []
    for track_idx, item, _track, _track_items_el in audio_items:
        seed_elements = [item]
        subclip = _subclip_for_track_item(item, obj_map)
        if subclip is not None:
            seed_elements.append(subclip)
            clip = _clip_for_subclip(subclip, obj_map)
            if clip is not None:
                seed_elements.append(clip)
                source = _source_for_clip(clip, obj_map)
                if source is not None:
                    seed_elements.append(source)

        paths = _collect_referenced_audio_paths(
            seed_elements,
            obj_map,
            uid_map,
            base_dir=prproj_path.parent,
        )
        if not paths:
            continue
        name = _display_name(seed_elements)
        sources[track_idx] = ResolvedAudioSource(
            track_index=track_idx,
            representative_path=paths[0],
            name=name,
            paths=tuple(paths),
        )
        discovered_paths.extend(paths)

    return ResolvedSequenceAudioSources(
        method="prproj",
        sources_by_track=sources,
        discovered_paths=tuple(_dedupe_paths(discovered_paths)),
    )


def _discover_audio_from_xml(xml_path: Path, seq_name: str) -> ResolvedSequenceAudioSources:
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    sequence = _find_xml_sequence(root, seq_name)
    if sequence is None:
        raise ValueError(f"No <sequence> named {seq_name!r} found in {xml_path}")

    audio_section = sequence.find(".//media/audio")
    if audio_section is None:
        return ResolvedSequenceAudioSources(method="xml")

    sources: dict[int, ResolvedAudioSource] = {}
    discovered_paths: list[Path] = []
    for track_idx, track_el in enumerate(audio_section.findall("track")):
        paths: list[Path] = []
        names: list[str] = []
        for clipitem in track_el.findall("clipitem"):
            file_el = clipitem.find("file")
            if file_el is None:
                continue
            pathurl_el = file_el.find("pathurl")
            if pathurl_el is None or not pathurl_el.text:
                continue
            path = _resolve_candidate_path(_pathurl_to_local(pathurl_el.text), xml_path.parent)
            paths.append(path)
            name_el = file_el.find("name")
            if name_el is not None and name_el.text:
                names.append(name_el.text)
        paths = _dedupe_paths(paths)
        if not paths:
            continue
        sources[track_idx] = ResolvedAudioSource(
            track_index=track_idx,
            representative_path=paths[0],
            name=names[0] if names else paths[0].name,
            paths=tuple(paths),
        )
        discovered_paths.extend(paths)

    return ResolvedSequenceAudioSources(
        method="xml",
        sources_by_track=sources,
        discovered_paths=tuple(_dedupe_paths(discovered_paths)),
    )


def _find_xml_sequence(root: ET.Element, seq_name: str) -> ET.Element | None:
    sequences = root.findall(".//sequence")
    if not sequences:
        return None

    exact = [
        sequence
        for sequence in sequences
        if _xml_sequence_name(sequence) == seq_name
    ]
    if exact:
        return exact[0]

    normalized_query = _normalize_name(seq_name)
    normalized = [
        sequence
        for sequence in sequences
        if _normalize_name(_xml_sequence_name(sequence)) == normalized_query
    ]
    if normalized:
        return normalized[0]

    if len(sequences) == 1:
        return sequences[0]
    return None


def _xml_sequence_name(sequence: ET.Element) -> str:
    name_el = sequence.find("name")
    if name_el is not None and name_el.text:
        return name_el.text
    return ""


def _normalize_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKC", name)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.casefold()


def _subclip_for_track_item(item: ET.Element, obj_map: dict[str, ET.Element]) -> ET.Element | None:
    subclip_ref = item.find(".//SubClip")
    if subclip_ref is None or not subclip_ref.get("ObjectRef"):
        return None
    return obj_map.get(subclip_ref.get("ObjectRef"))


def _clip_for_subclip(subclip: ET.Element, obj_map: dict[str, ET.Element]) -> ET.Element | None:
    clip_ref_el = subclip.find("Clip")
    if clip_ref_el is None or not clip_ref_el.get("ObjectRef"):
        return None
    return obj_map.get(clip_ref_el.get("ObjectRef"))


def _source_for_clip(clip_root: ET.Element, obj_map: dict[str, ET.Element]) -> ET.Element | None:
    source_ref = clip_root.find(".//Source")
    if source_ref is None or not source_ref.get("ObjectRef"):
        return None
    return obj_map.get(source_ref.get("ObjectRef"))


def _display_name(elements: list[ET.Element]) -> str:
    for element in elements:
        name_el = element.find("Name")
        if name_el is not None and name_el.text:
            return name_el.text
        object_name = element.get("ObjectName")
        if object_name:
            return object_name
    return ""


def _collect_referenced_audio_paths(
    seed_elements: list[ET.Element],
    obj_map: dict[str, ET.Element],
    uid_map: dict[str, ET.Element],
    *,
    max_depth: int = 8,
    base_dir: Path | None = None,
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

        for candidate in _extract_media_paths_from_element(element):
            candidate = _resolve_candidate_path(candidate, base_dir)
            key = _path_key(candidate)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            results.append(candidate)

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


def _extract_media_paths_from_element(element: ET.Element) -> list[Path]:
    results: list[Path] = []
    for node in element.iter():
        text = (node.text or "").strip()
        if not text:
            continue
        candidate = _normalize_candidate_path(text)
        if candidate is None:
            continue
        if candidate.suffix.lower() not in SUPPORTED_MEDIA_EXTENSIONS:
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


def _pathurl_to_local(pathurl: str) -> Path:
    parsed = urlparse(unquote(pathurl))
    path = parsed.path
    if len(path) >= 3 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    return Path(path.replace("\\", os.sep))


def _looks_like_path(text: str) -> bool:
    lowered = text.casefold()
    if not any(lowered.endswith(ext) for ext in SUPPORTED_MEDIA_EXTENSIONS):
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


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = _path_key(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))
