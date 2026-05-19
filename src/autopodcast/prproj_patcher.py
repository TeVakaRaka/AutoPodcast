"""Patch multicam angle switches directly into a Premiere Pro .prproj file."""

from __future__ import annotations

import copy
import gzip
import json
import re
import unicodedata
import uuid
from pathlib import Path
from xml.etree import ElementTree as ET

TICKS_PER_SECOND = 254016000000


class IncrementalLog:
    """Write JSONL log entries incrementally (flush after each write).

    Also keeps entries in memory for summary building.
    """

    def __init__(self, log_path: Path | None):
        self._path = log_path
        self._file = None
        self._entries: list[dict] = []
        if log_path is not None:
            self._file = open(log_path, "w", encoding="utf-8")

    def write(self, entry: dict):
        self._entries.append(entry)
        if self._file is not None:
            self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._file.flush()

    @property
    def entries(self) -> list[dict]:
        return self._entries

    @property
    def active(self) -> bool:
        return self._file is not None

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


def seconds_to_ticks(s: float) -> int:
    """Convert seconds to Premiere Pro ticks."""
    return round(s * TICKS_PER_SECOND)


# NTSC rates: explicit exact fractions (float→Fraction drift breaks limit_denominator)
_NTSC_RATES = {
    23.976: (24000, 1001),
    29.97:  (30000, 1001),
    59.94:  (60000, 1001),
}


def fps_to_ticks_per_frame(fps: float) -> int:
    """Convert frame rate to ticks per frame.

    Uses exact integer arithmetic for NTSC rates to avoid
    floating-point drift.
    """
    for ntsc_fps, (num, den) in _NTSC_RATES.items():
        if abs(fps - ntsc_fps) < 0.01:
            return TICKS_PER_SECOND * den // num
    return round(TICKS_PER_SECOND / fps)


def snap_ticks_to_frame(ticks: int, tpf: int) -> int:
    """Snap a tick value to the nearest frame boundary."""
    return round(ticks / tpf) * tpf


def read_audio_offsets(
    prproj_path: Path,
    seq_name: str,
) -> dict[int, float]:
    """Read per-audio-track InPoint offsets from a .prproj file.

    Returns:
        Dict {track_index: offset_seconds} where offset is how many seconds
        of source audio to skip (InPoint / TICKS_PER_SECOND).
    """
    with gzip.open(prproj_path, "rb") as f:
        xml_bytes = f.read()
    root = ET.fromstring(xml_bytes.decode("utf-8"))

    obj_map = _build_obj_map(root)
    uid_map = _build_uid_map(root)
    seq = _find_sequence(root, seq_name)

    audio_items = _find_audio_track_items(root, seq, obj_map, uid_map)
    offsets: dict[int, float] = {}
    for track_idx, a_item, _track, _track_items_el in audio_items:
        subclip_ref = a_item.find(".//SubClip").get("ObjectRef")
        subclip = obj_map.get(subclip_ref)
        if subclip is None:
            continue
        clip_ref = subclip.find("Clip").get("ObjectRef")
        clip = obj_map.get(clip_ref)
        if clip is None:
            continue
        in_point_el = clip.find(".//InPoint")
        if in_point_el is not None and in_point_el.text:
            in_point = int(in_point_el.text)
            offsets[track_idx] = in_point / TICKS_PER_SECOND
    return offsets


def segments_to_cuts(segments: list) -> tuple[int, list[dict]]:
    """Convert analyzed Segments to (first_angle, cuts) for patch_prproj.

    Args:
        segments: List of Segment objects with .camera_index and .start_s.

    Returns:
        (first_angle, cuts) where first_angle is the camera for segment 0,
        and cuts contains transition points for segments 1..N (only where
        camera_index actually changes).
    """
    if not segments:
        return (0, [])

    first_angle = segments[0].camera_index
    cuts = []
    prev_angle = first_angle

    for seg in segments[1:]:
        if seg.camera_index != prev_angle:
            cuts.append({"time": seg.start_s, "angle": seg.camera_index})
            prev_angle = seg.camera_index

    return first_angle, cuts


def build_segments(
    orig_start: int,
    orig_end: int,
    orig_angle: int,
    cuts: list[dict],
    tpf: int = 0,
) -> list[tuple[int, int, int]]:
    """Build (start_ticks, end_ticks, angle) segments from cuts.

    Args:
        orig_start: Original clip start in ticks.
        orig_end: Original clip end in ticks.
        orig_angle: Original SelectedTrackIndex.
        cuts: List of {"time": float, "angle": int} dicts.
        tpf: Ticks per frame for frame-snapping (0 = disabled).

    Returns:
        List of (start, end, angle) tuples.
    """
    if not cuts:
        return [(orig_start, orig_end, orig_angle)]

    segments = []
    current_start = orig_start
    current_angle = orig_angle

    for cut in cuts:
        cut_ticks = seconds_to_ticks(cut["time"])
        if tpf > 0:
            cut_ticks = snap_ticks_to_frame(cut_ticks, tpf)
        if cut_ticks <= current_start or cut_ticks >= orig_end:
            continue
        segments.append((current_start, cut_ticks, current_angle))
        current_start = cut_ticks
        current_angle = cut["angle"]

    segments.append((current_start, orig_end, current_angle))

    # Merge sub-frame micro-segments into previous segment.
    if tpf > 0 and len(segments) > 1:
        merged = [segments[0]]
        for seg_s, seg_e, angle in segments[1:]:
            if (seg_e - seg_s) < tpf:
                prev_s, _, prev_angle = merged[-1]
                merged[-1] = (prev_s, seg_e, prev_angle)
            else:
                merged.append((seg_s, seg_e, angle))
        segments = merged

    return segments


def segments_to_audio_intervals(
    segments: list,
    track_index: int,
) -> list[tuple[float, float]]:
    """Convert Segment list to active time intervals for a given audio track.

    Args:
        segments: List of Segment objects with .speaker_state, .start_s, .end_s.
        track_index: 0 for host (SPEAKER_A), 1 for guest (SPEAKER_B).

    Returns:
        Merged list of (start_s, end_s) intervals where the track is active.
    """
    from autopodcast.models.domain import SpeakerState

    active_states = {
        0: {SpeakerState.SPEAKER_A, SpeakerState.BOTH, SpeakerState.SILENCE},
        1: {SpeakerState.SPEAKER_B, SpeakerState.BOTH, SpeakerState.SILENCE},
    }
    states = active_states.get(track_index, set())

    intervals: list[tuple[float, float]] = []
    for seg in segments:
        if seg.speaker_state in states:
            if intervals and abs(intervals[-1][1] - seg.start_s) < 1e-6:
                # Merge adjacent intervals
                intervals[-1] = (intervals[-1][0], seg.end_s)
            else:
                intervals.append((seg.start_s, seg.end_s))

    return intervals


def expand_audio_intervals(
    intervals: list[tuple[float, float]],
    overlap_s: float,
    total_duration_s: float,
    pre_roll_s: float | None = None,
    post_roll_s: float | None = None,
) -> list[tuple[float, float]]:
    """Expand each interval around boundaries, clamp and merge.

    By default this behaves like the legacy symmetric overlap expansion.
    When pre_roll_s/post_roll_s are provided, the start and end can be
    extended asymmetrically. This helps open a speaker mic slightly before
    the detected onset so the first syllable is not clipped.
    """
    if pre_roll_s is None:
        pre_roll_s = overlap_s
    if post_roll_s is None:
        post_roll_s = overlap_s

    if not intervals or (pre_roll_s <= 0 and post_roll_s <= 0):
        return intervals

    expanded: list[tuple[float, float]] = []
    for s, e in intervals:
        new_s = max(0.0, s - pre_roll_s)
        new_e = min(total_duration_s, e + post_roll_s)
        if expanded and new_s <= expanded[-1][1]:
            # Merge with previous
            expanded[-1] = (expanded[-1][0], max(expanded[-1][1], new_e))
        else:
            expanded.append((new_s, new_e))

    return expanded


def _find_audio_track_items(
    root: ET.Element,
    seq: ET.Element,
    obj_map: dict[str, ET.Element],
    uid_map: dict[str, ET.Element],
) -> list[tuple[int, ET.Element, ET.Element, ET.Element]]:
    """Find AudioClipTrackItems in the sequence's AudioTrackGroup.

    Returns:
        List of (track_index, track_item, track, track_items_el) tuples, one per
        audio track that has at least one clip. Ordered by real Premiere track
        index, preserving gaps.
    """
    track_groups = seq.find("TrackGroups")
    if track_groups is None:
        return []

    # Find the AudioTrackGroup
    atg = None
    for tg in track_groups:
        second = tg.find("Second")
        if second is None:
            continue
        ref = second.get("ObjectRef")
        if not ref:
            continue
        el = obj_map.get(ref)
        if el is not None and el.tag == "AudioTrackGroup":
            atg = el
            break

    if atg is None:
        return []

    tg_inner = atg.find("TrackGroup")
    if tg_inner is None:
        return []
    tracks_el = tg_inner.find("Tracks")
    if tracks_el is None:
        return []

    result: list[tuple[int, ET.Element, ET.Element, ET.Element]] = []
    for track_ref_el in sorted(tracks_el, key=lambda e: int(e.get("Index", "0"))):
        track_index = int(track_ref_el.get("Index", "0"))
        track_uid = track_ref_el.get("ObjectURef")
        if not track_uid:
            continue

        track = uid_map.get(track_uid)
        if track is None or track.tag != "AudioClipTrack":
            continue

        clip_items = track.find(".//ClipItems")
        if clip_items is None:
            continue
        track_items = clip_items.find("TrackItems")
        if track_items is None:
            continue

        for ti_el in track_items:
            ti_ref = ti_el.get("ObjectRef")
            if not ti_ref:
                continue
            ti = obj_map.get(ti_ref)
            if ti is not None and ti.tag == "AudioClipTrackItem":
                result.append((track_index, ti, track, track_items))
                break  # one item per track

    return result


def _build_full_segments(
    orig_start: int,
    orig_end: int,
    active_intervals: list[tuple[int, int]],
) -> list[tuple[int, int, bool]]:
    """Build contiguous (start, end, enabled) segments covering [orig_start, orig_end].

    Active intervals are marked enabled=True, gaps between them are enabled=False.
    """
    segments: list[tuple[int, int, bool]] = []
    cursor = orig_start
    for s, e in active_intervals:
        if s > cursor:
            segments.append((cursor, s, False))
        segments.append((s, e, True))
        cursor = e
    if cursor < orig_end:
        segments.append((cursor, orig_end, False))
    return segments


def _set_trackitem_muted(cti_el: ET.Element, muted: bool) -> None:
    """Set IsMuted on a ClipTrackItem element."""
    if muted:
        muted_el = cti_el.find("IsMuted")
        if muted_el is None:
            muted_el = ET.SubElement(cti_el, "IsMuted")
        muted_el.text = "true"
    else:
        muted_el = cti_el.find("IsMuted")
        if muted_el is not None:
            cti_el.remove(muted_el)


def _split_audio_track_item(
    root: ET.Element,
    orig_item: ET.Element,
    track_items_el: ET.Element,
    intervals_ticks: list[tuple[int, int]],
    obj_map: dict[str, ET.Element],
    next_oid: int,
    next_node_id: int,
    inc_log: IncrementalLog | None = None,
    track_idx: int = 0,
    override_source_offset_ticks: int | None = None,
    tpf: int = 0,
) -> tuple[int, int]:
    """Split an AudioClipTrackItem into contiguous enabled/disabled segments.

    All segments cover the full original clip range [orig_start, orig_end].
    Active intervals become enabled clips; gaps become disabled (IsMuted) clips.
    No clips are removed — disabled clips remain on the timeline.

    Returns:
        Updated (next_oid, next_node_id).
    """
    # Read original properties (with defensive checks)
    subclip_el = orig_item.find(".//SubClip")
    if subclip_el is None:
        raise ValueError("Audio track item missing SubClip element")
    subclip_ref = subclip_el.get("ObjectRef")
    orig_subclip = obj_map.get(subclip_ref)
    if orig_subclip is None:
        raise ValueError(f"SubClip ObjectRef={subclip_ref} not found in obj_map")
    clip_el = orig_subclip.find("Clip")
    if clip_el is None:
        raise ValueError("SubClip missing Clip element")
    orig_clip_ref = clip_el.get("ObjectRef")
    orig_clip = obj_map.get(orig_clip_ref)
    if orig_clip is None:
        raise ValueError(f"Clip ObjectRef={orig_clip_ref} not found in obj_map")

    subclip_name = orig_subclip.find("Name").text or "Audio"

    comp_el = orig_item.find(".//Components")
    if comp_el is None:
        raise ValueError("Audio track item missing Components element")
    comp_ref = comp_el.get("ObjectRef")
    orig_comp_chain = obj_map.get(comp_ref)
    if orig_comp_chain is None:
        raise ValueError(f"Components ObjectRef={comp_ref} not found in obj_map")

    # Read original InPoint and timeline bounds
    in_point_el = orig_clip.find(".//InPoint")
    if in_point_el is None:
        raise ValueError("Clip missing InPoint element")
    orig_in_point = int(in_point_el.text)
    start_el = orig_item.find(".//Start")
    if start_el is None:
        raise ValueError("Audio track item missing Start element")
    orig_start = int(start_el.text)
    end_el = orig_item.find(".//End")
    if end_el is None:
        raise ValueError("Audio track item missing End element")
    orig_end = int(end_el.text)

    # Source offset: shift intervals from raw audio time to timeline time
    if override_source_offset_ticks is not None:
        source_offset_ticks = override_source_offset_ticks
    else:
        source_offset_ticks = orig_in_point - orig_start
    if tpf and source_offset_ticks != 0:
        source_offset_ticks = snap_ticks_to_frame(source_offset_ticks, tpf)
    if source_offset_ticks != 0:
        intervals_ticks = [
            (s - source_offset_ticks, e - source_offset_ticks)
            for s, e in intervals_ticks
        ]

    # Clip intervals to [orig_start, orig_end]
    clipped: list[tuple[int, int]] = []
    for s, e in intervals_ticks:
        cs = max(s, orig_start)
        ce = min(e, orig_end)
        if ce > cs:
            clipped.append((cs, ce))
    intervals_ticks = clipped

    # Build full segment list (enabled + disabled) covering entire clip range
    full_segments = _build_full_segments(orig_start, orig_end, intervals_ticks)

    # Merge sub-frame segments into neighbours to prevent
    # InPoint/OutPoint snap collapsing them to zero duration
    if tpf and len(full_segments) > 1:
        merged: list[tuple[int, int, bool]] = [full_segments[0]]
        for seg_s, seg_e, enabled in full_segments[1:]:
            if (seg_e - seg_s) < tpf:
                prev_s, _, prev_en = merged[-1]
                merged[-1] = (prev_s, seg_e, prev_en)
            else:
                merged.append((seg_s, seg_e, enabled))
        full_segments = merged

    if not full_segments:
        # Shouldn't happen if orig_end > orig_start, but be safe
        return next_oid, next_node_id

    cti = orig_item.find("ClipTrackItem")

    # Process segment 0: modify original item in-place
    seg0_start, seg0_end, seg0_enabled = full_segments[0]
    new_src_in_0 = orig_in_point + (seg0_start - orig_start)
    new_src_out_0 = new_src_in_0 + (seg0_end - seg0_start)
    if tpf:
        new_src_in_0 = snap_ticks_to_frame(new_src_in_0, tpf)
        new_src_out_0 = new_src_in_0 + (seg0_end - seg0_start)

    # Invariant checks
    _check_segment_invariants(seg0_start, seg0_end, new_src_in_0, new_src_out_0, 0)

    orig_item.find(".//Start").text = str(seg0_start)
    orig_item.find(".//End").text = str(seg0_end)
    orig_clip.find(".//InPoint").text = str(new_src_in_0)
    orig_clip.find(".//OutPoint").text = str(new_src_out_0)

    # Set IsMuted on original item's ClipTrackItem
    if cti is not None:
        _set_trackitem_muted(cti, not seg0_enabled)

    if inc_log is not None:
        inc_log.write({
            "event": "audio_split",
            "track": track_idx,
            "clip": subclip_name,
            "orig_tl_start": orig_start,
            "orig_tl_end": orig_end,
            "orig_src_in": orig_in_point,
            "seg_idx": 0,
            "new_tl_start": seg0_start,
            "new_tl_end": seg0_end,
            "new_src_in": new_src_in_0,
            "new_src_out": new_src_out_0,
            "enabled": seg0_enabled,
        })

    # Create new elements for segments 1..N via deep copy
    for seg_idx, (seg_start, seg_end, enabled) in enumerate(full_segments[1:], 1):
        new_src_in = orig_in_point + (seg_start - orig_start)
        new_src_out = new_src_in + (seg_end - seg_start)
        if tpf:
            new_src_in = snap_ticks_to_frame(new_src_in, tpf)
            new_src_out = new_src_in + (seg_end - seg_start)

        # Invariant checks
        _check_segment_invariants(seg_start, seg_end, new_src_in, new_src_out, seg_idx)

        # Allocate ObjectIDs
        comp_chain_oid = str(next_oid); next_oid += 1
        subclip_oid = str(next_oid); next_oid += 1
        audio_clip_oid = str(next_oid); next_oid += 1
        track_item_oid = str(next_oid); next_oid += 1

        node_id = next_node_id; next_node_id += 1

        # Deep copy AudioComponentChain — preserve all fields, change ObjectID
        new_comp = copy.deepcopy(orig_comp_chain)
        new_comp.set("ObjectID", comp_chain_oid)
        root.append(new_comp)

        # Deep copy AudioClip — change ObjectID, ClipID, InPoint, OutPoint
        new_clip = copy.deepcopy(orig_clip)
        new_clip.set("ObjectID", audio_clip_oid)
        clip_el = new_clip.find("Clip") or new_clip.find(".//Clip")
        if clip_el is None:
            raise ValueError(f"Audio seg {seg_idx}: deep copy missing Clip in AudioClip")
        clip_id_el = clip_el.find("ClipID") or clip_el.find(".//ClipID")
        if clip_id_el is not None:
            clip_id_el.text = str(uuid.uuid4())
        in_el = clip_el.find("InPoint") or clip_el.find(".//InPoint")
        if in_el is None:
            raise ValueError(f"Audio seg {seg_idx}: missing InPoint in AudioClip")
        in_el.text = str(new_src_in)
        out_el = clip_el.find("OutPoint") or clip_el.find(".//OutPoint")
        if out_el is None:
            raise ValueError(f"Audio seg {seg_idx}: missing OutPoint in AudioClip")
        out_el.text = str(new_src_out)
        root.append(new_clip)

        # Deep copy SubClip — change ObjectID, Clip/@ObjectRef
        new_subclip = copy.deepcopy(orig_subclip)
        new_subclip.set("ObjectID", subclip_oid)
        sc_clip_el = new_subclip.find("Clip")
        if sc_clip_el is None:
            raise ValueError(f"Audio seg {seg_idx}: deep copy missing Clip in SubClip")
        sc_clip_el.set("ObjectRef", audio_clip_oid)
        root.append(new_subclip)

        # Deep copy AudioClipTrackItem — change ObjectID and internal refs
        new_item = copy.deepcopy(orig_item)
        new_item.set("ObjectID", track_item_oid)

        # Use descendant search (.//...) for robustness across Premiere versions
        # Components ref
        comp_found = new_item.find(".//Components")
        if comp_found is None:
            raise ValueError(f"Audio seg {seg_idx}: missing Components in deep copy")
        comp_found.set("ObjectRef", comp_chain_oid)

        # Node/ID — create if missing (needed for unique node identification)
        node_el = new_item.find(".//Node")
        if node_el is not None:
            id_el = node_el.find("ID")
            if id_el is None:
                id_el = ET.SubElement(node_el, "ID")
            id_el.text = str(node_id)

        # Start/End — must exist (same descendant search as seg_idx 0 uses)
        start_el = new_item.find(".//Start")
        if start_el is None:
            raise ValueError(f"Audio seg {seg_idx}: missing Start in deep copy")
        start_el.text = str(seg_start)
        end_el = new_item.find(".//End")
        if end_el is None:
            raise ValueError(f"Audio seg {seg_idx}: missing End in deep copy")
        end_el.text = str(seg_end)

        # SubClip ref
        sc_el = new_item.find(".//SubClip")
        if sc_el is None:
            raise ValueError(f"Audio seg {seg_idx}: missing SubClip in deep copy")
        sc_el.set("ObjectRef", subclip_oid)

        # IsMuted — find ClipTrackItem for _set_trackitem_muted
        new_cti = new_item.find("ClipTrackItem") or new_item.find(".//ClipTrackItem")
        if new_cti is not None:
            _set_trackitem_muted(new_cti, not enabled)

        root.append(new_item)

        # Add TrackItem ref to the track
        ti_ref = ET.SubElement(track_items_el, "TrackItem")
        ti_ref.set("Index", str(seg_idx))
        ti_ref.set("ObjectRef", track_item_oid)

        if inc_log is not None:
            inc_log.write({
                "event": "audio_split",
                "track": track_idx,
                "clip": subclip_name,
                "orig_tl_start": orig_start,
                "orig_tl_end": orig_end,
                "orig_src_in": orig_in_point,
                "seg_idx": seg_idx,
                "new_tl_start": seg_start,
                "new_tl_end": seg_end,
                "new_src_in": new_src_in,
                "new_src_out": new_src_out,
                "enabled": enabled,
            })

    return next_oid, next_node_id


def _check_segment_invariants(
    tl_start: int, tl_end: int, src_in: int, src_out: int, seg_idx: int,
) -> None:
    """Validate InPoint/OutPoint invariants. Raises RuntimeError on failure."""
    duration = tl_end - tl_start
    if duration <= 0:
        raise RuntimeError(
            f"Audio segment {seg_idx}: zero-length segment "
            f"(tl_start={tl_start}, tl_end={tl_end})"
        )
    src_duration = src_out - src_in
    if abs(src_duration - duration) > 1:
        raise RuntimeError(
            f"Audio segment {seg_idx}: duration mismatch "
            f"src={src_duration} tl={duration}"
        )
    if src_in < 0:
        raise RuntimeError(
            f"Audio segment {seg_idx}: negative InPoint {src_in}"
        )


def _find_sequence(root: ET.Element, seq_name: str) -> ET.Element:
    """Find a Sequence element by name."""
    def _direct_name(el: ET.Element) -> str | None:
        name_el = el.find("Name")
        if name_el is not None and name_el.text:
            return name_el.text
        object_name = el.get("ObjectName")
        if object_name:
            return object_name
        return None

    def _normalize(name: str) -> str:
        normalized = unicodedata.normalize("NFKC", name)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized.casefold()

    obj_map = {
        el.get("ObjectID"): el
        for el in root.iter()
        if el.get("ObjectID")
    }
    candidates: list[tuple[str, ET.Element]] = []

    def _add_candidate(name: str | None, seq: ET.Element) -> None:
        if not name:
            return
        candidates.append((name, seq))

    for seq in root.iter("Sequence"):
        _add_candidate(_direct_name(seq), seq)

    # Some project variants keep the display name on a wrapper element and
    # reference the actual sequence via ObjectRef.
    for el in root.iter():
        wrapper_name = _direct_name(el)
        if not wrapper_name:
            continue
        for child in el:
            if child.tag != "Sequence":
                continue
            ref = child.get("ObjectRef")
            target = obj_map.get(ref) if ref else child
            if target is not None and target.tag == "Sequence":
                _add_candidate(wrapper_name, target)

    def _dedupe(matches: list[tuple[str, ET.Element]]) -> list[ET.Element]:
        unique: list[ET.Element] = []
        seen_ids: set[int] = set()
        for _name, seq in matches:
            seq_id = id(seq)
            if seq_id in seen_ids:
                continue
            seen_ids.add(seq_id)
            unique.append(seq)
        return unique

    exact = _dedupe([(name, seq) for name, seq in candidates if name == seq_name])
    if exact:
        return exact[0]

    normalized_query = _normalize(seq_name)
    normalized = _dedupe([
        (name, seq)
        for name, seq in candidates
        if _normalize(name) == normalized_query
    ])
    if normalized:
        return normalized[0]

    available = sorted({name for name, _seq in candidates if name})
    if available:
        preview = ", ".join(repr(name) for name in available[:20])
        if len(available) > 20:
            preview += f", ... (+{len(available) - 20} more)"
        raise ValueError(
            f"Sequence '{seq_name}' not found in project. "
            f"Available sequences: {preview}"
        )
    raise ValueError(
        f"Sequence '{seq_name}' not found in project. No sequences were found."
    )


def _read_sequence_tpf(seq: ET.Element, obj_map: dict) -> int:
    """Read ticks-per-frame from Sequence's VideoTrackGroup.

    Returns 0 if not found (frame-snapping disabled).
    """
    track_groups = seq.find("TrackGroups")
    if track_groups is None:
        return 0
    for tg in track_groups:
        second = tg.find("Second")
        if second is None:
            continue
        ref = second.get("ObjectRef")
        if not ref:
            continue
        el = obj_map.get(ref)
        if el is not None and el.tag == "VideoTrackGroup":
            tg_inner = el.find("TrackGroup")
            if tg_inner is not None:
                fr = tg_inner.find("FrameRate")
                if fr is not None and fr.text:
                    return int(fr.text)
    return 0


def _build_obj_map(root: ET.Element) -> dict[str, ET.Element]:
    """Build ObjectID -> Element mapping for root-level elements."""
    obj_map = {}
    for el in root:
        oid = el.get("ObjectID")
        if oid:
            obj_map[oid] = el
    return obj_map


def _build_uid_map(root: ET.Element) -> dict[str, ET.Element]:
    """Build ObjectUID -> Element mapping for root-level elements."""
    uid_map = {}
    for el in root:
        uid = el.get("ObjectUID")
        if uid:
            uid_map[uid] = el
    return uid_map


def _find_multicam_track_item(
    root: ET.Element,
    seq: ET.Element,
    obj_map: dict[str, ET.Element],
    uid_map: dict[str, ET.Element],
) -> tuple[ET.Element, ET.Element]:
    """Find the multicam VideoClipTrackItem and its parent VideoClipTrack.

    Returns:
        (track_item_element, track_element) tuple.
    """
    # TrackGroups contains <TrackGroup> children with <Second ObjectRef="ID"/>
    # The referenced objects are VideoTrackGroup, AudioTrackGroup, etc.
    track_groups = seq.find("TrackGroups")
    if track_groups is None:
        raise ValueError("No TrackGroups in sequence")

    # Find the VideoTrackGroup
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
        raise ValueError("No VideoTrackGroup found in sequence")

    # VideoTrackGroup -> TrackGroup -> Tracks -> Track[ObjectURef=...]
    tg_inner = vtg.find("TrackGroup")
    if tg_inner is None:
        raise ValueError("No TrackGroup in VideoTrackGroup")
    tracks_el = tg_inner.find("Tracks")
    if tracks_el is None:
        raise ValueError("No Tracks in VideoTrackGroup")

    for track_ref_el in tracks_el:
        track_uid = track_ref_el.get("ObjectURef")
        if not track_uid:
            continue

        track = uid_map.get(track_uid)
        if track is None or track.tag != "VideoClipTrack":
            continue

        # Check TrackItems in this track
        clip_items = track.find(".//ClipItems")
        if clip_items is None:
            continue

        track_items = clip_items.find("TrackItems")
        if track_items is None:
            continue

        for ti_el in track_items:
            ti_ref = ti_el.get("ObjectRef")
            if not ti_ref:
                continue

            ti = obj_map.get(ti_ref)
            if ti is None or ti.tag != "VideoClipTrackItem":
                continue

            # Check if this is a multicam item
            subclip_el = ti.find(".//SubClip")
            if subclip_el is None:
                continue

            subclip_ref = subclip_el.get("ObjectRef")
            if not subclip_ref:
                continue

            subclip = obj_map.get(subclip_ref)
            if subclip is None:
                continue

            clip_el = subclip.find("Clip")
            if clip_el is None:
                continue

            clip_ref = clip_el.get("ObjectRef")
            if not clip_ref:
                continue

            clip = obj_map.get(clip_ref)
            if clip is None:
                continue

            is_mc = clip.find(".//IsMulticam")
            if is_mc is not None and is_mc.text == "true":
                return ti, track

    raise ValueError("No multicam VideoClipTrackItem found in sequence")


def _max_object_id(root: ET.Element) -> int:
    """Find the maximum ObjectID across all root-level elements."""
    max_id = 0
    for el in root:
        oid = el.get("ObjectID")
        if oid:
            try:
                max_id = max(max_id, int(oid))
            except ValueError:
                pass
    return max_id


def _get_next_id(root: ET.Element) -> int:
    """Get the current NextID from the Project element."""
    for el in root:
        if el.tag == "Project" and el.get("ObjectID"):
            next_id_el = el.find("NextID")
            if next_id_el is not None:
                return int(next_id_el.text)
    raise ValueError("NextID not found in Project")


def _set_next_id(root: ET.Element, value: int) -> None:
    """Set the NextID in the Project element."""
    for el in root:
        if el.tag == "Project" and el.get("ObjectID"):
            next_id_el = el.find("NextID")
            if next_id_el is not None:
                next_id_el.text = str(value)
                return
    raise ValueError("NextID not found in Project")


def patch_prproj(
    in_path: Path,
    cuts: list[dict],
    seq_name: str,
    out_path: Path,
    first_angle: int | None = None,
    audio_segments: list | None = None,
    log_path: Path | None = None,
    audio_track_map: dict[int, int] | None = None,
    source_offset_s: float | None = None,
    audio_source_offset_s: float | None = None,
    audio_overlap_s: float = 0.0,
    audio_pre_roll_s: float | None = None,
    audio_post_roll_s: float | None = None,
    audio_track_intervals_s: dict[int, list[tuple[float, float]]] | None = None,
    prelude_log_entries: list[dict] | None = None,
    fps: float = 0.0,
) -> int:
    """Patch multicam angle switches into a .prproj file.

    Args:
        in_path: Path to input .prproj file.
        cuts: List of {"time": float, "angle": int} dicts.
        seq_name: Name of the sequence to patch.
        out_path: Path to output .prproj file.
        first_angle: If set, override the original clip's SelectedTrackIndex
            for the first segment. Useful when analysis determines a different
            starting camera than what's in the .prproj.
        audio_segments: If set, list of Segment objects used to split audio
            clips — disabling inactive speakers via IsMuted on timeline.
        log_path: If set, write JSONL log file with split details.
        audio_track_map: If set, mapping {audio_track_index: speaker_index}
            where speaker_index 0 = host (SPEAKER_A), 1 = guest (SPEAKER_B).
            Needed when audio track order doesn't match speaker order
            (e.g. --camera-host 2 --camera-guest 1 swaps the angles).
        source_offset_s: If set, explicit offset (seconds) between raw audio
            time and timeline time for video cuts. None = auto-detect from
            InPoint (backward compat for from-xml pipeline). Use 0.0 when
            mic WAV files are independent recordings (auto-multicam pipeline).
        audio_source_offset_s: Same as source_offset_s but for audio mute
            splits. None = auto-detect from each audio clip's InPoint.
        audio_overlap_s: Expand each audio active interval by this many
            seconds on each side to create crossfade overlap (eliminates clicks).
        audio_pre_roll_s: Optional extra expansion before each active interval.
            If None, audio_overlap_s is used for backward compatibility.
        audio_post_roll_s: Optional extra expansion after each active interval.
            If None, audio_overlap_s is used for backward compatibility.
        audio_track_intervals_s: Optional explicit per-track active intervals in
            seconds. When provided, these intervals are used directly instead
            of deriving activity from audio_segments + audio_track_map.
        prelude_log_entries: Optional JSONL entries to write before patching
            details, useful for logging camera planning diagnostics.

    Returns:
        Number of segments created.
    """
    # 1. Decompress and parse
    with gzip.open(in_path, "rb") as f:
        xml_bytes = f.read()
    xml_str = xml_bytes.decode("utf-8")
    root = ET.fromstring(xml_str)

    inc_log = IncrementalLog(log_path)
    try:
        if prelude_log_entries:
            for entry in prelude_log_entries:
                inc_log.write(entry)
        return _patch_prproj_inner(
            root, cuts, seq_name, out_path, first_angle,
            audio_segments, inc_log, audio_track_map,
            source_offset_s, audio_source_offset_s,
            audio_overlap_s, audio_pre_roll_s, audio_post_roll_s,
            audio_track_intervals_s, fps,
        )
    finally:
        inc_log.close()


def _patch_prproj_inner(
    root: ET.Element,
    cuts: list[dict],
    seq_name: str,
    out_path: Path,
    first_angle: int | None,
    audio_segments: list | None,
    inc_log: IncrementalLog,
    audio_track_map: dict[int, int] | None,
    source_offset_s: float | None,
    audio_source_offset_s: float | None,
    audio_overlap_s: float = 0.0,
    audio_pre_roll_s: float | None = None,
    audio_post_roll_s: float | None = None,
    audio_track_intervals_s: dict[int, list[tuple[float, float]]] | None = None,
    fps: float = 0.0,
) -> int:
    """Inner implementation of patch_prproj (separated for try/finally)."""
    # 2. Build object maps
    obj_map = _build_obj_map(root)
    uid_map = _build_uid_map(root)

    # 3. Find sequence
    seq = _find_sequence(root, seq_name)
    inc_log.write({"event": "sequence_found", "name": seq_name})

    # Frame-snapping: auto-detect from .prproj, CLI fps overrides
    if fps > 0:
        tpf = fps_to_ticks_per_frame(fps)
    else:
        tpf = _read_sequence_tpf(seq, obj_map)

    # 4. Find multicam track item and its track
    orig_item, track = _find_multicam_track_item(root, seq, obj_map, uid_map)

    # 5. Read original properties (with defensive checks)
    start_el = orig_item.find(".//Start")
    if start_el is None:
        raise ValueError("Multicam track item missing Start element")
    orig_start = int(start_el.text)
    end_el = orig_item.find(".//End")
    if end_el is None:
        raise ValueError("Multicam track item missing End element")
    orig_end = int(end_el.text)
    inc_log.write({"event": "multicam_found", "orig_start": orig_start, "orig_end": orig_end})

    subclip_el = orig_item.find(".//SubClip")
    if subclip_el is None:
        raise ValueError("Multicam track item missing SubClip element")
    subclip_ref = subclip_el.get("ObjectRef")
    orig_subclip = obj_map.get(subclip_ref)
    if orig_subclip is None:
        raise ValueError(f"SubClip ObjectRef={subclip_ref} not found in obj_map")
    clip_el = orig_subclip.find("Clip")
    if clip_el is None:
        raise ValueError("SubClip missing Clip element")
    orig_clip_ref = clip_el.get("ObjectRef")
    orig_clip = obj_map.get(orig_clip_ref)
    if orig_clip is None:
        raise ValueError(f"Clip ObjectRef={orig_clip_ref} not found in obj_map")

    sel_track_el = orig_clip.find(".//SelectedTrackIndex")
    if sel_track_el is None:
        raise ValueError("Clip missing SelectedTrackIndex element")
    orig_angle = int(sel_track_el.text)
    if first_angle is not None:
        orig_angle = first_angle
    source_el = orig_clip.find(".//Source")
    if source_el is None:
        raise ValueError("Clip missing Source element")
    source_ref = source_el.get("ObjectRef")
    label_el = orig_clip.find(".//asl.clip.label.name")
    label_text = label_el.text if label_el is not None else "BE.Prefs.LabelColors.5"
    master_clip_el = orig_subclip.find("MasterClip")
    if master_clip_el is None:
        raise ValueError("SubClip missing MasterClip element")
    master_clip_uref = master_clip_el.get("ObjectURef")
    subclip_name = orig_subclip.find("Name").text or "Многокам"
    frame_rect = orig_item.find("FrameRect").text
    pixel_ar = orig_item.find("PixelAspectRatio")
    pixel_ar_text = pixel_ar.text if pixel_ar is not None else "1,1"

    comp_el = orig_item.find(".//Components")
    if comp_el is None:
        raise ValueError("Multicam track item missing Components element")
    comp_ref = comp_el.get("ObjectRef")
    orig_comp_chain = obj_map.get(comp_ref)
    if orig_comp_chain is None:
        raise ValueError(f"Components ObjectRef={comp_ref} not found in obj_map")

    # Read original InPoint BEFORE modification — needed for source offset
    in_point_el = orig_clip.find(".//InPoint")
    if in_point_el is None:
        raise ValueError("Clip missing InPoint element")
    orig_in_point = int(in_point_el.text)

    # Source offset: difference between source position and timeline position.
    # When InPoint != Start, audio analysis times (raw WAV) need adjustment.
    # cut_time is in raw audio seconds; we shift by _offset to get
    # timeline-relative seconds.
    if source_offset_s is None:
        _offset = (orig_in_point - orig_start) / TICKS_PER_SECOND
    else:
        _offset = source_offset_s

    # Adjust cuts for source offset
    adjusted_cuts = [
        {"time": c["time"] - _offset, "angle": c["angle"]}
        for c in cuts
    ]

    # 6. Build segments
    segments = build_segments(orig_start, orig_end, orig_angle, adjusted_cuts, tpf=tpf)
    has_video_cuts = len(segments) > 1
    inc_log.write({
        "event": "video_segments",
        "count": len(segments),
        "segments": [
            {
                "start": round((seg_start - orig_start) / TICKS_PER_SECOND, 3),
                "end": round((seg_end - orig_start) / TICKS_PER_SECOND, 3),
                "angle": angle,
            }
            for seg_start, seg_end, angle in segments
        ],
    })

    if not has_video_cuts and audio_segments is None and not audio_track_intervals_s:
        # No cuts and no audio mute — just copy
        with gzip.open(out_path, "wb") as f:
            f.write(ET.tostring(root, encoding="unicode", xml_declaration=True).encode("utf-8"))
        return len(segments)

    next_oid = _max_object_id(root) + 1
    next_node_id = _get_next_id(root)

    if has_video_cuts:
        # 7. Modify original TrackItem (segment 0)
        seg0_start, seg0_end, seg0_angle = segments[0]
        orig_item.find(".//Start").text = str(seg0_start)
        orig_item.find(".//End").text = str(seg0_end)
        in_point = orig_in_point + (seg0_start - orig_start)
        out_point = orig_in_point + (seg0_end - orig_start)
        if tpf:
            in_point = snap_ticks_to_frame(in_point, tpf)
            out_point = in_point + (seg0_end - seg0_start)
        orig_clip.find(".//InPoint").text = str(in_point)
        orig_clip.find(".//OutPoint").text = str(out_point)
        orig_clip.find(".//SelectedTrackIndex").text = str(seg0_angle)

        # 8. Create new elements for segments 1..N
        # Find TrackItems element to add new refs
        clip_items = track.find(".//ClipItems")
        track_items_el = clip_items.find("TrackItems")

        # ClassIDs from originals
        item_class_id = orig_item.get("ClassID")
        item_version = orig_item.get("Version")
        clip_class_id = orig_clip.get("ClassID")
        clip_version = orig_clip.get("Version")
        subclip_class_id = orig_subclip.get("ClassID")
        subclip_version = orig_subclip.get("Version")
        comp_class_id = orig_comp_chain.get("ClassID")
        comp_version = orig_comp_chain.get("Version")

        # ClipTrackItem version
        cti = orig_item.find("ClipTrackItem")
        cti_version = cti.get("Version") if cti is not None else "8"
        co = cti.find("ComponentOwner") if cti is not None else None
        co_version = co.get("Version") if co is not None else "1"
        ti_inner = cti.find("TrackItem") if cti is not None else None
        ti_version = ti_inner.get("Version") if ti_inner is not None else "3"
        node_in_ti = ti_inner.find("Node") if ti_inner is not None else None
        node_version = node_in_ti.get("Version") if node_in_ti is not None else "1"

        # Clip version inside VideoClip
        clip_inner = orig_clip.find("Clip")
        clip_inner_version = clip_inner.get("Version") if clip_inner is not None else "18"
        clip_node = clip_inner.find("Node") if clip_inner is not None else None
        clip_node_version = clip_node.get("Version") if clip_node is not None else "1"
        clip_props = clip_node.find("Properties") if clip_node is not None else None
        clip_props_version = clip_props.get("Version") if clip_props is not None else "1"

        # ComponentChain inner version
        cc_inner = orig_comp_chain.find("ComponentChain")
        cc_inner_version = cc_inner.get("Version") if cc_inner is not None else "3"
        cc_node = cc_inner.find("Node") if cc_inner is not None else None
        cc_node_version = cc_node.get("Version") if cc_node is not None else "1"
        cc_props = cc_node.find("Properties") if cc_node is not None else None
        cc_props_version = cc_props.get("Version") if cc_props is not None else "1"

        for seg_idx, (seg_start, seg_end, seg_angle) in enumerate(segments[1:], 1):
            # Allocate ObjectIDs
            comp_chain_oid = str(next_oid)
            next_oid += 1
            subclip_oid = str(next_oid)
            next_oid += 1
            video_clip_oid = str(next_oid)
            next_oid += 1
            track_item_oid = str(next_oid)
            next_oid += 1

            node_id = next_node_id
            next_node_id += 1

            # Create VideoComponentChain (template copy)
            vcc = ET.SubElement(root, "VideoComponentChain")
            vcc.set("ObjectID", comp_chain_oid)
            vcc.set("ClassID", comp_class_id)
            vcc.set("Version", comp_version)

            dm = ET.SubElement(vcc, "DefaultMotion")
            dm.text = "true"
            do = ET.SubElement(vcc, "DefaultOpacity")
            do.text = "true"
            dmc = ET.SubElement(vcc, "DefaultMotionComponentID")
            dmc.text = "1"
            doc = ET.SubElement(vcc, "DefaultOpacityComponentID")
            doc.text = "2"

            cc = ET.SubElement(vcc, "ComponentChain")
            cc.set("Version", cc_inner_version)
            cc_n = ET.SubElement(cc, "Node")
            cc_n.set("Version", cc_node_version)
            cc_p = ET.SubElement(cc_n, "Properties")
            cc_p.set("Version", cc_props_version)
            acid = ET.SubElement(cc_p, "MZ.ComponentChain.ActiveComponentID")
            acid.text = "2"
            acpi = ET.SubElement(cc_p, "MZ.ComponentChain.ActiveComponentParamIndex")
            acpi.text = "4294967295"

            # Create VideoClip
            vc = ET.SubElement(root, "VideoClip")
            vc.set("ObjectID", video_clip_oid)
            vc.set("ClassID", clip_class_id)
            vc.set("Version", clip_version)

            clip_el = ET.SubElement(vc, "Clip")
            clip_el.set("Version", clip_inner_version)
            c_node = ET.SubElement(clip_el, "Node")
            c_node.set("Version", clip_node_version)
            c_props = ET.SubElement(c_node, "Properties")
            c_props.set("Version", clip_props_version)
            lbl = ET.SubElement(c_props, "asl.clip.label.name")
            lbl.text = label_text

            src = ET.SubElement(clip_el, "Source")
            src.set("ObjectRef", source_ref)
            cid = ET.SubElement(clip_el, "ClipID")
            cid.text = str(uuid.uuid4())
            seg_in_point = orig_in_point + (seg_start - orig_start)
            seg_out_point = orig_in_point + (seg_end - orig_start)
            if tpf:
                seg_in_point = snap_ticks_to_frame(seg_in_point, tpf)
                seg_out_point = seg_in_point + (seg_end - seg_start)
            inp = ET.SubElement(clip_el, "InPoint")
            inp.text = str(seg_in_point)
            outp = ET.SubElement(clip_el, "OutPoint")
            outp.text = str(seg_out_point)
            imc = ET.SubElement(clip_el, "IsMulticam")
            imc.text = "true"
            sti = ET.SubElement(clip_el, "SelectedTrackIndex")
            sti.text = str(seg_angle)

            # Create SubClip
            sc = ET.SubElement(root, "SubClip")
            sc.set("ObjectID", subclip_oid)
            sc.set("ClassID", subclip_class_id)
            sc.set("Version", subclip_version)

            sc_clip = ET.SubElement(sc, "Clip")
            sc_clip.set("ObjectRef", video_clip_oid)
            mc = ET.SubElement(sc, "MasterClip")
            mc.set("ObjectURef", master_clip_uref)
            nm = ET.SubElement(sc, "Name")
            nm.text = subclip_name
            ocg = ET.SubElement(sc, "OrigChGrp")
            ocg.text = "0"

            # Create VideoClipTrackItem
            vcti = ET.SubElement(root, "VideoClipTrackItem")
            vcti.set("ObjectID", track_item_oid)
            vcti.set("ClassID", item_class_id)
            vcti.set("Version", item_version)

            cti_el = ET.SubElement(vcti, "ClipTrackItem")
            cti_el.set("Version", cti_version)

            co_el = ET.SubElement(cti_el, "ComponentOwner")
            co_el.set("Version", co_version)
            comp_ref_el = ET.SubElement(co_el, "Components")
            comp_ref_el.set("ObjectRef", comp_chain_oid)

            ti_el = ET.SubElement(cti_el, "TrackItem")
            ti_el.set("Version", ti_version)
            n_el = ET.SubElement(ti_el, "Node")
            n_el.set("Version", node_version)
            id_el = ET.SubElement(n_el, "ID")
            id_el.text = str(node_id)
            s_el = ET.SubElement(ti_el, "Start")
            s_el.text = str(seg_start)
            e_el = ET.SubElement(ti_el, "End")
            e_el.text = str(seg_end)

            sc_ref = ET.SubElement(cti_el, "SubClip")
            sc_ref.set("ObjectRef", subclip_oid)

            fr = ET.SubElement(vcti, "FrameRect")
            fr.text = frame_rect
            pa = ET.SubElement(vcti, "PixelAspectRatio")
            pa.text = pixel_ar_text

            # Add TrackItem ref to the track
            ti_ref = ET.SubElement(track_items_el, "TrackItem")
            ti_ref.set("Index", str(seg_idx))
            ti_ref.set("ObjectRef", track_item_oid)

    # 9. Audio mute: split audio clips by speaker activity
    if audio_segments is not None or audio_track_intervals_s:
        # Log all speech intervals
        if inc_log.active and audio_segments is not None:
            all_intervals = []
            for seg in audio_segments:
                all_intervals.append({
                    "start": seg.start_s,
                    "end": seg.end_s,
                    "state": seg.speaker_state.value,
                })
            inc_log.write({"event": "speech_intervals", "count": len(all_intervals), "intervals": all_intervals})
        if inc_log.active and audio_track_intervals_s:
            inc_log.write({
                "event": "audio_track_intervals",
                "tracks": {
                    str(track_idx): [
                        {"start": round(start_s, 3), "end": round(end_s, 3)}
                        for start_s, end_s in intervals
                    ]
                    for track_idx, intervals in sorted(audio_track_intervals_s.items())
                },
            })

        audio_items = _find_audio_track_items(root, seq, obj_map, uid_map)
        inc_log.write({
            "event": "audio_tracks_found",
            "count": len(audio_items),
            "tracks": [track_idx for track_idx, _a_item, _a_track, _track_items_el in audio_items],
        })
        for track_idx, a_item, a_track, a_track_items_el in audio_items:
            if audio_track_intervals_s is not None:
                if track_idx not in audio_track_intervals_s:
                    # In explicit per-track mode, missing tracks are left untouched.
                    continue
                intervals = list(audio_track_intervals_s.get(track_idx, []))
            else:
                if audio_track_map is not None:
                    speaker_idx = audio_track_map.get(track_idx)
                    if speaker_idx is None:
                        continue  # track not mapped to a speaker
                else:
                    speaker_idx = track_idx  # backwards compatibility
                intervals = segments_to_audio_intervals(audio_segments, speaker_idx)
            if (
                audio_overlap_s > 0
                or (audio_pre_roll_s is not None and audio_pre_roll_s > 0)
                or (audio_post_roll_s is not None and audio_post_roll_s > 0)
            ):
                if audio_segments is not None:
                    total_dur = max((seg.end_s for seg in audio_segments), default=0.0)
                else:
                    end_el = a_item.find("./ClipTrackItem/TrackItem/End")
                    if end_el is not None and end_el.text:
                        total_dur = int(end_el.text) / TICKS_PER_SECOND
                    else:
                        total_dur = max(
                            (
                                end_s
                                for per_track in audio_track_intervals_s.values()
                                for _start_s, end_s in per_track
                            ),
                            default=0.0,
                        )
                intervals = expand_audio_intervals(
                    intervals,
                    audio_overlap_s,
                    total_dur,
                    pre_roll_s=audio_pre_roll_s,
                    post_roll_s=audio_post_roll_s,
                )
            intervals_ticks = [
                (snap_ticks_to_frame(seconds_to_ticks(s), tpf) if tpf else seconds_to_ticks(s),
                 snap_ticks_to_frame(seconds_to_ticks(e), tpf) if tpf else seconds_to_ticks(e))
                for s, e in intervals
            ]
            if tpf:
                intervals_ticks = [(s, e) for s, e in intervals_ticks if e > s]
            override_offset_ticks = (
                seconds_to_ticks(audio_source_offset_s)
                if audio_source_offset_s is not None
                else None
            )
            inc_log.write({"event": "audio_split_start", "track": track_idx})
            next_oid, next_node_id = _split_audio_track_item(
                root, a_item, a_track_items_el, intervals_ticks,
                obj_map, next_oid, next_node_id,
                inc_log=inc_log if inc_log.active else None,
                track_idx=track_idx,
                override_source_offset_ticks=override_offset_ticks,
                tpf=tpf,
            )

        # Build summary and check for FATAL zero-enabled tracks
        if inc_log.active:
            summary: dict = {"event": "summary"}
            log_entries = inc_log.entries
            for track_idx, _a_item, _a_track, _track_items_el in audio_items:
                enabled_count = sum(
                    1 for e in log_entries
                    if e.get("event") == "audio_split"
                    and e.get("track") == track_idx
                    and e.get("enabled")
                )
                disabled_count = sum(
                    1 for e in log_entries
                    if e.get("event") == "audio_split"
                    and e.get("track") == track_idx
                    and not e.get("enabled")
                )
                enabled_ticks = sum(
                    e["new_tl_end"] - e["new_tl_start"]
                    for e in log_entries
                    if e.get("event") == "audio_split"
                    and e.get("track") == track_idx
                    and e.get("enabled")
                )
                summary[f"track_{track_idx}_enabled"] = enabled_count
                summary[f"track_{track_idx}_disabled"] = disabled_count
                summary[f"track_{track_idx}_enabled_seconds"] = round(
                    enabled_ticks / TICKS_PER_SECOND, 3
                )
            inc_log.write(summary)

    # 10. Update NextID
    _set_next_id(root, next_node_id)

    # 11. Write output
    xml_out = ET.tostring(root, encoding="unicode", xml_declaration=True)
    with gzip.open(out_path, "wb") as f:
        f.write(xml_out.encode("utf-8"))

    return len(segments)
