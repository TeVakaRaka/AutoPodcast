"""Segmentation: run-length encoding, debounce, min-length, merge."""

from __future__ import annotations

from autopodcast.models.domain import Segment, SpeakerState
from autopodcast.models.project import ProjectConfig


def run_length_encode(states: list[SpeakerState], hop_s: float) -> list[Segment]:
    """Convert per-frame states to raw segments via run-length encoding."""
    if not states:
        return []

    segments: list[Segment] = []
    current_state = states[0]
    start_frame = 0

    for i in range(1, len(states)):
        if states[i] != current_state:
            segments.append(
                Segment(
                    start_s=start_frame * hop_s,
                    end_s=i * hop_s,
                    camera_index=0,
                    speaker_state=current_state,
                )
            )
            current_state = states[i]
            start_frame = i

    # Final segment
    segments.append(
        Segment(
            start_s=start_frame * hop_s,
            end_s=len(states) * hop_s,
            camera_index=0,
            speaker_state=current_state,
        )
    )

    return segments


def debounce_segments(
    segments: list[Segment], min_duration_s: float
) -> list[Segment]:
    """Remove segments shorter than min_duration_s, merging into previous."""
    if len(segments) <= 1:
        return list(segments)

    result: list[Segment] = [segments[0]]

    for seg in segments[1:]:
        if seg.duration_s < min_duration_s:
            # Extend previous segment to cover this one
            result[-1] = Segment(
                start_s=result[-1].start_s,
                end_s=seg.end_s,
                camera_index=result[-1].camera_index,
                speaker_state=result[-1].speaker_state,
                speaker_label=result[-1].speaker_label,
            )
        else:
            result.append(seg)

    return result


def enforce_min_length(
    segments: list[Segment], min_duration_s: float
) -> list[Segment]:
    """Merge segments shorter than min_duration_s into previous segment."""
    if len(segments) <= 1:
        return list(segments)

    result: list[Segment] = [segments[0]]

    for seg in segments[1:]:
        if seg.duration_s < min_duration_s:
            result[-1] = Segment(
                start_s=result[-1].start_s,
                end_s=seg.end_s,
                camera_index=result[-1].camera_index,
                speaker_state=result[-1].speaker_state,
                speaker_label=result[-1].speaker_label,
            )
        else:
            result.append(seg)

    return result


def merge_adjacent(segments: list[Segment]) -> list[Segment]:
    """Merge adjacent segments with the same speaker_state."""
    if len(segments) <= 1:
        return list(segments)

    result: list[Segment] = [segments[0]]

    for seg in segments[1:]:
        prev = result[-1]
        if seg.speaker_state == prev.speaker_state:
            result[-1] = Segment(
                start_s=prev.start_s,
                end_s=seg.end_s,
                camera_index=prev.camera_index,
                speaker_state=prev.speaker_state,
                speaker_label=prev.speaker_label,
            )
        else:
            result.append(seg)

    return result


def segment_timeline(
    states: list[SpeakerState], config: ProjectConfig
) -> list[Segment]:
    """Full segmentation pipeline: RLE -> debounce -> min-length -> merge."""
    hop_s = config.hop_ms / 1000.0
    debounce_s = config.debounce_ms / 1000.0
    min_seg_s = config.min_segment_ms / 1000.0

    segments = run_length_encode(states, hop_s)
    segments = debounce_segments(segments, debounce_s)
    segments = enforce_min_length(segments, min_seg_s)
    segments = merge_adjacent(segments)

    return segments
