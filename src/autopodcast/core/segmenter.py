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


def collapse_both_between_same(segments: list[Segment]) -> list[Segment]:
    """Replace 'both' with neighbor state when surrounded by same speaker.

    A→both→A becomes A→A→A (caller runs merge_adjacent after).
    Preserves real transitions like A→both→B.
    """
    if len(segments) < 3:
        return list(segments)

    result = list(segments)
    for i in range(1, len(result) - 1):
        if (
            result[i].speaker_state == SpeakerState.BOTH
            and result[i - 1].speaker_state == result[i + 1].speaker_state
            and result[i - 1].speaker_state
            in (SpeakerState.SPEAKER_A, SpeakerState.SPEAKER_B)
        ):
            result[i] = Segment(
                start_s=result[i].start_s,
                end_s=result[i].end_s,
                camera_index=result[i].camera_index,
                speaker_state=result[i - 1].speaker_state,
                speaker_label=result[i - 1].speaker_label,
            )

    return result


def collapse_short_neutral_segments(
    segments: list[Segment],
    min_duration_s: float,
    collapsible_states: set[SpeakerState] | None = None,
) -> list[Segment]:
    """Absorb short BOTH/SILENCE runs into neighboring speaker segments.

    Camera switching should not jump to wide for tiny overlap/pause islands.
    Preference order:
      1. Same speaker on both sides -> absorb into that speaker.
      2. Speaker on the left -> absorb into the left speaker.
      3. Speaker on the right -> absorb into the right speaker.
    """
    if len(segments) <= 1 or min_duration_s <= 0:
        return list(segments)

    neutral_states = collapsible_states or {SpeakerState.BOTH, SpeakerState.SILENCE}
    speaker_states = {SpeakerState.SPEAKER_A, SpeakerState.SPEAKER_B}
    result = list(segments)

    for i, seg in enumerate(result):
        if seg.speaker_state not in neutral_states or seg.duration_s >= min_duration_s:
            continue

        prev = result[i - 1] if i > 0 else None
        nxt = result[i + 1] if i + 1 < len(result) else None

        replacement_state = None
        replacement_label = None

        if (
            prev is not None
            and nxt is not None
            and prev.speaker_state == nxt.speaker_state
            and prev.speaker_state in speaker_states
        ):
            replacement_state = prev.speaker_state
            replacement_label = prev.speaker_label
        elif prev is not None and prev.speaker_state in speaker_states:
            replacement_state = prev.speaker_state
            replacement_label = prev.speaker_label
        elif nxt is not None and nxt.speaker_state in speaker_states:
            replacement_state = nxt.speaker_state
            replacement_label = nxt.speaker_label

        if replacement_state is None:
            continue

        result[i] = Segment(
            start_s=seg.start_s,
            end_s=seg.end_s,
            camera_index=seg.camera_index,
            speaker_state=replacement_state,
            speaker_label=replacement_label,
        )

    return merge_adjacent(result)


def enforce_min_length(
    segments: list[Segment],
    min_duration_s: float,
    protected_indices: set[int] | None = None,
) -> list[Segment]:
    """Merge segments shorter than min_duration_s into previous segment."""
    if len(segments) <= 1:
        return list(segments)

    protected = protected_indices or set()
    result: list[Segment] = [segments[0]]

    for idx, seg in enumerate(segments[1:], start=1):
        if seg.duration_s < min_duration_s and idx not in protected:
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


def find_short_takeover_indices(
    segments: list[Segment],
    min_duration_s: float,
    takeover_min_duration_s: float,
    takeover_context_s: float,
) -> set[int]:
    """Keep short interruptions that punctuate a longer opposite-speaker run."""
    if (
        len(segments) < 3
        or takeover_min_duration_s <= 0
        or takeover_context_s <= 0
        or takeover_min_duration_s >= min_duration_s
    ):
        return set()

    speaker_states = {SpeakerState.SPEAKER_A, SpeakerState.SPEAKER_B}
    protected: set[int] = set()

    for i in range(1, len(segments) - 1):
        seg = segments[i]
        if seg.speaker_state not in speaker_states:
            continue
        if not (takeover_min_duration_s <= seg.duration_s < min_duration_s):
            continue

        prev = segments[i - 1]
        nxt = segments[i + 1]
        if prev.speaker_state not in speaker_states or nxt.speaker_state not in speaker_states:
            continue
        if prev.speaker_state != nxt.speaker_state or prev.speaker_state == seg.speaker_state:
            continue
        if prev.duration_s + nxt.duration_s < takeover_context_s:
            continue

        protected.add(i)

    return protected


def preserve_dense_overlap_clusters(
    segments: list[Segment],
    max_bridge_s: float,
    max_turn_s: float,
    min_turns: int,
    anchor_min_s: float,
) -> list[Segment]:
    """Preserve dense A/B/BOTH clusters as BOTH before camera smoothing.

    This keeps "everyone talking over each other" sections intact long enough
    for the camera scheduler to prefer a steadier wide shot instead of
    flattening the cluster into plain shot/reverse-shot first.
    """
    if (
        len(segments) < 3
        or max_bridge_s <= 0
        or max_turn_s <= 0
        or min_turns <= 0
        or anchor_min_s < 0
    ):
        return list(segments)

    speaker_states = {SpeakerState.SPEAKER_A, SpeakerState.SPEAKER_B}
    result: list[Segment] = []
    i = 0

    def is_both_anchor(seg: Segment) -> bool:
        return (
            seg.speaker_state == SpeakerState.BOTH
            and seg.duration_s >= anchor_min_s
        )

    while i < len(segments):
        if not is_both_anchor(segments[i]):
            result.append(segments[i])
            i += 1
            continue

        cluster_end = i
        turn_count = 0
        j = i + 1

        while j < len(segments):
            if not is_both_anchor(segments[j]):
                j += 1
                continue

            bridge = segments[cluster_end + 1:j]
            if not bridge:
                cluster_end = j
                j += 1
                continue

            if any(seg.speaker_state not in speaker_states for seg in bridge):
                break
            if any(seg.duration_s > max_turn_s for seg in bridge):
                break
            if segments[j].start_s - segments[cluster_end].end_s > max_bridge_s:
                break

            turn_count += len(bridge)
            cluster_end = j
            j += 1

        if cluster_end > i and turn_count >= min_turns:
            result.append(
                Segment(
                    start_s=segments[i].start_s,
                    end_s=segments[cluster_end].end_s,
                    camera_index=segments[i].camera_index,
                    speaker_state=SpeakerState.BOTH,
                )
            )
            i = cluster_end + 1
            continue

        result.append(segments[i])
        i += 1

    return merge_adjacent(result)


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


def build_camera_segments(
    states: list[SpeakerState], config: ProjectConfig,
) -> list[Segment]:
    """Build camera-oriented segments from speech states.

    Compared to audio mute, video should be more conservative with neutral
    states (`BOTH`/`SILENCE`) so we do not cut to wide for every short
    overlap or pause, while still allowing real speaker turns to surface
    faster than the legacy 2-second pipeline.
    """
    hop_s = config.hop_ms / 1000.0
    segments = run_length_encode(states, hop_s)
    both_anchor_min_s = max(hop_s * 3, config.camera_debounce_ms / 1000.0)

    both_min_ms = config.camera_both_min_segment_ms
    silence_min_ms = config.camera_silence_min_segment_ms
    if config.camera_neutral_min_segment_ms > 0:
        both_min_ms = config.camera_neutral_min_segment_ms
        silence_min_ms = config.camera_neutral_min_segment_ms

    if silence_min_ms > 0:
        segments = collapse_short_neutral_segments(
            segments,
            silence_min_ms / 1000.0,
            {SpeakerState.SILENCE},
        )
    segments = preserve_dense_overlap_clusters(
        segments,
        max_bridge_s=config.sticky_wide_max_bridge_sec,
        max_turn_s=config.sticky_wide_max_turn_sec,
        min_turns=config.sticky_wide_min_turns,
        anchor_min_s=both_anchor_min_s,
    )
    if both_min_ms > 0:
        segments = collapse_short_neutral_segments(
            segments,
            both_min_ms / 1000.0,
            {SpeakerState.BOTH},
        )

    if config.camera_debounce_ms > 0:
        segments = debounce_segments(segments, config.camera_debounce_ms / 1000.0)

    if config.camera_min_segment_ms > 0:
        protected_takeovers = find_short_takeover_indices(
            segments,
            config.camera_min_segment_ms / 1000.0,
            config.camera_takeover_min_segment_ms / 1000.0,
            config.camera_takeover_context_ms / 1000.0,
        )
        segments = enforce_min_length(
            segments,
            config.camera_min_segment_ms / 1000.0,
            protected_indices=protected_takeovers,
        )

    if silence_min_ms > 0:
        segments = collapse_short_neutral_segments(
            segments,
            silence_min_ms / 1000.0,
            {SpeakerState.SILENCE},
        )
    segments = preserve_dense_overlap_clusters(
        segments,
        max_bridge_s=config.sticky_wide_max_bridge_sec,
        max_turn_s=config.sticky_wide_max_turn_sec,
        min_turns=config.sticky_wide_min_turns,
        anchor_min_s=both_anchor_min_s,
    )
    if both_min_ms > 0:
        segments = collapse_short_neutral_segments(
            segments,
            both_min_ms / 1000.0,
            {SpeakerState.BOTH},
        )

    return merge_adjacent(segments)


def build_audio_mute_segments(
    states: list[SpeakerState], config: ProjectConfig,
) -> list[Segment]:
    """Build audio mute segments with lighter smoothing than video cuts.

    Audio mute should preserve genuine overlap (`BOTH`) so short interjections
    from the second speaker are not erased before track muting is computed.
    """
    hop_s = config.hop_ms / 1000.0
    segments = run_length_encode(states, hop_s)

    if config.audio_mute_debounce_ms > 0:
        segments = debounce_segments(segments, config.audio_mute_debounce_ms / 1000.0)

    if config.audio_mute_min_segment_ms > 0:
        segments = enforce_min_length(segments, config.audio_mute_min_segment_ms / 1000.0)

    if config.audio_mute_collapse_both_between_same:
        segments = collapse_both_between_same(segments)

    return merge_adjacent(segments)
