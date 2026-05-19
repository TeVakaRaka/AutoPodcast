"""Camera scheduling with wide cutaway for long monologues."""

from __future__ import annotations

from autopodcast.models.domain import CameraEvent, Segment, SpeakerState
from autopodcast.models.project import ProjectConfig


def schedule_camera_events(
    segments: list[Segment], config: ProjectConfig
) -> list[CameraEvent]:
    """Schedule camera events with wide cutaway inserts for long talk segments.

    Priority:
      1. BOTH -> wide camera
      2. SILENCE -> wide camera
      3. Long monologue (> threshold) -> wide cutaway insert
      4. Otherwise -> speaker's closeup camera

    Long-talk cutaway carries through BOTH/SILENCE segments but stops
    at a different speaker. If the natural wide segments following a long
    talk fully cover the remaining cutaway duration, the in-segment
    cutaway is suppressed (the natural wide provides enough variety).
    """
    camera_a = config.audio_inputs[0].camera_index if config.audio_inputs else 1
    camera_b = config.audio_inputs[1].camera_index if len(config.audio_inputs) > 1 else 2
    label_a = config.audio_inputs[0].speaker_label if config.audio_inputs else "speaker_a"
    label_b = config.audio_inputs[1].speaker_label if len(config.audio_inputs) > 1 else "speaker_b"
    wide_cam = config.both_speaking_camera

    events: list[CameraEvent] = []
    overlay_end = 0.0
    overlay_speaker: SpeakerState | None = None

    for i, seg in enumerate(segments):
        if seg.speaker_state in (SpeakerState.BOTH, SpeakerState.SILENCE):
            events.append(CameraEvent(
                start_s=seg.start_s, end_s=seg.end_s,
                camera_index=wide_cam, speaker_state=seg.speaker_state,
            ))
            continue

        # SPEAKER_A or SPEAKER_B
        if seg.speaker_state == SpeakerState.SPEAKER_A:
            cam, label = camera_a, label_a
        else:
            cam, label = camera_b, label_b

        # Cancel overlay if different speaker
        if seg.speaker_state != overlay_speaker:
            overlay_end = 0.0
            overlay_speaker = None

        # Apply carry-over overlay from previous long-talk cutaway
        if overlay_end > seg.start_s:
            cut = min(overlay_end, seg.end_s)
            events.append(CameraEvent(
                start_s=seg.start_s, end_s=cut,
                camera_index=wide_cam, speaker_state=seg.speaker_state,
            ))
            if cut < seg.end_s:
                events.append(CameraEvent(
                    start_s=cut, end_s=seg.end_s,
                    camera_index=cam, speaker_state=seg.speaker_state,
                    speaker_label=label,
                ))
            continue

        # Check for long talk
        threshold = config.long_talk_threshold_sec
        wide_dur = config.wide_duration_sec

        if seg.duration_s > threshold:
            wide_start = seg.start_s + threshold
            in_seg_wide = seg.end_s - wide_start
            remaining_needed = wide_dur - in_seg_wide

            natural_wide = _sum_following_wide(segments, i)

            if remaining_needed > 0 and natural_wide >= remaining_needed:
                # Natural wide segments fully cover remaining duration — suppress
                events.append(CameraEvent(
                    start_s=seg.start_s, end_s=seg.end_s,
                    camera_index=cam, speaker_state=seg.speaker_state,
                    speaker_label=label,
                ))
            elif config.long_talk_mode == "repeat":
                # Repeat mode: insert wide periodically
                cooldown = config.wide_cooldown_sec
                _emit_repeat_wide(
                    events, seg, cam, label, wide_cam,
                    threshold, wide_dur, cooldown,
                )
                # Carry-over for last wide if clipped by segment boundary
                last_wide_end = _last_wide_end_in_repeat(
                    seg, threshold, wide_dur, cooldown,
                )
                if last_wide_end > seg.end_s:
                    overlay_end = last_wide_end
                    overlay_speaker = seg.speaker_state
            else:
                # Once mode: insert cutaway within segment
                wide_end = min(wide_start + wide_dur, seg.end_s)

                events.append(CameraEvent(
                    start_s=seg.start_s, end_s=wide_start,
                    camera_index=cam, speaker_state=seg.speaker_state,
                    speaker_label=label,
                ))
                events.append(CameraEvent(
                    start_s=wide_start, end_s=wide_end,
                    camera_index=wide_cam, speaker_state=seg.speaker_state,
                ))
                if wide_end < seg.end_s:
                    events.append(CameraEvent(
                        start_s=wide_end, end_s=seg.end_s,
                        camera_index=cam, speaker_state=seg.speaker_state,
                        speaker_label=label,
                    ))

                # Set carry-over if cutaway was clipped by segment boundary
                if wide_end < wide_start + wide_dur:
                    overlay_end = wide_start + wide_dur
                    overlay_speaker = seg.speaker_state
        else:
            events.append(CameraEvent(
                start_s=seg.start_s, end_s=seg.end_s,
                camera_index=cam, speaker_state=seg.speaker_state,
                speaker_label=label,
            ))

    merged = _merge_adjacent(events)
    split = _split_long_segments(merged, config)
    if config.min_camera_event_s > 0:
        split = _enforce_min_camera_duration(
            split,
            config.min_camera_event_s,
            speaker_grace_s=config.camera_takeover_min_segment_ms / 1000.0,
        )
    split = _stick_wide_for_dense_clusters(split, config)
    split = _insert_dialogue_reestablish_wides(split, config)
    return _extend_wide_through_dense_dialogue(split, config)


def _split_long_segments(
    events: list[CameraEvent], config: ProjectConfig
) -> list[CameraEvent]:
    """Split segments longer than max_segment_sec.

    For SILENCE/BOTH segments, alternate between available cameras.
    For speaker segments, keep the same camera.
    """
    max_dur = config.max_segment_sec
    if max_dur <= 0:
        return events

    camera_a = config.audio_inputs[0].camera_index if config.audio_inputs else 1
    camera_b = config.audio_inputs[1].camera_index if len(config.audio_inputs) > 1 else 2
    wide_cam = config.both_speaking_camera
    rotation = [wide_cam, camera_a, camera_b]

    result: list[CameraEvent] = []
    for ev in events:
        if ev.duration_s <= max_dur:
            result.append(ev)
            continue

        # Split into chunks
        t = ev.start_s
        chunk_idx = 0
        while t < ev.end_s - 0.001:
            chunk_end = min(t + max_dur, ev.end_s)

            if ev.speaker_state in (SpeakerState.SILENCE, SpeakerState.BOTH):
                cam = rotation[chunk_idx % len(rotation)]
            elif chunk_idx % 2 == 0:
                cam = ev.camera_index  # speaker's camera
            else:
                cam = wide_cam  # wide for variety

            result.append(CameraEvent(
                start_s=t,
                end_s=chunk_end,
                camera_index=cam,
                speaker_state=ev.speaker_state,
                speaker_label=ev.speaker_label,
            ))
            t = chunk_end
            chunk_idx += 1

    return result


def _emit_repeat_wide(
    events: list[CameraEvent],
    seg: Segment,
    cam: int,
    label: str,
    wide_cam: int,
    threshold: float,
    wide_dur: float,
    cooldown: float,
) -> None:
    """Emit repeating wide cutaways within a long segment.

    Pattern: [speaker 0→threshold] [wide threshold→threshold+wide_dur]
             [speaker ...→...+cooldown] [wide ...→...+wide_dur] ...
    """
    t = seg.start_s
    next_wide = seg.start_s + threshold

    while t < seg.end_s - 0.001:
        if t < next_wide:
            # Speaker camera until next wide insertion
            speaker_end = min(next_wide, seg.end_s)
            events.append(CameraEvent(
                start_s=t, end_s=speaker_end,
                camera_index=cam, speaker_state=seg.speaker_state,
                speaker_label=label,
            ))
            t = speaker_end
        else:
            # Wide cutaway
            wide_end = min(t + wide_dur, seg.end_s)
            events.append(CameraEvent(
                start_s=t, end_s=wide_end,
                camera_index=wide_cam, speaker_state=seg.speaker_state,
            ))
            t = wide_end
            next_wide = t + cooldown


def _last_wide_end_in_repeat(
    seg: Segment,
    threshold: float,
    wide_dur: float,
    cooldown: float,
) -> float:
    """Calculate the end time of the last wide cutaway in repeat mode."""
    t = seg.start_s
    next_wide = seg.start_s + threshold
    last_wide_end = 0.0

    while t < seg.end_s - 0.001:
        if t < next_wide:
            t = min(next_wide, seg.end_s)
        else:
            last_wide_end = t + wide_dur
            t = min(t + wide_dur, seg.end_s)
            next_wide = t + cooldown

    return last_wide_end


def _sum_following_wide(segments: list[Segment], idx: int) -> float:
    """Sum durations of consecutive BOTH/SILENCE segments after idx."""
    total = 0.0
    for j in range(idx + 1, len(segments)):
        if segments[j].speaker_state in (SpeakerState.BOTH, SpeakerState.SILENCE):
            total += segments[j].duration_s
        else:
            break
    return total


def _merge_adjacent(events: list[CameraEvent]) -> list[CameraEvent]:
    """Merge adjacent events with the same camera_index."""
    if not events:
        return []

    merged: list[CameraEvent] = [events[0]]
    for ev in events[1:]:
        prev = merged[-1]
        if ev.camera_index == prev.camera_index:
            merged[-1] = CameraEvent(
                start_s=prev.start_s,
                end_s=ev.end_s,
                camera_index=prev.camera_index,
                speaker_state=prev.speaker_state,
                speaker_label=prev.speaker_label,
            )
        else:
            merged.append(ev)

    return merged


def _stick_wide_for_dense_clusters(
    events: list[CameraEvent], config: ProjectConfig,
) -> list[CameraEvent]:
    """Bridge dense overlap clusters into a steadier wide shot.

    When natural BOTH-wide anchors are separated only by several short speaker
    turns, keep the wide across the whole cluster instead of bouncing through
    closeups for a couple of seconds.
    """
    if len(events) < 3:
        return events

    max_bridge_s = config.sticky_wide_max_bridge_sec
    max_turn_s = config.sticky_wide_max_turn_sec
    min_turns = config.sticky_wide_min_turns
    if max_bridge_s <= 0 or max_turn_s <= 0 or min_turns <= 0:
        return events

    wide_cam = config.both_speaking_camera
    speaker_states = {SpeakerState.SPEAKER_A, SpeakerState.SPEAKER_B}
    result: list[CameraEvent] = []
    i = 0

    def is_both_wide_anchor(ev: CameraEvent) -> bool:
        return ev.camera_index == wide_cam and ev.speaker_state == SpeakerState.BOTH

    def qualifies_bridge(start_idx: int, end_idx: int) -> bool:
        bridge_events = events[start_idx + 1:end_idx]
        if not bridge_events:
            return False
        bridge_duration_s = events[end_idx].start_s - events[start_idx].end_s
        if bridge_duration_s > max_bridge_s:
            return False
        speaker_turns = [
            ev for ev in bridge_events
            if ev.camera_index != wide_cam and ev.speaker_state in speaker_states
        ]
        if len(speaker_turns) < min_turns:
            return False
        if any(ev.duration_s > max_turn_s for ev in speaker_turns):
            return False
        if any(ev.speaker_state not in speaker_states for ev in bridge_events):
            return False
        return True

    while i < len(events):
        if not is_both_wide_anchor(events[i]):
            result.append(events[i])
            i += 1
            continue

        cluster_end = i
        j = i + 1
        while j < len(events):
            if not is_both_wide_anchor(events[j]):
                j += 1
                continue
            if not qualifies_bridge(cluster_end, j):
                break
            cluster_end = j
            j += 1

        if cluster_end == i:
            result.append(events[i])
            i += 1
            continue

        result.append(CameraEvent(
            start_s=events[i].start_s,
            end_s=events[cluster_end].end_s,
            camera_index=wide_cam,
            speaker_state=SpeakerState.BOTH,
        ))
        i = cluster_end + 1

    return _merge_adjacent(result)


def _insert_dialogue_reestablish_wides(
    events: list[CameraEvent], config: ProjectConfig,
) -> list[CameraEvent]:
    """Occasionally insert a short wide during active shot-reverse-shot dialogue.

    This acts like a re-establishing master shot: after a few speaker exchanges
    and a decent amount of time since the last wide, briefly cut to the wide
    camera to refresh spatial context without making the edit feel busy.
    """
    if len(events) < 2:
        return events

    interval_s = config.dialogue_wide_interval_sec
    wide_duration_s = config.dialogue_wide_duration_sec
    min_turns = config.dialogue_wide_min_turns
    if interval_s <= 0 or wide_duration_s <= 0 or min_turns <= 0:
        return events

    wide_cam = config.both_speaking_camera
    min_tail_s = max(
        config.min_camera_event_s,
        config.camera_takeover_min_segment_ms / 1000.0,
    )

    result: list[CameraEvent] = []
    last_wide_end = events[0].start_s
    turns_since_wide = 0
    last_speaker: SpeakerState | None = None

    for ev in events:
        if ev.speaker_state in (SpeakerState.BOTH, SpeakerState.SILENCE):
            result.append(ev)
            last_wide_end = ev.end_s
            turns_since_wide = 0
            last_speaker = None
            continue

        if ev.camera_index == wide_cam:
            result.append(ev)
            last_wide_end = ev.end_s
            turns_since_wide = 0
            last_speaker = None
            continue

        is_turn_change = last_speaker is not None and ev.speaker_state != last_speaker
        if is_turn_change:
            turns_since_wide += 1

        should_insert = (
            is_turn_change
            and turns_since_wide >= min_turns
            and ev.start_s - last_wide_end >= interval_s
            and ev.duration_s >= wide_duration_s + min_tail_s
        )

        if should_insert:
            wide_end = ev.start_s + wide_duration_s
            result.append(CameraEvent(
                start_s=ev.start_s,
                end_s=wide_end,
                camera_index=wide_cam,
                speaker_state=ev.speaker_state,
            ))
            result.append(CameraEvent(
                start_s=wide_end,
                end_s=ev.end_s,
                camera_index=ev.camera_index,
                speaker_state=ev.speaker_state,
                speaker_label=ev.speaker_label,
            ))
            last_wide_end = wide_end
            turns_since_wide = 0
        else:
            result.append(ev)

        last_speaker = ev.speaker_state

    return _merge_adjacent(result)


def _extend_wide_through_dense_dialogue(
    events: list[CameraEvent], config: ProjectConfig,
) -> list[CameraEvent]:
    """Hold a wide shot through a short burst of rapid back-and-forth dialogue.

    This catches cases where overlap is audible and the exchange is clearly
    energetic, but the segmenter has already collapsed most of the explicit
    BOTH islands into alternating A/B closeups.
    """
    if len(events) < 2:
        return events

    wide_cam = config.both_speaking_camera
    max_span_s = config.dialogue_cluster_max_span_sec
    max_turn_s = config.dialogue_cluster_max_turn_sec
    min_turns = config.dialogue_cluster_min_turns
    if max_span_s <= 0 or max_turn_s <= 0 or min_turns <= 0:
        return events

    speaker_states = {SpeakerState.SPEAKER_A, SpeakerState.SPEAKER_B}
    result: list[CameraEvent] = []
    i = 0

    while i < len(events):
        anchor = events[i]
        if anchor.camera_index != wide_cam:
            result.append(anchor)
            i += 1
            continue

        turn_events: list[CameraEvent] = []
        j = i + 1
        while j < len(events):
            ev = events[j]
            if ev.camera_index == wide_cam or ev.speaker_state not in speaker_states:
                break
            if ev.duration_s > max_turn_s:
                break
            if ev.end_s - anchor.start_s > max_span_s:
                break
            turn_events.append(ev)
            j += 1

        speakers = {ev.speaker_state for ev in turn_events}
        if len(turn_events) >= min_turns and len(speakers) >= 2:
            result.append(CameraEvent(
                start_s=anchor.start_s,
                end_s=turn_events[-1].end_s,
                camera_index=wide_cam,
                speaker_state=SpeakerState.BOTH,
            ))
            i = j
            continue

        result.append(anchor)
        i += 1

    return _merge_adjacent(result)


def _enforce_min_camera_duration(
    events: list[CameraEvent],
    min_duration_s: float,
    speaker_grace_s: float = 0.0,
) -> list[CameraEvent]:
    """Remove camera events shorter than min_duration_s by merging into neighbors.

    Short events are absorbed into their predecessor (extending its end_s).
    If the first event is short, it is absorbed into the next event instead.
    After absorption, adjacent events with the same camera are merged.
    """
    if not events or min_duration_s <= 0:
        return events

    def preserve_short_speaker_event(ev: CameraEvent) -> bool:
        return (
            speaker_grace_s > 0
            and ev.speaker_state in (SpeakerState.SPEAKER_A, SpeakerState.SPEAKER_B)
            and ev.duration_s >= speaker_grace_s
        )

    result: list[CameraEvent] = [events[0]]
    for ev in events[1:]:
        if ev.duration_s < min_duration_s and not preserve_short_speaker_event(ev):
            # Absorb into previous event
            prev = result[-1]
            result[-1] = CameraEvent(
                start_s=prev.start_s,
                end_s=ev.end_s,
                camera_index=prev.camera_index,
                speaker_state=prev.speaker_state,
                speaker_label=prev.speaker_label,
            )
        else:
            result.append(ev)

    # First event may still be short — absorb into next
    if (
        len(result) > 1
        and result[0].duration_s < min_duration_s
        and not preserve_short_speaker_event(result[0])
    ):
        result[1] = CameraEvent(
            start_s=result[0].start_s,
            end_s=result[1].end_s,
            camera_index=result[1].camera_index,
            speaker_state=result[1].speaker_state,
            speaker_label=result[1].speaker_label,
        )
        result = result[1:]

    return _merge_adjacent(result)


def camera_events_to_segments(events: list[CameraEvent]) -> list[Segment]:
    """Convert CameraEvents back to Segments for Timeline compatibility."""
    return [
        Segment(
            start_s=ev.start_s,
            end_s=ev.end_s,
            camera_index=ev.camera_index,
            speaker_state=ev.speaker_state,
            speaker_label=ev.speaker_label,
        )
        for ev in events
    ]
