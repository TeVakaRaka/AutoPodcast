"""Volume ducking automation for inactive microphones."""

from __future__ import annotations

from autopodcast.models.domain import DuckingEvent, Segment, SpeakerState
from autopodcast.models.project import ProjectConfig


def generate_ducking_events(
    segments: list[Segment], config: ProjectConfig
) -> list[DuckingEvent]:
    """Generate volume automation events for ducking inactive mics.

    Track 0 = speaker A mic, Track 1 = speaker B mic.
    Active speaker's mic stays at 0 dB, inactive at ducking_db.
    BOTH/SILENCE: both at 0 dB.
    """
    if not config.ducking_enabled:
        return []

    events: list[DuckingEvent] = []
    duck_db = config.ducking_db

    for seg in segments:
        if seg.speaker_state == SpeakerState.SPEAKER_A:
            events.append(DuckingEvent(time_s=seg.start_s, target_db=0.0, track_index=0))
            events.append(DuckingEvent(time_s=seg.start_s, target_db=duck_db, track_index=1))
        elif seg.speaker_state == SpeakerState.SPEAKER_B:
            events.append(DuckingEvent(time_s=seg.start_s, target_db=duck_db, track_index=0))
            events.append(DuckingEvent(time_s=seg.start_s, target_db=0.0, track_index=1))
        else:  # BOTH or SILENCE
            events.append(DuckingEvent(time_s=seg.start_s, target_db=0.0, track_index=0))
            events.append(DuckingEvent(time_s=seg.start_s, target_db=0.0, track_index=1))

    return events
