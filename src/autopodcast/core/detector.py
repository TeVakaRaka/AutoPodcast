"""Hysteresis-based speech activity detection with hangover."""

from __future__ import annotations

from enum import Enum

from autopodcast.models.domain import SpeakerActivity, SpeakerState
from autopodcast.models.project import ProjectConfig


class _DetectorState(Enum):
    INACTIVE = 0
    ACTIVE = 1
    HANGOVER = 2


def detect_activity(activity: SpeakerActivity, config: ProjectConfig) -> None:
    """Apply hysteresis detection to speaker frames, setting is_active in-place."""
    state = _DetectorState.INACTIVE
    hangover_counter = 0
    hangover_limit = config.hangover_frames

    for frame in activity.frames:
        level = frame.envelope_db

        if state == _DetectorState.INACTIVE:
            if level > config.speech_threshold_db:
                state = _DetectorState.ACTIVE
                frame.is_active = True
            else:
                frame.is_active = False

        elif state == _DetectorState.ACTIVE:
            if level < config.release_threshold_db:
                state = _DetectorState.HANGOVER
                hangover_counter = 0
                frame.is_active = True
            else:
                frame.is_active = True

        elif state == _DetectorState.HANGOVER:
            if level > config.speech_threshold_db:
                state = _DetectorState.ACTIVE
                hangover_counter = 0
                frame.is_active = True
            elif hangover_counter >= hangover_limit:
                state = _DetectorState.INACTIVE
                frame.is_active = False
            else:
                hangover_counter += 1
                frame.is_active = True


def combine_speakers(
    activity_a: SpeakerActivity, activity_b: SpeakerActivity
) -> list[SpeakerState]:
    """Combine two speaker activities into per-frame SpeakerState."""
    n_frames = min(len(activity_a.frames), len(activity_b.frames))
    states = []

    for i in range(n_frames):
        a_active = activity_a.frames[i].is_active
        b_active = activity_b.frames[i].is_active

        if a_active and b_active:
            states.append(SpeakerState.BOTH)
        elif a_active:
            states.append(SpeakerState.SPEAKER_A)
        elif b_active:
            states.append(SpeakerState.SPEAKER_B)
        else:
            states.append(SpeakerState.SILENCE)

    return states
