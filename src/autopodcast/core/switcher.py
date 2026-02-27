"""Camera assignment based on speaker state."""

from __future__ import annotations

from autopodcast.models.domain import Segment, SpeakerState
from autopodcast.models.project import ProjectConfig


def assign_cameras(segments: list[Segment], config: ProjectConfig) -> list[Segment]:
    """Assign camera_index and speaker_label to each segment based on rules.

    - SPEAKER_A -> audio_inputs[0].camera_index
    - SPEAKER_B -> audio_inputs[1].camera_index
    - BOTH -> config.both_speaking_camera
    - SILENCE -> config.default_camera
    """
    camera_a = config.audio_inputs[0].camera_index if config.audio_inputs else 1
    camera_b = config.audio_inputs[1].camera_index if len(config.audio_inputs) > 1 else 2
    label_a = config.audio_inputs[0].speaker_label if config.audio_inputs else "speaker_a"
    label_b = config.audio_inputs[1].speaker_label if len(config.audio_inputs) > 1 else "speaker_b"

    result = []
    for seg in segments:
        if seg.speaker_state == SpeakerState.SPEAKER_A:
            cam = camera_a
            label = label_a
        elif seg.speaker_state == SpeakerState.SPEAKER_B:
            cam = camera_b
            label = label_b
        elif seg.speaker_state == SpeakerState.BOTH:
            cam = config.both_speaking_camera
            label = None
        else:  # SILENCE
            cam = config.default_camera
            label = None

        result.append(
            Segment(
                start_s=seg.start_s,
                end_s=seg.end_s,
                camera_index=cam,
                speaker_state=seg.speaker_state,
                speaker_label=label,
            )
        )

    return result
