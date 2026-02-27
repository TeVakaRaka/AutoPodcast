from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SpeakerState(Enum):
    SILENCE = "silence"
    SPEAKER_A = "speaker_a"
    SPEAKER_B = "speaker_b"
    BOTH = "both"


@dataclass
class AnalysisFrame:
    time_s: float
    rms_db: float
    envelope_db: float
    is_active: bool


@dataclass
class SpeakerActivity:
    speaker_label: str
    frames: list[AnalysisFrame] = field(default_factory=list)


@dataclass
class Segment:
    start_s: float
    end_s: float
    camera_index: int
    speaker_state: SpeakerState
    speaker_label: str | None = None

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class DuckingEvent:
    time_s: float
    target_db: float
    track_index: int


@dataclass
class Timeline:
    segments: list[Segment] = field(default_factory=list)
    ducking_events: list[DuckingEvent] = field(default_factory=list)
    total_duration_s: float = 0.0
    fps: float = 29.97
