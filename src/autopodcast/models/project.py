from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AudioInput:
    path: Path
    speaker_label: str
    camera_index: int


@dataclass
class CameraInput:
    path: Path
    label: str
    index: int


@dataclass
class ProjectConfig:
    audio_inputs: list[AudioInput] = field(default_factory=list)
    cameras: list[CameraInput] = field(default_factory=list)
    output_dir: Path = field(default_factory=lambda: Path("."))

    # Analysis
    sample_rate: int = 16000
    window_ms: float = 30.0
    hop_ms: float = 10.0
    smoothing_ms: float = 150.0

    # Detection (hysteresis)
    speech_threshold_db: float = -28.0
    release_threshold_db: float = -33.0
    silence_threshold_db: float = -40.0
    hangover_ms: float = 600.0

    # Segmentation
    min_segment_ms: float = 2000.0
    debounce_ms: float = 300.0

    # Camera
    default_camera: int = 0
    both_speaking_camera: int = 0

    # Ducking
    ducking_enabled: bool = False
    ducking_db: float = -12.0

    # Sequence format
    sequence_width: int = 1920
    sequence_height: int = 1080

    @property
    def window_samples(self) -> int:
        return int(self.sample_rate * self.window_ms / 1000.0)

    @property
    def hop_samples(self) -> int:
        return int(self.sample_rate * self.hop_ms / 1000.0)

    @property
    def smoothing_kernel_size(self) -> int:
        return max(1, int(self.smoothing_ms / self.hop_ms))

    @property
    def hangover_frames(self) -> int:
        return max(1, int(self.hangover_ms / self.hop_ms))

    @property
    def debounce_frames(self) -> int:
        return max(1, int(self.debounce_ms / self.hop_ms))

    @property
    def min_segment_frames(self) -> int:
        return max(1, int(self.min_segment_ms / self.hop_ms))
