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
    smoothing_ms: float = 100.0
    smoothing_attack_ms: float = 30.0
    smoothing_decay_ms: float = 500.0
    highpass_hz: float = 100.0  # 0.0 to disable
    input_gain_db: float = 0.0
    input_gain_a_db: float = 0.0  # per-mic gain for speaker A (host)
    input_gain_b_db: float = 0.0  # per-mic gain for speaker B (guest)

    # Detection (hysteresis)
    speech_threshold_db: float = -24.0
    release_threshold_db: float = -30.0
    silence_threshold_db: float = -40.0
    hangover_ms: float = 1000.0
    detector_backend: str = "auto"  # auto | rms | silero
    vad_threshold: float = 0.5
    vad_min_speech_ms: float = 120.0
    vad_min_silence_ms: float = 80.0
    vad_speech_pad_ms: float = 30.0
    vad_use_onnx: bool = True

    # Post-detection processing
    detection_pre_roll_s: float = 1.0
    detection_post_roll_s: float = 0.25
    detection_min_on_s: float = 0.10
    detection_min_off_s: float = 0.30
    detection_hangover_s: float = 0.20
    camera_debounce_ms: float = 300.0
    camera_min_segment_ms: float = 800.0
    camera_takeover_min_segment_ms: float = 700.0
    camera_takeover_context_ms: float = 3000.0
    camera_both_min_segment_ms: float = 900.0
    camera_silence_min_segment_ms: float = 1600.0
    camera_neutral_min_segment_ms: float = 0.0  # legacy override for both/silence
    audio_mute_debounce_ms: float = 300.0  # debounce for audio mute segments
    audio_mute_min_segment_ms: float = 500.0
    audio_mute_collapse_both_between_same: bool = False
    safety_margin_db: float = 6.0
    safety_window_s: float = 0.20

    # Segmentation
    min_segment_ms: float = 2000.0
    debounce_ms: float = 300.0
    max_segment_sec: float = 20.0

    # Camera
    default_camera: int = 0
    both_speaking_camera: int = 0
    long_talk_threshold_sec: float = 15.0
    wide_duration_sec: float = 5.0
    long_talk_mode: str = "repeat"
    wide_cooldown_sec: float = 20.0
    dialogue_wide_interval_sec: float = 24.0
    dialogue_wide_duration_sec: float = 2.0
    dialogue_wide_min_turns: int = 3
    sticky_wide_max_bridge_sec: float = 8.0
    sticky_wide_max_turn_sec: float = 2.4
    sticky_wide_min_turns: int = 3
    dialogue_cluster_max_span_sec: float = 9.0
    dialogue_cluster_max_turn_sec: float = 4.0
    dialogue_cluster_min_turns: int = 4
    min_camera_event_s: float = 0.8

    # Ducking
    ducking_enabled: bool = True
    ducking_db: float = -96.0
    gate_fade_s: float = 0.15
    audio_overlap_s: float = 0.10
    audio_pre_roll_s: float = 0.35
    audio_post_roll_s: float = 0.10

    # RMS guard (optional safety net against false mutes)
    rms_guard_enabled: bool = False
    rms_guard_window_ms: float = 50.0
    rms_guard_threshold_db: float = 12.0  # above noise floor
    rms_guard_on_ms: float = 150.0
    rms_guard_off_ms: float = 500.0

    # Mask filter (SNR-aware post-filter for short segments)
    mask_filter_enabled: bool = True
    mask_filter_min_on_ms: float = 300.0
    mask_filter_min_off_fill_ms: float = 360.0
    mask_filter_snr_peak_strong_db: float = 10.0
    mask_filter_snr_peak_weak_db: float = 6.0
    mask_filter_crest_min_db: float = 8.0
    mask_filter_noise_window_s: float = 15.0
    mask_filter_noise_percentile: float = 10.0
    lowpass_hz: float = 0.0  # 0=disabled; 4000.0 for speech band

    # Cross-gate (crosstalk suppression)
    cross_gate_db: float = 6.0  # 0 = disabled

    # Roles
    speaker_role_overrides: dict[str, str] = field(default_factory=dict)

    # Sequence format
    sequence_width: int = 1920
    sequence_height: int = 1080

    def validate(self) -> None:
        """Raise ValueError on invalid config."""
        if not (self.speech_threshold_db > self.release_threshold_db > self.silence_threshold_db):
            raise ValueError(
                f"Thresholds must be ordered: speech ({self.speech_threshold_db}) "
                f"> release ({self.release_threshold_db}) "
                f"> silence ({self.silence_threshold_db})"
            )

        if self.detector_backend not in {"auto", "rms", "silero"}:
            raise ValueError(
                f"detector_backend must be one of auto/rms/silero, got '{self.detector_backend}'"
            )

        if not (0.0 < self.vad_threshold < 1.0):
            raise ValueError(f"vad_threshold must be between 0 and 1, got {self.vad_threshold}")

        for name, val in [
            ("window_ms", self.window_ms),
            ("hop_ms", self.hop_ms),
            ("smoothing_ms", self.smoothing_ms),
            ("smoothing_attack_ms", self.smoothing_attack_ms),
            ("smoothing_decay_ms", self.smoothing_decay_ms),
            ("hangover_ms", self.hangover_ms),
            ("min_segment_ms", self.min_segment_ms),
            ("debounce_ms", self.debounce_ms),
            ("vad_min_speech_ms", self.vad_min_speech_ms),
            ("vad_min_silence_ms", self.vad_min_silence_ms),
            ("vad_speech_pad_ms", self.vad_speech_pad_ms),
            ("sample_rate", self.sample_rate),
            ("camera_debounce_ms", self.camera_debounce_ms),
            ("camera_min_segment_ms", self.camera_min_segment_ms),
            ("camera_takeover_min_segment_ms", self.camera_takeover_min_segment_ms),
            ("camera_takeover_context_ms", self.camera_takeover_context_ms),
            ("camera_both_min_segment_ms", self.camera_both_min_segment_ms),
            ("camera_silence_min_segment_ms", self.camera_silence_min_segment_ms),
        ]:
            if val <= 0:
                raise ValueError(f"{name} must be positive, got {val}")

        for name, val in [
            ("gate_fade_s", self.gate_fade_s),
            ("audio_overlap_s", self.audio_overlap_s),
            ("audio_pre_roll_s", self.audio_pre_roll_s),
            ("audio_post_roll_s", self.audio_post_roll_s),
            ("min_camera_event_s", self.min_camera_event_s),
            ("max_segment_sec", self.max_segment_sec),
            ("detection_pre_roll_s", self.detection_pre_roll_s),
            ("detection_post_roll_s", self.detection_post_roll_s),
            ("safety_window_s", self.safety_window_s),
            ("wide_duration_sec", self.wide_duration_sec),
            ("wide_cooldown_sec", self.wide_cooldown_sec),
            ("long_talk_threshold_sec", self.long_talk_threshold_sec),
            ("dialogue_wide_interval_sec", self.dialogue_wide_interval_sec),
            ("dialogue_wide_duration_sec", self.dialogue_wide_duration_sec),
            ("sticky_wide_max_bridge_sec", self.sticky_wide_max_bridge_sec),
            ("sticky_wide_max_turn_sec", self.sticky_wide_max_turn_sec),
            ("dialogue_cluster_max_span_sec", self.dialogue_cluster_max_span_sec),
            ("dialogue_cluster_max_turn_sec", self.dialogue_cluster_max_turn_sec),
            ("audio_mute_debounce_ms", self.audio_mute_debounce_ms),
            ("audio_mute_min_segment_ms", self.audio_mute_min_segment_ms),
            ("camera_neutral_min_segment_ms", self.camera_neutral_min_segment_ms),
        ]:
            if val < 0:
                raise ValueError(f"{name} must be non-negative, got {val}")

        if self.dialogue_wide_min_turns < 1:
            raise ValueError(
                f"dialogue_wide_min_turns must be >= 1, got {self.dialogue_wide_min_turns}"
            )
        if self.sticky_wide_min_turns < 1:
            raise ValueError(
                f"sticky_wide_min_turns must be >= 1, got {self.sticky_wide_min_turns}"
            )
        if self.dialogue_cluster_min_turns < 1:
            raise ValueError(
                f"dialogue_cluster_min_turns must be >= 1, got {self.dialogue_cluster_min_turns}"
            )

        for name, val in [
            ("default_camera", self.default_camera),
            ("both_speaking_camera", self.both_speaking_camera),
        ]:
            if val < 0:
                raise ValueError(f"{name} must be >= 0, got {val}")

        for inp in self.audio_inputs:
            if inp.camera_index < 0:
                raise ValueError(f"camera_index must be >= 0 for '{inp.speaker_label}', got {inp.camera_index}")

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
    def smoothing_attack_kernel(self) -> int:
        return max(1, int(self.smoothing_attack_ms / self.hop_ms))

    @property
    def smoothing_decay_kernel(self) -> int:
        return max(1, int(self.smoothing_decay_ms / self.hop_ms))

    @property
    def hangover_frames(self) -> int:
        return max(1, int(self.hangover_ms / self.hop_ms))

    @property
    def debounce_frames(self) -> int:
        return max(1, int(self.debounce_ms / self.hop_ms))

    @property
    def min_segment_frames(self) -> int:
        return max(1, int(self.min_segment_ms / self.hop_ms))

    @property
    def detection_pre_roll_frames(self) -> int:
        return max(0, int(self.detection_pre_roll_s * 1000.0 / self.hop_ms))

    @property
    def detection_post_roll_frames(self) -> int:
        return max(0, int(self.detection_post_roll_s * 1000.0 / self.hop_ms))

    @property
    def detection_min_on_frames(self) -> int:
        return max(1, int(self.detection_min_on_s * 1000.0 / self.hop_ms))

    @property
    def detection_min_off_frames(self) -> int:
        return max(1, int(self.detection_min_off_s * 1000.0 / self.hop_ms))

    @property
    def detection_hangover_frames(self) -> int:
        return max(0, int(self.detection_hangover_s * 1000.0 / self.hop_ms))

    @property
    def safety_window_frames(self) -> int:
        return max(1, int(self.safety_window_s * 1000.0 / self.hop_ms))

    @property
    def mask_filter_min_on_frames(self) -> int:
        return max(1, int(self.mask_filter_min_on_ms / self.hop_ms))

    @property
    def mask_filter_min_off_fill_frames(self) -> int:
        return max(1, int(self.mask_filter_min_off_fill_ms / self.hop_ms))

    @property
    def mask_filter_noise_window_frames(self) -> int:
        return max(1, int(self.mask_filter_noise_window_s * 1000.0 / self.hop_ms))
