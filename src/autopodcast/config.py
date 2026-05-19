"""Configuration helpers for building ProjectConfig from CLI args."""

from __future__ import annotations

from pathlib import Path

from autopodcast.core.roles import normalize_speaker_role
from autopodcast.models.project import AudioInput, CameraInput, ProjectConfig


def build_config(
    audio_a: str,
    label_a: str,
    camera_a: int,
    audio_b: str,
    label_b: str,
    camera_b: int,
    camera_wide: int = 0,
    output_dir: str = ".",
    **overrides,
) -> ProjectConfig:
    """Build ProjectConfig from CLI-style arguments."""
    config = ProjectConfig(
        audio_inputs=[
            AudioInput(path=Path(audio_a), speaker_label=label_a, camera_index=camera_a),
            AudioInput(path=Path(audio_b), speaker_label=label_b, camera_index=camera_b),
        ],
        output_dir=Path(output_dir),
        default_camera=camera_wide,
        both_speaking_camera=camera_wide,
    )

    for key, value in overrides.items():
        if value is not None and hasattr(config, key):
            object.__setattr__(config, key, value)

    # Normalize speaker labels using role overrides and canonical map
    for inp in config.audio_inputs:
        resolved = config.speaker_role_overrides.get(inp.speaker_label)
        if resolved is None:
            resolved = normalize_speaker_role(inp.speaker_label)
        inp.speaker_label = resolved

    config.validate()
    return config
