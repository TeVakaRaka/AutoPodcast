"""Volume ducking automation with gate-style paired keyframes."""

from __future__ import annotations

import numpy as np

from autopodcast.models.domain import DuckingEvent, Segment, SpeakerState
from autopodcast.models.project import ProjectConfig


def _target_db_for_segment(
    seg: Segment, track_index: int, duck_db: float
) -> float:
    """Return target dB for a track in a given segment."""
    if seg.speaker_state == SpeakerState.SPEAKER_A:
        return 0.0 if track_index == 0 else duck_db
    elif seg.speaker_state == SpeakerState.SPEAKER_B:
        return duck_db if track_index == 0 else 0.0
    else:  # BOTH or SILENCE
        return 0.0


def _compute_noise_floor(audio: np.ndarray, sample_rate: int, window_ms: float) -> float:
    """Compute noise floor as 10th percentile of short-term RMS in dB."""
    window_samples = max(1, int(sample_rate * window_ms / 1000.0))
    n_windows = len(audio) // window_samples
    if n_windows == 0:
        return -96.0
    trimmed = audio[: n_windows * window_samples]
    frames = trimmed.reshape(n_windows, window_samples)
    rms_values = np.sqrt(np.mean(frames ** 2, axis=1))
    rms_values = rms_values[rms_values > 0]
    if len(rms_values) == 0:
        return -96.0
    db_values = 20.0 * np.log10(rms_values)
    return float(np.percentile(db_values, 10))


def _rms_at_time(
    audio: np.ndarray, sample_rate: int, time_s: float, window_ms: float
) -> float:
    """Compute short-term RMS in dB at a specific time (±window_ms/2)."""
    half_win = int(sample_rate * window_ms / 1000.0 / 2)
    center = int(time_s * sample_rate)
    start = max(0, center - half_win)
    end = min(len(audio), center + half_win)
    if start >= end:
        return -96.0
    chunk = audio[start:end]
    rms = np.sqrt(np.mean(chunk ** 2))
    if rms <= 0:
        return -96.0
    return float(20.0 * np.log10(rms))


def generate_ducking_events(
    segments: list[Segment],
    config: ProjectConfig,
    audio_arrays: list[np.ndarray] | None = None,
) -> list[DuckingEvent]:
    """Generate volume automation events with gate-style paired keyframes.

    Instead of single keyframes at segment boundaries (which Premiere
    interpolates with Bezier curves creating long ramps), this generates
    hold+target keyframe pairs that create clean step transitions with
    a short fade (gate_fade_s).

    For each track, at each segment boundary where the level changes:
      - hold keyframe at (T - gate_fade_s) with the OLD value
      - target keyframe at T with the NEW value

    This creates a step function with short fades instead of long ramps.
    """
    if not config.ducking_enabled:
        return []

    if not segments:
        return []

    duck_db = config.ducking_db
    fade = config.gate_fade_s
    events: list[DuckingEvent] = []

    # Pre-compute noise floors for RMS guard
    noise_floors: dict[int, float] = {}
    if config.rms_guard_enabled and audio_arrays is not None:
        for idx in range(min(2, len(audio_arrays))):
            noise_floors[idx] = _compute_noise_floor(
                audio_arrays[idx], config.sample_rate, config.rms_guard_window_ms,
            )

    for track_idx in range(2):
        prev_db = _target_db_for_segment(segments[0], track_idx, duck_db)

        # Initial keyframe at t=0
        events.append(DuckingEvent(time_s=0.0, target_db=prev_db, track_index=track_idx))

        for seg in segments[1:]:
            cur_db = _target_db_for_segment(seg, track_idx, duck_db)

            if cur_db != prev_db:
                # RMS guard: skip mute if audio has signal
                if (
                    config.rms_guard_enabled
                    and audio_arrays is not None
                    and track_idx in noise_floors
                    and cur_db < prev_db  # transition to mute
                ):
                    rms_db = _rms_at_time(
                        audio_arrays[track_idx],
                        config.sample_rate,
                        seg.start_s,
                        config.rms_guard_window_ms,
                    )
                    if rms_db > noise_floors[track_idx] + config.rms_guard_threshold_db:
                        # Signal detected — skip mute, keep prev_db
                        continue

                # Hold keyframe: keep old value just before transition
                hold_time = max(0.0, seg.start_s - fade)
                events.append(DuckingEvent(
                    time_s=hold_time, target_db=prev_db, track_index=track_idx,
                ))
                # Target keyframe: new value at transition point
                events.append(DuckingEvent(
                    time_s=seg.start_s, target_db=cur_db, track_index=track_idx,
                ))

            prev_db = cur_db

    return events
