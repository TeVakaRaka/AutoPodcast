"""Hysteresis-based speech activity detection with hangover."""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

import numpy as np

from autopodcast.models.domain import SpeakerActivity, SpeakerState
from autopodcast.models.project import ProjectConfig


class _DetectorState(Enum):
    INACTIVE = 0
    ACTIVE = 1
    HANGOVER = 2


class SileroVADUnavailableError(RuntimeError):
    """Raised when Silero VAD was requested but is unavailable."""


@lru_cache(maxsize=1)
def _load_silero_api():
    """Load Silero VAD functions lazily so RMS mode has no hard dependency."""
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad
    except Exception as exc:
        raise SileroVADUnavailableError(
            "Silero VAD is not available. Install the optional dependency with "
            "'pip install silero-vad' or use --detector-backend rms."
        ) from exc
    return load_silero_vad, get_speech_timestamps


@lru_cache(maxsize=2)
def _load_silero_model(use_onnx: bool):
    """Cache the Silero model instance for repeated detections."""
    load_silero_vad, _ = _load_silero_api()
    return load_silero_vad(onnx=use_onnx)


def _silero_detect_intervals(
    audio: np.ndarray,
    config: ProjectConfig,
) -> list[tuple[float, float]]:
    """Run Silero VAD and return speech intervals in seconds."""
    _, get_speech_timestamps = _load_silero_api()
    model = _load_silero_model(config.vad_use_onnx)

    wav = np.asarray(audio, dtype=np.float32).reshape(-1)
    wav = np.clip(wav, -1.0, 1.0)

    try:
        import torch
    except Exception:
        tensor = wav
    else:
        tensor = torch.from_numpy(wav)

    timestamps = get_speech_timestamps(
        tensor,
        model,
        sampling_rate=config.sample_rate,
        threshold=config.vad_threshold,
        min_speech_duration_ms=int(config.vad_min_speech_ms),
        min_silence_duration_ms=int(config.vad_min_silence_ms),
        speech_pad_ms=int(config.vad_speech_pad_ms),
        return_seconds=True,
    )

    intervals: list[tuple[float, float]] = []
    for item in timestamps:
        start_s = float(item.get("start", 0.0))
        end_s = float(item.get("end", 0.0))
        if end_s > start_s:
            intervals.append((start_s, end_s))
    return intervals


def _mark_activity_from_intervals(
    activity: SpeakerActivity,
    intervals: list[tuple[float, float]],
) -> None:
    """Project second-based intervals onto analysis frames."""
    frames = activity.frames
    if not frames:
        return

    for frame in frames:
        frame.is_active = False

    if not intervals:
        return

    idx = 0
    n_intervals = len(intervals)
    for frame in frames:
        while idx < n_intervals and intervals[idx][1] <= frame.time_s:
            idx += 1
        if idx >= n_intervals:
            break
        start_s, end_s = intervals[idx]
        if start_s <= frame.time_s < end_s:
            frame.is_active = True


def _finalize_activity_detection(
    activity: SpeakerActivity,
    config: ProjectConfig,
    diagnostics: dict | None,
    *,
    backend: str,
    safety_count: int,
    apply_mask_filter: bool,
    vad_intervals: list[tuple[float, float]] | None = None,
    warning: str | None = None,
) -> None:
    """Run shared post-processing and fill diagnostics."""
    _stabilize_frames(activity, config)
    if apply_mask_filter and config.mask_filter_enabled:
        _apply_mask_filter(activity, config, diagnostics)
    _dilate_frames(activity, config)

    if diagnostics is not None:
        active_count = sum(1 for f in activity.frames if f.is_active)
        total_count = len(activity.frames)
        hop_s = config.hop_ms / 1000.0
        diagnostics["detector_backend"] = backend
        diagnostics["total_frames"] = total_count
        diagnostics["active_frames"] = active_count
        diagnostics["active_seconds"] = round(active_count * hop_s, 3)
        diagnostics["total_seconds"] = round(total_count * hop_s, 3)
        diagnostics["safety_forced_frames"] = safety_count
        if vad_intervals is not None:
            diagnostics["vad_intervals"] = [
                {"start_s": round(start_s, 3), "end_s": round(end_s, 3)}
                for start_s, end_s in vad_intervals
            ]
        if warning:
            diagnostics["detector_warning"] = warning


def _detect_activity_rms(
    activity: SpeakerActivity,
    config: ProjectConfig,
    diagnostics: dict | None = None,
    *,
    backend: str = "rms",
    warning: str | None = None,
) -> None:
    """Apply the legacy RMS + hysteresis detector."""
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

    safety_count = _apply_safety_rule(activity, config)
    _finalize_activity_detection(
        activity,
        config,
        diagnostics,
        backend=backend,
        safety_count=safety_count,
        apply_mask_filter=True,
        warning=warning,
    )


def _detect_activity_silero(
    activity: SpeakerActivity,
    config: ProjectConfig,
    audio: np.ndarray,
    diagnostics: dict | None = None,
) -> None:
    """Apply Silero VAD and project timestamps onto frames."""
    intervals = _silero_detect_intervals(audio, config)
    _mark_activity_from_intervals(activity, intervals)
    _finalize_activity_detection(
        activity,
        config,
        diagnostics,
        backend="silero",
        safety_count=0,
        apply_mask_filter=False,
        vad_intervals=intervals,
    )


def _apply_safety_rule(activity: SpeakerActivity, config: ProjectConfig) -> int:
    """Force frames active where RMS clearly indicates speech.

    Returns number of frames forced active by safety rule.
    """
    frames = activity.frames
    n = len(frames)
    if n == 0:
        return 0

    safety_threshold = config.speech_threshold_db + config.safety_margin_db
    win = config.safety_window_frames
    forced = 0

    for i in range(n - win + 1):
        window_min = min(frames[j].rms_db for j in range(i, i + win))
        if window_min > safety_threshold:
            for j in range(i, i + win):
                if not frames[j].is_active:
                    frames[j].is_active = True
                    forced += 1

    return forced


def _stabilize_frames(activity: SpeakerActivity, config: ProjectConfig) -> None:
    """Remove short on/off bursts to prevent jitter."""
    frames = activity.frames
    n = len(frames)
    if n == 0:
        return

    min_on = config.detection_min_on_frames
    min_off = config.detection_min_off_frames
    hangover = config.detection_hangover_frames

    # Pass 1: remove short ON runs
    i = 0
    while i < n:
        if frames[i].is_active:
            run_start = i
            while i < n and frames[i].is_active:
                i += 1
            run_len = i - run_start
            if run_len < min_on:
                for j in range(run_start, i):
                    frames[j].is_active = False
        else:
            i += 1

    # Pass 2: fill short OFF runs
    i = 0
    while i < n:
        if not frames[i].is_active:
            run_start = i
            while i < n and not frames[i].is_active:
                i += 1
            run_len = i - run_start
            if run_len < min_off:
                for j in range(run_start, i):
                    frames[j].is_active = True
        else:
            i += 1

    # Pass 3: hangover extension
    if hangover > 0:
        i = 0
        while i < n:
            if frames[i].is_active:
                while i < n and frames[i].is_active:
                    i += 1
                # i is now first inactive after active run
                for j in range(i, min(i + hangover, n)):
                    frames[j].is_active = True
                i = i + hangover
            else:
                i += 1


def _compute_noise_floor(
    rms_db_arr: np.ndarray, window_frames: int, percentile: float,
) -> np.ndarray:
    """Compute a rolling noise floor estimate using a percentile over a window."""
    n = len(rms_db_arr)
    if n == 0:
        return np.array([], dtype=np.float64)

    noise_floor = np.empty(n, dtype=np.float64)
    half_w = window_frames // 2

    for t in range(n):
        lo = max(0, t - half_w)
        hi = min(n, t + half_w + 1)
        noise_floor[t] = np.percentile(rms_db_arr[lo:hi], percentile)

    return noise_floor


def _apply_mask_filter(
    activity: SpeakerActivity, config: ProjectConfig,
    diagnostics: dict | None = None,
) -> None:
    """SNR-aware post-filter: remove short ON segments that look like noise."""
    frames = activity.frames
    n = len(frames)
    if n == 0:
        return

    min_on = config.mask_filter_min_on_frames
    min_off_fill = config.mask_filter_min_off_fill_frames
    snr_strong = config.mask_filter_snr_peak_strong_db
    snr_weak = config.mask_filter_snr_peak_weak_db
    crest_min = config.mask_filter_crest_min_db

    # Extract arrays
    rms_db_arr = np.array([f.rms_db for f in frames], dtype=np.float64)
    peak_db_arr = np.array([f.peak_db for f in frames], dtype=np.float64)

    # Compute noise floor
    noise_floor = _compute_noise_floor(
        rms_db_arr, config.mask_filter_noise_window_frames, config.mask_filter_noise_percentile,
    )

    log_entries = []
    removed_count = 0

    # Pass 1: RLE and filter short ON segments
    i = 0
    while i < n:
        if frames[i].is_active:
            run_start = i
            while i < n and frames[i].is_active:
                i += 1
            run_len = i - run_start

            if run_len < min_on:
                # Evaluate SNR/crest
                seg_peak = float(np.max(peak_db_arr[run_start:i]))
                seg_rms = float(np.mean(rms_db_arr[run_start:i]))
                seg_noise = float(np.mean(noise_floor[run_start:i]))
                snr_peak = seg_peak - seg_noise
                crest = seg_peak - seg_rms

                keep = (snr_peak >= snr_strong) or (snr_peak >= snr_weak and crest >= crest_min)

                if not keep:
                    for j in range(run_start, i):
                        frames[j].is_active = False
                    removed_count += 1
                    if diagnostics is not None:
                        hop_s = config.hop_ms / 1000.0
                        log_entries.append({
                            "start_s": round(run_start * hop_s, 3),
                            "end_s": round(i * hop_s, 3),
                            "len_frames": run_len,
                            "action": "removed",
                            "peak_db": round(seg_peak, 1),
                            "rms_db": round(seg_rms, 1),
                            "noise_floor_db": round(seg_noise, 1),
                            "snr_peak": round(snr_peak, 1),
                            "crest": round(crest, 1),
                            "reason": "short_on_removed_low_snr",
                        })
                elif diagnostics is not None:
                    hop_s = config.hop_ms / 1000.0
                    log_entries.append({
                        "start_s": round(run_start * hop_s, 3),
                        "end_s": round(i * hop_s, 3),
                        "len_frames": run_len,
                        "action": "kept",
                        "peak_db": round(seg_peak, 1),
                        "rms_db": round(seg_rms, 1),
                        "noise_floor_db": round(seg_noise, 1),
                        "snr_peak": round(snr_peak, 1),
                        "crest": round(crest, 1),
                        "reason": "short_on_kept_high_snr",
                    })
        else:
            i += 1

    # Pass 2: fill short OFF gaps between ON segments
    gap_filled = 0
    i = 0
    while i < n:
        if not frames[i].is_active:
            run_start = i
            while i < n and not frames[i].is_active:
                i += 1
            run_len = i - run_start

            # Check if bounded by ON on both sides
            has_left = run_start > 0 and frames[run_start - 1].is_active
            has_right = i < n and frames[i].is_active

            if run_len < min_off_fill and has_left and has_right:
                for j in range(run_start, i):
                    frames[j].is_active = True
                gap_filled += 1
        else:
            i += 1

    if diagnostics is not None:
        diagnostics["mask_filter_log"] = log_entries
        diagnostics["mask_filter_removed"] = removed_count
        diagnostics["mask_filter_gap_filled"] = gap_filled


def _dilate_frames(activity: SpeakerActivity, config: ProjectConfig) -> None:
    """Extend active regions by pre-roll (backward) and post-roll (forward)."""
    frames = activity.frames
    n = len(frames)
    if n == 0:
        return

    pre = config.detection_pre_roll_frames
    post = config.detection_post_roll_frames

    if pre == 0 and post == 0:
        return

    # Collect active run boundaries first (to avoid mutating while scanning)
    runs = []
    i = 0
    while i < n:
        if frames[i].is_active:
            run_start = i
            while i < n and frames[i].is_active:
                i += 1
            runs.append((run_start, i))  # [start, end)
        else:
            i += 1

    # Dilate each run
    for run_start, run_end in runs:
        # Backward (pre-roll): conditional — stop at silence
        for j in range(run_start - 1, max(-1, run_start - pre - 1), -1):
            if frames[j].rms_db < config.silence_threshold_db:
                break
            frames[j].is_active = True

        # Forward (post-roll): unconditional
        for j in range(run_end, min(n, run_end + post)):
            frames[j].is_active = True


def detect_activity(
    activity: SpeakerActivity,
    config: ProjectConfig,
    diagnostics: dict | None = None,
    audio: np.ndarray | None = None,
) -> None:
    """Detect speech activity using the configured backend."""
    backend = config.detector_backend
    if backend == "rms":
        _detect_activity_rms(activity, config, diagnostics, backend="rms")
        return

    if backend == "silero":
        if audio is None:
            raise ValueError("Silero detector requires raw audio input")
        _detect_activity_silero(activity, config, audio, diagnostics)
        return

    if audio is not None:
        try:
            _detect_activity_silero(activity, config, audio, diagnostics)
            return
        except SileroVADUnavailableError as exc:
            warning = str(exc)
        except Exception as exc:
            warning = f"Silero VAD failed, falling back to RMS: {exc}"
        else:
            warning = None
        _detect_activity_rms(
            activity,
            config,
            diagnostics,
            backend="rms_fallback",
            warning=warning,
        )
        return

    _detect_activity_rms(activity, config, diagnostics, backend="rms_auto_no_audio")


def apply_cross_gate(
    states: list[SpeakerState],
    activity_a: SpeakerActivity,
    activity_b: SpeakerActivity,
    threshold_db: float = 6.0,
) -> list[SpeakerState]:
    """Suppress crosstalk in BOTH frames by comparing envelope levels.

    For frames marked BOTH, if one mic is significantly louder than the other
    (difference >= threshold_db), the quieter mic is considered crosstalk and
    the frame is reassigned to the louder speaker only.
    """
    result = []
    for i, state in enumerate(states):
        if state != SpeakerState.BOTH:
            result.append(state)
            continue
        a_db = activity_a.frames[i].envelope_db
        b_db = activity_b.frames[i].envelope_db
        diff = a_db - b_db
        if diff > 0 and diff >= threshold_db:
            result.append(SpeakerState.SPEAKER_A)
        elif diff < 0 and diff <= -threshold_db:
            result.append(SpeakerState.SPEAKER_B)
        else:
            result.append(SpeakerState.BOTH)
    return result


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
