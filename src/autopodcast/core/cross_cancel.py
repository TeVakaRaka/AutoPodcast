"""Cross-channel mic-bleed cancellation via Wiener-Hopf normal equations.

Pure functions over numpy arrays. No I/O, no global state.

The pipeline learns, for each pair of channels (i, j), a short FIR filter
W_ji that predicts what bleed from channel j looks like on channel i, by
fitting on samples where dominance is unambiguous (only j speaks). The
filter is then applied across the whole episode and subtracted from
channel i.

This is the closed-form (batch) Wiener-Hopf solution — same idea as an
NLMS adaptive cross-canceller, but solved once per episode instead of
sample-by-sample, so the result is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.linalg import LinAlgError, solve_toeplitz
from scipy.signal import correlate


@dataclass(frozen=True)
class CrossCancelConfig:
    """Configuration for cross-channel cancellation.

    fir_taps              FIR length in samples. At 16 kHz, 256 ≈ 16 ms.
                          Should cover direct delay + early reflections.
    ridge                 Tikhonov regularisation on R[0]; protects against
                          near-singular autocorrelation matrices.
    frame_ms              Frame size for the dominance detector.
    dominance_margin_db   How much louder channel i must be vs the others
                          to count as "only i" for training.
    silence_floor_db      Frames below this absolute RMS don't count as
                          dominant (avoid training on background noise).
    min_train_seconds     If a pair has fewer than this many seconds of
                          unambiguous training data, skip cancellation
                          for that pair (return identity).
    """

    fir_taps: int = 256
    ridge: float = 1e-6
    frame_ms: float = 20.0
    dominance_margin_db: float = 8.0
    silence_floor_db: float = -45.0
    min_train_seconds: float = 0.5


def compute_dominance_mask(
    tracks: np.ndarray,
    sample_rate: int,
    config: CrossCancelConfig,
) -> np.ndarray:
    """Per-sample boolean mask: only_mask[i, t] == True iff at time t channel i
    is louder than every other channel by `config.dominance_margin_db` AND
    above `config.silence_floor_db`.

    Parameters
    ----------
    tracks : (n_channels, n_samples) float
    sample_rate : Hz
    config : CrossCancelConfig

    Returns
    -------
    only_mask : (n_channels, n_samples) bool
    """
    if tracks.ndim != 2:
        raise ValueError(f"tracks must be 2D (n_channels, n_samples), got shape {tracks.shape}")
    n_ch, n = tracks.shape
    if n_ch < 2:
        return np.zeros((n_ch, n), dtype=bool)
    frame_len = max(1, int(round(config.frame_ms * sample_rate / 1000.0)))
    n_frames = n // frame_len
    if n_frames == 0:
        return np.zeros((n_ch, n), dtype=bool)

    # Per-frame RMS in dB
    rms_db = np.full((n_ch, n_frames), -120.0, dtype=np.float64)
    for i in range(n_ch):
        for f in range(n_frames):
            chunk = tracks[i, f * frame_len : (f + 1) * frame_len]
            rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2) + 1e-12))
            rms_db[i, f] = 20.0 * np.log10(rms + 1e-9)

    # Dominance: ch i wins by margin over all other channels AND is above floor
    dom = np.zeros((n_ch, n_frames), dtype=bool)
    for i in range(n_ch):
        others = [j for j in range(n_ch) if j != i]
        margin_ok = np.all(rms_db[i] > rms_db[others] + config.dominance_margin_db, axis=0)
        loud_enough = rms_db[i] > config.silence_floor_db
        dom[i] = margin_ok & loud_enough

    # Expand frame mask to per-sample
    only_mask = np.zeros((n_ch, n), dtype=bool)
    for i in range(n_ch):
        for f in range(n_frames):
            if dom[i, f]:
                only_mask[i, f * frame_len : (f + 1) * frame_len] = True
    return only_mask


def learn_filters(
    tracks: np.ndarray,
    only_mask: np.ndarray,
    sample_rate: int,
    config: CrossCancelConfig,
) -> dict[tuple[int, int], np.ndarray]:
    """Fit a Wiener-Hopf FIR filter for each pair (victim_i, interferer_j).

    For each ordered pair (i, j), i != j:
      - take samples where only_mask[j] is True (only j speaks),
      - solve R w = r, where
          R[k] = sum_t xj[t] * xj[t-k]   (autocorr of xj on training mask)
          r[k] = sum_t xi[t] * xj[t-k]   (cross-corr xi vs xj on training mask)
        for k = 0..L-1, via a symmetric Toeplitz solver.

    Returns
    -------
    filters : dict (i, j) -> np.ndarray of length config.fir_taps.
              Pairs with insufficient training data are omitted (no filter
              applied for that pair).
    """
    if tracks.shape != only_mask.shape:
        raise ValueError(
            f"tracks and only_mask must have same shape, got {tracks.shape} vs {only_mask.shape}"
        )
    n_ch, n = tracks.shape
    L = int(config.fir_taps)
    if L < 1:
        raise ValueError(f"fir_taps must be >= 1, got {L}")
    min_samples = max(4 * L, int(config.min_train_seconds * sample_rate))

    filters: dict[tuple[int, int], np.ndarray] = {}
    for i in range(n_ch):
        for j in range(n_ch):
            if i == j:
                continue
            mask_j = only_mask[j]
            if int(mask_j.sum()) < min_samples:
                continue

            xj = tracks[j].astype(np.float64)
            xi = tracks[i].astype(np.float64)
            xj_m = xj * mask_j
            xi_m = xi * mask_j

            ac = correlate(xj_m, xj_m, mode="full", method="fft")
            center = len(ac) // 2
            R = ac[center : center + L].astype(np.float64).copy()
            if R[0] <= 0.0:
                continue
            R[0] *= 1.0 + config.ridge

            cc = correlate(xi_m, xj_m, mode="full", method="fft")
            r = cc[center : center + L].astype(np.float64)

            try:
                w = solve_toeplitz(R, r)
            except (LinAlgError, ValueError):
                continue
            if not np.all(np.isfinite(w)):
                continue
            filters[(i, j)] = w
    return filters


def apply_filters(
    tracks: np.ndarray,
    filters: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    """Subtract the learned bleed estimates from each channel.

    For each (i, j) in filters: cleaned[i] -= conv(tracks[j], W_ji).
    """
    if tracks.ndim != 2:
        raise ValueError(f"tracks must be 2D, got {tracks.shape}")
    n_ch, n = tracks.shape
    cleaned = tracks.astype(np.float64, copy=True)
    for (i, j), w in filters.items():
        if i == j or not (0 <= i < n_ch and 0 <= j < n_ch):
            continue
        bleed_estimate = np.convolve(tracks[j].astype(np.float64), w, mode="full")[:n]
        cleaned[i] -= bleed_estimate
    return cleaned


def cross_cancel(
    audio_arrays: Sequence[np.ndarray],
    sample_rate: int,
    config: CrossCancelConfig | None = None,
) -> list[np.ndarray]:
    """High-level convenience wrapper.

    Takes a list of equal-length per-channel arrays (as produced by
    `load_and_align`), runs the full pipeline (dominance mask → filter
    learning → subtraction), and returns the cleaned arrays in the
    same shape/order.

    If there are fewer than 2 channels, the input is returned unchanged.
    Channels that lacked enough unambiguous training data fall through
    unchanged for the corresponding pairs.
    """
    if config is None:
        config = CrossCancelConfig()
    if len(audio_arrays) < 2:
        return [np.asarray(a) for a in audio_arrays]
    tracks = np.stack([np.asarray(a, dtype=np.float64) for a in audio_arrays])
    mask = compute_dominance_mask(tracks, sample_rate, config)
    filters = learn_filters(tracks, mask, sample_rate, config)
    cleaned = apply_filters(tracks, filters)
    return [cleaned[i] for i in range(cleaned.shape[0])]
