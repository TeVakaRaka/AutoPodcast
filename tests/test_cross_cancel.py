"""Tests for cross-channel mic-bleed cancellation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import butter, fftconvolve, sosfilt

from autopodcast.core.cross_cancel import (
    CrossCancelConfig,
    apply_filters,
    compute_dominance_mask,
    cross_cancel,
    learn_filters,
)

SR = 16_000  # the project's working sample rate


def _synth_voice(duration_s: float, intervals_s, seed: int) -> np.ndarray:
    """Band-pass-filtered noise modulated by syllable envelope, zero outside intervals."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * SR)
    sos = butter(4, [200, 3500], btype="band", fs=SR, output="sos")
    voiced = sosfilt(sos, rng.standard_normal(n))
    envelope = np.zeros(n)
    for t0, t1 in intervals_s:
        cursor = t0
        while cursor < t1:
            dur = rng.uniform(0.10, 0.25)
            amp = rng.uniform(0.5, 1.0)
            i0 = int(cursor * SR)
            i1 = min(int((cursor + dur) * SR), int(t1 * SR), n)
            length = i1 - i0
            if length > 10:
                attack = int(length * 0.15)
                release = int(length * 0.25)
                hold = length - attack - release
                if hold > 0:
                    env = np.concatenate(
                        [
                            np.linspace(0, amp, attack),
                            np.full(hold, amp),
                            np.linspace(amp, 0, release),
                        ]
                    )
                    envelope[i0 : i0 + len(env)] += env
            cursor += dur + rng.uniform(0.05, 0.25)
    voice = voiced * envelope
    rms = float(np.sqrt(np.mean(voice ** 2)))
    if rms > 1e-9:
        voice *= 0.1 / rms
    return voice


def _room_ir(delay_ms: float, tail_ms: float = 6.0, seed: int = 0) -> np.ndarray:
    """Short impulse response: direct delay + a few early reflections."""
    rng = np.random.default_rng(seed)
    length = int((delay_ms + tail_ms) * SR / 1000)
    h = np.zeros(length)
    direct = int(delay_ms * SR / 1000)
    h[direct] = 1.0
    for k in range(3):
        offset = direct + int(rng.uniform(0.5, tail_ms - 0.5) * SR / 1000)
        if 0 < offset < length:
            h[offset] += (0.5 ** (k + 1)) * float(rng.choice([-1, 1]))
    return h


def _make_two_speaker_scenario(
    duration_s: float = 10.0,
    bleed_db: float = -18.0,
    delay_ms: float = 5.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-channel scenario with controlled bleed. Returns (tracks, oracle_voices)."""
    voice_a = _synth_voice(duration_s, [(0.0, 2.5), (7.5, duration_s)], seed=seed)
    voice_b = _synth_voice(duration_s, [(4.0, 6.5), (7.5, duration_s)], seed=seed + 1)
    h_ab = _room_ir(delay_ms, seed=11)
    h_ba = _room_ir(delay_ms * 0.85, seed=22)
    atten = 10 ** (bleed_db / 20.0)
    bleed_to_a = atten * fftconvolve(voice_b, h_ba, mode="same")
    bleed_to_b = atten * fftconvolve(voice_a, h_ab, mode="same")
    n = len(voice_a)
    rng = np.random.default_rng(seed + 100)
    floor = 1e-3
    track_a = voice_a + bleed_to_a + rng.standard_normal(n) * floor
    track_b = voice_b + bleed_to_b + rng.standard_normal(n) * floor
    tracks = np.stack([track_a, track_b])
    voices = np.stack([voice_a, voice_b])
    return tracks, voices


def _bleed_db_on(channel: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return -np.inf
    rms = float(np.sqrt(np.mean(channel[mask] ** 2) + 1e-12))
    return 20.0 * np.log10(rms + 1e-9)


# ---------------------------------------------------------------------------
# compute_dominance_mask
# ---------------------------------------------------------------------------


class TestComputeDominanceMask:
    def test_returns_correct_shape(self):
        tracks = np.random.randn(2, SR * 2)
        mask = compute_dominance_mask(tracks, SR, CrossCancelConfig())
        assert mask.shape == tracks.shape
        assert mask.dtype == bool

    def test_silence_never_dominates(self):
        tracks = np.zeros((2, SR * 2))
        mask = compute_dominance_mask(tracks, SR, CrossCancelConfig())
        assert mask.sum() == 0

    def test_loud_channel_dominates(self):
        n = SR * 3
        rng = np.random.default_rng(0)
        loud = rng.standard_normal(n) * 0.3
        quiet = rng.standard_normal(n) * 0.001
        tracks = np.stack([loud, quiet])
        mask = compute_dominance_mask(tracks, SR, CrossCancelConfig())
        assert mask[0].sum() > 0.5 * n
        assert mask[1].sum() == 0

    def test_single_channel_returns_empty_mask(self):
        tracks = np.random.randn(1, SR)
        mask = compute_dominance_mask(tracks, SR, CrossCancelConfig())
        assert mask.shape == (1, SR)
        assert mask.sum() == 0

    def test_2d_required(self):
        with pytest.raises(ValueError):
            compute_dominance_mask(np.zeros(SR), SR, CrossCancelConfig())

    def test_too_short_returns_empty(self):
        tracks = np.random.randn(2, 5)  # shorter than one frame
        mask = compute_dominance_mask(tracks, SR, CrossCancelConfig())
        assert mask.sum() == 0


# ---------------------------------------------------------------------------
# learn_filters
# ---------------------------------------------------------------------------


class TestLearnFilters:
    def test_no_training_data_skips_pair(self):
        tracks = np.zeros((2, SR))
        mask = np.zeros((2, SR), dtype=bool)
        filters = learn_filters(tracks, mask, SR, CrossCancelConfig())
        assert filters == {}

    def test_filter_length_matches_config(self):
        tracks, _ = _make_two_speaker_scenario(duration_s=8.0, seed=42)
        mask = compute_dominance_mask(tracks, SR, CrossCancelConfig())
        config = CrossCancelConfig(fir_taps=128)
        filters = learn_filters(tracks, mask, SR, config)
        for w in filters.values():
            assert w.shape == (128,)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            learn_filters(np.zeros((2, 100)), np.zeros((2, 99), dtype=bool), SR, CrossCancelConfig())

    def test_fir_taps_must_be_positive(self):
        tracks = np.zeros((2, SR))
        mask = np.zeros((2, SR), dtype=bool)
        with pytest.raises(ValueError):
            learn_filters(tracks, mask, SR, CrossCancelConfig(fir_taps=0))

    def test_identical_channels_yield_near_unity_filter(self):
        """If both channels carry the same signal, fitting "bleed B onto A" gives w ≈ delta."""
        rng = np.random.default_rng(0)
        sig = rng.standard_normal(SR * 4) * 0.1
        tracks = np.stack([sig, sig])
        # Force training data
        mask = np.zeros_like(tracks, dtype=bool)
        mask[1, : SR * 2] = True  # pretend only B speaks in first 2 s
        filters = learn_filters(tracks, mask, SR, CrossCancelConfig(fir_taps=64))
        w = filters[(0, 1)]
        # Peak should be at lag 0 and dominate
        assert int(np.argmax(np.abs(w))) == 0
        assert abs(w[0]) > 0.5 * np.sum(np.abs(w[1:]))


# ---------------------------------------------------------------------------
# apply_filters
# ---------------------------------------------------------------------------


class TestApplyFilters:
    def test_empty_filters_returns_copy(self):
        tracks = np.random.randn(2, 1000).astype(np.float64)
        out = apply_filters(tracks, {})
        assert out.shape == tracks.shape
        assert np.allclose(out, tracks)
        # must be a copy, not a view
        out[0, 0] = 999.0
        assert tracks[0, 0] != 999.0

    def test_subtraction_actually_subtracts(self):
        # Identity filter -> output[i] = input[i] - input[j]
        tracks = np.array([[1.0, 2.0, 3.0, 4.0], [0.5, 0.5, 0.5, 0.5]])
        identity = np.array([1.0])
        cleaned = apply_filters(tracks, {(0, 1): identity})
        np.testing.assert_allclose(cleaned[0], tracks[0] - tracks[1])
        np.testing.assert_allclose(cleaned[1], tracks[1])

    def test_preserves_shape_and_dtype(self):
        tracks = np.random.randn(3, 500)
        w = np.random.randn(64)
        cleaned = apply_filters(tracks, {(0, 1): w, (2, 1): w})
        assert cleaned.shape == tracks.shape
        assert cleaned.dtype == np.float64

    def test_ignores_invalid_indices(self):
        tracks = np.random.randn(2, 200).astype(np.float64)
        cleaned = apply_filters(tracks, {(5, 6): np.array([1.0]), (0, 0): np.array([1.0])})
        np.testing.assert_allclose(cleaned, tracks)


# ---------------------------------------------------------------------------
# end-to-end cross_cancel
# ---------------------------------------------------------------------------


class TestCrossCancelEndToEnd:
    def test_single_channel_passes_through(self):
        sig = np.random.randn(SR)
        out = cross_cancel([sig], SR)
        assert len(out) == 1
        np.testing.assert_array_equal(out[0], sig)

    def test_reduces_bleed_on_typical_scenario(self):
        """Synthetic -18 dB / 5 ms bleed: bleed on victim mic during 'only other'
        frames must drop after cancellation."""
        tracks, _ = _make_two_speaker_scenario(
            duration_s=10.0, bleed_db=-18.0, delay_ms=5.0, seed=42
        )
        config = CrossCancelConfig(fir_taps=256)
        mask = compute_dominance_mask(tracks, SR, config)
        # Bleed level on track 0 during "only 1" frames, before vs after
        before = _bleed_db_on(tracks[0], mask[1])
        cleaned = np.stack(cross_cancel([tracks[0], tracks[1]], SR, config))
        after = _bleed_db_on(cleaned[0], mask[1])
        assert after < before, f"bleed should drop; before={before:.2f} after={after:.2f}"
        # Require a non-trivial reduction
        assert (before - after) > 0.5, f"bleed reduction too small: {before - after:.2f} dB"

    def test_preserves_own_voice(self):
        """Speaker's own voice should survive cancellation almost unchanged."""
        tracks, voices = _make_two_speaker_scenario(
            duration_s=10.0, bleed_db=-18.0, delay_ms=5.0, seed=42
        )
        config = CrossCancelConfig(fir_taps=256)
        mask = compute_dominance_mask(tracks, SR, config)
        cleaned = np.stack(cross_cancel([tracks[0], tracks[1]], SR, config))
        # On "only 0" frames, channel 0 should still be close to voice 0
        m = mask[0]
        if m.sum() > 0:
            diff = cleaned[0, m] - voices[0, m]
            voice_rms = float(np.sqrt(np.mean(voices[0, m] ** 2) + 1e-12))
            diff_rms = float(np.sqrt(np.mean(diff ** 2) + 1e-12))
            distortion_db = 20.0 * np.log10(diff_rms / voice_rms)
            assert distortion_db < -20.0, f"own-voice distortion too high: {distortion_db:.2f} dB"

    def test_deterministic(self):
        """Same input → bit-exact same output."""
        tracks, _ = _make_two_speaker_scenario(seed=7)
        out_a = np.stack(cross_cancel([tracks[0], tracks[1]], SR))
        out_b = np.stack(cross_cancel([tracks[0], tracks[1]], SR))
        np.testing.assert_array_equal(out_a, out_b)

    def test_three_channels(self):
        """Pipeline works for >2 channels; learns 6 directional filters when possible."""
        a = _synth_voice(8.0, [(0.0, 2.0)], seed=1)
        b = _synth_voice(8.0, [(2.5, 4.5)], seed=2)
        c = _synth_voice(8.0, [(5.0, 7.5)], seed=3)
        atten = 10 ** (-18.0 / 20.0)
        h = _room_ir(5.0, seed=99)
        n = len(a)

        def with_bleed(target, others):
            bleed = sum(atten * fftconvolve(o, h, mode="same") for o in others)
            return target + bleed

        tracks_list = [with_bleed(a, [b, c]), with_bleed(b, [a, c]), with_bleed(c, [a, b])]
        cleaned = cross_cancel(tracks_list, SR, CrossCancelConfig(fir_taps=192))
        assert len(cleaned) == 3
        for arr in cleaned:
            assert arr.shape == (n,)
            assert np.all(np.isfinite(arr))

    def test_handles_silent_channel(self):
        sig = _synth_voice(6.0, [(0.0, 3.0)], seed=5)
        silent = np.zeros_like(sig)
        cleaned = cross_cancel([sig, silent], SR)
        # Should not raise and shouldn't blow up the live channel
        assert np.all(np.isfinite(cleaned[0]))
        assert np.all(np.isfinite(cleaned[1]))

    def test_input_arrays_not_mutated(self):
        tracks, _ = _make_two_speaker_scenario(seed=11)
        orig0 = tracks[0].copy()
        orig1 = tracks[1].copy()
        _ = cross_cancel([tracks[0], tracks[1]], SR)
        np.testing.assert_array_equal(tracks[0], orig0)
        np.testing.assert_array_equal(tracks[1], orig1)

    def test_returns_list_of_arrays(self):
        tracks, _ = _make_two_speaker_scenario(seed=12)
        out = cross_cancel([tracks[0], tracks[1]], SR)
        assert isinstance(out, list)
        assert all(isinstance(a, np.ndarray) for a in out)
        assert len(out) == 2
        assert all(a.shape == tracks[0].shape for a in out)
