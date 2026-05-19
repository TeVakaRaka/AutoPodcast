"""Wiener-Hopf cross-channel cancellation prototype for podcast mic bleed.

Generates synthetic 2-mic scenarios with controlled bleed (room IR + attenuation),
fits a per-pair FIR filter on solo segments via the closed-form Wiener-Hopf
normal equations, applies it to the full episode, and reports:

  - bleed level (dB) on victim mic during "only other" frames, before vs after
  - false-trigger rate at the project's default -28 dB speech threshold,
    before vs after
  - residual distortion on "only this" frames (lower is better)

Run:
    python3 experiments/cross_cancel_prototype.py
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import butter, correlate, fftconvolve, sosfilt

SR = 48_000


def synth_voice(duration_s: float, active_intervals_s, syllable_rate=4.0, seed=0):
    """Voice-like signal: bandpass-filtered noise with syllable envelope.

    The signal is zero outside ``active_intervals_s`` (list of (t0, t1) pairs)
    so we have ground-truth solo regions.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * SR)
    base = rng.standard_normal(n)
    sos = butter(4, [200, 3500], btype="band", fs=SR, output="sos")
    voiced = sosfilt(sos, base)

    envelope = np.zeros(n)
    for t0, t1 in active_intervals_s:
        cursor = t0
        while cursor < t1:
            dur = rng.uniform(0.08, 0.25)
            amp = rng.uniform(0.4, 1.0)
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
            cursor += dur + rng.uniform(0.05, 0.30)

    voice = voiced * envelope
    rms = float(np.sqrt(np.mean(voice ** 2)))
    if rms > 1e-9:
        voice *= 0.1 / rms
    return voice


def make_room_response(delay_ms=5.0, num_reflections=4, decay=0.5, tail_ms=8.0, seed=42):
    """Short impulse response: direct delay + a few early reflections within tail_ms."""
    rng = np.random.default_rng(seed)
    length = int((delay_ms + tail_ms) * SR / 1000)
    h = np.zeros(length)
    direct = int(delay_ms * SR / 1000)
    h[direct] = 1.0
    for k in range(num_reflections):
        offset = direct + int(rng.uniform(0.5, tail_ms - 0.5) * SR / 1000)
        if 0 < offset < length:
            h[offset] += (decay ** (k + 1)) * float(rng.choice([-1, 1]))
    return h


def make_scenario(duration_s=12.0, bleed_db=-18.0, delay_ms=5.0, seed=0):
    """Two-mic scenario with cross-bleed and disjoint solo regions.

    Timeline (sec):
       0.0 - 3.0   only A
       3.0 - 5.0   silence
       5.0 - 8.0   only B
       8.0 - 9.0   silence
       9.0 - 12.0  overlap

    Returns
    -------
    tracks       : (2, N) float64 - mic recordings with bleed + noise
    voices_clean : (2, N) float64 - ground-truth isolated voices
    """
    voice_a = synth_voice(
        duration_s,
        active_intervals_s=[(0.0, 3.0), (9.0, 12.0)],
        syllable_rate=4.0,
        seed=seed,
    )
    voice_b = synth_voice(
        duration_s,
        active_intervals_s=[(5.0, 8.0), (9.0, 12.0)],
        syllable_rate=3.5,
        seed=seed + 1,
    )

    h_ab = make_room_response(delay_ms=delay_ms, decay=0.5, seed=11)
    h_ba = make_room_response(delay_ms=delay_ms * 0.85, decay=0.5, seed=22)
    attenuation = 10 ** (bleed_db / 20.0)

    bleed_to_a = attenuation * fftconvolve(voice_b, h_ba, mode="same")
    bleed_to_b = attenuation * fftconvolve(voice_a, h_ab, mode="same")

    n = len(voice_a)
    rng = np.random.default_rng(seed + 100)
    floor = 1e-3  # ~-60 dB, realistic lavalier self-noise + room tone
    noise_a = rng.standard_normal(n) * floor
    noise_b = rng.standard_normal(n) * floor

    tracks = np.stack([voice_a + bleed_to_a + noise_a, voice_b + bleed_to_b + noise_b])
    voices = np.stack([voice_a, voice_b])
    return tracks, voices


def dominance_mask(tracks, frame_ms=20, margin_db=8.0):
    """Per-sample boolean mask: only_mask[i] = True where channel i dominates by margin."""
    n_ch, n = tracks.shape
    frame_len = int(frame_ms * SR / 1000)
    n_frames = n // frame_len
    rms_db = np.full((n_ch, n_frames), -120.0)
    for i in range(n_ch):
        for f in range(n_frames):
            chunk = tracks[i, f * frame_len : (f + 1) * frame_len]
            rms = float(np.sqrt(np.mean(chunk ** 2) + 1e-12))
            rms_db[i, f] = 20.0 * np.log10(rms + 1e-9)

    dom = np.zeros((n_ch, n_frames), dtype=bool)
    for i in range(n_ch):
        others = [j for j in range(n_ch) if j != i]
        dom[i] = np.all(rms_db[i] > rms_db[others] + margin_db, axis=0)
        # Require some absolute energy so silence doesn't get flagged as "dominant"
        dom[i] &= rms_db[i] > -45.0

    only_mask = np.zeros((n_ch, n), dtype=bool)
    for i in range(n_ch):
        for f in range(n_frames):
            if dom[i, f]:
                only_mask[i, f * frame_len : (f + 1) * frame_len] = True
    return only_mask


def wiener_hopf_debleed(tracks, only_mask, fir_taps=384, ridge=1e-6):
    """Per-pair (i, j) closed-form FIR cancellation learned on "only j" frames.

    For each victim mic i and interferer j != i:
      1. Restrict signals to samples where only j speaks
      2. Solve symmetric Toeplitz system R w = r where
           R[k] = autocorr(xj_m)[k]   k = 0..L-1
           r[k] = cross-corr(xi_m, xj_m)[k]
      3. Subtract conv(xj, w) from xi everywhere
    """
    n_ch, n = tracks.shape
    cleaned = tracks.copy().astype(np.float64)
    filters = {}
    L = int(fir_taps)

    for i in range(n_ch):
        for j in range(n_ch):
            if i == j:
                continue
            mask_j = only_mask[j]
            if mask_j.sum() < 4 * L:
                continue
            xj = tracks[j].astype(np.float64)
            xi = tracks[i].astype(np.float64)
            xj_m = xj * mask_j
            xi_m = xi * mask_j

            ac = correlate(xj_m, xj_m, mode="full", method="fft")
            center = len(ac) // 2
            R = ac[center : center + L].astype(np.float64)
            R[0] += ridge * (R[0] + 1e-9)

            cc = correlate(xi_m, xj_m, mode="full", method="fft")
            r = cc[center : center + L].astype(np.float64)

            try:
                w = solve_toeplitz(R, r)
            except np.linalg.LinAlgError:
                continue

            bleed_estimate = np.convolve(xj, w, mode="full")[:n]
            cleaned[i] -= bleed_estimate
            filters[(i, j)] = w
    return cleaned, filters


def measure_bleed_db(tracks, only_mask):
    """Average RMS (dB) of channel i during 'only j' frames, j != i."""
    out = {}
    for i in range(tracks.shape[0]):
        for j in range(tracks.shape[0]):
            if i == j:
                continue
            m = only_mask[j]
            if m.sum() == 0:
                out[(i, j)] = -np.inf
                continue
            rms = float(np.sqrt(np.mean(tracks[i, m] ** 2) + 1e-12))
            out[(i, j)] = 20.0 * np.log10(rms + 1e-9)
    return out


def measure_distortion_db(tracks_clean, voices_oracle, only_mask):
    """Residual error vs oracle voice on 'only i' frames, normalised (dB)."""
    out = {}
    for i in range(tracks_clean.shape[0]):
        m = only_mask[i]
        if m.sum() == 0:
            out[i] = float("nan")
            continue
        diff = tracks_clean[i, m] - voices_oracle[i, m]
        voice = voices_oracle[i, m]
        s = float(np.sqrt(np.mean(voice ** 2) + 1e-12))
        d = float(np.sqrt(np.mean(diff ** 2) + 1e-12))
        if s < 1e-6:
            out[i] = float("nan")
        else:
            out[i] = 20.0 * np.log10(d / s)
    return out


def false_trigger_rate(tracks, only_mask, threshold_db=-28.0, frame_ms=20):
    """% of frames inside 'only j' regions where channel i RMS exceeds threshold."""
    n_ch, n = tracks.shape
    frame_len = int(frame_ms * SR / 1000)
    n_frames = n // frame_len
    out = {}
    for i in range(n_ch):
        for j in range(n_ch):
            if i == j:
                continue
            mask = only_mask[j]
            trig = total = 0
            for f in range(n_frames):
                f0, f1 = f * frame_len, (f + 1) * frame_len
                if mask[f0:f1].sum() < 0.5 * frame_len:
                    continue
                rms = float(np.sqrt(np.mean(tracks[i, f0:f1] ** 2) + 1e-12))
                total += 1
                if 20.0 * np.log10(rms + 1e-9) > threshold_db:
                    trig += 1
            out[(i, j)] = (trig / total * 100.0) if total else 0.0
    return out


SCENARIOS = [
    ("strong  (-12 dB, 5 ms)", -12.0, 5.0),
    ("typical (-18 dB, 5 ms)", -18.0, 5.0),
    ("weak    (-25 dB, 5 ms)", -25.0, 5.0),
    ("longdly (-18 dB, 8 ms)", -18.0, 8.0),
]
FIR_TAPS = [256, 512, 768, 1024]
SEEDS = [42, 7, 1337]
DURATION_S = 20.0


def run_one(bleed_db, delay_ms, seed, fir_taps):
    tracks, voices = make_scenario(
        duration_s=DURATION_S, bleed_db=bleed_db, delay_ms=delay_ms, seed=seed
    )
    mask = dominance_mask(tracks)
    bb = float(np.mean(list(measure_bleed_db(tracks, mask).values())))
    ftb = float(np.mean(list(false_trigger_rate(tracks, mask).values())))
    cleaned, filters = wiener_hopf_debleed(tracks, mask, fir_taps=fir_taps)
    ba = float(np.mean(list(measure_bleed_db(cleaned, mask).values())))
    fta = float(np.mean(list(false_trigger_rate(cleaned, mask).values())))
    da = float(np.nanmean(list(measure_distortion_db(cleaned, voices, mask).values())))
    return bb, ba, ftb, fta, da, filters


def describe_filter_peak(filters, expected_delay_ms):
    out = []
    for (i, j), w in filters.items():
        peak_idx = int(np.argmax(np.abs(w)))
        peak_ms = peak_idx * 1000.0 / SR
        peak_db = 20.0 * np.log10(abs(w[peak_idx]) + 1e-9)
        out.append(
            f"pair {j}->{i}: peak at {peak_ms:5.2f} ms (expected ~{expected_delay_ms:.1f}), amp {peak_db:+5.1f} dB"
        )
    return out


def main():
    print("=" * 100)
    print(
        f"Wiener-Hopf Cross-Channel Cancellation — averaged over {len(SEEDS)} seeds, {DURATION_S}s clips"
    )
    print("=" * 100)
    print(
        f"{'Scenario':<24} {'FIR':>5} "
        f"{'BleedBef':>9} {'BleedAft':>9} {'Δ':>7} "
        f"{'FT-bef':>7} {'FT-aft':>7} {'FT-drop':>8} {'Distort':>8}"
    )
    print("-" * 100)

    for scen_name, bleed_db, delay_ms in SCENARIOS:
        first = True
        for L in FIR_TAPS:
            rows = [run_one(bleed_db, delay_ms, seed, L) for seed in SEEDS]
            bb = float(np.mean([r[0] for r in rows]))
            ba = float(np.mean([r[1] for r in rows]))
            ftb = float(np.mean([r[2] for r in rows]))
            fta = float(np.mean([r[3] for r in rows]))
            da = float(np.mean([r[4] for r in rows]))
            ratio_str = f"{ftb / fta:>5.1f}x" if fta > 0.05 else "  inf"
            name = scen_name if first else ""
            first = False
            print(
                f"{name:<24} {L:>5} "
                f"{bb:>+9.2f} {ba:>+9.2f} {bb-ba:>+7.2f} "
                f"{ftb:>6.1f}% {fta:>6.1f}% {ratio_str:>8} {da:>+8.2f}"
            )
        print()

    print("-" * 100)
    print("Filter shape diagnostic (typical -18 dB / 5 ms scenario, FIR=768, seed=42):")
    print("-" * 100)
    _, _, _, _, _, filters = run_one(-18.0, 5.0, 42, 768)
    for line in describe_filter_peak(filters, expected_delay_ms=5.0):
        print("  " + line)
    print()
    print("Legend:")
    print("  BleedBef/Aft : RMS dB on victim mic during 'only other' frames")
    print("  Δ            : bleed reduction in dB (higher is better)")
    print("  FT-bef/aft   : % of 'only other' frames triggering > -28 dB detector")
    print("  FT-drop      : ratio FT-before / FT-after (higher is better)")
    print("  Distort      : residual error vs oracle voice on own-speech frames (lower = better)")


if __name__ == "__main__":
    main()
