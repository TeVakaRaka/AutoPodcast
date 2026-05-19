"""Wiener-Hopf cross-channel cancellation — synthetic evaluation harness.

Drives ``autopodcast.core.cross_cancel`` against a bank of synthetic 2/3/4-mic
scenarios with controlled bleed and reports SNR, false-trigger rate, and
own-voice distortion.

Run:
    python3 experiments/cross_cancel_prototype.py
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from autopodcast.core.cross_cancel import (
    CrossCancelConfig,
    compute_dominance_mask,
    cross_cancel,
)

SR = 16_000


# ---------------------------------------------------------------------------
# Synthesis helpers (only used here, kept out of production)
# ---------------------------------------------------------------------------


def _synth_voice(duration_s: float, active_intervals_s, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(duration_s * SR)
    sos = butter(4, [200, 3500], btype="band", fs=SR, output="sos")
    voiced = sosfilt(sos, rng.standard_normal(n))
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


def _room_ir(delay_ms: float, tail_ms: float = 6.0, seed: int = 42) -> np.ndarray:
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


def _two_speaker_scene(duration_s, bleed_db, delay_ms, seed):
    voice_a = _synth_voice(duration_s, [(0.0, 2.5), (7.5, duration_s)], seed=seed)
    voice_b = _synth_voice(duration_s, [(4.0, 6.5), (7.5, duration_s)], seed=seed + 1)
    h_ab = _room_ir(delay_ms, seed=11)
    h_ba = _room_ir(delay_ms * 0.85, seed=22)
    atten = 10 ** (bleed_db / 20.0)
    bleed_a = atten * fftconvolve(voice_b, h_ba, mode="same")
    bleed_b = atten * fftconvolve(voice_a, h_ab, mode="same")
    n = len(voice_a)
    rng = np.random.default_rng(seed + 100)
    floor = 1e-3
    tracks = np.stack(
        [voice_a + bleed_a + rng.standard_normal(n) * floor,
         voice_b + bleed_b + rng.standard_normal(n) * floor]
    )
    voices = np.stack([voice_a, voice_b])
    return tracks, voices


def _four_speaker_scene(duration_s, bleed_db, delay_ms, seed):
    """1 host + 3 guests scenario with partial overlap."""
    intervals = [
        [(0.0, 2.0), (8.0, 10.0)],
        [(2.5, 4.5), (8.0, 10.0)],
        [(5.0, 6.5)],
        [(6.5, 7.5), (8.0, 10.0)],
    ]
    voices = [_synth_voice(duration_s, ints, seed=seed + i) for i, ints in enumerate(intervals)]
    n = len(voices[0])
    atten = 10 ** (bleed_db / 20.0)
    rng = np.random.default_rng(seed + 500)
    floor = 1e-3
    tracks = []
    for i, target in enumerate(voices):
        bleed_sum = np.zeros(n)
        for j, source in enumerate(voices):
            if j == i:
                continue
            h = _room_ir(delay_ms * (0.8 + 0.1 * j), seed=100 * i + j)
            bleed_sum += atten * fftconvolve(source, h, mode="same")
        tracks.append(target + bleed_sum + rng.standard_normal(n) * floor)
    return np.stack(tracks), np.stack(voices)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _bleed_db(tracks, mask):
    out = []
    for i in range(tracks.shape[0]):
        for j in range(tracks.shape[0]):
            if i == j or mask[j].sum() == 0:
                continue
            rms = float(np.sqrt(np.mean(tracks[i, mask[j]] ** 2) + 1e-12))
            out.append(20.0 * np.log10(rms + 1e-9))
    return float(np.mean(out)) if out else float("nan")


def _distortion_db(cleaned, voices, mask):
    out = []
    for i in range(cleaned.shape[0]):
        m = mask[i]
        if m.sum() == 0:
            continue
        diff = cleaned[i, m] - voices[i, m]
        s = float(np.sqrt(np.mean(voices[i, m] ** 2) + 1e-12))
        d = float(np.sqrt(np.mean(diff ** 2) + 1e-12))
        if s > 1e-6:
            out.append(20.0 * np.log10(d / s))
    return float(np.nanmean(out)) if out else float("nan")


def _false_trigger_rate(tracks, mask, threshold_db=-28.0, frame_ms=20):
    n_ch, n = tracks.shape
    frame_len = int(frame_ms * SR / 1000)
    n_frames = n // frame_len
    counts = []
    for i in range(n_ch):
        for j in range(n_ch):
            if i == j:
                continue
            trig = total = 0
            for f in range(n_frames):
                f0, f1 = f * frame_len, (f + 1) * frame_len
                if mask[j, f0:f1].sum() < 0.5 * frame_len:
                    continue
                rms = float(np.sqrt(np.mean(tracks[i, f0:f1] ** 2) + 1e-12))
                total += 1
                if 20.0 * np.log10(rms + 1e-9) > threshold_db:
                    trig += 1
            if total:
                counts.append(trig / total * 100.0)
    return float(np.mean(counts)) if counts else 0.0


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------


SEEDS = [42, 7, 1337]
DURATION_S = 12.0
FIR_TAPS = [128, 256, 512]

SCENARIOS = [
    ("2ch strong  (-12 dB, 5 ms)", _two_speaker_scene, -12.0, 5.0),
    ("2ch typical (-18 dB, 5 ms)", _two_speaker_scene, -18.0, 5.0),
    ("2ch weak    (-25 dB, 5 ms)", _two_speaker_scene, -25.0, 5.0),
    ("2ch longdly (-18 dB, 8 ms)", _two_speaker_scene, -18.0, 8.0),
    ("4ch typical (-18 dB, 5 ms)", _four_speaker_scene, -18.0, 5.0),
    ("4ch strong  (-12 dB, 5 ms)", _four_speaker_scene, -12.0, 5.0),
]


def _run(scene_fn, bleed_db, delay_ms, seed, fir_taps):
    tracks, voices = scene_fn(DURATION_S, bleed_db, delay_ms, seed)
    cfg = CrossCancelConfig(fir_taps=fir_taps)
    mask = compute_dominance_mask(tracks, SR, cfg)
    bb = _bleed_db(tracks, mask)
    ftb = _false_trigger_rate(tracks, mask)
    cleaned = np.stack(cross_cancel(list(tracks), SR, cfg))
    ba = _bleed_db(cleaned, mask)
    fta = _false_trigger_rate(cleaned, mask)
    da = _distortion_db(cleaned, voices, mask)
    return bb, ba, ftb, fta, da


def main():
    print("=" * 100)
    print(
        f"Cross-channel cancellation — averaged over {len(SEEDS)} seeds, {DURATION_S}s clips, SR={SR}"
    )
    print("=" * 100)
    print(
        f"{'Scenario':<30} {'FIR':>5} "
        f"{'BleedBef':>9} {'BleedAft':>9} {'Δ':>7} "
        f"{'FT-bef':>7} {'FT-aft':>7} {'FT-drop':>8} {'Distort':>8}"
    )
    print("-" * 100)

    for name, scene_fn, bleed_db, delay_ms in SCENARIOS:
        first = True
        for L in FIR_TAPS:
            rows = [_run(scene_fn, bleed_db, delay_ms, s, L) for s in SEEDS]
            bb = float(np.mean([r[0] for r in rows]))
            ba = float(np.mean([r[1] for r in rows]))
            ftb = float(np.mean([r[2] for r in rows]))
            fta = float(np.mean([r[3] for r in rows]))
            da = float(np.mean([r[4] for r in rows]))
            drop = f"{ftb / fta:>5.1f}x" if fta > 0.05 else "  inf"
            label = name if first else ""
            first = False
            print(
                f"{label:<30} {L:>5} "
                f"{bb:>+9.2f} {ba:>+9.2f} {bb-ba:>+7.2f} "
                f"{ftb:>6.1f}% {fta:>6.1f}% {drop:>8} {da:>+8.2f}"
            )
        print()

    print("Legend:")
    print("  BleedBef/Aft : RMS dB on victim mic during 'only other' frames")
    print("  Δ            : bleed reduction in dB (higher is better)")
    print("  FT-bef/aft   : % of 'only other' frames triggering > -28 dB detector")
    print("  FT-drop      : ratio FT-before / FT-after (higher is better)")
    print("  Distort      : residual error vs oracle voice on own-speech frames (lower = better)")


if __name__ == "__main__":
    main()
