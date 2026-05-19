# Algorithm

## Pipeline Stages

### 1. Audio Loading
- Input: WAV or video files (video demuxed via ffmpeg)
- Output: mono float64 numpy array at 16 kHz
- Both tracks padded to same length

### 2. RMS + Envelope
- Sliding window: 30ms window, 10ms hop
- `rms = sqrt(mean(chunk^2))`
- `rms_db = 20 * log10(rms + 1e-10)`
- EMA smoothing: `envelope[i] = α * rms[i] + (1-α) * envelope[i-1]`
  - `α = 2 / (kernel_size + 1)`
  - Fast attack, smooth decay — natural for speech

### 3. Hysteresis Detection
Per-speaker state machine:

```
INACTIVE → envelope > speech_threshold (-28 dB) → ACTIVE
ACTIVE   → envelope < release_threshold (-33 dB) → HANGOVER
HANGOVER → envelope > speech_threshold → ACTIVE (reset timer)
HANGOVER → timer >= hangover_ms (600ms) → INACTIVE
```

- 5 dB hysteresis gap prevents oscillation
- 600ms hangover bridges natural speech pauses

### 4. Speaker Combination
Per-frame: A+B → BOTH, A only → SPEAKER_A, B only → SPEAKER_B, neither → SILENCE

### 5. Segmentation
1. Run-length encoding → raw segments
2. Debounce: remove segments < 300ms, merge into previous
3. Min-length: segments < 2000ms merge into previous
4. Merge adjacent same-state segments

### 6. Camera Scheduling

**Base assignment (priority order):**
1. BOTH → wide camera when overlap is genuine (about 0.9s+ by default)
2. SILENCE → wide camera only for longer pauses (about 1.6s+ by default)
3. SPEAKER_A / SPEAKER_B → closeup camera

**Camera-oriented smoothing:**
- Video uses a different segmentation profile than audio mute.
- Short speaker turns can survive from about `0.8s`, so brief questions/interjections
  are more likely to appear on camera.
- `BOTH` and `SILENCE` are treated differently:
  - `BOTH` is preserved more easily so real simultaneous speech can cut to wide.
  - `SILENCE` is preserved less easily so the edit does not jump to wide on every short pause.
- Short "takeover" turns (about `0.7s+`) are preserved when they interrupt a much
  longer run of the opposite speaker, so a quick question can still earn a camera cut.
- A final minimum camera-event hold prevents overly jittery cutting.

**Dialogue re-establishing wide:**
- During active shot-reverse-shot dialogue, the scheduler can insert a short wide
  "master" shot to refresh screen geography and reduce closeup fatigue.
- By default this happens only after several speaker exchanges (`3` turns) and no
  wide has appeared for about `24s`.
- Duration is short (`2s` by default), so the wide works as a reset rather than
  stealing the whole conversation.

**Sticky wide for dense overlap clusters:**
- If real `BOTH` segments are separated only by a few short speaker turns, the
  scheduler can bridge them into one longer wide instead of bouncing in and out.
- By default the bridged gap can be up to about `8s`, with only short turns
  inside it (about `2.4s` max each) and at least `3` turns overall.
- This is meant for heated interruptions and overlapping talk, not for normal
  longer back-and-forth answers.

**Long-talk cutaway:**
When a single speaker talks continuously for longer than `long_talk_threshold_sec` (default 15s), a wide cutaway is inserted for visual variety:
- Cutaway starts at `segment_start + threshold`
- Duration: `wide_duration_sec` (default 5s), mode `"once"` (max one per segment)
- Cutaway carries through BOTH/SILENCE segments (already wide) and into the next same-speaker segment
- Cutaway stops at a different speaker's segment
- If consecutive BOTH/SILENCE segments after the speaker segment fully cover the remaining cutaway duration, the in-segment cutaway is suppressed (natural wide provides enough variety)
- Adjacent events with the same camera are merged

### 7. Ducking (optional)
- Active speaker: 0 dB
- Inactive speaker: -12 dB
- BOTH/SILENCE: both 0 dB

### 8. Export
FCP 7 XML with frame-accurate timing, clip references, and volume keyframes.
