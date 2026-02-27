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

### 6. Camera Assignment
- SPEAKER_A → closeup camera
- SPEAKER_B → closeup camera
- BOTH / SILENCE → wide shot

### 7. Ducking (optional)
- Active speaker: 0 dB
- Inactive speaker: -12 dB
- BOTH/SILENCE: both 0 dB

### 8. Export
FCP 7 XML with frame-accurate timing, clip references, and volume keyframes.
