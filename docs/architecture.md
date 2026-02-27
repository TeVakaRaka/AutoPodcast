# Architecture

## Module Overview

```
src/autopodcast/
  cli.py                 # Click CLI: analyze, preview, export
  config.py              # ProjectConfig builder from CLI args

  core/                  # Pure functions — no I/O
    audio_loader.py      # WAV loading, ffmpeg demux, resampling
    analyzer.py          # RMS computation, EMA envelope smoothing
    detector.py          # Hysteresis state machine + hangover
    segmenter.py         # RLE, debounce, min-length, merge
    switcher.py          # Camera assignment rules
    ducking.py           # Volume automation events

  models/                # Dataclasses only
    domain.py            # SpeakerState, Segment, Timeline, etc.
    project.py           # ProjectConfig, AudioInput, CameraInput

  export/
    fcp7xml.py           # FCP 7 XML (Premiere Pro import)
    json_export.py       # JSON timeline dump

  utils/
    ffmpeg.py            # ffmpeg subprocess (probe, demux)
```

## Data Flow

```
Audio files
  → audio_loader (WAV/video → numpy float64)
  → analyzer (RMS → envelope)
  → detector (hysteresis → is_active per frame)
  → combine_speakers (→ SpeakerState per frame)
  → segmenter (RLE → debounce → min-length → merge)
  → switcher (camera assignment)
  → ducking (volume events)
  → Timeline
  → export (FCP 7 XML / JSON)
```

## Design Principles

1. **Core modules are pure** — they take numpy arrays and dataclasses in, return dataclasses out. No file I/O, no side effects. This makes them trivially testable.

2. **All I/O at the edges** — `audio_loader`, `utils/ffmpeg`, `export/`, and `cli.py` handle all file operations.

3. **No ML dependencies** — the entire pipeline is deterministic: RMS energy, hysteresis thresholds, debounce timers.
