# Architecture

AutoPodcast is an automatic rough-cut editor for multi-camera podcasts. It
analyses per-speaker microphone tracks, decides which camera to show and
which mics to keep open, and writes the result back into a Premiere Pro
project (`.prproj`) or an FCP 7 XML.

See also: [code-map.md](code-map.md) (per-module reference + call chains),
[multicam-logic.md](multicam-logic.md) (the cross-mode editorial logic),
[algorithm.md](algorithm.md) (the base 2-speaker pipeline).

## Module map

```
src/autopodcast/
  cli.py             Click CLI group — every subcommand lives here
  __main__.py        entry point: GUI on double-click, CLI when given args
  config.py          ProjectConfig builder from CLI args

  core/              pure functions over numpy arrays + dataclasses, no I/O
    audio_loader     WAV loading, ffmpeg demux, resample to 16 kHz mono
    analyzer         RMS computation, EMA envelope smoothing
    detector         active-speaker detection — RMS hysteresis or Silero VAD
    segmenter        RLE, debounce, min-length, merge
    switcher         basic camera assignment rules
    ducking          volume-automation events
    cross_cancel     Wiener-Hopf mic-bleed cancellation (pre-detector layer)
    camera_motion    detect physical camera motion (motion guard)
    motion_cache     on-disk cache for motion analysis
    camera_scheduler shared ordering / cooldown logic
    calibrator       auto-calibration of speech/release thresholds
    roles            host / cohost / guest taxonomy
    monologue_2cam   planner: 1 narrator, 2 cameras
    monologue_sources  source resolution for the monologue layout
    auto_switch_4cams  planner: 1 host + 3 guests, 4 cameras
    sakha_aimakh     planner: 2 hosts + 1 guest (the largest planner)
    auto_switch_custom planner: configurable N people / M cameras ("Конструктор")
    audio_sources    resolve source audio files from a .prproj / FCP 7 XML

  models/            dataclasses only — domain.py, project.py
  export/            fcp7xml.py, json_export.py, multicam_jsx.py (ExtendScript)
  utils/ffmpeg.py    ffmpeg subprocess helpers (probe, demux)
  prproj_patcher.py  parse + patch binary .prproj (gunzip + XML)
  import_xml.py      read an FCP 7 XML back into the Timeline model
  gui/               customtkinter desktop GUI (see docs/gui.md)
```

## Editing modes

| CLI command | Layout | Planner module |
|---|---|---|
| `analyze` | base 2-track timeline | `switcher` + `segmenter` |
| `auto-multicam` | 1 host + 1 guest | `detector` + `switcher` |
| `auto-switch-4cams` | 1 host + 3 guests, 4 cams | `auto_switch_4cams` |
| `auto-switch-sakha-aimakh` | 2 hosts + 1 guest | `sakha_aimakh` |
| `auto-switch-monologue` | 1 narrator, 2 cams | `monologue_2cam` |
| `auto-switch-custom` | any N people / M cameras (configurable) | `auto_switch_custom` |

The first four are fixed presets. `auto-switch-custom` ("Конструктор") generalizes
them: you declare the cast and cameras yourself and map each person to a camera.
See [docs/custom-mode.md](custom-mode.md).

## Data flow

```
audio files
  -> audio_loader        (WAV / video -> numpy float64, 16 kHz mono)
  -> cross_cancel        (optional: subtract mic bleed before detection)
  -> analyzer            (RMS -> EMA envelope)
  -> detector            (hysteresis / Silero -> is_active per frame)
  -> combine_speakers    (-> SpeakerState per frame)
  -> segmenter           (RLE -> debounce -> min-length -> merge)
  -> mode planner        (camera plan + audio-mute plan)
  -> prproj_patcher      (patch .prproj)  OR  export/fcp7xml
```

## Design principles

1. **Core modules are pure** — numpy arrays and dataclasses in, dataclasses
   out. No file I/O, no side effects. Trivially unit-testable.
2. **All I/O at the edges** — `audio_loader`, `utils/ffmpeg`, `export/`,
   `prproj_patcher`, `cli.py` and `gui/` handle every file operation.
3. **Deterministic by default** — RMS energy, hysteresis thresholds,
   debounce timers, closed-form filters. Silero VAD is the only optional
   ML component (`--detector-backend silero`).
4. **CLI is the single source of truth** — the GUI builds and runs CLI
   commands; it never reimplements pipeline logic.

## Mic bleed — the recurring problem

Separate mics pick up each other (cross-talk). When one speaker is silent,
their mic still hears the other speaker, so the detector reports false
activity. Two layers address this:

- `core/cross_cancel.py` — generic pre-detector Wiener-Hopf cancellation,
  on by default for `auto-multicam`.
- inside `sakha_aimakh.py` — debleed / waveform-gate / source-owner layers
  specific to the 2+1 layout.

See [docs/decisions.md](decisions.md) for the history behind these.
