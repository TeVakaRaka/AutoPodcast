# Code map

Per-module reference for `src/autopodcast/`. Pairs with
[architecture.md](architecture.md) (high-level), [multicam-logic.md](multicam-logic.md)
(editorial logic) and [algorithm.md](algorithm.md) (base 2-speaker pipeline).

Pipeline stages: **load → analyze → detect → plan → export**, with `cli.py` /
`gui/` as orchestration. Every `core/` planner is a pure function (no I/O).

## Entry points & orchestration

| Module | Lines | Purpose |
|---|---:|---|
| `__main__.py` | 585 | Entry on launch: GUI on double-click; CLI when given args; text wizards as fallback (handles Windows UTF-8 console). |
| `cli.py` | 3222 | Click command group — every subcommand. Wires load→analyze→detect→planner→patch/export. Shared helpers: `_resolve_speaker_mics`, `_maybe_apply_cross_cancel`, `_cross_cancel_options`, `_parse_person_spec`, JSONL serializers. |
| `config.py` | 45 | `build_config()` — CLI args → `ProjectConfig` (used by the base 2-speaker path). |

CLI commands: `analyze`, `auto-multicam`, `auto-switch-4cams`,
`auto-switch-sakha-aimakh`, `auto-switch-monologue`, `auto-switch-custom`,
`preview`, `export`, `from-xml`, `calibrate`, `patch-prproj`, `gui`.

## Models (pure dataclasses)

| Module | Lines | Key types |
|---|---:|---|
| `models/domain.py` | 67 | `SpeakerState` (SILENCE/A/B/BOTH), `AnalysisFrame` (time/rms/envelope/is_active/peak), `SpeakerActivity`, `Segment`, `CameraEvent`, `DuckingEvent`, `Timeline`. |
| `models/project.py` | 285 | `AudioInput`, `CameraInput`, `ProjectConfig` (analysis/detection/segmentation/ducking tunables). |

## core/ — audio analysis & detection

| Module | Lines | Key API | Stage |
|---|---:|---|---|
| `audio_loader.py` | 106 | `load_audio`, `load_and_align`, `apply_offset` (WAV/video→16 kHz mono). | load |
| `audio_sources.py` | 420 | `resolve_sequence_audio_sources` — map each audio track → source file from `.prproj`/XML. | load |
| `analyzer.py` | 122 | `analyze_speaker` — RMS + EMA envelope → `SpeakerActivity`. | analyze |
| `detector.py` | 554 | `detect_activity` (RMS hysteresis or Silero VAD → `is_active`), `combine_speakers` (2-speaker A/B/BOTH), `apply_cross_gate`. | detect |
| `cross_cancel.py` | 215 | `cross_cancel` — Wiener-Hopf inter-mic bleed subtraction (pre-detector). | detect |
| `calibrator.py` | 104 | `calibrate` — auto speech/release thresholds from noise floor. | (calibrate cmd) |
| `segmenter.py` | 442 | `run_length_encode`, `build_camera_segments`, `build_audio_mute_segments`, merge/debounce. | plan (base path) |
| `ducking.py` | 131 | `generate_ducking_events` — per-track volume keyframes. | export (base path) |
| `roles.py` | 24 | `normalize_speaker_role` — host/guest ↔ ведущий/гость. | orchestration |

## core/ — planners (one per mode)

| Module | Lines | Entry | Layout |
|---|---:|---|---|
| `auto_switch_4cams.py` | 1722 | `build_roundtable_plan` | 1 host + 3 guests; conversation-phase model. Exports the shared `build_audio_plan`. |
| `sakha_aimakh.py` | 3364 | `build_sakha_aimakh_plan` | 2 hosts + 1 guest; leak-matrix attribution (`_build_studio_frame_states`), choppy-run consolidator. Reference impl. |
| `auto_switch_custom.py` | 439 | `build_custom_plan` | any N people / M cameras; static person→camera map; reuses sakha studio + `build_audio_plan`. |
| `monologue_2cam.py` | 697 | `build_monologue_plan` | 1 narrator, 2 cams; scheduled alternation + motion guard. |
| `switcher.py` | 47 | `assign_cameras` | legacy 2-speaker camera assignment (analyze cmd). |
| `camera_scheduler.py` | 571 | `schedule_camera_events` | legacy 2-speaker scheduling (long-talk wide cutaways). |

## core/ — camera motion (optional)

| Module | Lines | Key API |
|---|---:|---|
| `camera_motion.py` | 1252 | `CameraMotionAnalyzer` — ffmpeg optical-flow → moving intervals (motion guard). |
| `monologue_sources.py` | 843 | `resolve_sequence_camera_sources` — resolve angle→video clips for motion analysis. |
| `motion_cache.py` | 122 | On-disk cache of motion plans (keyed by config + clip paths). |

## export / project I/O

| Module | Lines | Key API |
|---|---:|---|
| `prproj_patcher.py` | 1442 | `patch_prproj` (angle switches + per-track `IsMuted`), `segments_to_cuts`, `read_audio_offsets`, time/tick helpers. The N-track-capable writer. |
| `import_xml.py` | 162 | `parse_premiere_xml` — read FCP7 XML → sequence/clips. |
| `export/fcp7xml.py` | 389 | `generate_fcp7xml` / `save_fcp7xml`. |
| `export/multicam_jsx.py` | 116 | `generate_multicam_jsx` — ExtendScript multicam rig. |
| `export/json_export.py` | 81 | `save_timeline` / `load_timeline`. |
| `utils/ffmpeg.py` | 85 | `probe_duration`, `demux_audio`. |

## gui/ (customtkinter; spec.py is pure, the rest needs Tk)

| Module | Lines | Purpose |
|---|---:|---|
| `gui/spec.py` | 425 | `Field`/`Column`/`ModeSpec`/`MODES`, `build_argv` (form values → CLI args; `_build_custom_argv` expands the Конструктор tables). No tkinter → unit-tested. |
| `gui/widgets.py` | 295 | `FileField`, `SliderField`, `CollapsibleSection`, `DynamicRowsField` (add/remove tables), `make_field_widget`. |
| `gui/app.py` | 321 | `AutoPodcastApp` — the window; builds the form, runs the CLI subprocess, streams progress. |
| `gui/runner.py` | 106 | `ProcessRunner` — runs `autopodcast <cmd>` as a child process. |
| `gui/main.py` | 15 | `main()` — launch the window. |

## Dependency layers

- **Leaves** (no internal deps): `models/*`, `roles`, `utils/ffmpeg`,
  `import_xml`, `export/json_export`, `gui/spec`, `cross_cancel`, `analyzer`.
- **Analysis stack** (feed-forward): `audio_loader → analyzer → detector →`
  (base path) `segmenter → ducking`.
- **Source resolution**: `audio_sources`, `monologue_sources` → use
  `prproj_patcher` low-level parsing.
- **Planners**: `auto_switch_4cams` exports `build_audio_plan`;
  `auto_switch_custom` imports `build_audio_plan` (from 4cams) +
  `_build_studio_frame_states`/`_build_waveform_gate_context` (from
  `sakha_aimakh`). `sakha_aimakh`/`4cams`/`monologue` use `camera_motion`.
- **Orchestrators**: `cli.py` (all commands), `__main__.py`, `gui/*`. Final
  writer: `prproj_patcher.patch_prproj`.

## End-to-end call chains

**Multicam modes** (4cams / sakha / custom) — note they consume the per-person
`activities` dict directly; `combine_speakers` is **not** used:

```
cli <cmd>
  → _resolve_speaker_mics (audio_sources)        # mics from the .prproj/XML
  → load_and_align + apply_offset                # 16 kHz mono, aligned
  → [cross_cancel]                               # optional bleed subtraction
  → analyze_speaker ×N  → detect_activity ×N     # per-person SpeakerActivity
  → build_<mode>_plan(participants, activities, hop_s, config)
        → frame states (attribution: dominance | studio leak-matrix)
        → camera segments (camera rules + stability)
        → build_audio_plan → per-track open intervals
  → segments_to_cuts(plan.camera_segments)
  → patch_prproj(..., audio_track_intervals_s=plan.audio_open_intervals_s)
```

Planner per mode: `auto-switch-4cams` → `build_roundtable_plan`;
`auto-switch-sakha-aimakh` → `build_sakha_aimakh_plan`; `auto-switch-custom` →
`build_custom_plan`; `auto-switch-monologue` → `build_monologue_plan`.

**Base 2-speaker** (`analyze`, `auto-multicam`):

```
load → analyze_speaker ×2 → detect_activity ×2 → combine_speakers (A/B/BOTH)
  → segment_timeline → assign_cameras / schedule_camera_events
  → generate_ducking_events → Timeline
  → save_fcp7xml / generate_multicam_jsx   (analyze)   |   patch_prproj (auto-multicam)
```

## Dead / duplicated (cleanup candidates)

- `auto_switch_4cams` carries **dead `cam3_*` config** (the moving-close-up
  wait logic; `cam3_wait_timeout_s` defaults to 0 → switch immediately). Flagged
  in [decisions.md](decisions.md) → Open work.
- `core/switcher.py` + `camera_scheduler.py` — legacy 2-speaker path; only the
  `analyze`/`auto-multicam` commands use them.
- `core/analyzer.py:apply_lowpass` — defined, not used.
- `build_audio_plan` exists in both `auto_switch_4cams.py` and `sakha_aimakh.py`
  (custom imports the 4cams copy) — the two are near-identical; a shared
  `core/audio_plan.py` would remove the duplication (deferred to avoid touching
  the tuned planners).
