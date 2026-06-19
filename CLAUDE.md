# AutoPodcast

Automatic rough-cut podcast editor with deterministic audio analysis (RMS + hysteresis +
debounce) plus multi-mode camera planners. Optional Silero VAD backend; no other ML.
Targets Premiere Pro (FCP 7 XML round-trip + `.prproj` patching + ExtendScript).

## Stack
- Python 3.10+, numpy<2.0, soundfile, click
- Optional: silero-vad ONNX (CPU) backend (auto-detected via `--detector-backend silero`)
- Export: FCP 7 XML, JSON timeline, ExtendScript `.jsx`, patched `.prproj`
- Use `python3` (no `python` binary on this Mac)

## Architecture
- `src/autopodcast/core/` — pure functions over numpy arrays and dataclasses. No I/O.
  - Base: `analyzer`, `audio_loader`, `detector` (rms/silero), `segmenter`, `switcher`, `ducking`
  - Bleed: `cross_cancel` — Wiener-Hopf mic-bleed cancellation, runs before the detector
  - Infra: `camera_motion`, `motion_cache`, `camera_scheduler`, `calibrator`, `roles`
  - Mode planners: `monologue_2cam` + `monologue_sources`, `auto_switch_4cams`, `sakha_aimakh`,
    `auto_switch_custom` (configurable N-people / M-cameras — the "Конструктор" mode)
  - Premiere I/O: `audio_sources` (resolves source files from `.prproj` / FCP 7 XML)
- `src/autopodcast/models/` — dataclasses (`domain.py`, `project.py`)
- `src/autopodcast/export/` — `fcp7xml.py`, `json_export.py`, `multicam_jsx.py`
- `src/autopodcast/utils/ffmpeg.py` — ffmpeg subprocess helpers
- `src/autopodcast/prproj_patcher.py` — parse/patch binary `.prproj` (gunzip + XML)
- `src/autopodcast/import_xml.py` — read FCP 7 XML back into Timeline
- `src/autopodcast/cli.py` + `__main__.py` — Click group with mode subcommands
- `src/autopodcast/gui/` — customtkinter desktop GUI covering all five modes

Full module map and data flow: [docs/architecture.md](docs/architecture.md);
per-module reference: [docs/code-map.md](docs/code-map.md).

## CLI modes
```bash
python3 -m autopodcast --help
python3 -m autopodcast analyze --help                  # base 2-track planner
python3 -m autopodcast auto-switch-monologue --help    # 1 speaker, 2 cameras
python3 -m autopodcast auto-switch-4cams --help        # 1 host + 3 guests, 4 cams
python3 -m autopodcast auto-switch-sakha-aimakh --help # 2 hosts + 1 guest
python3 -m autopodcast auto-switch-custom --help       # configurable: any N people / M cameras
```

## SAKHA AYMAKH mode notes
`core/sakha_aimakh.py` is the largest planner. Active-speaker classification runs in
layers, each with its own config block on `SakhaAymakhConfig`:
1. RMS detector → 2. debleed (subtract mic bleed) → 3. waveform gate (per-pair correlation)
→ 4. source-owner detector (residual + correlation + tiebreak) → 5. strict mode
→ 6. cohost/guest rescue + hysteresis → 7. motion guard.
Frame states: `SILENCE`, `MAIN_HOST_ONLY`, `COHOST_ONLY`, `GUEST_ONLY`, three pair-overlaps,
`ALL_OVERLAP`.

## Spec
The canonical engineering spec for the 4-camera layout lives at
[`docs/auto-switching-spec-4cams.md`](docs/auto-switching-spec-4cams.md). It defines
participants, cameras, audio channels, events, decision rules, audio pre-roll, and the
two-stage attenuation behavior. Treat it as the source of truth when changing
`auto_switch_4cams.py`.

## Tests
```bash
python3 -m pytest tests/ -v
```
All planners are pure-function and unit-tested end-to-end with synthetic numpy fixtures
(see `tests/conftest.py`).

## Windows build
A standalone `autopodcast.exe` is produced via PyInstaller (`build.spec`). See
[`WINDOWS_CHECK_BUILD.md`](WINDOWS_CHECK_BUILD.md) for the full flow
(`DOWNLOAD_FFMPEG.bat` → `BUILD.bat` → `_build\dist\autopodcast\autopodcast.exe`).

## GUI
Double-clicking the exe (or `python3 -m autopodcast gui`) opens a customtkinter
window covering all five modes. It builds an `autopodcast` CLI command from the
form and runs it as a subprocess — it never reimplements pipeline logic.
"Creative" numeric parameters (tempo, cut frequency, camera share, detector
thresholds) are sliders. Form fields are declared in `gui/spec.py`; defaults
there must match the `@click.option` defaults (a test enforces this).
Details: [docs/gui.md](docs/gui.md).

## Releases
Named feature snapshots are kept locally in `release/` (gitignored) with the convention
`AutoPodcast-<TAG>-<YYYYMMDD>.zip`. The most recent is the canonical reference for
"what's currently shipping" — do not rely on `git log` for that, the production code
historically lived outside of git. The Windows `.exe` is also built by GitHub
Actions — see [docs/build-and-ci.md](docs/build-and-ci.md).

## Documentation (read these first in a new session)
- [docs/architecture.md](docs/architecture.md) — module map, data flow, design principles
- [docs/code-map.md](docs/code-map.md) — per-module reference + per-mode call chains
- [docs/multicam-logic.md](docs/multicam-logic.md) — cross-mode editorial logic (sakha-derived):
  attribution, camera rules (overlap→общак, shared shot, fragmented solo→close-up), stability
- [docs/decisions.md](docs/decisions.md) — what changed and why; investigations that did
  NOT lead to code (read this to avoid re-debugging settled questions) + Open/future work
- [docs/gui.md](docs/gui.md) — the GUI: structure, how to add fields/modes
- [docs/build-and-ci.md](docs/build-and-ci.md) — PyInstaller build + GitHub Actions
- [docs/algorithm.md](docs/algorithm.md) — base 2-speaker (analyze / auto-multicam) pipeline
- [docs/auto-switching-spec-4cams.md](docs/auto-switching-spec-4cams.md) — canonical 4-cam spec
- [docs/custom-mode.md](docs/custom-mode.md) — the "Конструктор" configurable N-people/M-cameras mode

**Keep the log current.** Whenever you change code, append an entry to
[docs/decisions.md](docs/decisions.md) (newest first): what changed, where (file/function),
why, how it was verified, and the commit SHA. This log is the project's memory across
sessions — updating it is part of "done", not optional.
