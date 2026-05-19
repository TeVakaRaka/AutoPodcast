# AutoPodcast

Automatic rough-cut podcast editor with deterministic audio analysis (RMS + hysteresis +
debounce) plus multi-mode camera planners. Optional Silero VAD backend; no other ML.
Targets Premiere Pro (FCP 7 XML round-trip + `.prproj` patching + ExtendScript).

## Stack
- Python 3.10+, numpy<2.0, soundfile, click
- Optional: torch + silero-vad (auto-detected via `--detector-backend silero`)
- Export: FCP 7 XML, JSON timeline, ExtendScript `.jsx`, patched `.prproj`
- Use `python3` (no `python` binary on this Mac)

## Architecture
- `src/autopodcast/core/` — pure functions over numpy arrays and dataclasses. No I/O.
  - Base: `analyzer`, `audio_loader`, `detector` (rms/silero), `segmenter`, `switcher`, `ducking`
  - Infra: `camera_motion`, `motion_cache`, `camera_scheduler`, `calibrator`, `roles`
  - Mode planners: `monologue_2cam` + `monologue_sources`, `auto_switch_4cams`, `sakha_aimakh`
  - Premiere I/O: `audio_sources` (resolves source files from `.prproj` / FCP 7 XML)
- `src/autopodcast/models/` — dataclasses (`domain.py`, `project.py`)
- `src/autopodcast/export/` — `fcp7xml.py`, `json_export.py`, `multicam_jsx.py`
- `src/autopodcast/utils/ffmpeg.py` — ffmpeg subprocess helpers
- `src/autopodcast/prproj_patcher.py` — parse/patch binary `.prproj` (gunzip + XML)
- `src/autopodcast/import_xml.py` — read FCP 7 XML back into Timeline
- `src/autopodcast/cli.py` + `__main__.py` — Click group with mode subcommands

## CLI modes
```bash
python3 -m autopodcast --help
python3 -m autopodcast analyze --help                  # base 2-track planner
python3 -m autopodcast auto-switch-monologue --help    # 1 speaker, 2 cameras
python3 -m autopodcast auto-switch-4cams --help        # 1 host + 3 guests, 4 cams
python3 -m autopodcast auto-switch-sakha-aimakh --help # 2 hosts + 1 guest
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

## Releases
Named feature snapshots are kept locally in `release/` (gitignored) with the convention
`AutoPodcast-<TAG>-<YYYYMMDD>.zip`. The most recent is the canonical reference for
"what's currently shipping" — do not rely on `git log` for that, the production code
historically lived outside of git.
