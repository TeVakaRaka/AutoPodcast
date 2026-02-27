# AutoPodcast

Automatic rough-cut podcast editor. Deterministic algorithm (RMS + hysteresis + debounce), no ML.

## Stack
- Python 3.10+, numpy, soundfile, click
- Export: FCP 7 XML (Premiere Pro native import)
- Use `python3` (no `python` binary on this Mac)

## Architecture
- `src/autopodcast/core/` — pure functions over numpy arrays and dataclasses. No I/O.
- `src/autopodcast/models/` — dataclasses (domain.py, project.py)
- `src/autopodcast/export/` — FCP 7 XML and JSON export
- `src/autopodcast/utils/` — ffmpeg subprocess helpers
- `src/autopodcast/cli.py` — Click CLI entry point

## Commands
```bash
python3 -m pytest tests/ -v          # run tests
python3 -m autopodcast analyze --help # CLI help
```
