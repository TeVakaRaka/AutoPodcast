# AutoPodcast Windows Check And Build

This archive is prepared for checking tests and building `autopodcast.exe` on Windows.

## Automatic builds (no local build needed)

A GitHub Actions workflow (`.github/workflows/build-windows.yml`) builds a
ready-to-run `autopodcast.exe` on a Windows runner on every push to `main`
or `claude/release-build`, and can also be triggered manually from the
repository's **Actions** tab (`Build Windows EXE` → `Run workflow`).

To get the latest build without building anything locally:

1. Open the repo on GitHub → **Actions** tab.
2. Click the most recent `Build Windows EXE` run.
3. Download the `autopodcast-windows` artifact at the bottom of the page.
   It already contains `autopodcast.exe`, all dependencies, and bundled
   `ffmpeg.exe` / `ffprobe.exe`.

The sections below describe the manual local build, kept as a fallback.

## Requirements

- Windows 10/11.
- Python 3.10 or newer.
- During Python install, enable `Add Python to PATH`.
- Internet access for first dependency install and optional ffmpeg download.

## Quick Check

Open `cmd.exe` in this folder and run:

```bat
RUN_TESTS.bat
```

If the script cannot find `pytest`, run:

```bat
python -m pip install -r requirements.txt
python -m pytest tests/ -v
```

## Build EXE

Open `cmd.exe` in this folder and run:

```bat
DOWNLOAD_FFMPEG.bat
BUILD.bat
```

Build output:

```text
_build\dist\autopodcast\autopodcast.exe
```

Run built executable:

```bat
RUN.bat --help
```

## Notes

- `DOWNLOAD_FFMPEG.bat` puts `ffmpeg.exe` and `ffprobe.exe` into `ffmpeg\`.
- `BUILD.bat` creates its own build venv in `_build\venv`.
- Generated folders such as `_build`, `.venv`, `.pytest_cache`, and `release` are intentionally not part of the archive.
- For Premiere project checks, use the CLI help:

```bat
python -m autopodcast --help
python -m autopodcast auto-switch-sakha-aimakh --help
python -m autopodcast auto-switch-4cams --help
python -m autopodcast auto-switch-monologue --help
```
