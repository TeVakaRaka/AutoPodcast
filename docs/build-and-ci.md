# Build & CI

## Distribution

AutoPodcast ships as a Windows folder build produced by PyInstaller from
`build.spec`. One analysis emits **two executables**:

| Executable | `console` | Use |
|---|---|---|
| `autopodcast.exe` | yes | CLI; also the child process the GUI launches |
| `autopodcast-gui.exe` | no | double-click graphical interface |

`build.spec` also bundles `customtkinter` theme data (`collect_data_files`),
the soundfile / silero-vad / onnxruntime binaries, and keeps `tkinter` in
the build (it must NOT be in `excludes`).

## Automatic builds (GitHub Actions)

`.github/workflows/build-windows.yml` builds on a `windows-latest` runner:

1. on every push to `main` or `claude/release-build`, or manually via the
   Actions tab (**Run workflow**);
2. installs `requirements.txt` + PyInstaller, runs `pyinstaller build.spec`;
3. downloads ffmpeg (BtbN build) into the dist folder;
4. smoke-tests `autopodcast.exe --help`;
5. uploads the whole dist folder as the `autopodcast-windows` artifact
   (~370 MB — onnxruntime + scipy + customtkinter are large).

**To get a build without building locally:** GitHub → Actions →
latest `Build Windows EXE` run → download the `autopodcast-windows`
artifact. It contains both `.exe` files, dependencies and ffmpeg.

The smoke test is load-bearing: it has already caught two real crashes
(non-ASCII in CLI output, and the frozen-exe Cyrillic console / headless
stdin issue). Keep it.

## Local build (Windows fallback)

See [WINDOWS_CHECK_BUILD.md](../WINDOWS_CHECK_BUILD.md):
`DOWNLOAD_FFMPEG.bat` → `BUILD.bat` → `_build\dist\autopodcast\`.

## Local development (macOS / Linux)

```bash
python3 -m pip install -e . -r requirements.txt
python3 -m pytest tests/ -v          # full test suite
python3 -m autopodcast --help        # CLI
python3 -m autopodcast gui           # GUI (needs a Tk-enabled Python)
```

Note: a Homebrew Python may lack `_tkinter`; install `python-tk` to run the
GUI locally. The pure-logic GUI tests (`tests/test_gui_spec.py`) need no Tk.

## Known CI warnings (not errors)

- Node.js 20 actions deprecation — `actions/*` versions may need bumping
  after mid-2026.
- `windows-latest` is migrating to `windows-2025`.

Neither breaks the build today; revisit if a run starts failing.
