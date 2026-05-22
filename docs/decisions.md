# Decisions & investigation log

Context for future sessions: what was changed, why, and which
investigations did *not* lead to code. Newest first. Dates are approximate
(May 2026).

## Branches / PRs

- `claude/sync-production-code-20260519` (PR #1) — base production code.
- `claude/cross-cancel-prototype` (PR #2) — cross-channel cancellation.
- `claude/fix-prproj-audio-bloat` (PR #3) — `.prproj` dedup fix.
- `claude/release-build` — integration of #1+#2+#3 + CI + GUI; this is the
  branch the Windows build is produced from.

## Decisions

### Production code was living outside git
The repo had only the initial MVP commit; ~14k lines of production code
(SAKHA AYMAKH, 4cams, monologue, `prproj_patcher`, …) existed only as
local files. PR #1 committed it in logical chunks. **Do not trust
`git log` for "what ships"** — historically `release/*.zip` snapshots were
the real record.

### Cross-channel mic-bleed cancellation (`core/cross_cancel.py`)
Separate mics pick up each other; when a speaker is silent the detector
still fires on the bleed. Added a deterministic Wiener-Hopf pre-detector
filter: for each channel pair it fits a short FIR on solo-speaker segments
and subtracts the predicted bleed before the RMS/VAD detector runs.
On synthetic tests it cuts false triggers ~1.5–3× without distorting the
speaker's own voice. **Default ON for `auto-multicam`** (1+1 is the most
bleed-prone layout); opt-in elsewhere via `--cross-cancel`.

### `--vad-threshold` raised to 0.65 for `auto-multicam`
Matches 4cams / SAKHA AYMAKH. A stricter Silero probability suppresses the
quieter cross-talk from the other mic.

### `.prproj` AudioComponentChain dedup (`prproj_patcher.py`)
Audio splitting `copy.deepcopy()`-ed a fresh `AudioComponentChain` per
segment. The chain has no per-segment state, so this duplicated thousands
of identical objects (measured: 14002 chains, 40 distinct → ~10.7 MB of
redundant XML) and bloated the file Premiere holds in memory. Fix: every
split segment references the original chain. **Partial fix (~15%)** — the
rest of the weight (`AudioClip`, transitions, `SubClip`) has real
per-segment data. A full fix means switching mute from physical cuts to
volume keyframes — a larger task, not done.

### CLI output forced to ASCII; console forced to UTF-8
`autopodcast.exe --help` crashed on a Russian Windows console:
`→`/`—` are not in cp1251/cp1252. CLI strings are now ASCII. Separately,
the Cyrillic interactive wizard needs a UTF-8 console, so `__main__.py`
switches the Windows console + Python streams to UTF-8 at startup, and the
exit prompts tolerate headless stdin (EOFError).

### GitHub Actions Windows build + two executables
`.github/workflows/build-windows.yml` builds `autopodcast.exe` on a Windows
runner so no one builds by hand. `build.spec` emits two exes —
`autopodcast.exe` (console/CLI) and `autopodcast-gui.exe` (windowed).

### Native GUI (`gui/`, customtkinter)
Replaced the text wizard with a window covering all four modes. Creative
numeric parameters are sliders. See [gui.md](gui.md).

### GUI exposes VAD strictness, not the RMS speech threshold
The GUI dropped the `--speech-threshold` slider. That option is the
RMS-backend dB threshold and has no effect when the Silero VAD backend is
active — which it is by default (`silero-vad` is bundled, `--detector-backend
auto` resolves to silero). The GUI now exposes `--vad-threshold` instead
(0.3–0.9) — the real, working knob for the spectral VAD detector. The CLI
still accepts `--speech-threshold` for the rms backend.

## Investigations that did NOT lead to code

### "Timeline lags / freezes" is not an autopodcast bug
Large `.prproj` files lag Premiere, but the cause is project complexity
(nested sequences, multi-component audio) plus length — not a regression.
`prproj_patcher.py` is byte-identical across the Mar/Apr/May builds. The
dedup fix above helps but does not make a 3-hour project small.
**fps (25 vs 23.976) is unrelated** — a coincidence.

### "Voice detection got worse after SAKHA AYMAKH" is not a regression
`detector.py`, `analyzer.py`, `segmenter.py` are byte-identical between the
pre-SAKHA (Apr 13) and SAKHA (May 4) builds. Real logs show the detector
reporting "both speakers" 58% of the time on a bleed-prone episode vs 14%
on a clean one — i.e. the problem is **mic bleed**, always present. SAKHA
AYMAKH mode has built-in debleed, so on its contrast the unchanged 1+1
detector merely *looks* worse. The fix is cross-cancel (above), not a
detector change.

## Open / future work

- Full `.prproj` size fix: volume keyframes instead of physical audio cuts.
- Validate cross-cancel on a real bleed-prone episode (only synthetic
  tested so far).
- Optionally port the SAKHA debleed layers into `auto-multicam`.
- GUI: drag-drop (deferred — `tkinterdnd2` + PyInstaller is fragile).
