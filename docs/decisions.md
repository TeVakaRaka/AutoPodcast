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

### "Конструктор" — a configurable N-people / M-cameras mode (`auto_switch_custom.py`, `cli.py`, `gui/*`)
**2026-06-16 · branch `claude/awesome-pike-256a34` (off `claude/release-build`).** The presets
hard-code the cast (1+3, 2+1, …); the user wanted a *сборная* version where you set how many
people there are, each one's audio track and camera, and how many cameras — configured in the GUI
"по кнопочкам". Added a fifth, **additive** mode (presets untouched). New pure planner
`core/auto_switch_custom.py` with one generalizing rule: per frame, take the dominance-filtered
active speaker set → all on one camera ⇒ that camera (solo close-up **or** a shared pair shot like
guests 1+2); on ≥2 cameras, or silence ⇒ the wide/общак. Static mapping, no moving close-up;
`shot-hold` (min camera duration) absorbs brief pauses/overlaps/interjections so the camera holds.
Reused as-is: `_resolve_speaker_mics`, the analyze/detect stack, `segments_to_cuts`, `patch_prproj`
(its `audio_track_intervals_s` is already per-track / N-capable), and the role-agnostic
`build_audio_plan` (imported from `auto_switch_4cams`, not duplicated). CLI `auto-switch-custom`
mirrors the 4-cam command: repeatable `--person "label:track:camera"` (1-based) + `--wide-camera`;
ASCII-only help (the `test_cli_ascii` guard). GUI: a new `Field(kind="rows")` + `DynamicRowsField`
add/remove tables for people and cameras; `build_argv` special-cases the custom mode to expand them
into `--person`/`--wide-camera` (all the testable logic stays in pure `spec.py`). Tests:
`test_auto_switch_custom.py` (rule + the user's 4-person case + stability + audio), `test_custom_cli.py`
(parse/validation), and custom cases in `test_gui_spec.py`. **Caveat:** the GUI window can't run on
this dev Mac (Homebrew Python has no Tk) — engine + `spec.py` are verified headless; verify the
window itself on the Windows build. Pre-existing unrelated failure left as-is:
`test_monologue_2cam.py::...motion_guard_smoke` (fails on pristine `release-build` too).

### SAKHA camera: a fragmented host turn no longer swallowed by the guest (`sakha_aimakh.py`)
**2026-06-02 · `d7da9a5`.** On the real 20.05 Саха recording (studio mode) the camera sat on the
guest ~1892–1905 s even though the host's mic was open and the host spoke ~1896.5–1901 s — the
camera had **diverged from the audio plan**. Cause: the host turn is split into sub-`shot_hold`
fragments (`main_host_only` interleaved with brief `main_host_guest` overlap-wides), and
`_enforce_min_camera_duration` merged each fragment into its longer guest neighbour. Fix: new
`_consolidate_choppy_camera_runs` runs *before* the min-duration cleanup. A choppy run (3+ short
shots, 2+ cameras, ≥1 close camera) is resolved by floor-share: one close camera holding ≥
`choppy_overlap_dominance` (new config, default 0.70) of the run → ride its close-up; otherwise a
genuine two-way overlap → ride the studio wide (all_wide), never the cohost pair shot. A sustained
solo stays one ≥short segment and never enters a run. User's editorial rule: overlap → общак; a
clearly-led-but-fragmented turn → that close-up; never the средний/pair shot (cohost is silent).
Verified: host turn now its own close-up, camera matches the audio plan, no sub-0.5 s chatter.

### Auto-resolve mics from the project in every switch mode (`cli.py`, `gui/spec.py`)
**2026-06-02 · `ef2cd35`.** Only `auto-switch-sakha-aimakh` auto-resolved each speaker's mic from
the `.prproj`/XML sequence; the other modes forced a manual file pick. Extracted the sakha logic
into a shared `_resolve_speaker_mics()` and wired it into `auto-multicam` (2 mics),
`auto-switch-4cams` (4), `auto-switch-monologue` (1): `--mic*` are now optional, resolved from the
sequence audio tracks (keyed by the existing `--audio-track-*`) when omitted, with a
duplicate-path guard (two tracks → one file disables muting). `auto-multicam` gained `--xml`. GUI
mic fields for these modes are no longer required and show "пусто = автопоиск из проекта". Aligned
`auto-multicam` audio defaults to sakha (speech −24→−27, pre-roll 0.35→0.24, post-roll 0.10→0.12).
Camera algorithms / leak models deliberately untouched (scope = "auto-audio + defaults"; the
leak-model port stays in Open work). 488 tests pass (+4 optional-mic tests). Caveat: live
end-to-end resolve not re-run (T7 drive disconnected mid-session), but `resolve_sequence_audio_sources`
is the same function the user's prior run used successfully (`method=prproj` in its log).

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

### Mode audit with grades (2026-06-02, no code change)
Comparative audit to decide what to harden. Camera planners: `sakha_aimakh` **A** (multi-engine
de-bleed + every editorial rule), `monologue_2cam` **A−** (predictive motion: delay-until-stable +
emergency escape), `auto_switch_4cams` **B−** (strong phase/fairness model + top motion guard, but
speaker detection is bare envelope-dominance and it carries dead `cam3_*` config),
`camera_scheduler`/`switcher` **C+** (decent 2-speaker rules but **no motion guard** — the only
planner that can cut to a moving camera). Audio `audio_clean_mode`s (real-audio behaviour):
`calibrated` **B** (correct normalized attribution, passes the t=364 gate, chatter ≈983), `legacy`
**C+** (best easy-frame accuracy + lowest wrong-hold, but fails the gate by co-opening the cohost),
`studio` **C−** (only physically-principled, but raw-dB + momentum cause the host-as-guest-bleed
long holds — the 1896 s case above), `source_owner` **D** (= what `strict`/`balanced` actually run
on real audio; `_build_strict_frame_states` only runs in no-audio tests; lowest chatter 52 but
worst wrong-hold 161 s, fails the gate, starves the cohost to 228 s). No `audio_clean_mode` is fully
correct — confirms the `experiments/` bake-off premise.

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

- Bring weaker modes toward `sakha_aimakh` (2026-06-02 audit): a real de-bleed/source-owner
  detector for `auto_switch_4cams` (currently bare envelope dominance); a motion guard for the
  2-speaker `camera_scheduler` path; port momentum, hold-through-brief-silence, max-visible-hold,
  and the choppy-run consolidator to the other multicam modes; remove the dead `cam3_*` config.
- Studio audio decision holds the guest mic open while the host talks (the 1896 s
  host-suppression / t=364-class hot-guest-bleed bug) — the `experiments/` bake-off.
- Full `.prproj` size fix: volume keyframes instead of physical audio cuts.
- Validate cross-cancel on a real bleed-prone episode (only synthetic
  tested so far).
- Optionally port the SAKHA debleed layers into `auto-multicam`.
- GUI: drag-drop (deferred — `tkinterdnd2` + PyInstaller is fragile).
