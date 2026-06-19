# Multicam editing logic

The cross-mode editorial logic for the multicam planners (`sakha_aimakh`,
`auto_switch_4cams`, `auto_switch_custom`). It was refined on real recordings in
**SAKHA AYMAKH** (`core/sakha_aimakh.py`, the most complete planner) and then
reused/generalized by the other modes. This doc is the high-level "why";
per-mode detail lives in each planner and in
[auto-switching-spec-4cams.md](auto-switching-spec-4cams.md).

The base 2-speaker path (`analyze` / `auto-multicam`, via `switcher` +
`camera_scheduler`) is a separate, older pipeline — see [algorithm.md](algorithm.md).

## The pipeline (per mode)

```
per-person mic  → analyzer (RMS + envelope)  → detector (Silero VAD or RMS hysteresis)
   → [optional] cross_cancel (subtract inter-mic bleed before detection)
   → ACTIVE SET per frame  (who is really speaking — see "Attribution")
   → CAMERA per frame      (active set + person→camera map — see "Camera rules")
   → stabilize             (min hold, cooldown, hysteresis, re-establish)
   → camera segments  +  per-track audio open/mute plan
   → segments_to_cuts → patch_prproj   (.prproj angle switches + mic mutes)
```

Every planner is a pure function over per-person `SpeakerActivity` (no I/O). The
2-speaker `combine_speakers` (A/B/BOTH) is **not** used here — the multicam
planners take the full `activities` dict and compute their own per-frame active
set.

## Attribution — who is really speaking (the leak problem)

Separate mics hear each other. When one person talks, everyone's mic shows
"activity", so raw VAD over-reports overlap (on a real 4-person episode each
track read 59–75% active). Attribution decides the *true* speaker set per frame.
Two strategies, selected per mode:

- **Loudness dominance** (simple) — keep the loudest active speaker plus anyone
  within `dominance_delta_db` (≈6 dB) of them; quieter channels are treated as
  bleed. Used by `auto_switch_4cams` and by `auto_switch_custom` with
  `clean_mode=off`. Cheap but, in heavy bleed, a loud leak within ~6 dB still
  reads as a second speaker.

- **Studio leak-matrix unmixing** (the well-tuned one, from sakha) — models each
  mic as its own speaker plus an attenuated leaked copy of the others, estimates
  the leak matrix from clean anchor frames, then per frame picks the source set
  that best reconstructs the observed levels. Leakage is *explained away*, so a
  mic that is only loud from a neighbour's bleed is never opened. A waveform
  residual cross-check guards the rare "leak louder than direct" case.
  `sakha_aimakh._build_studio_frame_states` (also `calibrated` / `strict` /
  `source_owner` / `legacy` variants); reused by `auto_switch_custom`
  (`clean_mode=studio`, the default). Optional `cross_cancel` (Wiener-Hopf FIR)
  can run *before* detection as an additional generic bleed subtractor.

## Camera rules (the editorial heart, sakha-derived)

Given the cleaned active set and each person's assigned camera:

| Situation | Camera |
|---|---|
| one person speaking | their close-up |
| people who **share** a camera (e.g. two guests on one medium) | that shared shot |
| speakers on **two or more different** cameras (cross-camera overlap) | the wide / **общак** |
| silence | the wide (but held through brief pauses — see stability) |

Refinements proven on real footage (the user judges by sound/picture):

- **Overlap → общак.** Genuine simultaneous speech across cameras cuts to the
  wide; it is the only shot that covers everyone.
- **A clearly-led but fragmented turn → that person's close-up**, not the wide.
  When one speaker dominates a choppy run (≥ `choppy_overlap_dominance`, default
  0.70, of the run) interleaved with brief overlap blips, ride their close-up
  instead of letting the fragments collapse into the neighbour or the wide
  (`sakha_aimakh._consolidate_choppy_camera_runs`).
- **Never the cohost / pair shot for a silent partner.** A two-person framing is
  used only when both are really talking; if one is silent, prefer their
  individual close-up or the студийный wide.
- **Hold through brief pauses.** A sub-`shot_hold` silence/overlap blip is
  absorbed into the surrounding shot, so the camera does not flick to wide on
  every micro-gap.
- **Periodic re-establish.** During long dialogue with no wide for ~25 s, insert
  a short (~1.5–2 s) wide "master" to refresh screen geography.

In `auto_switch_custom` this whole table reduces to one function,
`_camera_for_active` (`core/auto_switch_custom.py`): all active speakers on one
camera → that camera; spanning ≥2 cameras (or none) → wide. Because people who
should appear together are *assigned the same camera*, the "shared shot" and
"never the lone-pair shot" rules fall out of the mapping.

## Stability (anti-chatter)

Applied after the per-frame camera is chosen, so cuts are watchable:

- **Minimum shot hold** (`shot_hold`, ≈1.2–1.4 s) — no shot shorter than this;
  short ones merge into the previous (this is also what bridges brief
  pauses/overlaps).
- **Cooldown** between cuts (≈0.7–0.9 s).
- **Video-state hysteresis** — a new state must persist (≈180 ms; overlap
  120 ms; silence 400 ms) before a cut is allowed.
- **Silence timeout** (≈0.8 s) before going to the wide.
- **Motion guard** (4cams / sakha / monologue, optional) — never cut to a
  physically moving camera; `camera_motion` detects moving intervals.

## Audio plan (mic management)

Camera choice and mic management are decided together. The active speaker's mic
is open; the others are muted, with `audio_pre_roll` (~0.24 s) before onset and
`audio_post_roll` (~0.12 s) after, plus run stabilization so a mic does not
micro-open/close (`build_audio_plan`, shared by 4cams / sakha / custom; returns
per-track open intervals). `patch_prproj` applies these as physical clip
splits + `IsMuted` per track.

> The 4cams spec ([auto-switching-spec-4cams.md](auto-switching-spec-4cams.md)
> §7) describes a graded `0 / -9 / -18 / -96 dB` model; the current writer mutes
> (binary open/closed) with pre/post-roll. Switching to volume keyframes is
> tracked in [decisions.md](decisions.md) → Open work.

## How each mode realizes it

| Mode | Planner | Attribution | Notes |
|---|---|---|---|
| SAKHA AYMAKH (2 hosts + 1 guest) | `sakha_aimakh.py` | studio leak-matrix (+ calibrated/strict/source_owner) | reference impl; phases, choppy-run consolidator, motion guard, max-visible-hold |
| 4 cams (1 host + 3 guests) | `auto_switch_4cams.py` | loudness dominance | conversation-phase model, motion guard, moving guest close-up (CAM_3). Canonical spec: [auto-switching-spec-4cams.md](auto-switching-spec-4cams.md) |
| Конструктор (any N) | `auto_switch_custom.py` | studio (default) or off | static person→camera map; one generic camera rule; reuses sakha studio + shared `build_audio_plan`. See [custom-mode.md](custom-mode.md) |
| Монолог (1 narrator, 2 cams) | `monologue_2cam.py` | n/a (one speaker) | scheduled alternation + pause targeting + motion guard |
| analyze / auto-multicam (2 speakers) | `switcher` + `camera_scheduler` | `combine_speakers` A/B/BOTH | older base pipeline; see [algorithm.md](algorithm.md) |

## Lineage & known gaps

- The editorial rules were hardened in `sakha_aimakh` (graded **A** in the
  2026-06 mode audit in [decisions.md](decisions.md)); `auto_switch_4cams` is
  **B−** (bare envelope-dominance detection, carries dead `cam3_*` config) and
  is being brought up toward sakha (decisions.md → Open work).
- The studio leak-matrix is the shared attribution layer; `auto_switch_custom`
  imports it directly from `sakha_aimakh`.
- Open issues: the 4cams moving close-up (CAM_3) is only partly realized (dead
  `cam3_*` config); a studio hot-guest-bleed edge is under investigation
  (`experiments/` bake-off). Both in [decisions.md](decisions.md).
