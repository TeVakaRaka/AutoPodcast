# Конструктор — configurable mode (`auto-switch-custom`)

The four preset modes (`auto-multicam`, `auto-switch-4cams`,
`auto-switch-sakha-aimakh`, `auto-switch-monologue`) each hard-code a specific
cast and camera layout. **Конструктор** is the assembleable ("сборная") mode:
you declare *any* number of people and cameras and map each person to a camera
yourself. The presets are untouched — this is an additional, fifth mode.

Planner: `core/auto_switch_custom.py`. CLI: `auto-switch-custom`. GUI: the
"Конструктор" tab.

## The camera rule

Per analysis frame the planner computes the set of people currently speaking
(loudness-dominance filtered, like the presets), then picks one camera:

| Who is speaking | Camera shown |
|---|---|
| nobody | the wide / общак |
| everyone speaking shares one camera | that camera (a solo close-up, **or** a shared shot of a pair) |
| speakers are on two or more different cameras | the wide / общак |

This single rule generalizes every preset and matches the editorial rule
"overlap → общак": two people framed *together* on one camera looking good when
they both talk, while people on *different* cameras overlapping cuts to the wide.

The mapping is **static** — a person is always shown on their assigned camera.
There is no auto-moving close-up (that lives in `auto-switch-4cams`).

### Holding through blips

Brief events do not cause a cut: a sub-`shot-hold` segment (a 0.2 s
interjection, a momentary cross-camera overlap, a short silence between turns)
is merged into the shot that preceded it. So the camera holds on the current
speaker through small gaps, and only a *sustained* overlap or silence actually
cuts to the wide. `--shot-hold` (default 1.4 s) is the single knob for this.

## Worked example — the 4-person roundtable

4 people, 4 cameras: cam 1 = общак, cam 2 = host, cam 3 = a shared medium of
guests 1 **and** 2, cam 4 = guest 3.

```
host  -> cam 2      guest 1 -> cam 3      guest 2 -> cam 3      guest 3 -> cam 4
                              общак = cam 1
```

What the planner does:

- host alone → cam 2; guest 3 alone → cam 4
- guest 1 alone, guest 2 alone, **or guests 1+2 together** → cam 3 (their shared medium)
- host + guest 3 (different cameras) → cam 1 (общак)
- a real silence → cam 1 (общак)

### GUI

Open the **Конструктор** tab:

1. **Камеры** table — add a row per camera: its `Angle` (the multicam angle
   number in the sequence), an optional name, and tick **Общак** on exactly one.
2. **Люди** table — add a row per person: name, their **Аудиодорожка** (audio
   track number in the project) and the **Камера (angle)** that frames them.
3. Pick the `.prproj` and sequence name, press **Запустить**. Advanced knobs
   (`shot-hold`, periodic re-establish, mic muting, leak suppression, VAD) live
   under "Дополнительно".

### CLI

```bash
python3 -m autopodcast auto-switch-custom \
  --in show.prproj --seq "Episode 1" --out show_custom.prproj \
  --person "ведущий:1:2" \
  --person "гость1:2:3" \
  --person "гость2:3:3" \
  --person "гость3:4:4" \
  --wide-camera 1
```

`--person` is `label:audio_track:camera_angle` (1-based track & angle; no `:`
in the label) and is repeatable. `--wide-camera` is the общак angle. Mics are
auto-resolved from the project by audio track (as in every mode); pass `--mic`
once per person, in order, to override. Output is the patched `.prproj` plus a
`.log.jsonl`.

## Audio

When `--mute-audio` is on (default), each person's mic track is opened only
while they speak (with pre/post-roll and the same stabilization the presets
use, via the shared `build_audio_plan`) and muted otherwise. During silence all
tracks stay open. `--cross-cancel` subtracts inter-mic bleed before detection.

## Limitations / notes

- Camera angles are the user's responsibility — they must exist in the
  multicam sequence (the planner does not verify the angle count).
- Static mapping only; no moving close-up or guest-accent logic.
- Each person needs a distinct audio track (enforced).
