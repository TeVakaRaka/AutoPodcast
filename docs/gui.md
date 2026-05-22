# GUI

`src/autopodcast/gui/` is a customtkinter desktop window covering all four
editing modes. It is a thin front end: it builds an `autopodcast` CLI
command from form values and runs it as a child process.

## How it launches

- Double-click the `.exe` → `__main__.py` opens the GUI. If the GUI cannot
  start (no display, broken customtkinter) it falls back to the text wizard.
- `python -m autopodcast gui` → opens the GUI explicitly.
- `autopodcast.exe` is built in two variants (see [build-and-ci.md](build-and-ci.md)):
  `autopodcast-gui.exe` (windowed) and `autopodcast.exe` (console, CLI).

## Module layout

```
gui/
  spec.py     declarative form spec — Field, ModeSpec, MODES, build_argv.
              Pure data + logic, NO tkinter import, fully unit-tested.
  runner.py   ProcessRunner — runs the CLI in a child process, streams
              stdout into a queue, cancels via taskkill /T.
  widgets.py  FileField, CollapsibleSection, SliderField, widget factory.
  app.py      AutoPodcastApp(CTk) — the window.
  main.py     main() entry point.
```

## spec.py — the heart of the GUI

Each mode is a `ModeSpec` (cmd, key, title, out_suffix, fields). Each form
control is a `Field`:

| Field attribute | Purpose |
|---|---|
| `arg` | CLI flag, e.g. `--mic-a` |
| `label` | Russian label shown in the form |
| `kind` | `file` / `int` / `float` / `str` / `bool` / `choice` |
| `default` | must match the matching `@click.option` default in `cli.py` |
| `required` | always emitted; empty value raises a validation error |
| `advanced` | shown inside the collapsible "Дополнительно" section |
| `flag_pair` | for `bool` — `(on_flag, off_flag)` |
| `vmin/vmax/vstep/unit` | when set on int/float → rendered as a **slider** |

`build_argv(spec, values)` turns filled-in values into the CLI argument
list: required flags always, optional flags only when changed from default,
bool flags emit the side opposite their default, `--out` is derived from
`--in` + `out_suffix`.

## Sliders

Numeric "creative" parameters (tempo, cut frequency, camera share, detector
thresholds) are sliders with a live value readout, created by giving the
`Field` a `vmin`/`vmax`/`vstep`/`unit`. Every mode exposes sliders.

## How to add a field

1. Add a `Field(...)` to the relevant `ModeSpec` in `spec.py`, with a
   `default` equal to the CLI option's default.
2. That's it — the form, the argument builder and the consistency test pick
   it up automatically.

`tests/test_gui_spec.py::TestSpecMatchesCli` fails if a field's `arg` is not
a real CLI option or its `default` drifts from the CLI default.

## How to add a mode

Add a new `ModeSpec` to `MODES`. The mode selector and form rebuild from it.

## Process model

The GUI never imports the planners. `runner.py` launches
`autopodcast <subcommand> <args>` as a subprocess, so a crash in ffmpeg or
the ML backend cannot take the window down, and "Отмена" is a clean kill of
the whole process tree. Progress is an indeterminate bar plus the live
stdout log (there is no reliable percentage).

## Testing

`spec.py` is pure (no tkinter) and unit-tested in `tests/test_gui_spec.py`.
The window itself cannot be unit-tested headlessly — verify it manually on
Windows after a CI build (see the verification list in
[decisions.md](decisions.md)).
