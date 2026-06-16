"""Headless GUI construction smoke test.

Skipped automatically where Tk / customtkinter / a display are unavailable
(plain CI). Where they exist (the Windows build, or a Mac with python-tk
installed) it verifies the window builds every mode's form — including the
Конструктор dynamic tables — and that the default custom form yields a valid
CLI command.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tkinter")
pytest.importorskip("customtkinter")


@pytest.fixture
def app():
    from autopodcast.gui.app import AutoPodcastApp

    try:
        instance = AutoPodcastApp()
    except Exception as exc:  # noqa: BLE001 - no display available -> skip, don't fail
        pytest.skip(f"Tk display unavailable: {exc}")
    try:
        yield instance
    finally:
        instance.destroy()


def test_all_mode_forms_build(app):
    from autopodcast.gui.spec import MODES

    for mode in MODES:
        app._on_mode_change(mode.title)  # rebuilds the form; must not raise


def test_custom_default_form_builds_valid_command(app):
    from autopodcast.gui.spec import build_argv, mode_by_key

    app._on_mode_change("Конструктор")
    values = {arg: getter() for arg, getter in app._getters.items()}
    values["--in"] = "/proj/Episode.prproj"
    values["--seq"] = "Seq 1"

    people = values["--person"]
    assert people and isinstance(people, list)
    people[0]["label"] = "ведущий"  # the one thing the user must type

    argv = build_argv(mode_by_key("custom"), values)
    assert argv[0] == "auto-switch-custom"
    assert "--person" in argv
    # camera 1 is the default общак, so the command is valid out of the box
    assert argv[argv.index("--wide-camera") + 1] == "1"
    assert argv[argv.index("--out") + 1].endswith("_custom.prproj")
