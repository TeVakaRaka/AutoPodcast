"""Regression tests: CLI output must be ASCII-only.

A Russian Windows console uses a legacy code page (cp1251/cp1252) that
cannot encode characters like the arrow (U+2192) or em-dash (U+2014).
Such a character anywhere on a printable path crashes autopodcast.exe
with UnicodeEncodeError — this already happened twice with `--help`.

These tests fail if any non-ASCII character creeps back into a command's
help text, so the crash cannot regress.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from autopodcast.cli import cli


def _assert_ascii(text: str, where: str) -> None:
    try:
        text.encode("ascii")
    except UnicodeEncodeError as exc:
        bad = text[exc.start : exc.end]
        raise AssertionError(
            f"{where}: non-ASCII character {bad!r} (U+{ord(bad[0]):04X}) "
            f"in CLI output — will crash on a cp1251/cp1252 Windows console"
        ) from None


def test_root_help_is_ascii():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    _assert_ascii(result.output, "cli --help")


@pytest.mark.parametrize("command", sorted(cli.commands))
def test_each_command_help_is_ascii(command):
    result = CliRunner().invoke(cli, [command, "--help"])
    assert result.exit_code == 0
    _assert_ascii(result.output, f"{command} --help")
