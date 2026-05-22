"""Tests for the GUI subprocess runner — focused on error paths.

ProcessRunner always prefixes ``cli_command()``; the tests monkeypatch it
to a lightweight ``python -c`` so they exercise the runner mechanics
(streaming, exit codes, cancel) without launching the full autopodcast CLI.
"""

from __future__ import annotations

import sys
import time

import pytest

from autopodcast.gui import runner as runner_mod
from autopodcast.gui.runner import ProcessRunner, cli_command


def _use_python_c(monkeypatch):
    """Make ProcessRunner run `python -c <code>` instead of the real CLI."""
    monkeypatch.setattr(runner_mod, "cli_command", lambda: [sys.executable, "-c"])


def _wait_done(r: ProcessRunner, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while r.running and time.time() < deadline:
        time.sleep(0.05)
    # let the reader thread finish and set returncode
    deadline = time.time() + 3.0
    while r.returncode is None and time.time() < deadline:
        time.sleep(0.05)


def _drain(r: ProcessRunner) -> list[str]:
    lines = []
    while not r.output.empty():
        lines.append(r.output.get_nowait())
    return lines


# ---------------------------------------------------------------------------
# cli_command
# ---------------------------------------------------------------------------


def test_cli_command_returns_non_empty_list():
    cmd = cli_command()
    assert isinstance(cmd, list)
    assert len(cmd) >= 1
    assert all(isinstance(part, str) for part in cmd)


# ---------------------------------------------------------------------------
# initial state
# ---------------------------------------------------------------------------


def test_fresh_runner_is_idle():
    r = ProcessRunner()
    assert r.running is False
    assert r.poll() is None
    assert r.returncode is None


# ---------------------------------------------------------------------------
# normal streaming
# ---------------------------------------------------------------------------


def test_streams_stdout_lines(monkeypatch):
    _use_python_c(monkeypatch)
    r = ProcessRunner()
    r.start(["print('line-one'); print('line-two')"])
    _wait_done(r)
    assert r.returncode == 0
    lines = _drain(r)
    assert "line-one" in lines
    assert "line-two" in lines


def test_stderr_is_merged_into_output(monkeypatch):
    _use_python_c(monkeypatch)
    r = ProcessRunner()
    r.start(["import sys; sys.stderr.write('oops\\n')"])
    _wait_done(r)
    assert any("oops" in line for line in _drain(r))


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------


def test_failing_command_reports_nonzero_returncode(monkeypatch):
    _use_python_c(monkeypatch)
    r = ProcessRunner()
    r.start(["import sys; sys.exit(3)"])
    _wait_done(r)
    assert r.returncode == 3


def test_crashing_command_surfaces_traceback(monkeypatch):
    _use_python_c(monkeypatch)
    r = ProcessRunner()
    r.start(["raise RuntimeError('boom')"])
    _wait_done(r)
    assert r.returncode != 0
    assert any("boom" in line for line in _drain(r))


def test_double_start_raises(monkeypatch):
    _use_python_c(monkeypatch)
    r = ProcessRunner()
    r.start(["import time; time.sleep(5)"])
    try:
        with pytest.raises(RuntimeError):
            r.start(["print('second')"])
    finally:
        r.cancel()
        _wait_done(r)


# ---------------------------------------------------------------------------
# cancellation
# ---------------------------------------------------------------------------


def test_cancel_stops_a_running_process(monkeypatch):
    _use_python_c(monkeypatch)
    r = ProcessRunner()
    r.start(["import time; time.sleep(30)"])
    time.sleep(0.5)
    assert r.running is True
    r.cancel()
    _wait_done(r)
    assert r.running is False


def test_cancel_when_idle_is_noop():
    r = ProcessRunner()
    r.cancel()  # must not raise
    assert r.running is False
