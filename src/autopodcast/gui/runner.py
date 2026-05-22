"""Background subprocess runner for the AutoPodcast GUI.

Runs an ``autopodcast`` CLI command in a child process and streams its
stdout, line by line, into a thread-safe queue the GUI drains on a timer.
Keeping the work in a separate process means a crash in ffmpeg or the ML
backend cannot take the window down, and cancelling is a clean kill.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading


def cli_command() -> list[str]:
    """Command prefix that invokes the autopodcast CLI.

    Frozen build: the sibling ``autopodcast.exe`` (the console variant).
    Dev run: ``python -m autopodcast``.
    """
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(sys.executable)
        cli_exe = os.path.join(exe_dir, "autopodcast.exe")
        if os.path.exists(cli_exe):
            return [cli_exe]
        return [sys.executable]  # fallback: run self
    return [sys.executable, "-m", "autopodcast"]


class ProcessRunner:
    """Runs one CLI command at a time, streaming output into ``output``."""

    def __init__(self) -> None:
        self.output: queue.Queue[str] = queue.Queue()
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._returncode: int | None = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def returncode(self) -> int | None:
        return self._returncode

    def start(self, args: list[str]) -> None:
        """Launch ``autopodcast <args>`` as a child process."""
        if self.running:
            raise RuntimeError("Обработка уже запущена")
        cmd = cli_command() + args
        self._returncode = None
        while not self.output.empty():  # drop stale lines from a prior run
            try:
                self.output.get_nowait()
            except queue.Empty:
                break

        creationflags = 0
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            self.output.put(line.rstrip("\n"))
        proc.stdout.close()
        self._returncode = proc.wait()

    def cancel(self) -> None:
        """Terminate the running command, including its child processes."""
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        if sys.platform == "win32":
            # /T kills the whole process tree (the CLI's ffmpeg children too).
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
            )
        else:
            proc.terminate()

    def poll(self) -> int | None:
        """Return the exit code if the process has finished, else None."""
        if self._proc is None:
            return None
        return self._proc.poll()
