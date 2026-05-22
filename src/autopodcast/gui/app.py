"""Main AutoPodcast GUI window (customtkinter)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import customtkinter as ctk

from autopodcast.gui.runner import ProcessRunner
from autopodcast.gui.spec import MODES, build_argv
from autopodcast.gui.widgets import (
    CollapsibleSection,
    FileField,
    make_field_widget,
)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

_LOG_FONT = ("Courier New", 12)
_INVALID_BORDER = "#c0504d"  # red-ish border for an empty required field

# (prefix, progress 0..1) — first match wins, values must be non-decreasing
_PROGRESS_MARKERS = [
    ("Audio sources:", 0.05),
    ("Loading audio:", 0.10),
    ("Duration:", 0.18),
    ("Analyzing participant", 0.22),
    ("Analyzing camera motion", 0.55),
    ("Building SAKHA AYMAKH plan", 0.70),
    ("Planning complete", 0.78),
    ("Patching project", 0.88),
    ("Done:", 0.98),
]


def open_folder(path: str | os.PathLike) -> None:
    """Open a folder in the OS file manager (best-effort, never raises)."""
    folder = str(path)
    try:
        if sys.platform == "win32":
            os.startfile(folder)  # noqa: S606 - intended
        elif sys.platform == "darwin":
            subprocess.run(["open", folder], check=False)
        else:
            subprocess.run(["xdg-open", folder], check=False)
    except Exception:  # noqa: BLE001 - opening a folder must never crash the GUI
        pass


def _entry_of(widget):
    """Return the editable Entry of a field widget, or None."""
    if isinstance(widget, FileField):
        return widget.entry
    if isinstance(widget, ctk.CTkEntry):
        return widget
    return None


class AutoPodcastApp(ctk.CTk):
    """Single-window GUI covering all four AutoPodcast editing modes."""

    def __init__(self) -> None:
        super().__init__()
        self.title("AutoPodcast")
        self.geometry("920x760")
        self.minsize(760, 640)

        self.runner = ProcessRunner()
        self._getters: dict[str, object] = {}
        self._req_widgets: dict[str, object] = {}   # required arg -> widget
        self._default_borders: dict[str, object] = {}
        self._current_spec = MODES[0]
        self._busy = False
        self._last_out: str | None = None
        self._progress_val: float = 0.0

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=3)
        self.grid_rowconfigure(3, weight=2)

        # --- mode selector ---
        self._titles = [m.title for m in MODES]
        self.selector = ctk.CTkSegmentedButton(
            self, values=self._titles, command=self._on_mode_change,
        )
        self.selector.set(self._titles[0])
        self.selector.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 8))

        # --- scrollable parameter form ---
        self.form = ctk.CTkScrollableFrame(self, label_text="Параметры")
        self.form.grid(row=1, column=0, sticky="nsew", padx=16, pady=8)
        self.form.grid_columnconfigure(1, weight=1)

        # --- run / cancel / open + progress ---
        controls = ctk.CTkFrame(self, fg_color="transparent")
        controls.grid(row=2, column=0, sticky="ew", padx=16, pady=8)
        controls.grid_columnconfigure(3, weight=1)
        self.run_btn = ctk.CTkButton(controls, text="Запустить", command=self._on_run)
        self.run_btn.grid(row=0, column=0, padx=(0, 8))
        self.cancel_btn = ctk.CTkButton(
            controls, text="Отмена", command=self._on_cancel,
            state="disabled", fg_color="#a0392e", hover_color="#7d2c24",
        )
        self.cancel_btn.grid(row=0, column=1, padx=(0, 8))
        self.open_btn = ctk.CTkButton(
            controls, text="Открыть папку результата",
            command=self._open_result_folder, state="disabled",
        )
        self.open_btn.grid(row=0, column=2, padx=(0, 8))
        self.progress = ctk.CTkProgressBar(controls)
        self.progress.grid(row=0, column=3, sticky="ew")
        self.progress.set(0)

        # --- log ---
        self.log = ctk.CTkTextbox(self, font=_LOG_FONT, wrap="word")
        self.log.grid(row=3, column=0, sticky="nsew", padx=16, pady=(8, 16))
        self.log.configure(state="disabled")

        self._build_form(self._current_spec)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_runner()

    # ------------------------------------------------------------------ form

    def _on_mode_change(self, title: str) -> None:
        if self._busy:
            self.selector.set(self._current_spec.title)  # no switching mid-run
            return
        for mode in MODES:
            if mode.title == title:
                self._current_spec = mode
                break
        self._build_form(self._current_spec)

    def _build_form(self, spec) -> None:
        for child in self.form.winfo_children():
            child.destroy()
        self._getters = {}
        self._req_widgets = {}
        self._default_borders = {}

        row = 0
        for fld in spec.fields:
            if not fld.advanced:
                row = self._add_field_row(self.form, fld, row)

        advanced = [f for f in spec.fields if f.advanced]
        if advanced:
            section = CollapsibleSection(self.form, "Дополнительно", expanded=False)
            section.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 0))
            srow = 0
            for fld in advanced:
                srow = self._add_field_row(section.body, fld, srow)

    def _add_field_row(self, parent, fld, row: int) -> int:
        text = fld.label + ("  *" if fld.required else "")
        label = ctk.CTkLabel(parent, text=text, anchor="w")
        label.grid(row=row, column=0, sticky="w", padx=(0, 12), pady=5)
        widget, getter = make_field_widget(parent, fld)
        widget.grid(row=row, column=1, sticky="ew", pady=5)
        parent.grid_columnconfigure(1, weight=1)
        self._getters[fld.arg] = getter
        if fld.required:
            self._req_widgets[fld.arg] = widget
            entry = _entry_of(widget)
            if entry is not None:
                self._default_borders[fld.arg] = entry.cget("border_color")
        return row + 1

    # ------------------------------------------------------------ validation

    def _validate(self) -> None:
        """Highlight empty required fields and enable Run only when all filled."""
        if self._busy:
            return
        all_filled = True
        for arg, widget in self._req_widgets.items():
            getter = self._getters.get(arg)
            filled = bool(str(getter()).strip()) if getter else False
            if not filled:
                all_filled = False
            entry = _entry_of(widget)
            if entry is not None:
                if filled:
                    default = self._default_borders.get(arg)
                    if default is not None:
                        entry.configure(border_color=default)
                else:
                    entry.configure(border_color=_INVALID_BORDER)
        self.run_btn.configure(state="normal" if all_filled else "disabled")

    # ------------------------------------------------------------------- run

    def _on_run(self) -> None:
        if self._busy:
            return
        values = {arg: getter() for arg, getter in self._getters.items()}
        try:
            argv = build_argv(self._current_spec, values)
        except ValueError as exc:
            self._append(f"!!! {exc}")
            return

        self._last_out = None
        if "--out" in argv:
            self._last_out = argv[argv.index("--out") + 1]

        self._clear_log()
        self._append("Команда: autopodcast " + " ".join(argv))
        self._append("-" * 64)
        try:
            self.runner.start(argv)
        except Exception as exc:  # noqa: BLE001 - surface any startup failure
            self._append(f"!!! Не удалось запустить обработку: {exc}")
            return

        self._busy = True
        self._progress_val = 0.0
        self.run_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.open_btn.configure(state="disabled")
        self.progress.set(0.01)

    def _on_cancel(self) -> None:
        self.runner.cancel()
        self._append("... запрошена отмена")

    def _update_progress(self, line: str) -> None:
        """Advance the progress bar when a known milestone appears in a log line."""
        for prefix, value in _PROGRESS_MARKERS:
            if prefix in line and value > self._progress_val:
                self._progress_val = value
                self.progress.set(value)
                break

    def _poll_runner(self) -> None:
        while not self.runner.output.empty():
            try:
                line = self.runner.output.get_nowait()
                self._append(line)
                if self._busy:
                    self._update_progress(line)
            except Exception:  # noqa: BLE001
                break

        if self._busy and not self.runner.running and self.runner.returncode is not None:
            self._finish(self.runner.returncode)
        else:
            self._validate()

        self.after(100, self._poll_runner)

    def _finish(self, code: int) -> None:
        self._busy = False
        self.progress.set(1.0 if code == 0 else self._progress_val)
        self.cancel_btn.configure(state="disabled")
        self._append("-" * 64)
        if code == 0:
            self._append("ГОТОВО. Откройте результат в Premiere Pro (File > Open Project).")
            if self._last_out and Path(self._last_out).parent.exists():
                self.open_btn.configure(state="normal")
        else:
            self._append(f"ЗАВЕРШЕНО С ОШИБКОЙ (код {code}). Проверьте лог выше.")
        self._validate()  # restores Run button state

    def _open_result_folder(self) -> None:
        if self._last_out:
            open_folder(Path(self._last_out).parent)

    # ------------------------------------------------------------------- log

    def _append(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    # ----------------------------------------------------------------- close

    def _on_close(self) -> None:
        if self.runner.running:
            self.runner.cancel()
        self.destroy()
