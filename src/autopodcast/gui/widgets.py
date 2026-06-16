"""Reusable customtkinter widgets for the AutoPodcast GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog
from typing import Callable

import customtkinter as ctk

from autopodcast.gui.spec import FILE_AUDIO, FILE_PRPROJ, FILE_XML, Field


class Tooltip:
    """Floating tooltip that appears after a short hover delay."""

    _DELAY_MS = 350

    def __init__(self, widget: tk.BaseWidget, text: str) -> None:
        self._widget = widget
        self._text = text
        self._win: tk.Toplevel | None = None
        self._after_id: str | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._cancel, add="+")
        widget.bind("<Button>", self._cancel, add="+")

    def _schedule(self, _event=None) -> None:
        self._cancel()
        self._after_id = self._widget.after(self._DELAY_MS, self._show)

    def _cancel(self, _event=None) -> None:
        if self._after_id is not None:
            self._widget.after_cancel(self._after_id)
            self._after_id = None
        self._hide()

    def _show(self) -> None:
        if self._win is not None:
            return
        x = self._widget.winfo_rootx()
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 6
        self._win = tw = tk.Toplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.attributes("-topmost", True)
        tw.configure(bg="#3a3a3a")
        lbl = tk.Label(
            tw,
            text=self._text,
            justify=tk.LEFT,
            bg="#2b2b2b",
            fg="#dddddd",
            padx=10,
            pady=6,
            wraplength=320,
            font=("Segoe UI", 9) if tk.TclVersion else ("Helvetica", 10),
        )
        lbl.pack(padx=1, pady=1)
        tw.wm_geometry(f"+{x}+{y}")

    def _hide(self) -> None:
        if self._win is not None:
            self._win.destroy()
            self._win = None

_FILETYPES = {
    FILE_PRPROJ: [("Premiere проект", "*.prproj"), ("Все файлы", "*.*")],
    FILE_AUDIO: [
        ("Аудио", "*.wav *.aac *.mp3 *.m4a *.flac *.aif *.aiff *.ogg"),
        ("Все файлы", "*.*"),
    ],
    FILE_XML: [("FCP7 XML", "*.xml"), ("Все файлы", "*.*")],
}


class FileField(ctk.CTkFrame):
    """Path entry with a 'Обзор' button (file picker)."""

    def __init__(self, master, file_filter: str = "", hint: str = ""):
        super().__init__(master, fg_color="transparent")
        self._filetypes = _FILETYPES.get(file_filter, [("Все файлы", "*.*")])
        self.entry = ctk.CTkEntry(self, placeholder_text=hint or "путь к файлу")
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.button = ctk.CTkButton(self, text="Обзор", width=80, command=self._browse)
        self.button.grid(row=0, column=1)
        self.grid_columnconfigure(0, weight=1)

    def _browse(self) -> None:
        path = filedialog.askopenfilename(filetypes=self._filetypes)
        if path:
            self.entry.delete(0, "end")
            self.entry.insert(0, path)

    def get(self) -> str:
        return self.entry.get().strip()


class CollapsibleSection(ctk.CTkFrame):
    """A section with a header that shows/hides its body."""

    def __init__(self, master, title: str, expanded: bool = False):
        super().__init__(master, fg_color="transparent")
        self._expanded = expanded
        self._title = title
        self.header = ctk.CTkButton(
            self, text=self._header_text(), anchor="w",
            fg_color="transparent", hover=False, command=self.toggle,
        )
        self.header.grid(row=0, column=0, sticky="ew")
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.grid_columnconfigure(0, weight=1)
        self.body.grid_columnconfigure(0, weight=1)
        if expanded:
            self.body.grid(row=1, column=0, sticky="ew", padx=(16, 0))

    def _header_text(self) -> str:
        return ("▾ " if self._expanded else "▸ ") + self._title

    def toggle(self) -> None:
        self._expanded = not self._expanded
        self.header.configure(text=self._header_text())
        if self._expanded:
            self.body.grid(row=1, column=0, sticky="ew", padx=(16, 0))
        else:
            self.body.grid_remove()


class SliderField(ctk.CTkFrame):
    """A slider with a live value readout, for "creative" numeric params."""

    def __init__(self, master, fld: Field):
        super().__init__(master, fg_color="transparent")
        self._fld = fld
        self._decimals = 0 if fld.vstep >= 1 else 2
        steps = max(1, int(round((fld.vmax - fld.vmin) / fld.vstep)))
        self.slider = ctk.CTkSlider(
            self, from_=fld.vmin, to=fld.vmax,
            number_of_steps=steps, command=self._on_move,
        )
        start = fld.default if fld.default is not None else fld.vmin
        self.slider.set(start)
        self.slider.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.value_label = ctk.CTkLabel(self, width=70, anchor="e")
        self.value_label.grid(row=0, column=1)
        self.grid_columnconfigure(0, weight=1)
        self._on_move(start)

    def _quantized(self, raw: float) -> float:
        f = self._fld
        snapped = round((raw - f.vmin) / f.vstep) * f.vstep + f.vmin
        return round(snapped, self._decimals)

    def _on_move(self, raw: float) -> None:
        value = self._quantized(raw)
        text = f"{value:g}"
        if self._fld.unit:
            text += " " + self._fld.unit
        self.value_label.configure(text=text)

    def get(self) -> float:
        return self._quantized(self.slider.get())


class DynamicRowsField(ctk.CTkFrame):
    """A table of repeatable rows for the custom mode (people / cameras).

    Each row has one widget per column; rows can be added or removed with
    buttons. ``get`` returns a list of dicts keyed by each column's ``key``.
    """

    def __init__(self, master, columns, add_label="➕ Добавить", initial_rows=1):
        super().__init__(master, fg_color="transparent")
        self._columns = list(columns)
        self._rows: list[tuple[ctk.CTkFrame, dict]] = []
        self.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew")
        for ci, col in enumerate(self._columns):
            ctk.CTkLabel(header, text=col.header, anchor="w").grid(
                row=0, column=ci, sticky="w", padx=(0, 8)
            )
            header.grid_columnconfigure(ci, weight=1)

        self._body = ctk.CTkFrame(self, fg_color="transparent")
        self._body.grid(row=1, column=0, sticky="ew")
        self._body.grid_columnconfigure(0, weight=1)

        ctk.CTkButton(self, text=add_label, width=170, command=self.add_row).grid(
            row=2, column=0, sticky="w", pady=(6, 0)
        )

        for _ in range(max(0, initial_rows)):
            self.add_row()

    def add_row(self, values: dict | None = None) -> None:
        values = values or {}
        row_frame = ctk.CTkFrame(self._body, fg_color="transparent")
        row_frame.grid(row=len(self._rows), column=0, sticky="ew", pady=2)
        getters: dict = {}
        for ci, col in enumerate(self._columns):
            row_frame.grid_columnconfigure(ci, weight=1)
            default = values.get(col.key, col.default)
            if col.kind == "bool":
                var = ctk.BooleanVar(value=bool(default))
                ctk.CTkCheckBox(row_frame, text="", variable=var).grid(
                    row=0, column=ci, sticky="w", padx=(0, 8)
                )
                getters[col.key] = var.get
            else:
                entry = ctk.CTkEntry(row_frame)
                if default not in (None, ""):
                    entry.insert(0, str(default))
                entry.grid(row=0, column=ci, sticky="ew", padx=(0, 8))
                getters[col.key] = (lambda e=entry: e.get().strip())
        ctk.CTkButton(
            row_frame, text="✕", width=32,
            fg_color="#a0392e", hover_color="#7d2c24",
            command=lambda rf=row_frame: self.remove_row(rf),
        ).grid(row=0, column=len(self._columns), padx=(4, 0))
        self._rows.append((row_frame, getters))

    def remove_row(self, row_frame) -> None:
        for i, (rf, _g) in enumerate(self._rows):
            if rf is row_frame:
                rf.destroy()
                self._rows.pop(i)
                break

    def get(self) -> list[dict]:
        return [
            {key: getter() for key, getter in getters.items()}
            for _rf, getters in self._rows
        ]


def make_field_widget(master, fld: Field) -> tuple[ctk.CTkBaseClass, Callable[[], object]]:
    """Create the input widget for a Field and return (widget, getter)."""
    if fld.kind == "rows":
        initial = 2 if fld.arg == "--camera" else 1
        w = DynamicRowsField(master, columns=fld.columns, initial_rows=initial)
        return w, w.get

    if fld.kind == "file":
        w = FileField(master, file_filter=fld.file_filter, hint=fld.hint)
        return w, w.get

    if fld.kind == "bool":
        var = ctk.BooleanVar(value=bool(fld.default))
        w = ctk.CTkCheckBox(master, text="", variable=var)
        return w, var.get

    if fld.kind == "choice":
        var = ctk.StringVar(value=str(fld.default))
        w = ctk.CTkOptionMenu(master, values=list(fld.choices), variable=var)
        return w, var.get

    if fld.is_slider:
        w = SliderField(master, fld)
        return w, w.get

    # str / int / float -> a text entry
    w = ctk.CTkEntry(master, placeholder_text=fld.hint)
    if fld.default is not None:
        w.insert(0, str(fld.default))
    return w, lambda: w.get().strip()
