"""Reusable customtkinter widgets for the AutoPodcast GUI."""

from __future__ import annotations

from tkinter import filedialog
from typing import Callable

import customtkinter as ctk

from autopodcast.gui.spec import FILE_AUDIO, FILE_PRPROJ, FILE_XML, Field

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


def make_field_widget(master, fld: Field) -> tuple[ctk.CTkBaseClass, Callable[[], object]]:
    """Create the input widget for a Field and return (widget, getter)."""
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
