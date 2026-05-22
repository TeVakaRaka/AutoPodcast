"""Declarative form specification for the AutoPodcast GUI.

Each editing mode is described as a list of :class:`Field` objects. The GUI
builds its form from this data, and :func:`build_argv` turns the filled-in
values into a CLI argument list for the matching ``autopodcast`` command.

This module must NOT import tkinter — it is pure data + logic so it can be
unit-tested without a display.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# File-dialog categories. The GUI layer maps these to concrete filetypes.
FILE_PRPROJ = "prproj"
FILE_AUDIO = "audio"
FILE_XML = "xml"


@dataclass(frozen=True)
class Field:
    """One form field, bound to a CLI option."""

    arg: str                       # CLI flag, e.g. "--mic-a"
    label: str                     # Russian label shown in the form
    kind: str                      # file | int | float | str | bool | choice
    default: object = None         # default value (matches the @click.option)
    required: bool = False
    advanced: bool = False         # shown in the collapsible "Дополнительно"
    choices: tuple = ()            # for kind == "choice"
    file_filter: str = ""          # for kind == "file": FILE_* category
    flag_pair: tuple | None = None  # for kind == "bool": (on_flag, off_flag)
    hint: str = ""                 # optional placeholder / help text


@dataclass(frozen=True)
class ModeSpec:
    """One editing mode = one ``autopodcast`` subcommand."""

    cmd: str                       # subcommand name, e.g. "auto-multicam"
    key: str                       # short id, e.g. "multicam"
    title: str                     # Russian title for the mode selector
    out_suffix: str                # output name: {stem}{suffix}.prproj
    fields: list[Field] = field(default_factory=list)


def _equals_default(f: Field, text: str) -> bool:
    """True if the entered text equals the field's default value."""
    if f.default is None:
        return False
    if f.kind == "int":
        try:
            return int(text) == int(f.default)
        except ValueError:
            return False
    if f.kind == "float":
        try:
            return float(text) == float(f.default)
        except ValueError:
            return False
    return text == str(f.default)


def build_argv(spec: ModeSpec, values: dict) -> list[str]:
    """Turn filled-in form values into a CLI argument list.

    - required fields are always emitted (raises ValueError if empty);
    - optional fields are emitted only when non-empty and changed from default;
    - bool fields emit the flag opposite to their default when toggled;
    - ``--out`` is auto-derived from ``--in`` + ``spec.out_suffix``.
    """
    argv: list[str] = [spec.cmd]
    in_path: str | None = None

    for f in spec.fields:
        raw = values.get(f.arg)

        if f.kind == "bool":
            val = bool(raw)
            if val != f.default:
                if not f.flag_pair:
                    raise ValueError(f"bool field {f.arg} has no flag_pair")
                argv.append(f.flag_pair[0] if val else f.flag_pair[1])
            continue

        text = "" if raw is None else str(raw).strip()
        if f.arg == "--in":
            in_path = text

        if not text:
            if f.required:
                raise ValueError(f"Не заполнено обязательное поле: {f.label}")
            continue

        if f.required or not _equals_default(f, text):
            argv += [f.arg, text]

    if in_path:
        p = Path(in_path)
        argv += ["--out", str(p.with_name(p.stem + spec.out_suffix + p.suffix))]

    return argv


# ---------------------------------------------------------------------------
# Mode specifications — defaults mirror the @click.option defaults in cli.py.
# ---------------------------------------------------------------------------

_MULTICAM = ModeSpec(
    cmd="auto-multicam",
    key="multicam",
    title="1 ведущий + 1 гость",
    out_suffix="_multicam",
    fields=[
        Field("--in", "Файл проекта .prproj", "file", required=True, file_filter=FILE_PRPROJ),
        Field("--mic-a", "Микрофон ведущего", "file", required=True, file_filter=FILE_AUDIO),
        Field("--mic-b", "Микрофон гостя", "file", required=True, file_filter=FILE_AUDIO),
        Field("--seq", "Имя секвенции", "str", required=True),
        Field("--camera-host", "Камера ведущего (angle)", "int", default=1, advanced=True),
        Field("--camera-guest", "Камера гостя (angle)", "int", default=2, advanced=True),
        Field("--camera-wide", "Камера общего плана (angle)", "int", default=3, advanced=True),
        Field("--audio-track-host", "Аудиодорожка ведущего", "int", default=1, advanced=True),
        Field("--audio-track-guest", "Аудиодорожка гостя", "int", default=2, advanced=True),
        Field("--mute-audio", "Глушить неактивный микрофон", "bool", default=True,
              advanced=True, flag_pair=("--mute-audio", "--no-mute-audio")),
        Field("--cross-cancel", "Подавлять утечку микрофонов", "bool", default=True,
              advanced=True, flag_pair=("--cross-cancel", "--no-cross-cancel")),
        Field("--speech-threshold", "Порог речи, dB", "float", default=-24.0, advanced=True),
        Field("--input-gain", "Входное усиление, dB", "float", default=0.0, advanced=True),
    ],
)

_4CAMS = ModeSpec(
    cmd="auto-switch-4cams",
    key="4cams",
    title="1 ведущий + 3 гостя",
    out_suffix="_4cams",
    fields=[
        Field("--in", "Файл проекта .prproj", "file", required=True, file_filter=FILE_PRPROJ),
        Field("--mic-host", "Микрофон ведущего", "file", required=True, file_filter=FILE_AUDIO),
        Field("--mic-guest-1", "Микрофон гостя 1", "file", required=True, file_filter=FILE_AUDIO),
        Field("--mic-guest-2", "Микрофон гостя 2", "file", required=True, file_filter=FILE_AUDIO),
        Field("--mic-guest-3", "Микрофон гостя 3", "file", required=True, file_filter=FILE_AUDIO),
        Field("--seq", "Имя секвенции", "str", required=True),
        Field("--camera-all-wide", "Камера: общий план (angle)", "int", default=1, advanced=True),
        Field("--camera-guests-wide", "Камера: гости общим планом (angle)", "int", default=2, advanced=True),
        Field("--camera-guest-close", "Камера: крупный план гостя (angle)", "int", default=3, advanced=True),
        Field("--camera-host-close", "Камера: крупный план ведущего (angle)", "int", default=4, advanced=True),
        Field("--audio-track-host", "Аудиодорожка ведущего", "int", default=1, advanced=True),
        Field("--audio-track-guest-1", "Аудиодорожка гостя 1", "int", default=2, advanced=True),
        Field("--audio-track-guest-2", "Аудиодорожка гостя 2", "int", default=3, advanced=True),
        Field("--audio-track-guest-3", "Аудиодорожка гостя 3", "int", default=4, advanced=True),
        Field("--motion-check", "Учитывать движение камер", "bool", default=False,
              advanced=True, flag_pair=("--motion-check", "--no-motion-check")),
        Field("--motion-hwaccel", "Ускорение анализа движения", "choice", default="hybrid",
              advanced=True, choices=("cpu", "hybrid")),
        Field("--speech-threshold", "Порог речи, dB", "float", default=-27.0, advanced=True),
        Field("--input-gain", "Входное усиление, dB", "float", default=0.0, advanced=True),
    ],
)

_SAKHA = ModeSpec(
    cmd="auto-switch-sakha-aimakh",
    key="sakha",
    title="SAKHA AYMAKH",
    out_suffix="_sakha_aimakh",
    fields=[
        Field("--in", "Файл проекта .prproj", "file", required=True, file_filter=FILE_PRPROJ),
        Field("--seq", "Имя секвенции", "str", required=True),
        Field("--xml", "FCP7 XML (необязательно)", "file", file_filter=FILE_XML,
              hint="помогает найти источники аудио"),
        Field("--mic-main-host", "Микрофон главного ведущего (необязательно)", "file",
              file_filter=FILE_AUDIO, hint="пусто = автопоиск из проекта"),
        Field("--mic-cohost", "Микрофон со-ведущего (необязательно)", "file",
              file_filter=FILE_AUDIO, hint="пусто = автопоиск из проекта"),
        Field("--mic-guest", "Микрофон гостя (необязательно)", "file",
              file_filter=FILE_AUDIO, hint="пусто = автопоиск из проекта"),
        Field("--camera-main-host", "Камера главного ведущего (angle)", "int", default=1, advanced=True),
        Field("--camera-guest-close", "Камера крупного плана гостя (angle)", "int", default=2, advanced=True),
        Field("--camera-pair-wide", "Камера: со-ведущий + гость (angle)", "int", default=3, advanced=True),
        Field("--camera-all-wide", "Камера общего плана (angle)", "int", default=4, advanced=True),
        Field("--audio-track-main-host", "Аудиодорожка главного ведущего", "int", default=1, advanced=True),
        Field("--audio-track-cohost", "Аудиодорожка со-ведущего", "int", default=2, advanced=True),
        Field("--audio-track-guest", "Аудиодорожка гостя", "int", default=3, advanced=True),
        Field("--cut-intensity", "Интенсивность монтажа, 0-100%", "float", default=50.0, advanced=True),
        Field("--reaction-sensitivity", "Чувствительность к репликам, 0-100%", "float",
              default=None, advanced=True, hint="пусто = 50"),
        Field("--max-solo-hold", "Макс. удержание плана солиста, сек", "float", default=60.0, advanced=True),
        Field("--motion-check", "Учитывать движение камер", "bool", default=False,
              advanced=True, flag_pair=("--motion-check", "--no-motion-check")),
        Field("--motion-speed", "Скорость анализа движения", "choice", default="balanced",
              advanced=True, choices=("balanced", "turbo")),
        Field("--speech-threshold", "Порог речи, dB", "float", default=-27.0, advanced=True),
        Field("--input-gain", "Входное усиление, dB", "float", default=0.0, advanced=True),
    ],
)

_MONOLOGUE = ModeSpec(
    cmd="auto-switch-monologue",
    key="monologue",
    title="Монолог",
    out_suffix="_monologue",
    fields=[
        Field("--in", "Файл проекта .prproj", "file", required=True, file_filter=FILE_PRPROJ),
        Field("--mic", "Микрофон рассказчика", "file", required=True, file_filter=FILE_AUDIO),
        Field("--seq", "Имя секвенции", "str", required=True),
        Field("--camera-main", "Основная камера (angle)", "int", default=1, advanced=True),
        Field("--camera-accent", "Акцентная камера (angle)", "int", default=2, advanced=True),
        Field("--audio-track", "Аудиодорожка рассказчика", "int", default=1, advanced=True),
        Field("--switch-interval", "Интервал переключения, сек", "float", default=30.0, advanced=True),
        Field("--camera-main-share", "Доля основной камеры, %", "float", default=50.0, advanced=True),
        Field("--min-pause", "Мин. пауза для переключения, сек", "float", default=0.35, advanced=True),
        Field("--min-hold", "Мин. удержание камеры, сек", "float", default=12.0, advanced=True),
        Field("--motion-check", "Учитывать движение камер", "bool", default=True,
              advanced=True, flag_pair=("--motion-check", "--no-motion-check")),
        Field("--motion-hwaccel", "Ускорение анализа движения", "choice", default="hybrid",
              advanced=True, choices=("cpu", "hybrid")),
        Field("--speech-threshold", "Порог речи, dB", "float", default=-24.0, advanced=True),
        Field("--input-gain", "Входное усиление, dB", "float", default=0.0, advanced=True),
    ],
)

MODES: list[ModeSpec] = [_MULTICAM, _4CAMS, _SAKHA, _MONOLOGUE]


def mode_by_key(key: str) -> ModeSpec:
    """Look up a mode spec by its short key."""
    for spec in MODES:
        if spec.key == key:
            return spec
    raise KeyError(f"Unknown mode key: {key}")
