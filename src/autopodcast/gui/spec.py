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
    # When vmin/vmax are set on an int/float field, the GUI renders a slider
    # instead of a text entry. vstep is the slider granularity; unit is shown
    # next to the live value (e.g. "%", "с").
    vmin: float | None = None
    vmax: float | None = None
    vstep: float = 1.0
    unit: str = ""

    @property
    def is_slider(self) -> bool:
        return (
            self.kind in ("int", "float")
            and self.vmin is not None
            and self.vmax is not None
        )


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
        Field("--audio-track-host", "Аудиодорожка ведущего", "int", default=1,
              hint="Номер аудиодорожки в проекте, где микрофон ведущего.\n\nПрограмма не угадывает, кто на какой дорожке — укажите вручную. Неверный номер → откроется не тот микрофон."),
        Field("--audio-track-guest", "Аудиодорожка гостя", "int", default=2,
              hint="Номер аудиодорожки в проекте, где микрофон гостя.\n\nЕсли перепутать с дорожкой ведущего — программа откроет не тот микрофон."),
        Field("--camera-host", "Камера ведущего (angle)", "int", default=1, advanced=True),
        Field("--camera-guest", "Камера гостя (angle)", "int", default=2, advanced=True),
        Field("--camera-wide", "Камера общего плана (angle)", "int", default=3, advanced=True),
        Field("--dialogue-wide-interval", "Как часто общий план", "float", default=24.0,
              advanced=True, vmin=5.0, vmax=90.0, vstep=5.0, unit="с"),
        Field("--mute-audio", "Глушить неактивный микрофон", "bool", default=True,
              advanced=True, flag_pair=("--mute-audio", "--no-mute-audio")),
        Field("--cross-cancel", "Подавлять утечку микрофонов", "bool", default=True,
              advanced=True, flag_pair=("--cross-cancel", "--no-cross-cancel")),
        Field("--vad-threshold", "Строгость детектора речи (VAD)", "float", default=0.65,
              advanced=True, vmin=0.3, vmax=0.9, vstep=0.05,
              hint="выше — только чёткая речь; ниже — ловит тихие реплики"),
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
        Field("--audio-track-host", "Аудиодорожка ведущего", "int", default=1,
              hint="Номер аудиодорожки в проекте, где микрофон ведущего.\n\nПрограмма не угадывает, кто на какой дорожке — укажите вручную. Сверьте порядок дорожек A1/A2/A3/A4 в Premiere."),
        Field("--audio-track-guest-1", "Аудиодорожка гостя 1", "int", default=2,
              hint="Номер аудиодорожки с микрофоном первого гостя. Неверный номер → откроется не тот микрофон."),
        Field("--audio-track-guest-2", "Аудиодорожка гостя 2", "int", default=3,
              hint="Номер аудиодорожки с микрофоном второго гостя. Неверный номер → откроется не тот микрофон."),
        Field("--audio-track-guest-3", "Аудиодорожка гостя 3", "int", default=4,
              hint="Номер аудиодорожки с микрофоном третьего гостя. Неверный номер → откроется не тот микрофон."),
        Field("--camera-all-wide", "Камера: общий план (angle)", "int", default=1, advanced=True),
        Field("--camera-guests-wide", "Камера: гости общим планом (angle)", "int", default=2, advanced=True),
        Field("--camera-guest-close", "Камера: крупный план гостя (angle)", "int", default=3, advanced=True),
        Field("--camera-host-close", "Камера: крупный план ведущего (angle)", "int", default=4, advanced=True),
        Field("--shot-hold", "Мин. длина кадра", "float", default=1.4,
              advanced=True, vmin=0.5, vmax=5.0, vstep=0.1, unit="с"),
        Field("--cooldown", "Пауза между склейками", "float", default=0.9,
              advanced=True, vmin=0.2, vmax=3.0, vstep=0.1, unit="с"),
        Field("--motion-check", "Учитывать движение камер", "bool", default=False,
              advanced=True, flag_pair=("--motion-check", "--no-motion-check")),
        Field("--motion-hwaccel", "Ускорение анализа движения", "choice", default="hybrid",
              advanced=True, choices=("cpu", "hybrid")),
        Field("--vad-threshold", "Строгость детектора речи (VAD)", "float", default=0.65,
              advanced=True, vmin=0.3, vmax=0.9, vstep=0.05,
              hint="выше — только чёткая речь; ниже — ловит тихие реплики"),
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
        Field("--audio-track-main-host", "Аудиодорожка главного ведущего", "int", default=1,
              hint="Номер аудиодорожки в проекте, где микрофон главного ведущего.\n\nПрограмма не определяет, кто на какой дорожке — это задаёте вы. Неверный номер → откроется чужой микрофон и переключится не та камера. Сверьте порядок дорожек A1/A2/A3 в Premiere."),
        Field("--audio-track-cohost", "Аудиодорожка со-ведущего", "int", default=2,
              hint="Номер аудиодорожки с микрофоном со-ведущего.\n\nЕсли перепутать с дорожкой гостя — программа будет открывать не тот микрофон, когда говорит гость."),
        Field("--audio-track-guest", "Аудиодорожка гостя", "int", default=3,
              hint="Номер аудиодорожки с микрофоном гостя.\n\nЕсли перепутать с дорожкой со-ведущего — программа будет открывать не тот микрофон, когда говорит гость."),
        Field("--camera-main-host", "Камера главного ведущего (angle)", "int", default=1, advanced=True),
        Field("--camera-guest-close", "Камера крупного плана гостя (angle)", "int", default=2, advanced=True),
        Field("--camera-pair-wide", "Камера: со-ведущий + гость (angle)", "int", default=3, advanced=True),
        Field("--camera-all-wide", "Камера общего плана (angle)", "int", default=4, advanced=True),
        Field("--cut-intensity", "Темп монтажа", "float", default=50.0,
              advanced=True, vmin=0.0, vmax=100.0, vstep=5.0, unit="%",
              hint="Управляет ритмом монтажа.\n\n0% → длинные планы, редкие перебивки (спокойный документальный стиль)\n50% → баланс по умолчанию\n100% → короткие планы, частые склейки (динамичный разговорный формат)"),
        Field("--reaction-sensitivity", "Чувствительность к репликам", "float",
              default=None, advanced=True,
              hint="Насколько остро программа реагирует на смену говорящего.\n\n0% → стабильно удерживает кадр, не кидается при малейшей паузе\n50% → по умолчанию\n100% → переключается сразу при любой реплике или перебивке\n\nЕсли камера дёргается слишком часто — уменьши. Если запаздывает реагировать — увеличь."),
        Field("--max-solo-hold", "Макс. удержание плана солиста", "float", default=60.0,
              advanced=True, vmin=10.0, vmax=180.0, vstep=10.0, unit="с"),
        Field("--motion-check", "Учитывать движение камер", "bool", default=False,
              advanced=True, flag_pair=("--motion-check", "--no-motion-check")),
        Field("--motion-speed", "Скорость анализа движения", "choice", default="balanced",
              advanced=True, choices=("balanced", "turbo")),
        Field("--vad-threshold", "Строгость детектора речи (VAD)", "float", default=0.65,
              advanced=True, vmin=0.3, vmax=0.9, vstep=0.05,
              hint="Нейросеть оценивает каждый фрагмент аудио: 0–1, насколько там речь. Порог — минимальная уверенность для срабатывания.\n\n0.75–0.85 → только чёткая речь; шум, дыхание и утечка с соседних микрофонов игнорируются\n0.65 → по умолчанию, хорошо для нормальной записи\n0.4–0.55 → ловит тихую речь и шёпот, но больше ложных срабатываний"),
        Field("--audio-clean-mode", "Режим очистки утечек", "choice", default="strict",
              advanced=True, choices=("strict", "balanced", "legacy", "calibrated"),
              hint="Как программа решает, чей это голос, когда микрофоны слышат друг друга.\n\nstrict → анализ тайминга сигнала; точный, но иногда приписывает речь не тому каналу\nbalanced → то же, но мягче\nlegacy → старое простое сравнение громкости (дёргается)\ncalibrated → сравнение по громкости с калибровкой каждого микрофона под свой уровень + защита от дёрганья. Не путается из-за перекрученных микрофонов. Выбирай, если strict открывает не тот микрофон."),
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
        Field("--audio-track", "Аудиодорожка рассказчика", "int", default=1,
              hint="Номер аудиодорожки в проекте, где микрофон рассказчика."),
        Field("--camera-main", "Основная камера (angle)", "int", default=1, advanced=True),
        Field("--camera-accent", "Акцентная камера (angle)", "int", default=2, advanced=True),
        Field("--switch-interval", "Интервал переключения", "float", default=30.0,
              advanced=True, vmin=5.0, vmax=120.0, vstep=5.0, unit="с"),
        Field("--camera-main-share", "Доля основной камеры", "float", default=50.0,
              advanced=True, vmin=0.0, vmax=100.0, vstep=5.0, unit="%"),
        Field("--min-pause", "Мин. пауза для смены", "float", default=0.35,
              advanced=True, vmin=0.1, vmax=2.0, vstep=0.05, unit="с"),
        Field("--min-hold", "Мин. удержание камеры", "float", default=12.0,
              advanced=True, vmin=2.0, vmax=60.0, vstep=2.0, unit="с"),
        Field("--motion-check", "Учитывать движение камер", "bool", default=True,
              advanced=True, flag_pair=("--motion-check", "--no-motion-check")),
        Field("--motion-hwaccel", "Ускорение анализа движения", "choice", default="hybrid",
              advanced=True, choices=("cpu", "hybrid")),
        Field("--vad-threshold", "Строгость детектора речи (VAD)", "float", default=0.5,
              advanced=True, vmin=0.3, vmax=0.9, vstep=0.05,
              hint="Нейросеть оценивает каждый фрагмент аудио: 0–1, насколько там речь. Порог — минимальная уверенность для срабатывания.\n\n0.75–0.85 → только чёткая речь; шум, дыхание и утечка с соседних микрофонов игнорируются\n0.65 → по умолчанию, хорошо для нормальной записи\n0.4–0.55 → ловит тихую речь и шёпот, но больше ложных срабатываний"),
    ],
)

MODES: list[ModeSpec] = [_MULTICAM, _4CAMS, _SAKHA, _MONOLOGUE]


def mode_by_key(key: str) -> ModeSpec:
    """Look up a mode spec by its short key."""
    for spec in MODES:
        if spec.key == key:
            return spec
    raise KeyError(f"Unknown mode key: {key}")
