"""Allow running as: python3 -m autopodcast"""

import sys

# On Windows the console defaults to a legacy code page (cp1251/cp1252) that
# cannot encode the Cyrillic interactive-wizard text — that would crash even
# `autopodcast.exe --help`. Switch the console and Python's own streams to
# UTF-8 so console I/O never raises UnicodeEncodeError.
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleCP(65001)
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    except Exception:
        pass
    for _stream in (sys.stdout, sys.stderr, sys.stdin):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

from autopodcast.cli import cli


def _pause_before_exit(message: str) -> None:
    """Wait for Enter so a double-clicked .exe window stays open long enough
    to read the output. No-op when stdin is unavailable (CI, pipes)."""
    try:
        input(message)
    except (EOFError, KeyboardInterrupt):
        pass


def _run_wizard_command(args: list[str], out_file: str, base: str) -> int:
    """Run a Click command from the interactive wizard with shared error handling."""
    import os

    print("Запуск анализа и патчинга...")
    print()
    try:
        cli(args)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n*** ОШИБКА: {e} ***")
        text = str(e)
        if "--camera-main-file" in text and "--camera-accent-file" in text:
            print()
            print("Не удалось автоматически найти исходные видео камер.")
            print("Запусти режим ещё раз из терминала и передай:")
            print("  --camera-main-file <путь к основной камере>")
            print("  --camera-accent-file <путь к акцентной камере>")
            print("При наличии XML из Premiere можно ещё добавить --xml <файл.xml>.")
        error_log = base + "_error.log"
        try:
            with open(error_log, "w", encoding="utf-8") as f:
                traceback.print_exc(file=f)
            print(f"Лог ошибки: {error_log}")
        except Exception:
            pass
        return 1

    print()
    print("=" * 50)
    print("  ГОТОВО")
    print()
    print(f"  Результат: {out_file}")
    print()
    print("  Открой в Premiere Pro:")
    print(f"    File > Open Project > {os.path.basename(out_file)}")
    print("=" * 50)
    return 0


def _interactive_wizard_2speakers() -> int:
    """Interactive wizard for classic 2-speaker auto-multicam mode."""
    import os

    print("=" * 50)
    print("  AutoPodcast — версия 1")
    print("  .prproj + 2 микрофона = готовый проект")
    print("=" * 50)
    print()

    # --- .prproj file ---
    prproj = input("Перетащи .prproj файл из Premiere:\n.prproj файл: ").strip().strip('"')
    if not os.path.isfile(prproj):
        print(f"  Файл не найден: {prproj}")
        return 1
    print()

    # --- Mic A (host) ---
    mic_a = input("Перетащи WAV-файл микрофона ВЕДУЩЕГО (host):\nМикрофон ведущего: ").strip().strip('"')
    if not os.path.isfile(mic_a):
        print(f"  Файл не найден: {mic_a}")
        return 1
    print()

    # --- Mic B (guest) ---
    mic_b = input("Перетащи WAV-файл микрофона ГОСТЯ (guest):\nМикрофон гостя: ").strip().strip('"')
    if not os.path.isfile(mic_b):
        print(f"  Файл не найден: {mic_b}")
        return 1
    print()

    # --- Sequence name ---
    seq_name = input("Имя секвенции в проекте: ").strip()
    if not seq_name:
        print("  Имя секвенции не может быть пустым")
        return 1
    print()

    # --- Optional params ---
    print("Дополнительные настройки (Enter = по умолчанию):")
    print()

    extra = []

    cam_host = input("Angle ведущего (default 1): ").strip()
    if cam_host:
        extra += ["--camera-host", cam_host]

    cam_guest = input("Angle гостя (default 2): ").strip()
    if cam_guest:
        extra += ["--camera-guest", cam_guest]

    cam_wide = input("Angle широкого плана (default 3): ").strip()
    if cam_wide:
        extra += ["--camera-wide", cam_wide]

    audio_host = input("Аудиодорожка ведущего (default 1): ").strip()
    if audio_host:
        extra += ["--audio-track-host", audio_host]

    audio_guest = input("Аудиодорожка гостя (default 2): ").strip()
    if audio_guest:
        extra += ["--audio-track-guest", audio_guest]

    mute = input("Мьютить неактивные микрофоны? (Y/n, default Y): ").strip()
    if mute.lower() == "n":
        extra += ["--no-mute-audio"]

    threshold = input("Порог речи в dB (default -24.0): ").strip()
    if threshold:
        extra += ["--speech-threshold", threshold]

    gain = input("Входное усиление в dB (default 0.0): ").strip()
    if gain:
        extra += ["--input-gain", gain]

    print()

    # --- Output path ---
    base, ext = os.path.splitext(prproj)
    out_file = base + "_multicam" + ext

    args = [
        "auto-multicam",
        "--in", prproj,
        "--mic-a", mic_a,
        "--mic-b", mic_b,
        "--seq", seq_name,
        "--out", out_file,
    ] + extra

    return _run_wizard_command(args, out_file, base)


def _interactive_wizard_4cams() -> int:
    """Interactive wizard for 1 host + 3 guests + 4 cameras mode."""
    import os

    print("=" * 50)
    print("  AutoPodcast — версия 2")
    print("  1 ведущий + 3 гостя + 4 камеры")
    print("=" * 50)
    print()

    prproj = input("Перетащи .prproj файл из Premiere:\n.prproj файл: ").strip().strip('"')
    if not os.path.isfile(prproj):
        print(f"  Файл не найден: {prproj}")
        return 1
    print()

    mic_host = input("Перетащи WAV-файл микрофона ВЕДУЩЕГО:\nМикрофон ведущего: ").strip().strip('"')
    if not os.path.isfile(mic_host):
        print(f"  Файл не найден: {mic_host}")
        return 1
    print()

    mic_g1 = input("Перетащи WAV-файл микрофона ГОСТЯ 1:\nМикрофон гостя 1: ").strip().strip('"')
    if not os.path.isfile(mic_g1):
        print(f"  Файл не найден: {mic_g1}")
        return 1
    print()

    mic_g2 = input("Перетащи WAV-файл микрофона ГОСТЯ 2:\nМикрофон гостя 2: ").strip().strip('"')
    if not os.path.isfile(mic_g2):
        print(f"  Файл не найден: {mic_g2}")
        return 1
    print()

    mic_g3 = input("Перетащи WAV-файл микрофона ГОСТЯ 3:\nМикрофон гостя 3: ").strip().strip('"')
    if not os.path.isfile(mic_g3):
        print(f"  Файл не найден: {mic_g3}")
        return 1
    print()

    seq_name = input("Имя секвенции в проекте: ").strip()
    if not seq_name:
        print("  Имя секвенции не может быть пустым")
        return 1
    print()

    print("Дополнительные настройки (Enter = по умолчанию):")
    print()

    extra = []

    cam_all = input("Angle общего плана всей студии (default 1): ").strip()
    if cam_all:
        extra += ["--camera-all-wide", cam_all]

    cam_guests = input("Angle общего плана гостей (default 2): ").strip()
    if cam_guests:
        extra += ["--camera-guests-wide", cam_guests]

    cam_guest_close = input("Angle крупного плана гостей / движущейся камеры (default 3): ").strip()
    if cam_guest_close:
        extra += ["--camera-guest-close", cam_guest_close]

    cam_host = input("Angle крупного плана ведущего (default 4): ").strip()
    if cam_host:
        extra += ["--camera-host-close", cam_host]

    track_host = input("Аудиодорожка ведущего (default 1): ").strip()
    if track_host:
        extra += ["--audio-track-host", track_host]

    track_g1 = input("Аудиодорожка гостя 1 (default 2): ").strip()
    if track_g1:
        extra += ["--audio-track-guest-1", track_g1]

    track_g2 = input("Аудиодорожка гостя 2 (default 3): ").strip()
    if track_g2:
        extra += ["--audio-track-guest-2", track_g2]

    track_g3 = input("Аудиодорожка гостя 3 (default 4): ").strip()
    if track_g3:
        extra += ["--audio-track-guest-3", track_g3]

    mute = input("Мьютить неактивные микрофоны? (Y/n, default Y): ").strip()
    if mute.lower() == "n":
        extra += ["--no-mute-audio"]

    motion_check = input(
        "Проверять движение камер? Это медленнее, но не даёт входить в движущиеся планы (y/N, default N): "
    ).strip()
    if motion_check.lower() == "y":
        extra += ["--motion-check"]
        motion_hwaccel = input(
            "Ускорение проверки движения (cpu/hybrid, default hybrid): "
        ).strip()
        if motion_hwaccel:
            extra += ["--motion-hwaccel", motion_hwaccel]

    threshold = input("Порог речи в dB (default -27.0): ").strip()
    if threshold:
        extra += ["--speech-threshold", threshold]

    gain = input("Входное усиление в dB (default 0.0): ").strip()
    if gain:
        extra += ["--input-gain", gain]

    print()

    base, ext = os.path.splitext(prproj)
    out_file = base + "_4cams" + ext

    args = [
        "auto-switch-4cams",
        "--in", prproj,
        "--mic-host", mic_host,
        "--mic-guest-1", mic_g1,
        "--mic-guest-2", mic_g2,
        "--mic-guest-3", mic_g3,
        "--seq", seq_name,
        "--out", out_file,
    ] + extra

    return _run_wizard_command(args, out_file, base)


def _interactive_wizard_sakha_aimakh() -> int:
    """Interactive wizard for SAKHA AYMAKH: 2 hosts + 1 guest + 4 cameras."""
    import os

    print("=" * 50)
    print("  AutoPodcast — САХА АЙМАХ")
    print("  2 ведущих + 1 гость + 4 камеры")
    print("=" * 50)
    print()

    prproj = input("Перетащи .prproj файл из Premiere:\n.prproj файл: ").strip().strip('"')
    if not os.path.isfile(prproj):
        print(f"  Файл не найден: {prproj}")
        return 1
    print()

    seq_name = input("Имя секвенции в проекте: ").strip()
    if not seq_name:
        print("  Имя секвенции не может быть пустым")
        return 1
    print()

    xml_file = input("FCP7 XML из Premiere для поиска аудио (Enter = не указывать): ").strip().strip('"')
    if xml_file and not os.path.isfile(xml_file):
        print(f"  Файл не найден: {xml_file}")
        return 1
    print()

    try:
        from pathlib import Path
        from autopodcast.core.audio_sources import discover_sequence_audio_sources

        sources = discover_sequence_audio_sources(
            Path(prproj),
            seq_name,
            xml_path=xml_file or None,
        )
        if sources.sources_by_track:
            print(f"Найденные аудиодорожки ({sources.method}):")
            for track_idx, source in sorted(sources.sources_by_track.items()):
                name = f" — {source.name}" if source.name else ""
                print(f"  [{track_idx + 1}]{name}: {source.representative_path}")
            print()
        else:
            print("  В проекте не удалось найти аудиофайлы дорожек.")
            print("  Команда всё равно попробует подтянуть их при запуске.")
            print()
    except Exception as exc:
        print(f"  Не удалось заранее прочитать аудиодорожки: {exc}")
        print("  Команда попробует подтянуть их при запуске.")
        print()

    print("Дефолт камер: 1 основной ведущий, 2 гость, 3 гость+второй ведущий, 4 общак.")
    print("Если в проекте другой порядок, укажи нужные angle:")
    cam_main = input("Angle крупного плана основного ведущего (default 1): ").strip()
    cam_guest = input("Angle крупного плана гостя (default 2): ").strip()
    cam_pair = input("Angle общего кадра второго ведущего + гостя (default 3): ").strip()
    cam_all = input("Angle полного общака (default 4): ").strip()
    print()

    print("Дополнительные настройки (Enter = по умолчанию):")
    print()

    extra = []
    if cam_main:
        extra += ["--camera-main-host", cam_main]
    if cam_guest:
        extra += ["--camera-guest-close", cam_guest]
    if cam_pair:
        extra += ["--camera-pair-wide", cam_pair]
    if cam_all:
        extra += ["--camera-all-wide", cam_all]

    track_main = input("Аудиодорожка основного ведущего (default 1): ").strip()
    if track_main:
        extra += ["--audio-track-main-host", track_main]

    track_cohost = input("Аудиодорожка второго ведущего (default 2): ").strip()
    if track_cohost:
        extra += ["--audio-track-cohost", track_cohost]

    track_guest = input("Аудиодорожка гостя (default 3): ").strip()
    if track_guest:
        extra += ["--audio-track-guest", track_guest]

    cut_intensity = input("Интенсивность монтажа / частота катов 0-100% (default 50): ").strip()
    if cut_intensity:
        extra += ["--cut-intensity", cut_intensity]

    reaction = input("Чувствительность к коротким репликам 0-100% (default 50, ниже = меньше реакции на утечки): ").strip()
    if reaction:
        extra += ["--reaction-sensitivity", reaction]

    max_solo_hold = input("Максимум на одном говорящем в секундах (default 60): ").strip()
    if max_solo_hold:
        extra += ["--max-solo-hold", max_solo_hold]

    mute = input("Мьютить молчащие микрофоны? (Y/n, default Y): ").strip()
    if mute.lower() == "n":
        extra += ["--no-mute-audio"]

    motion_check = input(
        "Проверять движение камер? Это медленнее, но не даёт входить в движущиеся планы (y/N, default N): "
    ).strip()
    if motion_check.lower() == "y":
        extra += ["--motion-check"]
        motion_speed = input("Скорость motion-check (balanced/turbo, default balanced): ").strip()
        if motion_speed:
            extra += ["--motion-speed", motion_speed]
        motion_hwaccel = input("Ускорение motion-check (cpu/hybrid, default hybrid): ").strip()
        if motion_hwaccel:
            extra += ["--motion-hwaccel", motion_hwaccel]
        motion_cache = input("Использовать cache motion-check? (Y/n, default Y): ").strip()
        if motion_cache.lower() == "n":
            extra += ["--no-motion-cache"]

    threshold = input("Порог речи в dB (default -27.0): ").strip()
    if threshold:
        extra += ["--speech-threshold", threshold]

    gain = input("Входное усиление в dB (default 0.0): ").strip()
    if gain:
        extra += ["--input-gain", gain]

    print()

    base, ext = os.path.splitext(prproj)
    out_file = base + "_sakha_aimakh" + ext

    args = [
        "auto-switch-sakha-aimakh",
        "--in", prproj,
        "--seq", seq_name,
        "--out", out_file,
    ] + extra
    if xml_file:
        args += ["--xml", xml_file]

    return _run_wizard_command(args, out_file, base)


def _interactive_wizard_monologue() -> int:
    """Interactive wizard for 1 narrator + 2 cameras mode."""
    import os

    print("=" * 50)
    print("  AutoPodcast — ОЛОНХО")
    print("  1 рассказчик + 2 камеры")
    print("=" * 50)
    print()

    prproj = input("Перетащи .prproj файл из Premiere:\n.prproj файл: ").strip().strip('"')
    if not os.path.isfile(prproj):
        print(f"  Файл не найден: {prproj}")
        return 1
    print()

    mic = input("Перетащи WAV-файл микрофона рассказчика:\nМикрофон рассказчика: ").strip().strip('"')
    if not os.path.isfile(mic):
        print(f"  Файл не найден: {mic}")
        return 1
    print()

    seq_name = input("Имя секвенции в проекте: ").strip()
    if not seq_name:
        print("  Имя секвенции не может быть пустым")
        return 1
    print()

    print("Дополнительные настройки (Enter = по умолчанию):")
    print()

    extra = []

    cam_main = input("Angle основной камеры (default 1): ").strip()
    if cam_main:
        extra += ["--camera-main", cam_main]

    cam_accent = input("Angle акцентной камеры (default 2): ").strip()
    if cam_accent:
        extra += ["--camera-accent", cam_accent]

    audio_track = input("Аудиодорожка рассказчика (default 1): ").strip()
    if audio_track:
        extra += ["--audio-track", audio_track]

    switch_interval = input("Целевой интервал переключения в секундах (default 30): ").strip()
    if switch_interval:
        extra += ["--switch-interval", switch_interval]

    camera_main_share = input(
        "Доля основной камеры в процентах (default 50, например 80 = крупный 80 / общак 20): "
    ).strip()
    if camera_main_share:
        extra += ["--camera-main-share", camera_main_share]

    min_pause = input("Минимальная пауза для переключения в секундах (default 0.35): ").strip()
    if min_pause:
        extra += ["--min-pause", min_pause]

    min_hold = input("Минимальное удержание камеры в секундах (default 12): ").strip()
    if min_hold:
        extra += ["--min-hold", min_hold]

    motion_check = input(
        "Проверять движение камер? Это медленнее, но избегает движущихся планов (Y/n, default Y): "
    ).strip()
    if motion_check.lower() == "n":
        extra += ["--no-motion-check"]
    else:
        motion_hwaccel = input(
            "Ускорение проверки движения (cpu/hybrid, default hybrid): "
        ).strip()
        if motion_hwaccel:
            extra += ["--motion-hwaccel", motion_hwaccel]

    threshold = input("Порог речи в dB (default -24.0): ").strip()
    if threshold:
        extra += ["--speech-threshold", threshold]

    gain = input("Входное усиление в dB (default 0.0): ").strip()
    if gain:
        extra += ["--input-gain", gain]

    print()

    base, ext = os.path.splitext(prproj)
    out_file = base + "_monologue" + ext

    args = [
        "auto-switch-monologue",
        "--in", prproj,
        "--mic", mic,
        "--seq", seq_name,
        "--out", out_file,
    ] + extra

    return _run_wizard_command(args, out_file, base)


def _interactive_wizard():
    """Interactive wizard for double-click exe launch (no arguments)."""
    print("=" * 50)
    print("  AutoPodcast")
    print("  Выбор версии / режима запуска")
    print("=" * 50)
    print()
    print("1. Классическая версия: 2 микрофона, host/guest")
    print("2. Новая версия: 1 ведущий + 3 гостя + 4 камеры")
    print("3. ОЛОНХО: 1 рассказчик + 2 камеры")
    print("4. САХА АЙМАХ: 2 ведущих + 1 гость + 4 камеры")
    print()

    choice = input("Выбери версию (1/2/3/4, default 1): ").strip().lower()
    if choice in {"", "1", "classic", "v1"}:
        return _interactive_wizard_2speakers()
    if choice in {"2", "4cams", "v2", "roundtable"}:
        return _interactive_wizard_4cams()
    if choice in {"3", "monologue", "narrator", "olonkho", "solo"}:
        return _interactive_wizard_monologue()
    if choice in {"4", "sakha", "saha", "aimakh", "sakha-aimakh"}:
        return _interactive_wizard_sakha_aimakh()

    print(f"  Неизвестный выбор: {choice}")
    return 1


if getattr(sys, 'frozen', False) and len(sys.argv) <= 1:
    # Double-click on the exe -> graphical interface.
    try:
        from autopodcast.gui.main import main as _gui_main
        _gui_main()
    except Exception:
        # No display / broken customtkinter -> fall back to the text wizard.
        import traceback
        traceback.print_exc()
        print("\nГрафический интерфейс не запустился, текстовый режим...\n")
        try:
            _interactive_wizard()
        except SystemExit:
            pass
        except Exception as e:
            traceback.print_exc()
            print(f"\n*** ОШИБКА: {e} ***")
        _pause_before_exit("\nНажмите Enter для выхода...")
else:
    try:
        cli()
    except SystemExit:
        if getattr(sys, 'frozen', False):
            _pause_before_exit("\nPress Enter to exit / Нажмите Enter для выхода...")
        raise
