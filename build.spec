# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for AutoPodcast.

Builds two executables from one analysis:
  - autopodcast.exe      console build, used for the CLI and as the
                         child process the GUI launches.
  - autopodcast-gui.exe  windowed build (no console), the double-click
                         graphical interface.
"""

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

datas = collect_data_files('soundfile')
binaries = collect_dynamic_libs('soundfile')

for package_name in ('silero_vad', 'onnxruntime'):
    try:
        datas += collect_data_files(package_name)
    except Exception:
        pass
    try:
        binaries += collect_dynamic_libs(package_name)
    except Exception:
        pass

# customtkinter ships JSON theme files as package data — without them the
# GUI fails to start with "theme not found".
datas += collect_data_files('customtkinter')

a = Analysis(
    ['src/autopodcast/__main__.py'],
    pathex=['src/'],
    binaries=binaries,
    datas=datas,
    hiddenimports=['scipy.signal', 'scipy.linalg', 'silero_vad', 'onnxruntime', 'customtkinter'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'pytest', 'IPython'],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe_cli = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='autopodcast',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

exe_gui = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='autopodcast-gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)

coll = COLLECT(
    exe_cli,
    exe_gui,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='autopodcast',
)
