# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for AutoPodcast CLI."""

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

a = Analysis(
    ['src/autopodcast/__main__.py'],
    pathex=['src/'],
    binaries=binaries,
    datas=datas,
    hiddenimports=['scipy.signal', 'silero_vad', 'onnxruntime'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'pytest', 'IPython'],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
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

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='autopodcast',
)
