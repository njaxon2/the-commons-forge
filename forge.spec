# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Forge IDE."""

import sys
from pathlib import Path

block_cipher = None
project_root = Path(SPECPATH)

a = Analysis(
    [str(project_root / "forge" / "__main__.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        (str(project_root / "forge" / "gui"), "forge/gui"),
        (str(project_root / "forge" / "validation"), "forge/validation"),
    ],
    hiddenimports=[
        "PySide6.QtWidgets",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "matplotlib.backends.backend_qtagg",
        "scipy.signal",
        "scipy.sparse",
        "scipy.optimize",
        "scipy.integrate",
        "scipy.linalg",
        "scipy.interpolate",
        "scipy.special",
        "numpy",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="forge",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="forge",
)
