# -*- mode: python ; coding: utf-8 -*-
# Forge Windows PyInstaller spec
# Builds a single-directory installer with all dependencies

import sys
import os

block_cipher = None

# Explicit module list
forge_modules = [
    "forge", "forge.__main__", "forge.app", "forge.engine",
    "forge.engine.evaluator", "forge.engine.parser", "forge.engine.lexer",
    "forge.engine.session", "forge.engine.expr_fusion", "forge.engine.jit_compiler",
    "forge.gui", "forge.gui.main_window", "forge.gui.editor_widget",
    "forge.gui.command_widget", "forge.gui.workspace_browser",
    "forge.gui.file_browser", "forge.gui.plot_widget", "forge.gui.themes",
    "forge.cli", "forge.license", "forge.update",
]

# Dependency hidden imports -- unittest is required by scipy via numpy.testing
dep_modules = [
    "unittest", "unittest.mock",
    "scipy", "scipy.linalg", "scipy.sparse", "scipy.signal",
    "scipy.optimize", "scipy.interpolate", "scipy.fft",
    "scipy.integrate", "scipy.stats", "scipy.io", "scipy.spatial",
    "scipy._lib", "scipy._lib.array_api_compat",
    "scipy._lib.array_api_compat.numpy",
    "numpy", "numpy.linalg", "numpy.fft", "numpy.random",
    "numpy.testing",
    "matplotlib", "matplotlib.pyplot", "matplotlib.backends",
    "matplotlib.backends.backend_qtagg",
    "PySide6", "PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets",
    "PySide6.QtSvg", "PySide6.QtSvgWidgets",
    "requests", "jwt",
]

a = Analysis(
    ["forge_launcher.py"],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=forge_modules + dep_modules,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "test"],  # Do NOT exclude unittest -- scipy needs it
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
    name="Forge",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI app, no console window
    icon=None,  # TODO: add forge.ico
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Forge",
)
