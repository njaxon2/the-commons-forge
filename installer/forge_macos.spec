# -*- mode: python ; coding: utf-8 -*-
# Forge macOS PyInstaller spec
# Builds a .app bundle using Apple Accelerate (no MKL needed)

import sys
import os

block_cipher = None

# Explicit module list — all forge.* submodules
forge_modules = [
    "forge", "forge.__main__", "forge.app", "forge.engine",
    "forge.engine.evaluator", "forge.engine.parser", "forge.engine.lexer",
    "forge.engine.session", "forge.engine.expr_fusion", "forge.engine.jit_compiler",
    "forge.gui", "forge.gui.main_window", "forge.gui.editor",
    "forge.gui.console", "forge.gui.workspace", "forge.gui.theme",
    "forge.gui.figure_window", "forge.gui.git_panel", "forge.gui.addons",
    "forge.gui.preferences", "forge.gui.update_worker",
    "forge.toolboxes", "forge.toolboxes.core", "forge.toolboxes.signal",
    "forge.toolboxes.image", "forge.toolboxes.statistics",
    "forge.toolboxes.control", "forge.toolboxes.optimization",
    "forge.toolboxes.io", "forge.toolboxes.linear_algebra",
    "forge.toolboxes.strings", "forge.toolboxes.geometry",
    "forge.toolboxes.symbolic", "forge.toolboxes.audio",
    "forge.toolboxes.financial", "forge.toolboxes.communications",
    "forge.toolboxes.fuzzy", "forge.toolboxes.mapping",
    "forge.toolboxes.neural", "forge.toolboxes.ode",
    "forge.toolboxes.parallel", "forge.toolboxes.polynomials",
    "forge.toolboxes.robotics", "forge.toolboxes.sparse",
    "forge.toolboxes.specfun", "forge.toolboxes.splines",
    "forge.toolboxes.struct", "forge.toolboxes.time_series",
    "forge.toolboxes.video", "forge.toolboxes.wavelet",
    "forge.toolboxes.sets",
]

# Dependency hidden imports
dep_modules = [
    "scipy", "scipy.linalg", "scipy.sparse", "scipy.signal",
    "scipy.optimize", "scipy.interpolate", "scipy.fft",
    "scipy.integrate", "scipy.stats", "scipy.io", "scipy.spatial",
    "numpy", "numpy.linalg", "numpy.fft", "numpy.random",
    "matplotlib", "matplotlib.pyplot", "matplotlib.backends",
    "matplotlib.backends.backend_qtagg",
    "PySide6", "PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets",
    "PySide6.QtSvg", "PySide6.QtSvgWidgets",
    "requests", "jwt",
    "unittest", "unittest.mock", "numpy.testing",
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
    excludes=["tkinter", "test"]  # Do NOT exclude unittest -- scipy needs it,
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
    upx=False,  # UPX unreliable on macOS
    console=False,  # GUI app
    target_arch=None,  # Build for native arch of runner
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="Forge",
)

app = BUNDLE(
    coll,
    name="Forge.app",
    icon=None,  # TODO: add forge.icns
    bundle_identifier="cc.thecommons.forge",
    info_plist={
        "LSMinimumSystemVersion": "12.0",
        "CFBundleShortVersionString": "0.3.6",
        "CFBundleName": "Forge IDE",
        "NSHighResolutionCapable": True,
        "NSRequiresAquaSystemAppearance": False,  # Support dark mode
    },
)
