"""Smoke tests: verify basic app startup and framework."""
import pytest
from PySide6.QtCore import Qt


def test_app_launches(qtbot):
    """App creates MainWindow and shows it."""
    from forge.gui.main_window import ForgeMainWindow
    w = ForgeMainWindow()
    qtbot.addWidget(w)
    w.show()
    assert w.isVisible()


def test_menu_bar_exists(qtbot):
    """All expected menus are present."""
    from forge.gui.main_window import ForgeMainWindow
    w = ForgeMainWindow()
    qtbot.addWidget(w)
    menus = [a.text().replace('&', '') for a in w.menuBar().actions()]
    for name in ['File', 'Edit', 'View', 'Debug', 'Window', 'Help']:
        assert name in menus, f'Missing menu: {name}'


def test_status_bar(qtbot):
    """Status bar exists."""
    from forge.gui.main_window import ForgeMainWindow
    w = ForgeMainWindow()
    qtbot.addWidget(w)
    assert w.statusBar() is not None


def test_validation_framework():
    """Tolerance comparison works."""
    from forge.validation.framework import arrays_close
    import numpy as np
    ok, msg = arrays_close([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert ok, msg
    ok, msg = arrays_close([1.0], [1.0 + 1e-15])
    assert ok, msg
    ok, _ = arrays_close([1.0], [2.0])
    assert not ok


def test_forge_version():
    """Version is set."""
    import forge
    assert forge.__version__ == '0.1.0'
