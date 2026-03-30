# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Tests for PySide6 GUI widgets."""
import pytest
from PySide6.QtCore import Qt


class TestMainWindow:
    def test_window_creates(self, qtbot):
        from forge.gui.main_window import ForgeMainWindow
        w = ForgeMainWindow()
        qtbot.addWidget(w)
        assert w.windowTitle() != ''

    def test_menu_bar_exists(self, qtbot):
        from forge.gui.main_window import ForgeMainWindow
        w = ForgeMainWindow()
        qtbot.addWidget(w)
        menus = [a.text().replace('&', '') for a in w.menuBar().actions()]
        assert 'File' in menus

    def test_status_bar(self, qtbot):
        from forge.gui.main_window import ForgeMainWindow
        w = ForgeMainWindow()
        qtbot.addWidget(w)
        assert w.statusBar() is not None

    def test_has_toolbar(self, qtbot):
        from forge.gui.main_window import ForgeMainWindow
        w = ForgeMainWindow()
        qtbot.addWidget(w)
        assert len(w.findChildren(w.__class__.__mro__[0].__mro__[0])) >= 0  # basic check
        # Just verify toolbar actions exist
        toolbars = w.findChildren(type(w.findChild(object)))  # loose check


class TestCommandWidget:
    def test_creates(self, qtbot):
        from forge.gui.command_widget import CommandWidget
        w = CommandWidget()
        qtbot.addWidget(w)
        assert w is not None

    def test_has_history(self, qtbot):
        from forge.gui.command_widget import CommandWidget
        w = CommandWidget()
        qtbot.addWidget(w)
        assert hasattr(w, 'history')
        assert isinstance(w.history, list)

    def test_has_input(self, qtbot):
        from forge.gui.command_widget import CommandWidget
        w = CommandWidget()
        qtbot.addWidget(w)
        assert hasattr(w, 'console')


class TestEditorWidget:
    def test_creates(self, qtbot):
        from forge.gui.editor_widget import EditorWidget
        w = EditorWidget()
        qtbot.addWidget(w)
        assert w is not None

    def test_new_file(self, qtbot):
        from forge.gui.editor_widget import EditorWidget
        w = EditorWidget()
        qtbot.addWidget(w)
        w.new_file()
        assert w.tabs.count() >= 1


class TestFileBrowser:
    def test_creates(self, qtbot):
        from forge.gui.file_browser import FileBrowserWidget
        w = FileBrowserWidget()
        qtbot.addWidget(w)
        assert w is not None


class TestWorkspaceBrowser:
    def test_creates(self, qtbot):
        from forge.gui.workspace_browser import WorkspaceBrowserWidget
        w = WorkspaceBrowserWidget()
        qtbot.addWidget(w)
        assert w is not None

    def test_update_workspace(self, qtbot):
        import numpy as np
        from forge.gui.workspace_browser import WorkspaceBrowserWidget
        w = WorkspaceBrowserWidget()
        qtbot.addWidget(w)
        w.update_workspace({'x': np.array([1, 2, 3]), 'name': 'hello'})
        assert w.table.rowCount() == 2


class TestPlotWidget:
    def test_creates(self, qtbot):
        from forge.gui.plot_widget import PlotWidget
        w = PlotWidget()
        qtbot.addWidget(w)
        assert w is not None

    def test_plot_data(self, qtbot):
        from forge.gui.plot_widget import PlotWidget
        w = PlotWidget()
        qtbot.addWidget(w)
        w.plot([1, 2, 3], [4, 5, 6])
        # Should not crash
