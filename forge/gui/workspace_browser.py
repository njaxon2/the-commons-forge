"""Forge workspace browser widget (forge/gui/workspace_browser.py)."""

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QMenu,
    QHeaderView,
)


class WorkspaceBrowserWidget(QWidget):
    """Table showing all variables in the current workspace."""

    variable_inspect_requested = Signal(str)
    variable_delete_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget(0, 4, self)
        self.table.setHorizontalHeaderLabels(["Name", "Size", "Class", "Value"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.doubleClicked.connect(self._on_double_click)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)
        layout.addWidget(self.table)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_workspace(self, workspace_dict: dict):
        """Clear and repopulate from *workspace_dict* {name: value}."""
        self.table.setRowCount(0)
        for name, value in sorted(workspace_dict.items()):
            row = self.table.rowCount()
            self.table.insertRow(row)

            self.table.setItem(row, 0, QTableWidgetItem(name))
            self.table.setItem(row, 1, QTableWidgetItem(self._size_str(value)))
            self.table.setItem(row, 2, QTableWidgetItem(self._class_str(value)))
            self.table.setItem(row, 3, QTableWidgetItem(self._preview(value)))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _size_str(val) -> str:
        if isinstance(val, np.ndarray):
            return "x".join(str(d) for d in val.shape)
        if isinstance(val, (list, tuple)):
            return str(len(val))
        return "1x1"

    @staticmethod
    def _class_str(val) -> str:
        if isinstance(val, np.ndarray):
            return str(val.dtype)
        return type(val).__name__

    @staticmethod
    def _preview(val) -> str:
        s = str(val)
        return s if len(s) <= 60 else s[:57] + "..."

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def _on_double_click(self, index):
        name_item = self.table.item(index.row(), 0)
        if name_item:
            self.variable_inspect_requested.emit(name_item.text())

    def _context_menu(self, pos):
        index = self.table.indexAt(pos)
        if not index.isValid():
            return
        name_item = self.table.item(index.row(), 0)
        if name_item is None:
            return
        var_name = name_item.text()

        menu = QMenu(self)

        act_delete = QAction("Delete Variable", self)
        act_delete.triggered.connect(lambda: self._delete_variable(var_name))
        menu.addAction(act_delete)

        act_plot = QAction("Plot Variable", self)
        act_plot.triggered.connect(lambda: self._plot_variable(var_name))
        menu.addAction(act_plot)

        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _delete_variable(self, name: str):
        """Emit signal to request variable deletion from the session."""
        self.variable_delete_requested.emit(name)

    def _plot_variable(self, name: str):
        """Placeholder -- emit or callback to plot widget."""
        pass
