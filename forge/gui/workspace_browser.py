"""Enhanced Forge workspace browser (forge/gui/workspace_browser.py).

Features:
- Filter/search bar for variable names
- Sortable columns
- Rich context menu (Delete, Plot, Copy Value, Copy Name, Help on Class)
- Type-based row coloring
- Compact size display
"""

import numpy as np

from PySide6.QtCore import Qt, Signal, QSortFilterProxyModel
from PySide6.QtGui import QAction, QColor, QFont, QIcon
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QMenu, QHeaderView, QLineEdit, QLabel, QPushButton, QApplication,
)


# Type colour hints (subtle background tint)
_TYPE_COLORS = {
    "double":         QColor(137, 180, 250, 25),   # blue tint
    "single":         QColor(137, 180, 250, 20),
    "char":           QColor(166, 227, 161, 25),   # green tint
    "logical":        QColor(249, 226, 175, 25),   # yellow tint
    "cell":           QColor(203, 166, 247, 25),   # mauve tint
    "struct":         QColor(243, 139, 168, 25),   # red tint
    "double complex": QColor(137, 220, 235, 25),   # sky tint
}


class WorkspaceBrowserWidget(QWidget):
    """Table showing all variables in the current workspace with filtering."""

    variable_inspect_requested = Signal(str)
    variable_delete_requested = Signal(str)
    variable_plot_requested = Signal(str)
    help_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._workspace = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # Filter bar
        filter_layout = QHBoxLayout()
        filter_layout.setSpacing(4)
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter variables...")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self._apply_filter)
        filter_layout.addWidget(self.filter_edit)

        self.lbl_count = QLabel("0 vars")
        self.lbl_count.setStyleSheet("color: #6c7086; font-size: 10px; padding: 0 4px;")
        filter_layout.addWidget(self.lbl_count)
        layout.addLayout(filter_layout)

        # Table
        self.table = QTableWidget(0, 4, self)
        self.table.setHorizontalHeaderLabels(["Name", "Size", "Class", "Value"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents
        )
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.doubleClicked.connect(self._on_double_click)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)
        layout.addWidget(self.table)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_workspace(self, workspace_dict: dict):
        """Repopulate from workspace_dict, preserving filter."""
        self._workspace = workspace_dict
        self._populate(workspace_dict)

    def _populate(self, workspace_dict: dict):
        filter_text = self.filter_edit.text().lower()
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)

        # Filter out internal/constant variables if desired
        hidden = {'ans'}  # optionally hide 'ans' from default view

        visible_count = 0
        for name, value in sorted(workspace_dict.items()):
            if filter_text and filter_text not in name.lower():
                continue

            row = self.table.rowCount()
            self.table.insertRow(row)

            name_item = QTableWidgetItem(name)
            name_item.setFont(QFont("Consolas", 10, QFont.Bold))
            self.table.setItem(row, 0, name_item)

            size_item = QTableWidgetItem(self._size_str(value))
            size_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 1, size_item)

            cls_str = self._class_str(value)
            cls_item = QTableWidgetItem(cls_str)
            cls_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 2, cls_item)

            val_item = QTableWidgetItem(self._preview(value))
            self.table.setItem(row, 3, val_item)

            # Subtle type-based row colouring
            bg = _TYPE_COLORS.get(cls_str)
            if bg:
                for col in range(4):
                    item = self.table.item(row, col)
                    if item:
                        item.setBackground(bg)

            visible_count += 1

        self.table.setSortingEnabled(True)
        self.lbl_count.setText(f"{visible_count} vars")

    def _apply_filter(self, _text=None):
        self._populate(self._workspace)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    _DTYPE_MAP = {
        "float64": "double", "float32": "single",
        "int8": "int8", "int16": "int16", "int32": "int32", "int64": "int64",
        "uint8": "uint8", "uint16": "uint16", "uint32": "uint32", "uint64": "uint64",
        "bool": "logical", "complex128": "double complex", "complex64": "single complex",
    }

    @staticmethod
    def _size_str(val) -> str:
        from forge.engine.types import ForgeArray
        from forge.engine.containers import ForgeChar, ForgeCell, ForgeStruct
        if isinstance(val, ForgeChar):
            return f"1\u00d7{len(val.to_str())}"
        if isinstance(val, ForgeArray):
            return "\u00d7".join(str(d) for d in val.data.shape)
        if isinstance(val, ForgeCell):
            return "\u00d7".join(str(d) for d in val.shape)
        if isinstance(val, np.ndarray):
            return "\u00d7".join(str(d) for d in val.shape)
        if isinstance(val, (list, tuple)):
            return str(len(val))
        return "1\u00d71"

    @classmethod
    def _class_str(cls, val) -> str:
        from forge.engine.types import ForgeArray
        from forge.engine.containers import ForgeChar, ForgeCell, ForgeStruct
        if isinstance(val, ForgeChar):
            return "char"
        if isinstance(val, ForgeArray):
            dtype_name = str(val.data.dtype)
            return cls._DTYPE_MAP.get(dtype_name, dtype_name)
        if isinstance(val, ForgeCell):
            return "cell"
        if isinstance(val, ForgeStruct):
            return "struct"
        if isinstance(val, np.ndarray):
            return cls._DTYPE_MAP.get(str(val.dtype), str(val.dtype))
        if isinstance(val, bool):
            return "logical"
        if isinstance(val, (int, float)):
            return "double"
        return type(val).__name__

    @staticmethod
    def _preview(val) -> str:
        from forge.engine.types import ForgeArray
        from forge.engine.containers import ForgeChar, ForgeCell, ForgeStruct
        if isinstance(val, ForgeChar):
            s = val.to_str()
            return repr(s) if len(s) <= 60 else repr(s[:57]) + "..."
        if isinstance(val, ForgeArray):
            d = val.data
            if d.size == 0:
                return "[]"
            if d.size == 1:
                v = d.flat[0]
                if isinstance(v, (bool, np.bool_)):
                    return "true" if v else "false"
                return str(v)
            if d.size <= 10:
                return "[" + " ".join(f"{x:g}" if isinstance(x, float) else str(x) for x in d.flat) + "]"
            return f"[{d.shape[0]}\u00d7{d.shape[1] if d.ndim > 1 else 1} {str(d.dtype)}]"
        if isinstance(val, ForgeCell):
            n = len(val._data)
            return f"{{{n} elements}}"
        if isinstance(val, ForgeStruct):
            fields = list(val._fields.keys()) if hasattr(val, '_fields') else []
            return "{" + ", ".join(fields[:5]) + "}"
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
        cls_item = self.table.item(index.row(), 2)
        cls_name = cls_item.text() if cls_item else ""

        menu = QMenu(self)

        # Inspect
        act_inspect = QAction(f"Open '{var_name}' in Variable Editor", self)
        act_inspect.triggered.connect(lambda: self.variable_inspect_requested.emit(var_name))
        menu.addAction(act_inspect)

        menu.addSeparator()

        # Copy name
        act_copy_name = QAction("Copy Name", self)
        act_copy_name.triggered.connect(lambda: QApplication.clipboard().setText(var_name))
        menu.addAction(act_copy_name)

        # Copy value preview
        val_item = self.table.item(index.row(), 3)
        if val_item:
            act_copy_val = QAction("Copy Value", self)
            act_copy_val.triggered.connect(
                lambda: QApplication.clipboard().setText(val_item.text())
            )
            menu.addAction(act_copy_val)

        menu.addSeparator()

        # Plot
        act_plot = QAction(f"Plot '{var_name}'", self)
        act_plot.triggered.connect(lambda: self.variable_plot_requested.emit(var_name))
        menu.addAction(act_plot)

        # Help on type
        if cls_name:
            act_help = QAction(f"Help on '{cls_name}'", self)
            act_help.triggered.connect(lambda: self.help_requested.emit(cls_name))
            menu.addAction(act_help)

        menu.addSeparator()

        # Delete
        act_delete = QAction(f"Delete '{var_name}'", self)
        act_delete.triggered.connect(lambda: self.variable_delete_requested.emit(var_name))
        menu.addAction(act_delete)

        # Delete all
        act_clear = QAction("Clear Workspace", self)
        act_clear.triggered.connect(self._clear_workspace_requested)
        menu.addAction(act_clear)

        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _clear_workspace_requested(self):
        """Delete all user variables."""
        for name in list(self._workspace.keys()):
            # Keep built-in constants
            if name not in ('pi', 'e', 'eps', 'inf', 'Inf', 'nan', 'NaN',
                           'i', 'j', 'true', 'false', 'realmin', 'realmax'):
                self.variable_delete_requested.emit(name)
