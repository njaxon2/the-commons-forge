# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Forge variable editor widget — view/edit workspace variables."""

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QHeaderView, QDialogButtonBox, QTextEdit,
    QTabWidget, QWidget,
)


class VariableEditorDialog(QDialog):
    """Dialog for inspecting and editing a workspace variable."""

    value_changed = Signal(str, object)  # (var_name, new_value)

    def __init__(self, var_name: str, value, parent=None, readonly=False):
        super().__init__(parent)
        self.var_name = var_name
        self.value = value
        self.readonly = readonly
        self.setWindowTitle(f"Variable Editor — {var_name}")
        self.setMinimumSize(600, 400)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        from forge.engine.types import ForgeArray
        from forge.engine.containers import ForgeChar, ForgeCell, ForgeStruct

        # Header
        info = self._get_info()
        header = QLabel(info)
        header.setStyleSheet("font-weight: bold; padding: 4px;")
        layout.addWidget(header)

        if isinstance(self.value, ForgeArray):
            self._build_array_view(layout)
        elif isinstance(self.value, ForgeChar):
            self._build_char_view(layout)
        elif isinstance(self.value, ForgeCell):
            self._build_cell_view(layout)
        elif isinstance(self.value, ForgeStruct):
            self._build_struct_view(layout)
        else:
            self._build_text_view(layout)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

    def _get_info(self):
        from forge.engine.types import ForgeArray
        from forge.engine.containers import ForgeChar, ForgeCell, ForgeStruct
        if isinstance(self.value, ForgeArray):
            d = self.value.data
            shape = "x".join(str(s) for s in d.shape)
            return f"{self.var_name}: {shape} {d.dtype}"
        elif isinstance(self.value, ForgeChar):
            return f"{self.var_name}: 1x{len(self.value.to_str())} char"
        elif isinstance(self.value, ForgeCell):
            shape = "x".join(str(s) for s in self.value.shape)
            return f"{self.var_name}: {shape} cell"
        elif isinstance(self.value, ForgeStruct):
            nf = len(self.value._fields) if hasattr(self.value, '_fields') else 0
            return f"{self.var_name}: 1x1 struct with {nf} fields"
        return f"{self.var_name}: {type(self.value).__name__}"

    def _build_array_view(self, layout):
        data = self.value.data
        if data.ndim == 1:
            data = data.reshape(1, -1)
        rows, cols = data.shape[:2]
        # Limit display for very large arrays
        max_rows = min(rows, 500)
        max_cols = min(cols, 100)

        table = QTableWidget(max_rows, max_cols)
        for r in range(max_rows):
            for c in range(max_cols):
                v = data[r, c]
                item = QTableWidgetItem(self._fmt(v))
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if self.readonly:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                table.setItem(r, c, item)

        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        if not self.readonly:
            table.cellChanged.connect(lambda r, c: self._on_cell_changed(r, c, table))
        layout.addWidget(table)

        if rows > max_rows or cols > max_cols:
            layout.addWidget(QLabel(f"Showing {max_rows}x{max_cols} of {rows}x{cols}"))

    def _build_char_view(self, layout):
        text = QTextEdit()
        text.setPlainText(self.value.to_str())
        text.setReadOnly(self.readonly)
        layout.addWidget(text)

    def _build_cell_view(self, layout):
        from forge.engine.types import ForgeArray
        from forge.engine.containers import ForgeChar, ForgeCell, ForgeStruct

        items = self.value._data
        table = QTableWidget(len(items), 3)
        table.setHorizontalHeaderLabels(["Index", "Class", "Value"])
        table.horizontalHeader().setStretchLastSection(True)
        table.setEditTriggers(QTableWidget.NoEditTriggers)

        for i, item in enumerate(items):
            table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            if isinstance(item, ForgeChar):
                table.setItem(i, 1, QTableWidgetItem("char"))
                table.setItem(i, 2, QTableWidgetItem(f"'{item.to_str()}'"))
            elif isinstance(item, ForgeArray):
                d = item.data
                shape = "x".join(str(s) for s in d.shape)
                table.setItem(i, 1, QTableWidgetItem(f"{shape} double"))
                if d.size <= 6:
                    table.setItem(i, 2, QTableWidgetItem(
                        "[" + " ".join(self._fmt(x) for x in d.flat) + "]"))
                else:
                    table.setItem(i, 2, QTableWidgetItem(f"[{shape}]"))
            elif isinstance(item, ForgeCell):
                table.setItem(i, 1, QTableWidgetItem("cell"))
                table.setItem(i, 2, QTableWidgetItem(f"{{{len(item._data)} elements}}"))
            elif isinstance(item, ForgeStruct):
                table.setItem(i, 1, QTableWidgetItem("struct"))
                fields = list(item._fields.keys()) if hasattr(item, '_fields') else []
                table.setItem(i, 2, QTableWidgetItem(
                    "{" + ", ".join(fields[:5]) + "}"))
            else:
                table.setItem(i, 1, QTableWidgetItem(type(item).__name__))
                table.setItem(i, 2, QTableWidgetItem(str(item)[:60]))

        layout.addWidget(table)

    def _build_struct_view(self, layout):
        from forge.engine.types import ForgeArray
        from forge.engine.containers import ForgeChar

        fields = self.value._fields if hasattr(self.value, '_fields') else {}
        table = QTableWidget(len(fields), 3)
        table.setHorizontalHeaderLabels(["Field", "Class", "Value"])
        table.horizontalHeader().setStretchLastSection(True)
        table.setEditTriggers(QTableWidget.NoEditTriggers)

        for i, (name, val) in enumerate(fields.items()):
            table.setItem(i, 0, QTableWidgetItem(name))
            if isinstance(val, ForgeChar):
                table.setItem(i, 1, QTableWidgetItem("char"))
                table.setItem(i, 2, QTableWidgetItem(f"'{val.to_str()}'"))
            elif isinstance(val, ForgeArray):
                d = val.data
                shape = "x".join(str(s) for s in d.shape)
                table.setItem(i, 1, QTableWidgetItem(f"{shape} double"))
                if d.size <= 6:
                    table.setItem(i, 2, QTableWidgetItem(
                        "[" + " ".join(self._fmt(x) for x in d.flat) + "]"))
                else:
                    table.setItem(i, 2, QTableWidgetItem(f"[{shape}]"))
            else:
                table.setItem(i, 1, QTableWidgetItem(type(val).__name__))
                table.setItem(i, 2, QTableWidgetItem(str(val)[:60]))

        layout.addWidget(table)

    def _build_text_view(self, layout):
        text = QTextEdit()
        text.setPlainText(str(self.value))
        text.setReadOnly(True)
        layout.addWidget(text)

    @staticmethod
    def _fmt(v):
        if isinstance(v, (bool, np.bool_)):
            return '1' if v else '0'
        if isinstance(v, (complex, np.complexfloating)):
            return f"{v.real:.4g} + {v.imag:.4g}i"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if np.isnan(v):
            return 'NaN'
        if np.isinf(v):
            return 'Inf' if v > 0 else '-Inf'
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        av = abs(v)
        if av < 1e-3 or av >= 1e7:
            return f'{v:.4e}'
        return f'{v:.4f}'

    def _on_cell_changed(self, row, col, table):
        """Handle cell edit in array view."""
        item = table.item(row, col)
        if item is None:
            return
        try:
            val = float(item.text())
            self.value.data[row, col] = val
            self.value_changed.emit(self.var_name, self.value)
        except ValueError:
            # Revert
            item.setText(self._fmt(self.value.data[row, col]))
