"""Code Outline Panel - shows document structure."""
import os
import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QTextCursor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem, QLabel,
)


class OutlinePanel(QWidget):
    """Shows document structure: functions, classes, sections."""
    goto_line_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_text = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self._title = QLabel("Outline")
        self._title.setStyleSheet("color: #a6adc8; font-size: 11px; font-weight: bold; padding: 4px;")
        layout.addWidget(self._title)

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setRootIsDecorated(True)
        self._tree.setAnimated(True)
        self._tree.setIndentation(16)
        self._tree.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._tree)

    def update_outline(self, text, file_path=None):
        """Parse text and update the outline tree."""
        if text == self._current_text:
            return
        self._current_text = text
        self._tree.clear()

        if not text:
            return

        lines = text.split('\n')
        ext = os.path.splitext(file_path)[1].lower() if file_path else '.m'

        if ext == '.m':
            self._parse_m_code(lines)
        elif ext == '.py':
            self._parse_python(lines)
        else:
            self._parse_generic(lines)

        self._tree.expandAll()

    def _parse_m_code(self, lines):
        """Parse M-code for functions, classes, sections."""
        func_pattern = re.compile(r'^\s*function\s+(?:\[?[\w\s,]+\]?\s*=\s*)?([\w]+)\s*\(')
        class_pattern = re.compile(r'^\s*classdef\s+(\w+)')
        section_pattern = re.compile(r'^\s*%%\s*(.*)')
        props_pattern = re.compile(r'^\s*(properties|methods|events)\b')

        current_class = None
        for i, line in enumerate(lines):
            # Sections (%%...)
            m = section_pattern.match(line)
            if m:
                item = QTreeWidgetItem(self._tree, [f"§ {m.group(1).strip()}"])
                item.setForeground(0, QColor("#f9e2af"))
                item._line_num = i + 1
                continue

            # Classes
            m = class_pattern.match(line)
            if m:
                current_class = QTreeWidgetItem(self._tree, [f"⬢ {m.group(1)}"])
                current_class.setForeground(0, QColor("#cba6f7"))
                font = current_class.font(0)
                font.setBold(True)
                current_class.setFont(0, font)
                current_class._line_num = i + 1
                continue

            # Properties/methods blocks
            m = props_pattern.match(line)
            if m and current_class:
                block = QTreeWidgetItem(current_class, [f"⊞ {m.group(1)}"])
                block.setForeground(0, QColor("#89b4fa"))
                block._line_num = i + 1
                continue

            # Functions
            m = func_pattern.match(line)
            if m:
                parent = current_class if current_class else self._tree
                item = QTreeWidgetItem(parent, [f"ƒ {m.group(1)}"])
                item.setForeground(0, QColor("#89dceb"))
                item._line_num = i + 1
                continue

    def _parse_python(self, lines):
        """Parse Python for classes and functions."""
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('class '):
                m = re.match(r'class\s+(\w+)', stripped)
                if m:
                    item = QTreeWidgetItem(self._tree, [f"⬢ {m.group(1)}"])
                    item.setForeground(0, QColor("#cba6f7"))
                    item._line_num = i + 1
            elif stripped.startswith('def '):
                m = re.match(r'def\s+(\w+)', stripped)
                if m:
                    item = QTreeWidgetItem(self._tree, [f"ƒ {m.group(1)}"])
                    item.setForeground(0, QColor("#89dceb"))
                    item._line_num = i + 1

    def _parse_generic(self, lines):
        """Generic parse - just show sections."""
        for i, line in enumerate(lines):
            if line.strip().startswith('#') and len(line.strip()) > 3:
                text = line.strip().lstrip('#').strip()
                if text:
                    item = QTreeWidgetItem(self._tree, [f"# {text[:50]}"])
                    item.setForeground(0, QColor("#6c7086"))
                    item._line_num = i + 1

    def _on_item_clicked(self, item, column):
        if hasattr(item, '_line_num'):
            self.goto_line_requested.emit(item._line_num)
