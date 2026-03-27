"""Enhanced code outline panel with icons and better M-code parsing."""

import re
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem, QLineEdit,
)
from PySide6.QtGui import QFont, QColor, QIcon


class OutlinePanel(QWidget):
    """Code outline showing functions, classes, sections, and variables."""

    goto_line_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # Filter
        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Filter symbols...")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._apply_filter)
        layout.addWidget(self._filter)

        # Tree
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setRootIsDecorated(True)
        self._tree.setFont(QFont("Consolas", 10))
        self._tree.itemClicked.connect(self._on_click)
        layout.addWidget(self._tree)

    def update_outline(self, text, filename=""):
        """Parse code and update the outline tree."""
        self._tree.clear()
        if not text:
            return

        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''

        if ext in ('m', 'M', ''):
            self._parse_m_code(text)
        elif ext == 'py':
            self._parse_python(text)
        else:
            self._parse_m_code(text)  # Default to M

        self._tree.expandAll()
        self._apply_filter()

    def _parse_m_code(self, text):
        """Parse M/Octave code for outline."""
        lines = text.split('\n')

        sections_item = None
        functions_item = None
        classes_item = None

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Section markers: %% Title
            if stripped.startswith('%%'):
                if sections_item is None:
                    sections_item = QTreeWidgetItem(["\u00a7 Sections"])
                    sections_item.setForeground(0, QColor("#f9e2af"))
                    sections_item.setFont(0, QFont("Consolas", 10, QFont.Bold))
                    self._tree.addTopLevelItem(sections_item)
                title = stripped[2:].strip() or f"Section at line {i+1}"
                item = QTreeWidgetItem([f"  \u00a7 {title}"])
                item.setData(0, Qt.UserRole, i)
                item.setForeground(0, QColor("#f9e2af"))
                sections_item.addChild(item)

            # Function definitions
            m = re.match(r'^\s*function\b(.*)$', line)
            if m:
                if functions_item is None:
                    functions_item = QTreeWidgetItem(["\u0192 Functions"])
                    functions_item.setForeground(0, QColor("#89b4fa"))
                    functions_item.setFont(0, QFont("Consolas", 10, QFont.Bold))
                    self._tree.addTopLevelItem(functions_item)

                # Parse function signature
                sig = m.group(1).strip()
                # Extract function name
                fname_match = re.search(r'(?:=\s*)?([a-zA-Z_]\w*)\s*(?:\(|$)', sig)
                if fname_match:
                    fname = fname_match.group(1)
                else:
                    fname = sig[:30]

                # Extract return values
                ret_match = re.match(r'\[?([^=]*)\]?\s*=', sig)
                ret_str = ""
                if ret_match:
                    ret_str = ret_match.group(0).strip()

                # Extract arguments
                arg_match = re.search(r'\(([^)]*)\)', sig)
                args_str = f"({arg_match.group(1)})" if arg_match else "()"

                display = f"  \u0192 {fname}{args_str}"
                if ret_str:
                    display += f" \u2192 {ret_str.rstrip('= ')}"

                item = QTreeWidgetItem([display])
                item.setData(0, Qt.UserRole, i)
                item.setForeground(0, QColor("#89b4fa"))
                functions_item.addChild(item)

            # Classdef
            m = re.match(r'^\s*classdef\s+(\w+)', line)
            if m:
                if classes_item is None:
                    classes_item = QTreeWidgetItem(["\u25c6 Classes"])
                    classes_item.setForeground(0, QColor("#cba6f7"))
                    classes_item.setFont(0, QFont("Consolas", 10, QFont.Bold))
                    self._tree.addTopLevelItem(classes_item)
                cname = m.group(1)
                item = QTreeWidgetItem([f"  \u25c6 {cname}"])
                item.setData(0, Qt.UserRole, i)
                item.setForeground(0, QColor("#cba6f7"))
                classes_item.addChild(item)

            # Properties / methods blocks
            if re.match(r'^\s*properties\b', stripped):
                item = QTreeWidgetItem([f"  \u25cb properties"])
                item.setData(0, Qt.UserRole, i)
                item.setForeground(0, QColor("#a6adc8"))
                if classes_item:
                    classes_item.addChild(item)
                else:
                    self._tree.addTopLevelItem(item)

            if re.match(r'^\s*methods\b', stripped):
                item = QTreeWidgetItem([f"  \u25cb methods"])
                item.setData(0, Qt.UserRole, i)
                item.setForeground(0, QColor("#a6adc8"))
                if classes_item:
                    classes_item.addChild(item)
                else:
                    self._tree.addTopLevelItem(item)

    def _parse_python(self, text):
        """Parse Python code for outline."""
        lines = text.split('\n')

        classes_item = None
        functions_item = None
        current_class = None

        for i, line in enumerate(lines):
            # Class
            m = re.match(r'^class\s+(\w+)', line)
            if m:
                if classes_item is None:
                    classes_item = QTreeWidgetItem(["\u25c6 Classes"])
                    classes_item.setForeground(0, QColor("#cba6f7"))
                    classes_item.setFont(0, QFont("Consolas", 10, QFont.Bold))
                    self._tree.addTopLevelItem(classes_item)
                cname = m.group(1)
                current_class = QTreeWidgetItem([f"  \u25c6 {cname}"])
                current_class.setData(0, Qt.UserRole, i)
                current_class.setForeground(0, QColor("#cba6f7"))
                classes_item.addChild(current_class)

            # Top-level function
            m = re.match(r'^def\s+(\w+)\s*\(([^)]*)\)', line)
            if m:
                if functions_item is None:
                    functions_item = QTreeWidgetItem(["\u0192 Functions"])
                    functions_item.setForeground(0, QColor("#89b4fa"))
                    functions_item.setFont(0, QFont("Consolas", 10, QFont.Bold))
                    self._tree.addTopLevelItem(functions_item)
                fname = m.group(1)
                args = m.group(2)
                item = QTreeWidgetItem([f"  \u0192 {fname}({args})"])
                item.setData(0, Qt.UserRole, i)
                item.setForeground(0, QColor("#89b4fa"))
                functions_item.addChild(item)

            # Class method
            m = re.match(r'^    def\s+(\w+)\s*\(([^)]*)\)', line)
            if m and current_class:
                fname = m.group(1)
                args = m.group(2)
                item = QTreeWidgetItem([f"    \u0192 {fname}({args})"])
                item.setData(0, Qt.UserRole, i)
                item.setForeground(0, QColor("#89dceb"))
                current_class.addChild(item)

    def _on_click(self, item, column):
        line = item.data(0, Qt.UserRole)
        if line is not None:
            self.goto_line_requested.emit(line)

    def _apply_filter(self, text=None):
        """Filter outline items by text."""
        text = self._filter.text().lower() if text is None else text.lower()
        for i in range(self._tree.topLevelItemCount()):
            top = self._tree.topLevelItem(i)
            any_visible = False
            for j in range(top.childCount()):
                child = top.child(j)
                visible = not text or text in child.text(0).lower()
                child.setHidden(not visible)
                if visible:
                    any_visible = True
            top.setHidden(not any_visible and bool(text))
