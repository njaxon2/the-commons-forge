"""Snippet / template manager for the Forge editor
(forge/gui/snippet_manager.py).

Loads built-in Octave/MATLAB snippets and user snippets from
~/.forge/snippets.json.  Provides a SnippetDialog (Ctrl+Shift+S) and
Tab-trigger expansion inside CodeEditor.
"""

import json
import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QTreeWidget,
    QTreeWidgetItem, QPushButton, QLabel, QTextEdit,
)

# ------------------------------------------------------------------
# Built-in snippets
# ------------------------------------------------------------------

_BUILTIN_SNIPPETS = [
    {
        "prefix": "func",
        "description": "Function template with header comment",
        "body": (
            "%% ${1:fname} - ${2:Brief description}\n"
            "%  Usage: result = ${1:fname}(${3:args})\n"
            "%\n"
            "function ${4:result} = ${1:fname}(${3:args})\n"
            "    ${5:% body}\n"
            "end\n"
        ),
    },
    {
        "prefix": "for",
        "description": "For loop with index",
        "body": (
            "for ${1:i} = ${2:1}:${3:n}\n"
            "    ${4:% body}\n"
            "end\n"
        ),
    },
    {
        "prefix": "while",
        "description": "While loop",
        "body": (
            "while ${1:condition}\n"
            "    ${2:% body}\n"
            "end\n"
        ),
    },
    {
        "prefix": "if",
        "description": "If-elseif-else block",
        "body": (
            "if ${1:condition}\n"
            "    ${2:% then}\n"
            "elseif ${3:condition2}\n"
            "    ${4:% elseif body}\n"
            "else\n"
            "    ${5:% else body}\n"
            "end\n"
        ),
    },
    {
        "prefix": "switch",
        "description": "Switch-case block",
        "body": (
            "switch ${1:expr}\n"
            "    case ${2:val1}\n"
            "        ${3:% case 1}\n"
            "    case ${4:val2}\n"
            "        ${5:% case 2}\n"
            "    otherwise\n"
            "        ${6:% default}\n"
            "end\n"
        ),
    },
    {
        "prefix": "try",
        "description": "Try-catch block",
        "body": (
            "try\n"
            "    ${1:% risky code}\n"
            "catch ${2:err}\n"
            "    ${3:fprintf('Error: %s\\n', ${2:err}.message);}\n"
            "end\n"
        ),
    },
    {
        "prefix": "cls",
        "description": "Classdef template",
        "body": (
            "classdef ${1:ClassName}\n"
            "    properties\n"
            "        ${2:prop1}\n"
            "    end\n"
            "\n"
            "    methods\n"
            "        function obj = ${1:ClassName}(${3:args})\n"
            "            ${4:% constructor}\n"
            "        end\n"
            "    end\n"
            "end\n"
        ),
    },
    {
        "prefix": "fig",
        "description": "Figure / plot template",
        "body": (
            "figure;\n"
            "plot(${1:x}, ${2:y}, '${3:-b}');\n"
            "xlabel('${4:X}');\n"
            "ylabel('${5:Y}');\n"
            "title('${6:Title}');\n"
            "grid on;\n"
        ),
    },
    {
        "prefix": "fio",
        "description": "File I/O template (fopen/fread/fclose)",
        "body": (
            "fid = fopen('${1:filename}', '${2:r}');\n"
            "if fid == -1\n"
            "    error('Cannot open file: %s', '${1:filename}');\n"
            "end\n"
            "${3:data} = fread(fid, '*char')';\n"
            "fclose(fid);\n"
        ),
    },
    {
        "prefix": "test",
        "description": "Unit test function template",
        "body": (
            "function tests = ${1:test_suite}\n"
            "    tests = functiontests(localfunctions);\n"
            "end\n"
            "\n"
            "function ${2:test_example}(testCase)\n"
            "    ${3:actual} = ${4:myFunc()};\n"
            "    ${5:expected} = ${6:0};\n"
            "    verifyEqual(testCase, ${3:actual}, ${5:expected});\n"
            "end\n"
        ),
    },
]


# ------------------------------------------------------------------
# SnippetManager
# ------------------------------------------------------------------

class SnippetManager:
    """Loads built-in and user snippets; provides lookup helpers."""

    _USER_FILE = os.path.expanduser("~/.forge/snippets.json")

    def __init__(self):
        self._snippets: list[dict] = []
        self._reload()

    # -- public API --------------------------------------------------

    def all_snippets(self) -> list[dict]:
        """Return the full list of snippet dicts."""
        return list(self._snippets)

    def find_by_prefix(self, prefix: str) -> dict | None:
        """Return the first snippet whose prefix matches *prefix*."""
        for s in self._snippets:
            if s["prefix"] == prefix:
                return s
        return None

    def expand_body(self, body: str) -> str:
        """Convert internal body (with \\n and $N placeholders) to
        ready-to-insert text.  Placeholder markers are stripped so the
        user can simply type over them."""
        import re
        text = body.replace("\n", "\n")
        # Replace ${N:default} with the default text
        text = re.sub(r"\$\{\d+:([^}]*)\}", r"\1", text)
        # Remove bare $N placeholders
        text = re.sub(r"\$\d+", "", text)
        return text

    def reload(self):
        self._reload()

    # -- internal ----------------------------------------------------

    def _reload(self):
        self._snippets = list(_BUILTIN_SNIPPETS)
        if os.path.isfile(self._USER_FILE):
            try:
                with open(self._USER_FILE, "r", encoding="utf-8") as fh:
                    user = json.load(fh)
                if isinstance(user, list):
                    self._snippets.extend(user)
            except Exception:
                pass  # silently ignore bad user file


# ------------------------------------------------------------------
# SnippetDialog
# ------------------------------------------------------------------

class SnippetDialog(QDialog):
    """Modal dialog listing all snippets with a search/filter bar."""

    def __init__(self, manager: SnippetManager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Insert Snippet")
        self.resize(520, 420)
        self._manager = manager
        self._selected_snippet: dict | None = None

        layout = QVBoxLayout(self)

        # Search bar
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search snippets...")
        self._search.textChanged.connect(self._apply_filter)
        layout.addWidget(self._search)

        # Snippet list
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Prefix", "Description"])
        self._tree.setColumnWidth(0, 80)
        self._tree.setRootIsDecorated(False)
        self._tree.itemSelectionChanged.connect(self._on_selection)
        self._tree.itemDoubleClicked.connect(self._on_accept)
        layout.addWidget(self._tree)

        # Preview
        layout.addWidget(QLabel("Preview:"))
        self._preview = QTextEdit()
        self._preview.setReadOnly(True)
        self._preview.setMaximumHeight(120)
        layout.addWidget(self._preview)

        # Buttons
        btn_row = QHBoxLayout()
        btn_insert = QPushButton("Insert")
        btn_insert.setDefault(True)
        btn_insert.clicked.connect(self._on_accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(btn_insert)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

        self._populate()

    # -- helpers -----------------------------------------------------

    def selected_snippet(self) -> dict | None:
        return self._selected_snippet

    def _populate(self):
        self._tree.clear()
        for s in self._manager.all_snippets():
            item = QTreeWidgetItem([s["prefix"], s["description"]])
            item.setData(0, Qt.UserRole, s)
            self._tree.addTopLevelItem(item)
        if self._tree.topLevelItemCount():
            self._tree.setCurrentItem(self._tree.topLevelItem(0))

    def _apply_filter(self, text: str):
        text = text.lower()
        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            s = item.data(0, Qt.UserRole)
            visible = text in s["prefix"].lower() or text in s["description"].lower()
            item.setHidden(not visible)

    def _on_selection(self):
        items = self._tree.selectedItems()
        if items:
            s = items[0].data(0, Qt.UserRole)
            self._preview.setPlainText(self._manager.expand_body(s["body"]))

    def _on_accept(self, _item=None, _col=None):
        items = self._tree.selectedItems()
        if items:
            self._selected_snippet = items[0].data(0, Qt.UserRole)
            self.accept()
