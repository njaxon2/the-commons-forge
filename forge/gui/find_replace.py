"""Forge Find & Replace widget (forge/gui/find_replace.py).

Inline search bar that appears at the top of the editor.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QTextDocument, QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QLineEdit, QPushButton, QLabel,
    QCheckBox, QFrame,
)


class FindReplaceBar(QFrame):
    """Inline find/replace bar for CodeEditor."""

    closed = Signal()

    def __init__(self, editor, parent=None):
        super().__init__(parent)
        self.editor = editor
        self.setFrameShape(QFrame.StyledPanel)
        self._apply_theme()

    def _apply_theme(self):
        from forge.gui.theme_utils import detect_palette
        p = detect_palette()
        bg0 = p.get("bg0", "#1e1e2e")
        bg3 = p.get("bg3", "#313145")
        bg5 = p.get("bg5", "#44445a")
        fg0 = p.get("fg0", "#cdd6f4")
        fg2 = p.get("fg2", "#a6adc8")
        border1 = p.get("border1", "#44445a")
        self.setStyleSheet(f"""
            FindReplaceBar {{
                background: {bg3};
                border: 1px solid {border1};
                border-radius: 6px;
                padding: 4px;
            }}
            QLineEdit {{
                background: {bg0};
                color: {fg0};
                border: 1px solid {border1};
                border-radius: 4px;
                padding: 3px 6px;
                min-width: 200px;
            }}
            QPushButton {{
                background: {bg5};
                color: {fg0};
                border: none;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 11px;
            }}
            QPushButton:hover {{ background: {border1}; }}
            QCheckBox {{ color: {fg2}; font-size: 11px; }}
            QLabel {{ color: {fg2}; font-size: 11px; }}
        """)
        self._build_ui()
        self._match_count = 0

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(4)

        # Find row
        layout.addWidget(QLabel("Find:"))
        self.find_edit = QLineEdit()
        self.find_edit.setPlaceholderText("Search...")
        self.find_edit.textChanged.connect(self._on_find_changed)
        self.find_edit.returnPressed.connect(self.find_next)
        layout.addWidget(self.find_edit)

        self.btn_prev = QPushButton("\u25B2")  # ▲
        self.btn_prev.setFixedWidth(28)
        self.btn_prev.setToolTip("Previous (Shift+Enter)")
        self.btn_prev.clicked.connect(self.find_prev)
        layout.addWidget(self.btn_prev)

        self.btn_next = QPushButton("\u25BC")  # ▼
        self.btn_next.setFixedWidth(28)
        self.btn_next.setToolTip("Next (Enter)")
        self.btn_next.clicked.connect(self.find_next)
        layout.addWidget(self.btn_next)

        self.chk_case = QCheckBox("Aa")
        self.chk_case.setToolTip("Case sensitive")
        self.chk_case.toggled.connect(self._on_find_changed)
        layout.addWidget(self.chk_case)

        self.chk_whole = QCheckBox("W")
        self.chk_whole.setToolTip("Whole word")
        self.chk_whole.toggled.connect(self._on_find_changed)
        layout.addWidget(self.chk_whole)

        self.lbl_count = QLabel("")
        layout.addWidget(self.lbl_count)

        layout.addSpacing(8)

        # Replace
        layout.addWidget(QLabel("Replace:"))
        self.replace_edit = QLineEdit()
        self.replace_edit.setPlaceholderText("Replace with...")
        layout.addWidget(self.replace_edit)

        self.btn_replace = QPushButton("Replace")
        self.btn_replace.clicked.connect(self.replace_current)
        layout.addWidget(self.btn_replace)

        self.btn_replace_all = QPushButton("All")
        self.btn_replace_all.clicked.connect(self.replace_all)
        layout.addWidget(self.btn_replace_all)

        # Close button
        btn_close = QPushButton("\u2715")  # ✕
        btn_close.setFixedWidth(24)
        btn_close.clicked.connect(self._close)
        layout.addWidget(btn_close)

    def _get_flags(self):
        flags = QTextDocument.FindFlags()
        if self.chk_case.isChecked():
            flags |= QTextDocument.FindCaseSensitively
        if self.chk_whole.isChecked():
            flags |= QTextDocument.FindWholeWords
        return flags

    def _on_find_changed(self, _=None):
        self._highlight_all()
        self.find_next()

    def _highlight_all(self):
        """Count and highlight all matches."""
        text = self.find_edit.text()
        if not text:
            self.lbl_count.setText("")
            self._match_count = 0
            # Clear highlights
            self.editor.setExtraSelections(
                [s for s in self.editor.extraSelections()
                 if not hasattr(s, '_is_find')]
            )
            return

        doc = self.editor.document()
        cursor = QTextCursor(doc)
        flags = self._get_flags()
        count = 0
        selections = []

        fmt = QTextCharFormat()
        fmt.setBackground(QColor("#f9e2af"))
        fmt.setForeground(QColor(p.get("bg0", "#1e1e2e")) if hasattr(self, "_apply_theme") else QColor("#1e1e2e"))

        while True:
            cursor = doc.find(text, cursor, flags)
            if cursor.isNull():
                break
            sel = self.editor.__class__.__mro__[1].ExtraSelection()
            sel.cursor = cursor
            sel.format = fmt
            sel._is_find = True
            selections.append(sel)
            count += 1
            if count > 5000:
                break

        self._match_count = count
        self.lbl_count.setText(f"{count} matches" if count else "No matches")
        self.lbl_count.setStyleSheet(
            "color: #a6e3a1;" if count else "color: #f38ba8;"
        )

    def find_next(self):
        text = self.find_edit.text()
        if not text:
            return
        cursor = self.editor.textCursor()
        flags = self._get_flags()
        found = self.editor.document().find(text, cursor, flags)
        if found.isNull():
            # Wrap around
            cursor = QTextCursor(self.editor.document())
            found = self.editor.document().find(text, cursor, flags)
        if not found.isNull():
            self.editor.setTextCursor(found)

    def find_prev(self):
        text = self.find_edit.text()
        if not text:
            return
        cursor = self.editor.textCursor()
        flags = self._get_flags() | QTextDocument.FindBackward
        found = self.editor.document().find(text, cursor, flags)
        if found.isNull():
            cursor = QTextCursor(self.editor.document())
            cursor.movePosition(QTextCursor.End)
            found = self.editor.document().find(text, cursor, flags)
        if not found.isNull():
            self.editor.setTextCursor(found)

    def replace_current(self):
        cursor = self.editor.textCursor()
        if cursor.hasSelection():
            cursor.insertText(self.replace_edit.text())
            self.find_next()
            self._highlight_all()

    def replace_all(self):
        text = self.find_edit.text()
        replacement = self.replace_edit.text()
        if not text:
            return
        doc = self.editor.document()
        cursor = QTextCursor(doc)
        cursor.beginEditBlock()
        flags = self._get_flags()
        count = 0
        while True:
            found = doc.find(text, cursor, flags)
            if found.isNull():
                break
            found.insertText(replacement)
            cursor = found
            count += 1
            if count > 10000:
                break
        cursor.endEditBlock()
        self._highlight_all()
        self.lbl_count.setText(f"{count} replaced")

    def show_find(self, initial_text=""):
        self.setVisible(True)
        if initial_text:
            self.find_edit.setText(initial_text)
        self.find_edit.setFocus()
        self.find_edit.selectAll()

    def _close(self):
        self.setVisible(False)
        self.editor.setFocus()
        self.closed.emit()
