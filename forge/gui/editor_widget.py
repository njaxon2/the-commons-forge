"""Forge code editor widget with tabs and syntax highlighting
(forge/gui/editor_widget.py)."""

import os
import re

from PySide6.QtCore import Qt, Signal, QRect, QSize
from PySide6.QtGui import (
    QColor, QFont, QPainter, QSyntaxHighlighter, QTextCharFormat,
    QTextCursor, QKeyEvent,
)
from PySide6.QtWidgets import QTextEdit,  QWidget, QVBoxLayout, QTabWidget, QPlainTextEdit


# ======================================================================
# Syntax highlighter
# ======================================================================

class OctaveSyntaxHighlighter(QSyntaxHighlighter):
    """Basic syntax colouring for Octave / Forge .m files."""

    KEYWORDS = [
        "if", "else", "elseif", "endif", "end",
        "for", "endfor", "while", "endwhile", "do", "until",
        "switch", "case", "otherwise", "endswitch",
        "function", "endfunction", "return",
        "break", "continue",
        "try", "catch", "end_try_catch",
        "unwind_protect", "unwind_protect_cleanup",
        "global", "persistent",
        "true", "false",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rules: list[tuple[re.Pattern, QTextCharFormat]] = []
        self._build_rules()

    def _build_rules(self):
        # Keywords — bold blue
        kw_fmt = QTextCharFormat()
        kw_fmt.setForeground(QColor("blue"))
        kw_fmt.setFontWeight(QFont.Bold)
        kw_pattern = r"\b(" + "|".join(self.KEYWORDS) + r")\b"
        self._rules.append((re.compile(kw_pattern), kw_fmt))

        # Numbers — dark cyan
        num_fmt = QTextCharFormat()
        num_fmt.setForeground(QColor("darkcyan"))
        self._rules.append((re.compile(r"\b\d+\.?\d*([eE][+-]?\d+)?\b"), num_fmt))

        # Strings (single-quoted) — red
        str_fmt = QTextCharFormat()
        str_fmt.setForeground(QColor("red"))
        self._rules.append((re.compile(r"'[^']*'"), str_fmt))
        # Double-quoted strings
        self._rules.append((re.compile(r'"[^"]*"'), str_fmt))

        # Comments — green (must come last so it overrides)
        cmt_fmt = QTextCharFormat()
        cmt_fmt.setForeground(QColor("green"))
        self._rules.append((re.compile(r"%.*$"), cmt_fmt))

    def highlightBlock(self, text: str):
        for pattern, fmt in self._rules:
            for m in pattern.finditer(text):
                self.setFormat(m.start(), m.end() - m.start(), fmt)


# ======================================================================
# Line number area
# ======================================================================

class _LineNumberArea(QWidget):
    """Gutter widget drawn beside the code editor."""

    def __init__(self, editor: "CodeEditor"):
        super().__init__(editor)
        self._editor = editor

    def sizeHint(self):
        return QSize(self._editor.line_number_area_width(), 0)

    def paintEvent(self, event):
        self._editor.line_number_area_paint(event)


# ======================================================================
# Single code editor pane
# ======================================================================

class CodeEditor(QPlainTextEdit):
    """Text editor with line numbers, current-line highlight, and tabs->spaces."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.file_path: str | None = None

        font = QFont("Courier New", 10)
        font.setFixedPitch(True)
        self.setFont(font)
        self.setTabStopDistance(self.fontMetrics().horizontalAdvance(" ") * 4)

        # Line number area
        self._line_area = _LineNumberArea(self)
        self.blockCountChanged.connect(self._update_line_area_width)
        self.updateRequest.connect(self._update_line_area)
        self.cursorPositionChanged.connect(self._highlight_current_line)
        self._update_line_area_width(0)

        # Syntax highlighter
        self._highlighter = OctaveSyntaxHighlighter(self.document())

        self._highlight_current_line()

    # --- line numbers ---------------------------------------------------

    def line_number_area_width(self) -> int:
        digits = max(1, len(str(self.blockCount())))
        return 8 + self.fontMetrics().horizontalAdvance("9") * digits

    def _update_line_area_width(self, _count):
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def _update_line_area(self, rect, dy):
        if dy:
            self._line_area.scroll(0, dy)
        else:
            self._line_area.update(0, rect.y(), self._line_area.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self._update_line_area_width(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cr = self.contentsRect()
        self._line_area.setGeometry(
            QRect(cr.left(), cr.top(), self.line_number_area_width(), cr.height())
        )

    def line_number_area_paint(self, event):
        painter = QPainter(self._line_area)
        painter.fillRect(event.rect(), QColor("#f0f0f0"))
        block = self.firstVisibleBlock()
        block_num = block.blockNumber()
        top = int(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                painter.setPen(QColor("#888888"))
                painter.drawText(
                    0, top,
                    self._line_area.width() - 4,
                    self.fontMetrics().height(),
                    Qt.AlignRight, str(block_num + 1),
                )
            block = block.next()
            top = bottom
            bottom = top + int(self.blockBoundingRect(block).height())
            block_num += 1
        painter.end()

    # --- current line highlight ----------------------------------------

    def _highlight_current_line(self):
        sel = QTextEdit.ExtraSelection()
        sel.format.setBackground(QColor("#fffde7"))
        sel.format.setProperty(QTextCharFormat.FullWidthSelection, True)
        sel.cursor = self.textCursor()
        sel.cursor.clearSelection()
        self.setExtraSelections([sel])

    # --- tab inserts spaces --------------------------------------------

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Tab:
            self.insertPlainText("    ")
        else:
            super().keyPressEvent(event)


# ======================================================================
# Tabbed editor widget
# ======================================================================

class EditorWidget(QWidget):
    """Tab container for multiple CodeEditor panes."""

    file_run_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget(self)
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self._close_tab)
        layout.addWidget(self.tabs)

        # Start with one empty tab
        self.new_file()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def new_file(self):
        """Open a new untitled editor tab."""
        editor = CodeEditor(self)
        idx = self.tabs.addTab(editor, "untitled")
        self.tabs.setCurrentIndex(idx)
        return editor

    def open_file(self, path: str):
        """Open *path* in a new tab (or focus if already open)."""
        # Check if already open
        for i in range(self.tabs.count()):
            ed = self.tabs.widget(i)
            if isinstance(ed, CodeEditor) and ed.file_path == path:
                self.tabs.setCurrentIndex(i)
                return ed

        editor = CodeEditor(self)
        editor.file_path = path
        with open(path, "r", encoding="utf-8") as fh:
            editor.setPlainText(fh.read())
        name = os.path.basename(path)
        idx = self.tabs.addTab(editor, name)
        self.tabs.setCurrentIndex(idx)
        return editor

    def save_file(self):
        """Save the current tab to its file_path (no-op if untitled)."""
        editor = self.get_current_editor()
        if editor is None or editor.file_path is None:
            return False
        with open(editor.file_path, "w", encoding="utf-8") as fh:
            fh.write(editor.toPlainText())
        return True

    def get_current_editor(self) -> CodeEditor | None:
        """Return the CodeEditor in the active tab, or None."""
        w = self.tabs.currentWidget()
        return w if isinstance(w, CodeEditor) else None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _close_tab(self, index: int):
        if self.tabs.count() > 1:
            self.tabs.removeTab(index)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_F5:
            editor = self.get_current_editor()
            if editor and editor.file_path:
                self.file_run_requested.emit(editor.file_path)
        else:
            super().keyPressEvent(event)
