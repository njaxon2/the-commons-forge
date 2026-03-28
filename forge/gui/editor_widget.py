"""Forge code editor widget — enhanced with theme-aware highlighting,
bracket matching, improved gutter, and prompt stripping.
(forge/gui/editor_widget.py)
"""

import os
import re

from PySide6.QtCore import Qt, Signal, QRect, QSize, QTimer
from PySide6.QtGui import (
    QColor, QFont, QPainter, QSyntaxHighlighter, QTextCharFormat,
    QTextCursor, QKeyEvent, QPen, QTextFormat, QAction,
)
from PySide6.QtWidgets import QToolTip, QDialog
from forge.gui.snippet_manager import SnippetManager, SnippetDialog
from PySide6.QtWidgets import (
    QTextEdit, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QPlainTextEdit, QLabel, QMenu, QCompleter,
)


# ======================================================================
# Theme-aware colour palettes for syntax highlighting
# ======================================================================

_DARK_COLORS = {
    "keyword":    "#cba6f7",   # mauve
    "builtin":    "#89b4fa",   # blue
    "number":     "#fab387",   # peach
    "string":     "#a6e3a1",   # green
    "comment":    "#6c7086",   # overlay0
    "operator":   "#f38ba8",   # red
    "constant":   "#f9e2af",   # yellow
    "function":   "#89dceb",   # sky
    "bracket":    "#cdd6f4",   # text
    "line_bg":    "#313244",   # surface0
    "gutter_bg":  "#1e1e2e",   # base
    "gutter_fg":  "#585b70",   # surface2
    "gutter_active": "#cba6f7",
}

_LIGHT_COLORS = {
    "keyword":    "#7c3aed",
    "builtin":    "#0369a1",
    "number":     "#c2410c",
    "string":     "#15803d",
    "comment":    "#94a3b8",
    "operator":   "#dc2626",
    "constant":   "#b45309",
    "function":   "#0891b2",
    "bracket":    "#334155",
    "line_bg":    "#fffde7",
    "gutter_bg":  "#f1f5f9",
    "gutter_fg":  "#94a3b8",
    "gutter_active": "#7c3aed",
}

_MIDNIGHT_COLORS = {
    "keyword":    "#c792ea",
    "builtin":    "#82aaff",
    "number":     "#f78c6c",
    "string":     "#c3e88d",
    "comment":    "#546e7a",
    "operator":   "#ff5370",
    "constant":   "#ffcb6b",
    "function":   "#89ddff",
    "bracket":    "#d0d0d0",
    "line_bg":    "#1a1a2e",
    "gutter_bg":  "#0d0d1a",
    "gutter_fg":  "#3a3a5c",
    "gutter_active": "#c792ea",
}

PALETTES = {
    "dark": _DARK_COLORS,
    "light": _LIGHT_COLORS,
    "midnight": _MIDNIGHT_COLORS,
}

_current_palette = _DARK_COLORS


def set_editor_palette(theme_name: str):
    """Switch the editor colour palette (called from theme switching)."""
    global _current_palette
    _current_palette = PALETTES.get(theme_name, _DARK_COLORS)


def get_palette():
    return _current_palette


# ======================================================================
# Syntax highlighter
# ======================================================================

# ======================================================================
# Symbol parsing for breadcrumb and Go to Symbol
# ======================================================================

import re as _re_mod

def parse_symbols(text):
    """Parse function definitions and section headers from M/Python code."""
    symbols = []
    for i, line in enumerate(text.split("\n")):
        stripped = line.strip()
        # Octave/MATLAB function
        m = _re_mod.match(r"function\b.*?\b(\w+)\s*[=(]", stripped)
        if m:
            symbols.append({"kind": "function", "name": m.group(1), "line": i + 1})
            continue
        # Section header (%% ...)
        m = _re_mod.match(r"%%\s+(.+)", stripped)
        if m:
            symbols.append({"kind": "section", "name": m.group(1).strip(), "line": i + 1})
            continue
        # Python def
        m = _re_mod.match(r"def\s+(\w+)\s*\(", stripped)
        if m:
            symbols.append({"kind": "function", "name": m.group(1), "line": i + 1})
    return symbols

def symbol_at_line(symbols, line):
    """Return the name of the symbol containing the given line."""
    current = None
    for s in symbols:
        if s["line"] <= line:
            current = s["name"]
        else:
            break
    return current


class BreadcrumbBar(QWidget):
    """Clickable file path breadcrumb with current symbol display."""

    directory_clicked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(8, 2, 8, 2)
        self._layout.setSpacing(0)
        self._layout.addStretch()
        self.setFixedHeight(22)
        self.setStyleSheet("background: transparent; font-size: 11px;")

    def update_path(self, file_path, symbol_name=None):
        """Update breadcrumb to show file_path > symbol."""
        # Clear existing
        while self._layout.count() > 1:
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not file_path:
            return

        parts = file_path.replace("\\", "/").split("/")
        # Show last 3 path parts
        show_parts = parts[-3:] if len(parts) > 3 else parts

        for i, part in enumerate(show_parts):
            if i > 0:
                sep = QLabel(" > ")
                _p = get_palette()
                sep.setStyleSheet(f"color: {_p.get('gutter_fg', '#6c7086')}; font-size: 10px;")
                self._layout.insertWidget(self._layout.count() - 1, sep)

            lbl = QLabel(part)
            is_last = (i == len(show_parts) - 1)
            _p = get_palette()
            if is_last:
                lbl.setStyleSheet(f"color: {_p.get('bracket', '#cdd6f4')}; font-weight: bold; font-size: 11px;")
            else:
                lbl.setStyleSheet(f"color: {_p.get('gutter_fg', '#6c7086')}; font-size: 11px;")
                # Make directory parts clickable
                dir_path = "/".join(parts[:len(parts) - len(show_parts) + i + 1])
                lbl.setCursor(Qt.PointingHandCursor)
                lbl.mousePressEvent = lambda e, p=dir_path: self.directory_clicked.emit(p)
            self._layout.insertWidget(self._layout.count() - 1, lbl)

        if symbol_name:
            _p = get_palette()
            sep = QLabel(" > ")
            sep.setStyleSheet(f"color: {_p.get('gutter_fg', '#6c7086')}; font-size: 10px;")
            self._layout.insertWidget(self._layout.count() - 1, sep)
            sym = QLabel(symbol_name)
            sym.setStyleSheet(f"color: {_p.get('keyword', '#cba6f7')}; font-style: italic; font-size: 11px;")
            self._layout.insertWidget(self._layout.count() - 1, sym)


class GoToSymbolDialog(QDialog):
    """Quick symbol navigation dialog (Ctrl+Shift+O)."""

    def __init__(self, symbols, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Go to Symbol")
        self.setFixedSize(400, 350)
        _p = get_palette()
        self.setStyleSheet(f"""
            QDialog {{ background: {_p.get('gutter_bg', '#1e1e2e')}; border: 1px solid {_p.get('line_bg', '#313244')}; border-radius: 8px; }}
            QLineEdit {{ background: {_p.get('line_bg', '#2a2a3c')}; color: {_p.get('bracket', '#cdd6f4')}; border: 1px solid {_p.get('line_bg', '#313244')};
                        border-radius: 4px; padding: 6px; font-size: 13px; }}
            QListWidget {{ background: {_p.get('gutter_bg', '#252536')}; color: {_p.get('bracket', '#cdd6f4')}; border: none; font-size: 12px; }}
            QListWidget::item {{ padding: 4px 8px; border-radius: 4px; }}
            QListWidget::item:selected {{ background: {_p.get('line_bg', '#264f78')}; }}
        """)

        layout = QVBoxLayout(self)
        self._search = QLineEdit()
        self._search.setPlaceholderText("Type to filter symbols...")
        self._search.textChanged.connect(self._filter)
        layout.addWidget(self._search)

        from PySide6.QtWidgets import QListWidget
        self._list = QListWidget()
        self._list.itemDoubleClicked.connect(self._accept)
        layout.addWidget(self._list)

        self._symbols = symbols
        self._filtered = list(symbols)
        self._populate()

        self.selected_line = None

    def _populate(self):
        self._list.clear()
        for s in self._filtered:
            icon = "ƒ" if s["kind"] == "function" else "§"
            self._list.addItem(f"{icon}  {s['name']}  :{s['line']}")

    def _filter(self, text):
        text = text.lower()
        self._filtered = [s for s in self._symbols if text in s["name"].lower()]
        self._populate()

    def _accept(self, item=None):
        idx = self._list.currentRow()
        if 0 <= idx < len(self._filtered):
            self.selected_line = self._filtered[idx]["line"]
        self.accept()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._accept()
        elif event.key() == Qt.Key_Down:
            self._list.setFocus()
            if self._list.count() > 0:
                self._list.setCurrentRow(0)
        else:
            super().keyPressEvent(event)


class OctaveSyntaxHighlighter(QSyntaxHighlighter):
    """Rich syntax highlighting for Octave/MATLAB .m files."""

    KEYWORDS = [
        "if", "else", "elseif", "endif", "end",
        "for", "endfor", "while", "endwhile", "do", "until",
        "switch", "case", "otherwise", "endswitch",
        "function", "endfunction", "return",
        "break", "continue",
        "try", "catch", "end_try_catch",
        "unwind_protect", "unwind_protect_cleanup",
        "global", "persistent", "classdef", "properties", "methods",
        "events", "enumeration", "parfor", "spmd",
    ]

    BUILTINS = [
        "disp", "fprintf", "sprintf", "size", "length", "numel",
        "zeros", "ones", "eye", "rand", "randn", "linspace",
        "reshape", "sum", "prod", "min", "max", "sort",
        "find", "abs", "sqrt", "exp", "log", "sin", "cos", "tan",
        "plot", "figure", "hold", "title", "xlabel", "ylabel",
        "legend", "grid", "subplot", "scatter", "bar", "hist",
        "error", "warning", "assert", "nargin", "nargout",
        "cell", "struct", "fieldnames", "class", "isa",
        "fopen", "fclose", "fread", "fwrite", "fgets",
        "input", "keyboard", "eval", "feval",
    ]

    CONSTANTS = [
        "pi", "inf", "Inf", "nan", "NaN", "eps", "true", "false",
        "i", "j", "e", "realmin", "realmax",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rules: list[tuple[re.Pattern, str]] = []
        self._build_rules()

    def _build_rules(self):
        """Build rules as (pattern, palette_key) pairs."""
        # Keywords
        kw = r"\b(" + "|".join(self.KEYWORDS) + r")\b"
        self._rules.append((re.compile(kw), "keyword"))

        # Builtins
        bi = r"\b(" + "|".join(self.BUILTINS) + r")\b"
        self._rules.append((re.compile(bi), "builtin"))

        # Constants
        cn = r"\b(" + "|".join(self.CONSTANTS) + r")\b"
        self._rules.append((re.compile(cn), "constant"))

        # Function calls: word followed by (
        self._rules.append((re.compile(r"\b([a-zA-Z_]\w*)\s*(?=\()"), "function"))

        # Numbers (int, float, scientific, hex)
        self._rules.append((re.compile(
            r"\b0[xX][0-9a-fA-F]+\b|\b\d+\.?\d*([eE][+-]?\d+)?\b"
        ), "number"))

        # Operators
        self._rules.append((re.compile(
            r"[+\-*/\\^~!<>=&|@]|\.[\*/\\^]|==|~=|<=|>=|&&|\|\||\.\.\."
        ), "operator"))

        # Strings (single and double quoted)
        self._rules.append((re.compile(r"'[^']*'"), "string"))
        self._rules.append((re.compile(r'"[^"]*"'), "string"))

        # Comments (must come last)
        self._rules.append((re.compile(r"%.*$"), "comment"))
        self._rules.append((re.compile(r"#.*$"), "comment"))

    def highlightBlock(self, text: str):
        p = get_palette()

        # Multi-line block comment support (%{ ... %})
        if self.previousBlockState() == 1:
            end_idx = text.find('%}')
            if end_idx >= 0:
                fmt = QTextCharFormat()
                fmt.setForeground(QColor(p.get('comment', '#585b70')))
                fmt.setFontItalic(True)
                self.setFormat(0, end_idx + 2, fmt)
                self.setCurrentBlockState(0)
                return
            else:
                fmt = QTextCharFormat()
                fmt.setForeground(QColor(p.get('comment', '#585b70')))
                fmt.setFontItalic(True)
                self.setFormat(0, len(text), fmt)
                self.setCurrentBlockState(1)
                return

        bc_start = text.find('%{')
        if bc_start >= 0:
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(p.get('comment', '#585b70')))
            fmt.setFontItalic(True)
            bc_end = text.find('%}', bc_start + 2)
            if bc_end >= 0:
                self.setFormat(bc_start, bc_end + 2 - bc_start, fmt)
                self.setCurrentBlockState(0)
            else:
                self.setFormat(bc_start, len(text) - bc_start, fmt)
                self.setCurrentBlockState(1)

        for pattern, key in self._rules:
            fmt = QTextCharFormat()
            color = p.get(key, "#cccccc")
            fmt.setForeground(QColor(color))
            if key == "keyword":
                fmt.setFontWeight(QFont.Bold)
            elif key == "comment":
                fmt.setFontItalic(True)
            for m in pattern.finditer(text):
                self.setFormat(m.start(), m.end() - m.start(), fmt)


# ======================================================================
# Line number area with active-line highlight
# ======================================================================

class MinimapWidget(QWidget):
    """A compact code overview widget shown at the right edge of the editor."""

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.setFixedWidth(80)
        self.setCursor(Qt.PointingHandCursor)
        self._drag = False

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QColor, QPen
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        palette = get_palette()
        bg = QColor(palette.get('gutter_bg', '#1e1e2e'))
        bg.setAlpha(180)
        painter.fillRect(self.rect(), bg)

        doc = self.editor.document()
        total_lines = doc.blockCount()
        if total_lines == 0:
            painter.end()
            return

        h = self.height()
        scale = min(h / max(total_lines, 1), 3.0)

        # Visible region highlight
        first_visible = self.editor.firstVisibleBlock().blockNumber()
        block_rect = self.editor.blockBoundingRect(self.editor.firstVisibleBlock())
        block_h = max(1, int(block_rect.height()))
        visible_lines = max(1, self.editor.viewport().height() // block_h)

        vis_y = int(first_visible * scale)
        vis_h = max(10, int(visible_lines * scale))
        vis_color = QColor(palette.get('gutter_active', '#cba6f7'))
        vis_color.setAlpha(40)
        painter.fillRect(0, vis_y, self.width(), vis_h, vis_color)

        # Draw code lines as colored marks
        block = doc.begin()
        y = 0.0
        keyword_color = QColor(palette.get('keyword', '#cba6f7'))
        keyword_color.setAlpha(200)
        comment_color = QColor(palette.get('comment', '#6c7086'))
        comment_color.setAlpha(150)
        string_color = QColor(palette.get('string', '#a6e3a1'))
        string_color.setAlpha(150)
        normal_color = QColor(palette.get('bracket', '#cdd6f4'))
        normal_color.setAlpha(80)

        kw_set = {'function', 'if', 'for', 'while', 'switch', 'try', 'end', 'else', 'return', 'elseif', 'case'}
        while block.isValid() and y < h:
            text = block.text().rstrip()
            if text:
                stripped = text.lstrip()
                indent = len(text) - len(stripped)
                first_word = stripped.split()[0] if stripped.split() else ''
                if stripped.startswith('%') or stripped.startswith('#'):
                    color = comment_color
                elif stripped.startswith("'") or stripped.startswith('"'):
                    color = string_color
                elif first_word in kw_set:
                    color = keyword_color
                else:
                    color = normal_color

                x_start = int(indent * 0.5) + 2
                x_end = min(self.width() - 2, x_start + int(len(stripped) * 0.5))
                painter.setPen(Qt.NoPen)
                painter.setBrush(color)
                line_h = max(1, int(scale * 0.8))
                painter.drawRect(x_start, int(y), max(1, x_end - x_start), line_h)

            block = block.next()
            y += scale

        # Border line
        painter.setPen(QColor(palette.get('line_bg', '#313244')))
        painter.drawLine(0, 0, 0, h)
        painter.end()


    def wheelEvent(self, event):
        """Handle Ctrl+Scroll for zoom."""
        from PySide6.QtCore import Qt
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoomIn(1)
            elif delta < 0:
                self.zoomOut(1)
            event.accept()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event):
        self._drag = True
        self._scroll_to_pos(event.pos().y())

    def mouseMoveEvent(self, event):
        if self._drag:
            self._scroll_to_pos(event.pos().y())

    def mouseReleaseEvent(self, event):
        self._drag = False

    def _scroll_to_pos(self, y):
        total_lines = self.editor.document().blockCount()
        h = self.height()
        if h == 0 or total_lines == 0:
            return
        scale = min(h / max(total_lines, 1), 3.0)
        target_line = int(y / scale)
        target_line = max(0, min(target_line, total_lines - 1))
        block = self.editor.document().findBlockByNumber(target_line)
        cursor = QTextCursor(block)
        self.editor.setTextCursor(cursor)
        self.editor.centerCursor()


class _LineNumberArea(QWidget):
    """Gutter widget drawn beside the code editor."""

    def __init__(self, editor: "CodeEditor"):
        super().__init__(editor)
        self._editor = editor

    def sizeHint(self):
        return QSize(self._editor.line_number_area_width(), 0)

    def paintEvent(self, event):
        self._editor.line_number_area_paint(event)

    def mousePressEvent(self, event):
        """Click in gutter to set cursor to that line (future: breakpoints)."""
        # Could toggle breakpoints here
        pass


# ======================================================================
# Single code editor pane
# ======================================================================

class CodeEditor(QPlainTextEdit):
    """Text editor with line numbers, bracket matching, prompt stripping,
    auto-indent, and theme-aware syntax highlighting."""

    help_requested = Signal(str)   # right-click > Help on function

    def __init__(self, parent=None):
        super().__init__(parent)
        self.file_path: str | None = None

        font = QFont("Consolas", 10)
        font.setFixedPitch(True)
        self.setFont(font)
        self.setTabStopDistance(self.fontMetrics().horizontalAdvance(" ") * 4)

        # Line number area
        self._line_area = _LineNumberArea(self)
        self.blockCountChanged.connect(self._update_line_area_width)
        self.updateRequest.connect(self._update_line_area)
        self.cursorPositionChanged.connect(self._on_cursor_moved)
        self._update_line_area_width(0)

        # Syntax highlighter
        self._highlighter = OctaveSyntaxHighlighter(self.document())

        # Track modifications for tab title asterisk
        self.document().modificationChanged.connect(self._on_modification_changed)
        self._is_modified = False

        # Enable drag & drop and mouse tracking for hover tooltips
        self.setAcceptDrops(True)

        # Code folding state
        self._bookmarks = set()  # set of bookmarked line numbers (0-based)
        self._folded_regions = set()  # set of line numbers (0-based) that are fold headers with folded content
        self._fold_indicators = {}    # line_num -> end_line_num for foldable blocks
        self.setMouseTracking(True)

        # Minimap
        self._minimap = MinimapWidget(self)
        self._minimap_visible = True

        # Autocomplete
        self._completer = None
        self._setup_completer()

        # Bracket matching
        self._bracket_selections = []

        self._highlight_current_line()

        # Multi-cursor state
        self._extra_cursors = []  # list of QTextCursor
        self._column_sel_anchor = None

        # Snippet manager
        self._snippet_mgr = SnippetManager()

    # --- line numbers ---------------------------------------------------

    def line_number_area_width(self) -> int:
        digits = max(1, len(str(self.blockCount())))
        return 12 + self.fontMetrics().horizontalAdvance("9") * digits

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
        # Position minimap
        if hasattr(self, '_minimap') and self._minimap_visible:
            cr = self.contentsRect()
            self._minimap.setGeometry(cr.right() - 80, cr.top(), 80, cr.height())
            self._minimap.show()
        elif hasattr(self, '_minimap'):
            self._minimap.hide()

    def line_number_area_paint(self, event):
        p = get_palette()
        painter = QPainter(self._line_area)
        painter.fillRect(event.rect(), QColor(p["gutter_bg"]))

        block = self.firstVisibleBlock()
        block_num = block.blockNumber()
        top = int(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())

        current_block = self.textCursor().blockNumber()

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                is_current = (block_num == current_block)
                color = p["gutter_active"] if is_current else p["gutter_fg"]
                painter.setPen(QColor(color))
                font = painter.font()
                font.setBold(is_current)
                painter.setFont(font)
                painter.drawText(
                    0, top,
                    self._line_area.width() - 6,
                    self.fontMetrics().height(),
                    Qt.AlignRight, str(block_num + 1),
                )
            block = block.next()
            top = bottom
            bottom = top + int(self.blockBoundingRect(block).height())
            block_num += 1
        # Draw bookmark indicators (blue dots in left margin)
        if hasattr(self, '_bookmarks') and self._bookmarks:
            bm_block = self.firstVisibleBlock()
            bm_top = int(self.blockBoundingGeometry(bm_block).translated(self.contentOffset()).top())
            while bm_block.isValid() and bm_top <= event.rect().bottom():
                if bm_block.isVisible():
                    bm_line = bm_block.blockNumber()
                    if bm_line in self._bookmarks:
                        bm_h = int(self.blockBoundingRect(bm_block).height())
                        bm_y = bm_top + (bm_h - 6) // 2
                        painter.setPen(Qt.NoPen)
                        painter.setBrush(QColor("#89b4fa"))
                        painter.drawEllipse(3, bm_y, 6, 6)
                        painter.setBrush(Qt.NoBrush)
                next_bm = bm_block.next()
                if next_bm.isValid():
                    bm_top += int(self.blockBoundingRect(bm_block).height())
                    bm_block = next_bm
                else:
                    break

        # Draw fold indicators (▶/▼ in right margin)
        if hasattr(self, '_fold_regions') and self._fold_regions:
            fold_block = self.firstVisibleBlock()
            fold_top = int(self.blockBoundingGeometry(fold_block).translated(self.contentOffset()).top())
            fold_starts = {r[0] for r in self._fold_regions}
            while fold_block.isValid() and fold_top <= event.rect().bottom():
                if fold_block.isVisible():
                    line_num = fold_block.blockNumber()
                    if line_num in fold_starts:
                        fold_h = int(self.blockBoundingRect(fold_block).height())
                        fold_y = fold_top + (fold_h - 8) // 2
                        fold_x = self._line_area.width() - 14
                        is_folded = hasattr(self, '_folded_lines') and line_num in self._folded_lines
                        painter.setPen(QPen(QColor("#6c7086"), 1))
                        if is_folded:
                            # Right-pointing triangle ▶
                            pts = [QPointF(fold_x, fold_y), QPointF(fold_x + 8, fold_y + 4), QPointF(fold_x, fold_y + 8)]
                            painter.setBrush(QColor("#6c7086"))
                            from PySide6.QtGui import QPolygonF
                            painter.drawPolygon(QPolygonF(pts))
                            painter.setBrush(Qt.NoBrush)
                        else:
                            # Down-pointing triangle ▼
                            pts = [QPointF(fold_x, fold_y), QPointF(fold_x + 8, fold_y), QPointF(fold_x + 4, fold_y + 8)]
                            painter.setBrush(QColor("#6c7086"))
                            from PySide6.QtGui import QPolygonF
                            painter.drawPolygon(QPolygonF(pts))
                            painter.setBrush(Qt.NoBrush)
                next_fold = fold_block.next()
                if next_fold.isValid():
                    fold_top += int(self.blockBoundingRect(fold_block).height())
                    fold_block = next_fold
                else:
                    break

        painter.end()

    # --- current line highlight & bracket matching ---------------------



    def insertFromMimeData(self, source):
        """Override paste to strip >> prompts from copied command window output."""
        if source.hasText():
            text = source.text()
            stripped = self._strip_prompts(text)
            if stripped != text:
                from PySide6.QtCore import QMimeData
                new_source = QMimeData()
                new_source.setText(stripped)
                super().insertFromMimeData(new_source)
                return
        super().insertFromMimeData(source)

    @staticmethod
    def _strip_prompts(text):
        """Strip >> and >>> prompts from pasted text (MATLAB/Octave style)."""
        import re
        lines = text.split('\n')
        # Check if at least half the lines have prompts
        prompt_pattern = re.compile(r'^\s*>{1,3}\s?')
        prompt_count = sum(1 for line in lines if prompt_pattern.match(line))
        if prompt_count >= len(lines) * 0.4 and prompt_count >= 1:
            stripped = []
            for line in lines:
                m = prompt_pattern.match(line)
                if m:
                    stripped.append(line[m.end():])
                else:
                    stripped.append(line)
            return '\n'.join(stripped)
        return text


    def mousePressEvent(self, event):
        """Handle Ctrl+Click for go-to-definition."""
        from PySide6.QtCore import Qt
        if event.modifiers() & Qt.ControlModifier and event.button() == Qt.LeftButton:
            cursor = self.cursorForPosition(event.pos())
            cursor.select(QTextCursor.WordUnderCursor)
            word = cursor.selectedText().strip()
            if word and word.isidentifier():
                self._goto_definition(word)
                return
        super().mousePressEvent(event)

    def _goto_definition(self, symbol):
        """Try to navigate to the definition of a symbol."""
        import re
        # Search in current file first
        text = self.toPlainText()
        # M-language function definition patterns
        patterns = [
            rf'^\s*function\b.*\b{re.escape(symbol)}\s*\(',  # function definition
            rf'^\s*function\b.*=\s*{re.escape(symbol)}\s*\(',  # function with return
            rf'^\s*classdef\s+{re.escape(symbol)}\b',  # classdef
        ]
        for pattern in patterns:
            for i, line in enumerate(text.split('\n')):
                if re.match(pattern, line):
                    # Navigate to that line
                    block = self.document().findBlockByLineNumber(i)
                    cursor = QTextCursor(block)
                    self.setTextCursor(cursor)
                    self.centerCursor()
                    return

        # Try to find the function file on path
        # Look in common locations
        search_dirs = [os.path.dirname(self._file_path)] if hasattr(self, '_file_path') and self._file_path else []
        search_dirs.append(os.path.expanduser('~'))

        for d in search_dirs:
            candidate = os.path.join(d, f'{symbol}.m')
            if os.path.exists(candidate):
                # Signal to open this file
                self.help_requested.emit(f'edit:{candidate}')
                return

        # Not found - show tooltip
        self.setToolTip(f"Cannot find definition of '{symbol}'")

    def event(self, event):
        """Handle tooltip events for function hover info."""
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.ToolTip:
            from PySide6.QtGui import QCursor
            cursor = self.cursorForPosition(self.viewport().mapFromGlobal(QCursor.pos()))
            cursor.select(QTextCursor.WordUnderCursor)
            word = cursor.selectedText().strip()
            if word and word.isidentifier() and len(word) >= 2:
                tip = self._get_hover_info(word)
                if tip:
                    from PySide6.QtWidgets import QToolTip, QDialog
                    QToolTip.showText(QCursor.pos(), tip, self)
                    return True
        return super().event(event)

    def _get_hover_info(self, word):
        """Get hover tooltip information for a function/variable."""
        # Check if it's a known builtin
        try:
            main_win = self.parent()
            while main_win and not hasattr(main_win, 'session'):
                main_win = main_win.parent()
            if main_win and hasattr(main_win, 'session') and main_win.session:
                engine = main_win.session._engine
                if hasattr(engine, 'functions') and word in engine.functions:
                    func = engine.functions[word]
                    doc = getattr(func, '__doc__', '') or ''
                    first_line = doc.split('\n')[0].strip() if doc else 'Built-in function'
                    return f"<b>{word}</b><br/>{first_line}"
        except Exception:
            pass

        # Check in current file for function/variable definitions
        import re
        text = self.toPlainText()
        for i, line in enumerate(text.split('\n')):
            if re.match(rf'^\s*function\b.*\b{re.escape(word)}\s*\(', line):
                return f"<b>{word}</b><br/>Defined at line {i+1}"
            if re.match(rf'^\s*{re.escape(word)}\s*=', line):
                return f"<b>{word}</b><br/>Assigned at line {i+1}"
        return None


    def _lint_code(self):
        """Run basic lint checks on the current code."""
        import re
        text = self.toPlainText()
        lines = text.split('\n')
        diagnostics = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Check for common issues
            # 1. Missing semicolons on lines that produce output
            if stripped and not stripped.startswith('%') and not stripped.startswith('#'):
                if '=' in stripped and not stripped.endswith(';') and not stripped.endswith('...'):
                    # Assignment without semicolon (informational)
                    if not any(stripped.startswith(k) for k in ['if', 'for', 'while', 'function',
                        'switch', 'case', 'try', 'catch', 'else', 'elseif', 'end', 'return',
                        'break', 'continue', 'global', 'persistent', 'classdef', 'properties',
                        'methods', 'events']):
                        pass  # Could add hint about missing semicolon

            # 2. Check for unbalanced brackets on single line
            for char_open, char_close in [('(', ')'), ('[', ']'), ('{', '}')]:
                opens = stripped.count(char_open)
                closes = stripped.count(char_close)
                # Only flag if clearly unbalanced and not continuation
                if opens > closes + 1 and not stripped.endswith('...'):
                    diagnostics.append({
                        'line': i,
                        'col': stripped.index(char_open),
                        'message': f'Possibly unbalanced {char_open}{char_close}',
                        'severity': 'warning'
                    })

            # 3. Check for == in conditions (common MATLAB issue: = vs ==)
            if re.match(r'\s*if\s+.*[^=!<>~]=[^=]', stripped):
                if '==' not in stripped and '~=' not in stripped:
                    diagnostics.append({
                        'line': i,
                        'col': stripped.index('='),
                        'message': 'Assignment in if condition - did you mean ==?',
                        'severity': 'warning'
                    })

        return diagnostics

    def _apply_lint_decorations(self, diagnostics):
        """Apply squiggly underline decorations for lint warnings."""
        from PySide6.QtGui import QTextCharFormat, QPen
        extra = [s for s in self.extraSelections()
                 if not hasattr(s, '_is_lint')]

        for diag in diagnostics:
            from PySide6.QtWidgets import QPlainTextEdit
            sel = QPlainTextEdit.ExtraSelection()
            fmt = QTextCharFormat()
            if diag['severity'] == 'error':
                fmt.setUnderlineColor(QColor("#f38ba8"))
            else:
                fmt.setUnderlineColor(QColor("#f9e2af"))
            fmt.setUnderlineStyle(QTextCharFormat.WaveUnderline)

            block = self.document().findBlockByLineNumber(diag['line'])
            cursor = QTextCursor(block)
            cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
            sel.cursor = cursor
            sel.format = fmt
            sel._is_lint = True
            extra.append(sel)

        self.setExtraSelections(extra)

    def open_snippet_dialog(self):
        """Open the snippet insertion dialog."""
        if hasattr(self, '_snippet_mgr'):
            dlg = SnippetDialog(self._snippet_mgr, self)
            if dlg.exec():
                snippet = dlg.selected_snippet()
                if snippet:
                    import re as _re
                    body = snippet['body']
                    body = _re.sub(r'\$\{\d+:([^}]*)\}', r'', body)
                    body = _re.sub(r'\$\d+', '', body)
                    self.textCursor().insertText(body)

    def format_code(self):
        """Format the current code using MCodeFormatter."""
        try:
            from forge.gui.code_formatter import MCodeFormatter
            formatter = MCodeFormatter()
            cursor = self.textCursor()
            if cursor.hasSelection():
                text = cursor.selectedText().replace("\u2029", "\n")
                formatted = formatter.format(text)
                cursor.insertText(formatted)
            else:
                pos = cursor.position()
                text = self.toPlainText()
                formatted = formatter.format(text)
                self.setPlainText(formatted)
                cursor.setPosition(min(pos, len(formatted)))
                self.setTextCursor(cursor)
        except Exception as e:
            pass  # Silently fail

    def contextMenuEvent(self, event):
        """Rich right-click context menu with IDE actions."""
        from PySide6.QtWidgets import QMenu
        from PySide6.QtGui import QAction

        menu = QMenu(self)
        palette = get_palette()
        menu.setStyleSheet(f"""
            QMenu {{
                background: {palette.get('bg2', '#313244')};
                color: {palette.get('fg0', '#cdd6f4')};
                border: 1px solid {palette.get('border0', '#45475a')};
                border-radius: 6px;
                padding: 4px;
            }}
            QMenu::item {{
                padding: 6px 24px 6px 12px;
                border-radius: 4px;
            }}
            QMenu::item:selected {{
                background: {palette.get('accent', '#cba6f7')};
                color: {palette.get('bg0', '#1e1e2e')};
            }}
            QMenu::separator {{
                height: 1px;
                background: {palette.get('border0', '#45475a')};
                margin: 4px 8px;
            }}
        """)

        cursor = self.textCursor()
        has_selection = cursor.hasSelection()

        # Standard edit actions
        act_cut = menu.addAction("Cut\tCtrl+X")
        act_cut.setEnabled(has_selection)
        act_cut.triggered.connect(self.cut)

        act_copy = menu.addAction("Copy\tCtrl+C")
        act_copy.setEnabled(has_selection)
        act_copy.triggered.connect(self.copy)

        act_paste = menu.addAction("Paste\tCtrl+V")
        act_paste.triggered.connect(self.paste)

        act_select_all = menu.addAction("Select All\tCtrl+A")
        act_select_all.triggered.connect(self.selectAll)

        menu.addSeparator()

        # Code actions
        act_comment = menu.addAction("Toggle Comment\tCtrl+/")
        act_comment.triggered.connect(self._toggle_comment)

        act_dup = menu.addAction("Duplicate Line\tCtrl+D")
        act_dup.triggered.connect(self._duplicate_line)

        act_del = menu.addAction("Delete Line\tCtrl+Shift+K")
        act_del.triggered.connect(self._delete_line)

        menu.addSeparator()

        # Word under cursor actions
        cursor.select(QTextCursor.WordUnderCursor)
        word = cursor.selectedText().strip()
        if word and word.isidentifier():
            act_help = menu.addAction(f"Help on '{word}'\tF1")
            act_help.triggered.connect(lambda: self.help_requested.emit(word))

            act_find_all = menu.addAction(f"Find All References to '{word}'")
            act_find_all.triggered.connect(lambda: self._find_all_refs(word))

            act_rename = menu.addAction(f"Rename '{word}'...")
            act_rename.triggered.connect(lambda: self._rename_symbol(word))

            menu.addSeparator()

        # Folding
        if hasattr(self, '_fold_regions') and self._fold_regions:
            line = self.textCursor().blockNumber()
            for start, end in self._fold_regions:
                if line == start:
                    act_fold = menu.addAction("Fold Region")
                    act_fold.triggered.connect(lambda: self._toggle_fold(start))
                    break

        act_fold_all = menu.addAction("Fold All")
        act_fold_all.triggered.connect(self._fold_all)

        act_unfold_all = menu.addAction("Unfold All")
        act_unfold_all.triggered.connect(self._unfold_all)

        menu.addSeparator()

        # Bookmark
        act_bookmark = menu.addAction("Toggle Bookmark\tCtrl+F2")
        act_bookmark.triggered.connect(self.toggle_bookmark)

        # Run
        if has_selection:
            act_run = menu.addAction("Run Selection\tF9")
            act_run.triggered.connect(lambda: self.eval_requested.emit(cursor.selectedText()))

        menu.exec(event.globalPos())

    def _find_all_refs(self, word):
        """Highlight and list all references to a symbol."""
        import re
        text = self.toPlainText()
        pattern = r'\b' + re.escape(word) + r'\b'
        count = len(list(re.finditer(pattern, text)))
        # Flash status
        self.parent().parent().statusBar().showMessage(f"Found {count} references to '{word}'", 3000) if hasattr(self.parent(), 'parent') else None

    def _rename_symbol(self, old_word):
        """Rename all occurrences of a symbol in the current file."""
        from PySide6.QtWidgets import QInputDialog
        new_word, ok = QInputDialog.getText(
            self, "Rename Symbol",
            f"Rename '{old_word}' to:",
            text=old_word
        )
        if ok and new_word and new_word != old_word:
            import re
            text = self.toPlainText()
            pattern = r'\b' + re.escape(old_word) + r'\b'
            new_text = re.sub(pattern, new_word, text)
            cursor = self.textCursor()
            cursor.select(QTextCursor.Document)
            cursor.insertText(new_text)

    def _fold_all(self):
        """Fold all foldable regions."""
        if hasattr(self, '_fold_regions'):
            for start, end in self._fold_regions:
                self._fold_region(start, end)

    def _unfold_all(self):
        """Unfold all folded regions."""
        if hasattr(self, '_folded_lines'):
            for line in list(self._folded_lines):
                self._unfold_region(line)


    def _on_cursor_moved(self):
        """Handle cursor position change - update bracket matching and highlight occurrences."""
        self._match_brackets()
        self._highlight_word_occurrences()

    def _highlight_word_occurrences(self):
        """Highlight all occurrences of the word under cursor."""
        cursor = self.textCursor()
        extra = list(self.extraSelections())

        # Remove old word highlights (keep bracket and current line highlights)
        extra = [sel for sel in extra if not hasattr(sel, '_is_word_highlight')]

        # Get word under cursor
        cursor.select(QTextCursor.WordUnderCursor)
        word = cursor.selectedText().strip()

        if word and len(word) >= 2 and word.isidentifier():
            import re
            text = self.toPlainText()
            pattern = r'\b' + re.escape(word) + r'\b'

            _phw = get_palette()
            highlight_color = QColor(_phw.get('line_bg', '#2a2a3c'))
            highlight_color.setAlpha(200)

            for match in re.finditer(pattern, text):
                sel = QTextEdit.ExtraSelection()
                fmt = QTextCharFormat()
                fmt.setBackground(highlight_color)
                fmt.setProperty(QTextFormat.FullWidthSelection, False)
                cursor = QTextCursor(self.document())
                cursor.setPosition(match.start())
                cursor.setPosition(match.end(), QTextCursor.KeepAnchor)
                sel.cursor = cursor
                sel.format = fmt
                sel._is_word_highlight = True
                extra.append(sel)

        self.setExtraSelections(extra)



    # ── Multi-cursor helpers ─────────────────────────────────
    def _clear_extra_cursors(self):
        """Remove all secondary cursors."""
        self._extra_cursors.clear()
        self._column_sel_anchor = None
        self.viewport().update()

    def _add_cursor_above_or_below(self, direction):
        """Add an extra cursor one line above (-1) or below (+1)."""
        cursor = self.textCursor()
        block = cursor.block()
        col = cursor.columnNumber()
        target_block = block.previous() if direction < 0 else block.next()
        if target_block.isValid():
            new_cursor = QTextCursor(target_block)
            new_cursor.movePosition(QTextCursor.StartOfBlock)
            move_count = min(col, target_block.length() - 1)
            if move_count > 0:
                new_cursor.movePosition(QTextCursor.Right, QTextCursor.MoveAnchor, move_count)
            # Check for duplicate
            pos = new_cursor.position()
            for ec in self._extra_cursors:
                if ec.position() == pos:
                    return
            self._extra_cursors.append(new_cursor)
            self.viewport().update()

    def _select_next_occurrence(self):
        """Select the next occurrence of current selection (Ctrl+D)."""
        cursor = self.textCursor()
        text = cursor.selectedText()
        if not text:
            cursor.select(QTextCursor.WordUnderCursor)
            self.setTextCursor(cursor)
            return
        # Search forward from end of current selection
        doc = self.document()
        search_cursor = doc.find(text, cursor.selectionEnd())
        if search_cursor.isNull():
            search_cursor = doc.find(text, 0)  # wrap around
        if not search_cursor.isNull() and search_cursor.position() != cursor.position():
            self._extra_cursors.append(search_cursor)
            self.viewport().update()

    def _paint_extra_cursors(self, painter):
        """Draw extra cursor indicators."""
        if not self._extra_cursors:
            return
        from PySide6.QtGui import QPen, QColor
        _pec = get_palette()
        accent = QColor(_pec.get('constant', '#f9e2af'))
        accent.setAlpha(180)
        pen = QPen(accent, 2)
        painter.setPen(pen)
        for ec in self._extra_cursors:
            block = ec.block()
            if not block.isVisible():
                continue
            rect = self.blockBoundingGeometry(block).translated(self.contentOffset())
            col = ec.columnNumber()
            char_w = self.fontMetrics().horizontalAdvance("x")
            x = rect.x() + col * char_w
            painter.drawLine(int(x), int(rect.y()), int(x), int(rect.y() + rect.height()))


    def paintEvent(self, event):
        """Override to draw indentation guides."""
        super().paintEvent(event)
        if self._extra_cursors:
            from PySide6.QtGui import QPainter
            p = QPainter(self.viewport())
            self._paint_extra_cursors(p)
            p.end()

        from PySide6.QtGui import QPainter, QColor, QPen
        painter = QPainter(self.viewport())
        _pg = get_palette()
        guide_color = QColor(_pg.get('line_bg', '#313244'))
        guide_color.setAlpha(80)
        pen = QPen(guide_color, 1, Qt.DotLine)
        painter.setPen(pen)

        tab_width = self.tabStopDistance()
        if tab_width <= 0:
            tab_width = self.fontMetrics().horizontalAdvance(' ') * 4

        block = self.firstVisibleBlock()
        top = int(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible():
                text = block.text()
                # Count leading spaces
                indent_chars = len(text) - len(text.lstrip()) if text.strip() else 0
                if indent_chars > 0:
                    char_width = self.fontMetrics().horizontalAdvance(' ')
                    tab_size = 4
                    num_guides = indent_chars // tab_size
                    for i in range(1, num_guides + 1):
                        x = int(self.contentOffset().x()) + i * tab_size * char_width
                        painter.drawLine(x, top, x, bottom)

            block = block.next()
            if block.isValid():
                top = bottom
                bottom = top + int(self.blockBoundingRect(block).height())
            else:
                break

        painter.end()

    def _highlight_current_line(self):
        p = get_palette()
        sel = QTextEdit.ExtraSelection()
        sel.format.setBackground(QColor(p["line_bg"]))
        sel.format.setProperty(QTextCharFormat.FullWidthSelection, True)
        sel.cursor = self.textCursor()
        sel.cursor.clearSelection()
        selections = [sel] + self._bracket_selections
        self.setExtraSelections(selections)

    def _match_brackets(self):
        """Highlight matching brackets with depth-based coloring."""
        extra_selections = []
        cursor = self.textCursor()
        text = self.document().toPlainText()
        pos = cursor.position()

        if pos <= 0 or pos > len(text):
            self.setExtraSelections(self._current_line_selections())
            return

        # Check character before cursor and at cursor
        pairs = {'(': ')', '[': ']', '{': '}'}
        close_to_open = {')': '(', ']': '[', '}': '{'}

        # Rainbow bracket colors (depth-cycled)
        rainbow = ['#f38ba8', '#fab387', '#f9e2af', '#a6e3a1', '#89b4fa', '#cba6f7']

        def depth_color(depth):
            return rainbow[depth % len(rainbow)]

        def find_matching_forward(start, open_char, close_char):
            depth = 1
            i = start + 1
            while i < len(text):
                if text[i] == open_char:
                    depth += 1
                elif text[i] == close_char:
                    depth -= 1
                    if depth == 0:
                        return i
                i += 1
            return -1

        def find_matching_backward(start, open_char, close_char):
            depth = 1
            i = start - 1
            while i >= 0:
                if text[i] == close_char:
                    depth += 1
                elif text[i] == open_char:
                    depth -= 1
                    if depth == 0:
                        return i
                i -= 1
            return -1

        def count_depth_at(position, char):
            """Count nesting depth at a position."""
            depth = 0
            for i in range(position):
                c = text[i]
                if c in pairs:
                    depth += 1
                elif c in close_to_open:
                    depth -= 1
            return max(0, depth)

        char_before = text[pos - 1] if pos > 0 else ''
        char_at = text[pos] if pos < len(text) else ''

        match_pos = -1
        bracket_pos = -1
        bracket_depth = 0

        if char_before in pairs:
            bracket_pos = pos - 1
            match_pos = find_matching_forward(pos - 1, char_before, pairs[char_before])
            bracket_depth = count_depth_at(pos - 1, char_before)
        elif char_before in close_to_open:
            bracket_pos = pos - 1
            match_pos = find_matching_backward(pos - 1, close_to_open[char_before], char_before)
            bracket_depth = count_depth_at(pos - 1, char_before)
        elif char_at in pairs:
            bracket_pos = pos
            match_pos = find_matching_forward(pos, char_at, pairs[char_at])
            bracket_depth = count_depth_at(pos, char_at)
        elif char_at in close_to_open:
            bracket_pos = pos
            match_pos = find_matching_backward(pos, close_to_open[char_at], char_at)
            bracket_depth = count_depth_at(pos, char_at)

        # Start with current line highlight
        extra_selections = self._current_line_selections()

        if match_pos >= 0 and bracket_pos >= 0:
            color = QColor(depth_color(bracket_depth))

            # Highlight the bracket at cursor
            sel1 = QTextEdit.ExtraSelection()
            fmt1 = QTextCharFormat()
            fmt1.setBackground(QColor("#45475a"))
            fmt1.setForeground(color)
            cursor1 = QTextCursor(self.document())
            cursor1.setPosition(bracket_pos)
            cursor1.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor)
            sel1.cursor = cursor1
            sel1.format = fmt1
            extra_selections.append(sel1)

            # Highlight the matching bracket
            sel2 = QTextEdit.ExtraSelection()
            fmt2 = QTextCharFormat()
            fmt2.setBackground(QColor("#45475a"))
            fmt2.setForeground(color)
            cursor2 = QTextCursor(self.document())
            cursor2.setPosition(match_pos)
            cursor2.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor)
            sel2.cursor = cursor2
            sel2.format = fmt2
            extra_selections.append(sel2)

        self.setExtraSelections(extra_selections)

    def _current_line_selections(self):
        """Return extra selections for current line highlight."""
        selections = []
        sel = QTextEdit.ExtraSelection()
        palette = get_palette()
        sel.format.setBackground(QColor(palette.get('line_bg', '#313244')))
        sel.format.setProperty(QTextFormat.FullWidthSelection, True)
        sel.cursor = self.textCursor()
        sel.cursor.clearSelection()
        selections.append(sel)
        return selections

    def insertFromMimeData(self, source):
        """Strip leading >> and .. prompts when pasting from command window."""
        if source.hasText():
            text = source.text()
            lines = text.split('\n')
            stripped = []
            for line in lines:
                line = re.sub(r'^\s*>>\s?', '', line)
                line = re.sub(r'^\s*\.\.\s?', '', line)
                stripped.append(line)
            from PySide6.QtCore import QMimeData
            new_source = QMimeData()
            new_source.setText('\n'.join(stripped))
            super().insertFromMimeData(new_source)
        else:
            super().insertFromMimeData(source)

    # --- modification tracking & drag-drop -----------------------------

    def _on_modification_changed(self, changed):
        self._is_modified = changed
        parent = self.parent()
        while parent:
            if hasattr(parent, 'tabs') and hasattr(parent, '_update_tab_title'):
                parent._update_tab_title(self)
                break
            parent = parent.parent()

    # --- hover tooltips for function names ----------------------------

    def event(self, event):
        """Handle tooltip events for function hover."""
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.ToolTip:
            pos = event.pos()
            cursor = self.cursorForPosition(pos)
            cursor.select(QTextCursor.WordUnderCursor)
            word = cursor.selectedText().strip()
            if word and re.match(r'^[a-zA-Z_]\w*$', word):
                doc = self._get_function_doc(word)
                if doc:
                    from PySide6.QtCore import QPoint
                    global_pos = self.mapToGlobal(pos)
                    QToolTip.showText(global_pos, doc, self)
                    return True
            QToolTip.hideText()
            return True
        return super().event(event)

    def _get_function_doc(self, name):
        """Get a short tooltip for a function name."""
        # Check if we have a registered function with a docstring
        # Walk up to find the session
        parent = self.parent()
        while parent:
            if hasattr(parent, 'session') and parent.session:
                session = parent.session
                if hasattr(session, '_engine') and name in session._engine.functions:
                    func = session._engine.functions[name]
                    doc = getattr(func, '__doc__', None)
                    if doc:
                        # Truncate for tooltip
                        lines = doc.strip().split('\n')
                        short = '\n'.join(lines[:5])
                        if len(lines) > 5:
                            short += '\n...'
                        return f"<b>{name}</b><br><pre style='font-size:10px;'>{short}</pre>"
                    return f"<b>{name}</b> — built-in function"
                break
            parent = parent.parent()

        # Check if it's a keyword
        if name in OctaveSyntaxHighlighter.KEYWORDS:
            return f"<b>{name}</b> — keyword"
        if name in OctaveSyntaxHighlighter.CONSTANTS:
            return f"<b>{name}</b> — constant"
        return None

    # ---- Code snippets -----------------------------------------------
    _SNIPPETS = {
        'fori': 'for i = 1:n\n    \nend',
        'forj': 'for j = 1:n\n    \nend',
        'ifel': 'if condition\n    \nelseif condition\n    \nelse\n    \nend',
        'ife': 'if condition\n    \nelse\n    \nend',
        'whi': 'while condition\n    \nend',
        'swi': 'switch expr\n    case val\n        \n    otherwise\n        \nend',
        'tryc': 'try\n    \ncatch err\n    disp(err.message);\nend',
        'func': "function result = name(args)\n% NAME - Description\n%\n    \nend",
        'cls': "classdef Name\n    properties\n        \n    end\n    methods\n        function obj = Name()\n        end\n    end\nend",
        'plt': "figure;\nplot(x, y);\nxlabel('X');\nylabel('Y');\ntitle('Title');\ngrid on;",
        'sub': 'subplot(2, 1, 1);\nplot();',
        'fprintf': "fprintf('%s\\n', );",
        'fopen': "fid = fopen('file.txt', 'r');\n% ...\nfclose(fid);",
    }

    def _try_expand_snippet(self):
        """Try to expand a snippet trigger word at the cursor."""
        cursor = self.textCursor()
        # Get text before cursor on current line
        cursor.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)
        line_before = cursor.selectedText()

        words = line_before.split()
        if not words:
            return False

        trigger = words[-1]
        if trigger not in self._SNIPPETS:
            return False

        snippet = self._SNIPPETS[trigger]
        leading_spaces = len(line_before) - len(line_before.lstrip())
        indent = ' ' * leading_spaces

        # Select and replace the trigger word
        cursor = self.textCursor()
        for _ in range(len(trigger)):
            cursor.movePosition(QTextCursor.Left, QTextCursor.KeepAnchor)

        lines = snippet.split('\n')
        expanded = lines[0]
        for line in lines[1:]:
            expanded += '\n' + indent + line

        cursor.insertText(expanded)
        self.setTextCursor(cursor)
        return True

        # ---- Line operations -----------------------------------------------

    def _duplicate_line(self):
        """Duplicate the current line or selection."""
        cursor = self.textCursor()
        if cursor.hasSelection():
            text = cursor.selectedText()
            cursor.clearSelection()
            cursor.insertText(text + '\n' + text)
        else:
            cursor.movePosition(QTextCursor.StartOfBlock)
            cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
            line_text = cursor.selectedText()
            cursor.movePosition(QTextCursor.EndOfBlock)
            cursor.insertText('\n' + line_text)
        self.setTextCursor(cursor)

    def _move_line(self, direction):
        """Move the current line up (-1) or down (+1)."""
        cursor = self.textCursor()
        block_num = cursor.blockNumber()
        doc = self.document()

        if direction == -1 and block_num == 0:
            return
        if direction == 1 and block_num >= doc.blockCount() - 1:
            return

        cursor.beginEditBlock()
        # Select current line
        cursor.movePosition(QTextCursor.StartOfBlock)
        cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
        line_text = cursor.selectedText()

        # Delete current line
        cursor.movePosition(QTextCursor.StartOfBlock)
        cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
        if direction == -1:
            cursor.movePosition(QTextCursor.Left, QTextCursor.KeepAnchor)
        elif direction == 1:
            cursor.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor)
        cursor.removeSelectedText()

        # Insert at new position
        if direction == -1:
            cursor.movePosition(QTextCursor.EndOfBlock)
            cursor.insertText('\n' + line_text)
        else:
            cursor.movePosition(QTextCursor.StartOfBlock)
            cursor.insertText(line_text + '\n')
            cursor.movePosition(QTextCursor.Up)

        cursor.endEditBlock()
        self.setTextCursor(cursor)

    def _delete_line(self):
        """Delete the current line."""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.StartOfBlock)
        cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
        cursor.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor)
        cursor.removeSelectedText()
        self.setTextCursor(cursor)



    def _select_line(self):
        """Select entire current line."""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.StartOfBlock)
        cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
        self.setTextCursor(cursor)

    def _select_word(self):
        """Select word under cursor."""
        cursor = self.textCursor()
        cursor.select(QTextCursor.WordUnderCursor)
        self.setTextCursor(cursor)

    def _move_to_matching_bracket(self):
        """Jump to matching bracket."""
        cursor = self.textCursor()
        pos = cursor.position()
        text = self.toPlainText()
        if pos < len(text):
            char = text[pos]
            pairs = {'(': ')', '[': ']', '{': '}', ')': '(', ']': '[', '}': '{'}
            if char in pairs:
                target = pairs[char]
                is_open = char in '([{'
                depth = 0
                if is_open:
                    for i in range(pos, len(text)):
                        if text[i] == char: depth += 1
                        elif text[i] == target: depth -= 1
                        if depth == 0:
                            cursor.setPosition(i)
                            self.setTextCursor(cursor)
                            return
                else:
                    for i in range(pos, -1, -1):
                        if text[i] == char: depth += 1
                        elif text[i] == target: depth -= 1
                        if depth == 0:
                            cursor.setPosition(i)
                            self.setTextCursor(cursor)
                            return

    def _indent_selection(self):
        """Indent selected lines by one tab stop."""
        cursor = self.textCursor()
        if not cursor.hasSelection():
            cursor.insertText("    ")
            return
        start = cursor.selectionStart()
        end = cursor.selectionEnd()
        cursor.setPosition(start)
        cursor.movePosition(QTextCursor.StartOfBlock)
        cursor.setPosition(end, QTextCursor.KeepAnchor)
        cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
        text = cursor.selectedText()
        lines = text.split('\u2029')  # Qt paragraph separator
        indented = ['    ' + line for line in lines]
        cursor.insertText('\n'.join(indented))

    def _outdent_selection(self):
        """Outdent selected lines by one tab stop."""
        cursor = self.textCursor()
        start = cursor.selectionStart()
        end = cursor.selectionEnd()
        cursor.setPosition(start)
        cursor.movePosition(QTextCursor.StartOfBlock)
        cursor.setPosition(end, QTextCursor.KeepAnchor)
        cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
        text = cursor.selectedText()
        lines = text.split('\u2029')
        outdented = []
        for line in lines:
            if line.startswith('    '):
                outdented.append(line[4:])
            elif line.startswith('\t'):
                outdented.append(line[1:])
            else:
                outdented.append(line)
        cursor.insertText('\n'.join(outdented))




    def _to_upper(self):
        """Transform selection to upper case."""
        cursor = self.textCursor()
        if cursor.hasSelection():
            text = cursor.selectedText()
            cursor.insertText(text.upper())

    def _to_lower(self):
        """Transform selection to lower case."""
        cursor = self.textCursor()
        if cursor.hasSelection():
            text = cursor.selectedText()
            cursor.insertText(text.lower())

    def _sort_lines(self):
        """Sort selected lines alphabetically."""
        cursor = self.textCursor()
        if not cursor.hasSelection():
            return
        start = cursor.selectionStart()
        end = cursor.selectionEnd()
        cursor.setPosition(start)
        cursor.movePosition(QTextCursor.StartOfBlock)
        cursor.setPosition(end, QTextCursor.KeepAnchor)
        cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
        text = cursor.selectedText()
        lines = text.split('\u2029')
        lines.sort()
        cursor.insertText('\n'.join(lines))

    def _join_lines(self):
        """Join the current line with the next line (Ctrl+J)."""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.EndOfBlock)
        cursor.movePosition(QTextCursor.NextCharacter, QTextCursor.KeepAnchor)
        # Remove the newline and leading whitespace of next line
        if cursor.hasSelection():
            next_text = cursor.selectedText()
            cursor.insertText(' ')
            # Remove leading whitespace on the joined part
            cursor = self.textCursor()
            pos = cursor.position()
            text = self.toPlainText()
            # Trim extra spaces
            while pos < len(text) - 1 and text[pos] == ' ' and text[pos + 1] == ' ':
                cursor.deleteChar()
                text = self.toPlainText()

    def _toggle_comment(self):
        """Toggle % comment on current line or selection."""
        cursor = self.textCursor()
        if cursor.hasSelection():
            start = cursor.selectionStart()
            end = cursor.selectionEnd()
            cursor.setPosition(start)
            start_block = cursor.blockNumber()
            cursor.setPosition(end)
            end_block = cursor.blockNumber()

            cursor.beginEditBlock()
            # Check if all lines are commented
            all_commented = True
            for block_num in range(start_block, end_block + 1):
                block = self.document().findBlockByNumber(block_num)
                if not block.text().lstrip().startswith('%'):
                    all_commented = False
                    break

            for block_num in range(start_block, end_block + 1):
                block = self.document().findBlockByNumber(block_num)
                cursor.setPosition(block.position())
                if all_commented:
                    # Uncomment
                    text = block.text()
                    idx = text.index('%')
                    cursor.setPosition(block.position() + idx)
                    cursor.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor)
                    if idx + 1 < len(text) and text[idx + 1] == ' ':
                        cursor.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor)
                    cursor.removeSelectedText()
                else:
                    # Comment
                    cursor.insertText('% ')

            cursor.endEditBlock()
        else:
            # Single line toggle
            cursor.movePosition(QTextCursor.StartOfBlock)
            line = cursor.block().text()
            cursor.beginEditBlock()
            if line.lstrip().startswith('%'):
                idx = line.index('%')
                cursor.setPosition(cursor.block().position() + idx)
                cursor.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor)
                if idx + 1 < len(line) and line[idx + 1] == ' ':
                    cursor.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor)
                cursor.removeSelectedText()
            else:
                cursor.movePosition(QTextCursor.StartOfBlock)
                cursor.insertText('% ')
            cursor.endEditBlock()

        # ---- Bookmarks -----------------------------------------------

    def toggle_bookmark(self):
        """Toggle bookmark on current line."""
        line = self.textCursor().blockNumber()
        if line in self._bookmarks:
            self._bookmarks.discard(line)
        else:
            self._bookmarks.add(line)
        self.viewport().update()
        self._line_area.update()

    def next_bookmark(self):
        """Jump to the next bookmark."""
        if not self._bookmarks:
            return
        current = self.textCursor().blockNumber()
        sorted_bm = sorted(self._bookmarks)
        for bm in sorted_bm:
            if bm > current:
                self._goto_line(bm + 1)
                return
        # Wrap around
        self._goto_line(sorted_bm[0] + 1)

    def prev_bookmark(self):
        """Jump to the previous bookmark."""
        if not self._bookmarks:
            return
        current = self.textCursor().blockNumber()
        sorted_bm = sorted(self._bookmarks, reverse=True)
        for bm in sorted_bm:
            if bm < current:
                self._goto_line(bm + 1)
                return
        # Wrap around
        self._goto_line(sorted_bm[0] + 1)

    def clear_bookmarks(self):
        """Clear all bookmarks."""
        self._bookmarks.clear()
        self.viewport().update()
        self._line_area.update()

    def _goto_line(self, line_number):
        """Go to a specific line number (1-based)."""
        block = self.document().findBlockByNumber(line_number - 1)
        if block.isValid():
            cursor = QTextCursor(block)
            self.setTextCursor(cursor)
            self.centerCursor()

    # ---- Code folding -----------------------------------------------

    def _detect_fold_regions(self):
        """Detect foldable regions (function, if, for, while, switch, try, classdef)."""
        import re
        text = self.toPlainText()
        lines = text.split('\n')
        fold_keywords = {'function', 'if', 'for', 'while', 'switch', 'try', 'classdef', 'properties', 'methods', 'events'}
        end_keywords = {'end', 'endfunction', 'endif', 'endfor', 'endwhile', 'endswitch', 'end_try_catch'}

        indicators = {}
        stack = []  # (keyword, line_num)

        for i, line in enumerate(lines):
            stripped = line.strip()
            # Skip comments and empty lines
            if not stripped or stripped.startswith('%') or stripped.startswith('#'):
                continue
            # Get first word
            words = stripped.split()
            if not words:
                continue
            first_word = words[0].rstrip('(')

            if first_word in fold_keywords:
                stack.append((first_word, i))
            elif first_word in end_keywords:
                if stack:
                    kw, start_line = stack.pop()
                    if i - start_line > 1:  # only fold if more than 1 line
                        indicators[start_line] = i

        # Also detect %% section markers and block comments
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('%%') or stripped.startswith('%{'):
                # Find matching end
                if stripped.startswith('%{'):
                    for j in range(i + 1, len(lines)):
                        if lines[j].strip().startswith('%}'):
                            if j - i > 1:
                                indicators[i] = j
                            break

        self._fold_indicators = indicators

    def toggle_minimap(self):
        """Toggle minimap visibility."""
        self._minimap_visible = not self._minimap_visible
        if self._minimap_visible:
            self._minimap.show()
        else:
            self._minimap.hide()
        self.resizeEvent(None)

    def _toggle_fold(self, block_number):
        """Toggle folding for a given line."""
        if block_number not in self._fold_indicators:
            return

        if block_number in self._folded_regions:
            # Unfold
            self._folded_regions.discard(block_number)
            end_line = self._fold_indicators[block_number]
            block = self.document().findBlockByNumber(block_number + 1)
            while block.isValid() and block.blockNumber() <= end_line:
                block.setVisible(True)
                block = block.next()
        else:
            # Fold
            self._folded_regions.add(block_number)
            end_line = self._fold_indicators[block_number]
            block = self.document().findBlockByNumber(block_number + 1)
            while block.isValid() and block.blockNumber() <= end_line:
                block.setVisible(False)
                block = block.next()

        self.document().markContentsDirty(0, self.document().characterCount())
        self.viewport().update()
        self.update()

    def _line_number_area_mouse_press(self, event):
        """Handle click on fold markers in the gutter."""
        block = self.firstVisibleBlock()
        top = int(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())

        while block.isValid() and top <= event.pos().y():
            if block.isVisible() and bottom >= event.pos().y():
                line_num = block.blockNumber()
                # Check if click is in the fold marker area (right side of gutter)
                gutter_width = self._line_area.width()
                if event.pos().x() > gutter_width - 18:
                    if line_num in self._fold_indicators:
                        self._toggle_fold(line_num)
                        return
                break
            block = block.next()
            top = bottom
            bottom = top + int(self.blockBoundingRect(block).height())

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path and os.path.isfile(path):
                    parent = self.parent()
                    while parent:
                        if hasattr(parent, 'open_file'):
                            parent.open_file(path)
                            break
                        parent = parent.parent()
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    # --- autocomplete --------------------------------------------------

    def _setup_completer(self):
        """Set up autocomplete with Octave keywords and builtins."""
        words = list(OctaveSyntaxHighlighter.KEYWORDS)
        words.extend(OctaveSyntaxHighlighter.BUILTINS)
        words.extend(OctaveSyntaxHighlighter.CONSTANTS)
        # Will be extended with engine functions when connected
        self._completer_words = sorted(set(words))

        completer = QCompleter(self._completer_words, self)
        completer.setWidget(self)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setCompletionMode(QCompleter.PopupCompletion)
        completer.activated.connect(self._insert_completion)
        self._completer = completer

    def set_function_names(self, names: list):
        """Update the completer with engine function names."""
        words = list(OctaveSyntaxHighlighter.KEYWORDS)
        words.extend(OctaveSyntaxHighlighter.BUILTINS)
        words.extend(OctaveSyntaxHighlighter.CONSTANTS)
        words.extend(names)
        self._completer_words = sorted(set(words))
        from PySide6.QtCore import QStringListModel
        self._completer.setModel(QStringListModel(self._completer_words))

    def _insert_completion(self, completion):
        """Insert the selected completion."""
        tc = self.textCursor()
        # Remove the partial word already typed
        tc.select(QTextCursor.WordUnderCursor)
        tc.insertText(completion)
        self.setTextCursor(tc)

    def _completion_prefix(self):
        tc = self.textCursor()
        tc.select(QTextCursor.WordUnderCursor)
        return tc.selectedText()

    # --- context menu with Help ----------------------------------------

    def contextMenuEvent(self, event):
        """Custom context menu with Help on function."""
        menu = self.createStandardContextMenu()
        cursor = self.textCursor()

        # Get word under cursor (or at click position)
        click_cursor = self.cursorForPosition(event.pos())
        click_cursor.select(QTextCursor.WordUnderCursor)
        word = click_cursor.selectedText().strip()

        if word and re.match(r'^[a-zA-Z_]\w*$', word):
            menu.addSeparator()
            help_act = QAction(f"Help on \u2018{word}\u2019", menu)
            help_act.triggered.connect(lambda: self.help_requested.emit(word))
            menu.addAction(help_act)

            # Go to definition stub
            goto_act = QAction(f"Go to Definition: {word}", menu)
            goto_act.setEnabled(False)  # stub for future
            menu.addAction(goto_act)

        # If selection, offer evaluate
        if cursor.hasSelection():
            sel = cursor.selectedText().strip()
            if sel and len(sel) < 500:
                menu.addSeparator()
                eval_act = QAction("Evaluate Selection in Command Window", menu)
                eval_act.triggered.connect(lambda: self._eval_selection(sel))
                menu.addAction(eval_act)

        menu.exec(event.globalPos())

    def _eval_selection(self, text):
        """Emit text for evaluation in command window (parent handles routing)."""
        # Walk up to find the EditorWidget/MainWindow
        parent = self.parent()
        while parent:
            if hasattr(parent, 'eval_requested'):
                parent.eval_requested.emit(text)
                return
            parent = parent.parent()

    # --- auto-indent & tab handling ------------------------------------

    def keyPressEvent(self, event: QKeyEvent):
        # Line operation shortcuts
        # Multi-cursor shortcuts
        if event.key() == Qt.Key_D and event.modifiers() == Qt.ControlModifier:
            if hasattr(self, "_extra_cursors"):
                self._select_next_occurrence()
            return
        if event.key() in (Qt.Key_Up, Qt.Key_Down) and event.modifiers() == (Qt.ControlModifier | Qt.AltModifier):
            if hasattr(self, "_extra_cursors"):
                self._add_cursor_above_or_below(-1 if event.key() == Qt.Key_Up else 1)
            return
        if event.key() == Qt.Key_Escape and hasattr(self, "_extra_cursors") and self._extra_cursors:
            self._clear_extra_cursors()
            return
        if event.key() == Qt.Key_D and event.modifiers() == Qt.ControlModifier:
            self._duplicate_line()
            return
        if event.key() == Qt.Key_Up and event.modifiers() == Qt.AltModifier:
            self._move_line(-1)
            return
        if event.key() == Qt.Key_Down and event.modifiers() == Qt.AltModifier:
            self._move_line(1)
            return
        if event.key() == Qt.Key_K and event.modifiers() == (Qt.ControlModifier | Qt.ShiftModifier):
            self._delete_line()
            return
        if event.key() == Qt.Key_Slash and event.modifiers() == Qt.ControlModifier:
            self._toggle_comment()
            return
        if event.key() == Qt.Key_J and event.modifiers() == Qt.ControlModifier:
            if hasattr(self, '_join_lines'):
                self._join_lines()
            return
        if event.key() == Qt.Key_L and event.modifiers() == Qt.ControlModifier:
            if hasattr(self, '_select_line'):
                self._select_line()
            return
        if event.key() == Qt.Key_BracketRight and event.modifiers() == Qt.ControlModifier:
            if hasattr(self, '_indent_selection'):
                self._indent_selection()
            return
        if event.key() == Qt.Key_BracketLeft and event.modifiers() == Qt.ControlModifier:
            if hasattr(self, '_outdent_selection'):
                self._outdent_selection()
            return
        if event.key() == Qt.Key_M and event.modifiers() == Qt.ControlModifier:
            if hasattr(self, '_move_to_matching_bracket'):
                self._move_to_matching_bracket()
            return
        if event.key() == Qt.Key_U and event.modifiers() == (Qt.ControlModifier | Qt.ShiftModifier):
            if hasattr(self, '_to_upper'):
                self._to_upper()
            return

        # Let completer handle its keys
        if self._completer and self._completer.popup().isVisible():
            if event.key() in (Qt.Key_Enter, Qt.Key_Return, Qt.Key_Escape,
                               Qt.Key_Tab, Qt.Key_Backtab):
                event.ignore()
                return

        if event.key() == Qt.Key_Tab:
            # If completer has a single match, complete it
            if self._completer and self._completer.completionCount() == 1:
                self._insert_completion(self._completer.currentCompletion())
                self._completer.popup().hide()
                return
            # Check for snippet expansion
            if self._try_expand_snippet():
                return
            self.insertPlainText("    ")
        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            # Auto-indent: match previous line indentation
            cursor = self.textCursor()
            block_text = cursor.block().text()
            indent = re.match(r'^(\s*)', block_text).group(1)
            # Increase indent after keywords
            stripped = block_text.strip()
            if stripped and any(stripped.startswith(kw) for kw in
                              ['if ', 'if(', 'for ', 'for(', 'while ', 'while(',
                               'function ', 'switch ', 'try', 'else', 'elseif',
                               'case ', 'otherwise', 'catch']):
                indent += "    "
            super().keyPressEvent(event)
            self.insertPlainText(indent)
        else:
            super().keyPressEvent(event)


# ======================================================================
# Status bar info widget for editor
# ======================================================================

class EditorStatusWidget(QWidget):
    """Shows line:col and encoding info for the status bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        self.lbl_pos = QLabel("Ln 1, Col 1")
        self.lbl_encoding = QLabel("UTF-8")
        self.lbl_mode = QLabel("INS")

        for lbl in (self.lbl_pos, self.lbl_encoding, self.lbl_mode):
            lbl.setStyleSheet("font-size: 11px; padding: 0 4px;")
            layout.addWidget(lbl)

    def update_position(self, line: int, col: int):
        self.lbl_pos.setText(f"Ln {line}, Col {col}")


# ======================================================================
# Tabbed editor widget
# ======================================================================

class EditorWidget(QWidget):
    """Tab container for multiple CodeEditor panes."""

    file_run_requested = Signal(str)
    help_requested = Signal(str)       # bubbled up from CodeEditor context menu
    eval_requested = Signal(str)       # evaluate selection in command window

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Find/Replace bar (hidden by default)
        from forge.gui.find_replace import FindReplaceBar

        # Breadcrumb path bar
        self._breadcrumb = BreadcrumbBar(self)
        layout.addWidget(self._breadcrumb)

        self.tabs = QTabWidget(self)
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self._close_tab)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.tabs.currentChanged.connect(self._update_breadcrumb)
        layout.addWidget(self.tabs)

        # Find bar placeholder — created per-editor on demand
        self._find_bar = None

        # Status widget
        self.status = EditorStatusWidget(self)
        layout.addWidget(self.status)

        # Start with one empty tab
        self.new_file()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------


    def _mark_tab_modified(self, modified=True):
        """Mark the current tab as modified with a dot indicator."""
        idx = self.tabs.currentIndex()
        if idx < 0:
            return
        text = self.tabs.tabText(idx)
        if modified and not text.endswith(' ●'):
            self.tabs.setTabText(idx, text + ' ●')
        elif not modified and text.endswith(' ●'):
            self.tabs.setTabText(idx, text[:-2])




    def save_current(self):
        """Save the current file. Returns the path saved to, or None."""
        idx = self.tabs.currentIndex()
        if idx < 0:
            return None

        editor = self.tabs.widget(idx)
        if not hasattr(editor, 'toPlainText'):
            return None

        path = getattr(editor, '_file_path', None)
        if not path:
            return self.save_as()

        try:
            with open(path, 'w') as f:
                f.write(editor.toPlainText())
            editor.document().setModified(False)
            # Remove modified indicator
            tab_text = self.tabs.tabText(idx)
            if tab_text.endswith(' \u25cf'):
                self.tabs.setTabText(idx, tab_text[:-2])
            return path
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Save Error", str(e))
            return None

    def save_as(self):
        """Save current file with a new name."""
        from PySide6.QtWidgets import QFileDialog
        idx = self.tabs.currentIndex()
        if idx < 0:
            return None

        editor = self.tabs.widget(idx)
        if not hasattr(editor, 'toPlainText'):
            return None

        path, _ = QFileDialog.getSaveFileName(
            self, "Save As",
            os.path.expanduser("~"),
            "M-files (*.m);;Python (*.py);;All Files (*)"
        )
        if not path:
            return None

        try:
            with open(path, 'w') as f:
                f.write(editor.toPlainText())
            # Update tab name
            fname = os.path.basename(path)
            self.tabs.setTabText(idx, fname)
            # Store path on editor
            editor._file_path = path
            # Mark as unmodified
            editor.document().setModified(False)
            if hasattr(self, '_add_to_recent'):
                self._add_to_recent(path)
            return path
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Save Error", str(e))
            return None

    def _create_welcome_tab(self):
        """Create a styled welcome page as the first tab."""
        from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QFont

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(60, 40, 60, 40)

        palette = get_palette()
        bg = palette.get('bg0', '#1e1e2e')
        fg = palette.get('fg0', '#cdd6f4')
        fg2 = palette.get('fg2', '#a6adc8')
        accent = palette.get('accent', '#cba6f7')
        bg2 = palette.get('bg2', '#313244')

        # Title
        title = QLabel("Forge")
        title.setFont(QFont("Fira Code", 32, QFont.Bold))
        title.setStyleSheet(f"color: {accent}; background: transparent; margin-bottom: 4px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Octave-Compatible Computing Environment")
        subtitle.setFont(QFont("Fira Code", 12))
        subtitle.setStyleSheet(f"color: {fg2}; background: transparent; margin-bottom: 30px;")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        # Quick actions row
        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(16)

        btn_style = f"""
            QPushButton {{
                background: {bg2};
                color: {fg};
                border: 1px solid {palette.get('border0', '#45475a')};
                border-radius: 8px;
                padding: 16px 24px;
                font-size: 13px;
                font-family: 'Fira Code';
                text-align: left;
            }}
            QPushButton:hover {{
                background: {palette.get('bg3', '#45475a')};
                border-color: {accent};
            }}
        """

        btn_new = QPushButton("  New File\n  Ctrl+N")
        btn_new.setStyleSheet(btn_style)
        btn_new.setMinimumSize(160, 70)
        btn_new.clicked.connect(self.new_file)
        actions_layout.addWidget(btn_new)

        btn_open = QPushButton("  Open File\n  Ctrl+O")
        btn_open.setStyleSheet(btn_style)
        btn_open.setMinimumSize(160, 70)
        btn_open.clicked.connect(lambda: self.parent()._on_open() if hasattr(self.parent(), '_on_open') else None)
        actions_layout.addWidget(btn_open)

        btn_quick = QPushButton("  Quick Open\n  Ctrl+P")
        btn_quick.setStyleSheet(btn_style)
        btn_quick.setMinimumSize(160, 70)
        btn_quick.clicked.connect(lambda: self.parent()._show_quick_open() if hasattr(self.parent(), '_show_quick_open') else None)
        actions_layout.addWidget(btn_quick)

        btn_palette = QPushButton("  Command Palette\n  Ctrl+Shift+P")
        btn_palette.setStyleSheet(btn_style)
        btn_palette.setMinimumSize(160, 70)
        btn_palette.clicked.connect(lambda: self.parent()._show_command_palette() if hasattr(self.parent(), '_show_command_palette') else None)
        actions_layout.addWidget(btn_palette)

        layout.addLayout(actions_layout)

        # Recent files section
        recent_label = QLabel("Recent Files")
        recent_label.setFont(QFont("Fira Code", 14, QFont.Bold))
        recent_label.setStyleSheet(f"color: {fg}; background: transparent; margin-top: 30px;")
        layout.addWidget(recent_label)

        # Load recent files from history
        recent_files = self._get_recent_files()
        if recent_files:
            for fpath in recent_files[:8]:
                fname = os.path.basename(fpath)
                fdir = os.path.dirname(fpath)
                link = QPushButton(f"  {fname}  —  {fdir}")
                link.setStyleSheet(f"""
                    QPushButton {{
                        background: transparent;
                        color: {palette.get('accent', '#89b4fa')};
                        border: none;
                        text-align: left;
                        padding: 4px 8px;
                        font-size: 12px;
                    }}
                    QPushButton:hover {{
                        color: {accent};
                        text-decoration: underline;
                    }}
                """)
                link.setCursor(Qt.PointingHandCursor)
                link.clicked.connect(lambda checked, p=fpath: self.open_file(p))
                layout.addWidget(link)
        else:
            no_recent = QLabel("  No recent files")
            no_recent.setStyleSheet(f"color: {fg2}; background: transparent; font-size: 12px;")
            layout.addWidget(no_recent)

        layout.addStretch()

        # Version info
        version_label = QLabel("Forge v0.1.0  ·  1,303 functions  ·  352 tests passing")
        version_label.setFont(QFont("Fira Code", 10))
        version_label.setStyleSheet(f"color: {fg2}; background: transparent;")
        version_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(version_label)

        page.setStyleSheet(f"background: {bg};")

        # Add as tab
        idx = self.tabs.addTab(page, "Welcome")
        self.tabs.setCurrentIndex(idx)

    def _get_recent_files(self):
        """Get list of recently opened files."""
        import json
        recent_path = os.path.expanduser("~/.forge/recent_files.json")
        if os.path.exists(recent_path):
            try:
                with open(recent_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _add_to_recent(self, filepath):
        """Add a file to the recent files list."""
        import json
        recent_path = os.path.expanduser("~/.forge/recent_files.json")
        recent = self._get_recent_files()
        # Remove if already in list, add to front
        filepath = os.path.abspath(filepath)
        recent = [f for f in recent if f != filepath]
        recent.insert(0, filepath)
        recent = recent[:20]  # Keep last 20
        os.makedirs(os.path.dirname(recent_path), exist_ok=True)
        try:
            with open(recent_path, 'w') as f:
                json.dump(recent, f)
        except Exception:
            pass

    def new_file(self):
        """Open a new untitled editor tab."""
        editor = CodeEditor(self)
        editor.cursorPositionChanged.connect(self._update_status)
        editor.help_requested.connect(self.help_requested.emit)
        # Feed engine function names for autocomplete
        if hasattr(self, '_engine_func_names'):
            editor.set_function_names(self._engine_func_names)
        idx = self.tabs.addTab(editor, "untitled")
        self.tabs.setTabIcon(self.tabs.count() - 1, self._icon_for_file(""))
        self.tabs.setCurrentIndex(idx)
        return editor


    # -- file type icons --
    def _icon_for_file(self, filepath):
        """Return a QIcon based on file extension."""
        from PySide6.QtWidgets import QStyle
        from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor, QBrush
        from PySide6.QtCore import Qt

        ext = os.path.splitext(filepath)[1].lower() if filepath else ""

        # Map extensions to colors for simple colored-circle icons
        ext_colors = {
            ".py": "#3572A5",   # Python blue
            ".m": "#E44D26",    # MATLAB/function orange
            ".mat": "#E44D26",
            ".txt": "#888888",  # gray
            ".md": "#519ABA",   # markdown teal
            ".json": "#F1E05A", # json yellow
            ".yaml": "#CB171E", # red
            ".yml": "#CB171E",
            ".xml": "#F34B7D",  # pink
            ".html": "#E44D26", # orange
            ".css": "#563D7C",  # purple
            ".js": "#F1E05A",   # yellow
            ".ts": "#2B7489",   # teal
            ".c": "#555555",
            ".cpp": "#F34B7D",
            ".h": "#999999",
            ".rs": "#DEA584",   # rust
            ".go": "#00ADD8",   # go
            ".sh": "#89E051",   # shell green
            ".bat": "#C1F12E",
            ".log": "#AAAAAA",
            ".cfg": "#AAAAAA",
            ".ini": "#AAAAAA",
            ".toml": "#9C4121",
        }

        color_hex = ext_colors.get(ext, "#CCCCCC")

        # Create a small colored circle icon (16x16)
        pixmap = QPixmap(16, 16)
        pixmap.fill(Qt.transparent)
        p = QPainter(pixmap)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QBrush(QColor(color_hex)))
        p.setPen(Qt.NoPen)
        p.drawEllipse(2, 2, 12, 12)
        p.end()

        return QIcon(pixmap)


    def open_file(self, path: str):
        """Open *path* in a new tab (or focus if already open)."""
        for i in range(self.tabs.count()):
            ed = self.tabs.widget(i)
            if isinstance(ed, CodeEditor) and ed.file_path == path:
                self.tabs.setCurrentIndex(i)
                return ed

        editor = CodeEditor(self)
        editor.cursorPositionChanged.connect(self._update_status)
        editor.help_requested.connect(self.help_requested.emit)
        if hasattr(self, '_engine_func_names'):
            editor.set_function_names(self._engine_func_names)
        editor.file_path = path
        with open(path, "r", encoding="utf-8") as fh:
            editor.setPlainText(fh.read())
        name = os.path.basename(path)
        idx = self.tabs.addTab(editor, name)
        if hasattr(self, "_icon_for_file"):
            self.tabs.setTabIcon(idx, self._icon_for_file(path))
        self.tabs.setCurrentIndex(idx)
        return editor

    def save_file(self):
        """Save the current tab to its file_path (no-op if untitled)."""
        editor = self.get_current_editor()
        if editor is None or editor.file_path is None:
            return False
        with open(editor.file_path, "w", encoding="utf-8") as fh:
            fh.write(editor.toPlainText())
        editor.document().setModified(False)
        return True

    def _update_breadcrumb(self, *args):
        """Update breadcrumb bar with current file path and symbol."""
        editor = self.get_current_editor()
        if editor and hasattr(editor, 'file_path') and editor.file_path:
            # Get current symbol
            sym_name = None
            try:
                text = editor.toPlainText()
                symbols = parse_symbols(text)
                line = editor.textCursor().blockNumber() + 1
                sym_name = symbol_at_line(symbols, line)
            except Exception:
                pass
            self._breadcrumb.update_path(editor.file_path, sym_name)
            self._breadcrumb.show()
        else:
            self._breadcrumb.hide()

    def go_to_symbol(self):
        """Open the Go to Symbol dialog for the current editor."""
        editor = self.get_current_editor()
        if not editor or not hasattr(editor, 'toPlainText'):
            return
        symbols = parse_symbols(editor.toPlainText())
        if not symbols:
            return
        dlg = GoToSymbolDialog(symbols, self)
        if dlg.exec() and dlg.selected_line is not None:
            from PySide6.QtGui import QTextCursor
            block = editor.document().findBlockByNumber(dlg.selected_line - 1)
            cursor = QTextCursor(block)
            editor.setTextCursor(cursor)
            editor.centerCursor()

    def get_current_editor(self) -> CodeEditor | None:
        w = self.tabs.currentWidget()
        return w if isinstance(w, CodeEditor) else None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_tab_title(self, editor):
        """Update tab title with asterisk for unsaved changes."""
        for i in range(self.tabs.count()):
            if self.tabs.widget(i) is editor:
                base = os.path.basename(editor.file_path) if editor.file_path else "untitled"
                if editor._is_modified:
                    self.tabs.setTabText(i, f"* {base}")
                else:
                    self.tabs.setTabText(i, base)
                break

    def _auto_save_all(self):
        """Auto-save all modified files to a recovery location."""
        import os, json
        recovery_dir = os.path.join(os.path.expanduser("~"), ".forge", "recovery")
        os.makedirs(recovery_dir, exist_ok=True)
        self.save_session()


    # -- session restore --
    def save_session(self):
        """Save open file paths and active tab index to ~/.forge/session.json."""
        import json
        session_dir = os.path.expanduser("~/.forge")
        os.makedirs(session_dir, exist_ok=True)
        session_file = os.path.join(session_dir, "session.json")

        files = []
        for i in range(self.tabs.count()):
            editor = self.tabs.widget(i)
            if editor and hasattr(editor, "file_path") and editor.file_path:
                files.append(editor.file_path)

        data = {
            "files": files,
            "active_tab": self.tabs.currentIndex(),
        }
        try:
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def restore_session(self):
        """Restore previously open files from ~/.forge/session.json."""
        import json
        session_file = os.path.expanduser("~/.forge/session.json")
        if not os.path.exists(session_file):
            return
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        for path in data.get("files", []):
            if os.path.isfile(path):
                self.open_file(path)

        idx = data.get("active_tab", 0)
        if 0 <= idx < self.tabs.count():
            self.tabs.setCurrentIndex(idx)


        for i in range(self.tabs.count()):
            editor = self.tabs.widget(i)
            if hasattr(editor, 'document') and editor.document().isModified():
                if hasattr(editor, 'file_path') and editor.file_path:
                    # Save recovery copy
                    safe_name = editor.file_path.replace(os.sep, '_').replace('/', '_')
                    recovery_path = os.path.join(recovery_dir, safe_name + '.recovery')
                    try:
                        with open(recovery_path, 'w') as f:
                            f.write(editor.toPlainText())
                    except Exception:
                        pass

    def _close_tab(self, index: int):
        if self.tabs.count() > 1:
            # Check for unsaved changes
            editor = self.tabs.widget(index)
            if isinstance(editor, CodeEditor) and editor._is_modified:
                from PySide6.QtWidgets import QMessageBox
                reply = QMessageBox.question(
                    self, "Unsaved Changes",
                    f"Save changes to {os.path.basename(editor.file_path) if editor.file_path else 'untitled'}?",
                    QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
                )
                if reply == QMessageBox.Cancel:
                    return
                if reply == QMessageBox.Save:
                    if editor.file_path:
                        with open(editor.file_path, "w", encoding="utf-8") as f:
                            f.write(editor.toPlainText())
            self.tabs.removeTab(index)

    def _on_tab_changed(self, index):
        self._update_status()

    def _update_status(self):
        editor = self.get_current_editor()
        if editor:
            cursor = editor.textCursor()
            line = cursor.blockNumber() + 1
            col = cursor.columnNumber() + 1
            self.status.update_position(line, col)

    def show_find(self):
        """Show the find/replace bar for the current editor."""
        editor = self.get_current_editor()
        if editor is None:
            return
        from forge.gui.find_replace import FindReplaceBar
        if self._find_bar is None:
            self._find_bar = FindReplaceBar(editor, self)
            self.layout().insertWidget(1, self._find_bar)  # after tabs
            self._find_bar.closed.connect(lambda: self._find_bar.setVisible(False))
        else:
            self._find_bar.editor = editor
        # Pre-fill with selection
        sel = editor.textCursor().selectedText()
        self._find_bar.show_find(sel)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_F5:
            editor = self.get_current_editor()
            if editor and editor.file_path:
                self.file_run_requested.emit(editor.file_path)
        else:
            super().keyPressEvent(event)

        # Trigger autocomplete
        if self._completer and event.text() and event.text().isalpha():
            prefix = self._completion_prefix()
            if len(prefix) >= 2:
                self._completer.setCompletionPrefix(prefix)
                if self._completer.completionCount() > 0:
                    popup = self._completer.popup()
                    cr = self.cursorRect()
                    cr.setWidth(popup.sizeHintForColumn(0) +
                               popup.verticalScrollBar().sizeHint().width() + 20)
                    self._completer.complete(cr)
                else:
                    self._completer.popup().hide()
            else:
                self._completer.popup().hide()
        self._draw_indent_guides(event)


    # -- indent guides --
    def _draw_indent_guides(self, event):
        """Draw subtle vertical lines at each indentation level."""
        painter = QPainter(self.viewport())
        _pdg = get_palette()
        color = QColor(_pdg.get('line_bg', '#313244'))
        color.setAlpha(40)
        painter.setPen(color)

        block = self.firstVisibleBlock()
        font_metrics = self.fontMetrics()
        space_width = font_metrics.horizontalAdvance(" ")
        tab_stop = 4  # spaces per indent level

        while block.isValid():
            geom = self.blockBoundingGeometry(block).translated(self.contentOffset())
            if geom.top() > event.rect().bottom():
                break

            text = block.text()
            if text:
                # Count leading spaces
                stripped = text.lstrip(" ")
                leading = len(text) - len(stripped)
                indent_levels = leading // tab_stop

                for level in range(1, indent_levels + 1):
                    x = int(level * tab_stop * space_width) + self.contentOffset().x()
                    top = int(geom.top())
                    bottom = int(geom.bottom())
                    painter.drawLine(x, top, x, bottom)

            block = block.next()

        painter.end()

