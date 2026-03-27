"""Forge code editor widget with tabs and syntax highlighting
(forge/gui/editor_widget.py)."""

import os
import re

from PySide6.QtCore import Qt, Signal, QRect, QSize, QStringListModel
from PySide6.QtGui import (
    QColor, QFont, QPainter, QSyntaxHighlighter, QTextCharFormat,
    QTextCursor, QKeyEvent, QAction, QKeySequence,
)
from PySide6.QtWidgets import (
    QTextEdit, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QPlainTextEdit, QLabel, QDialog, QLineEdit, QListView,
    QAbstractItemView, QStyledItemDelegate, QStyle,
    QStyleOptionViewItem,
)


# ======================================================================
# Symbol parsing helpers
# ======================================================================

_FUNCTION_RE = re.compile(
    r"^\s*function\b.*?(?:=\s*)?(\w+)\s*\(", re.MULTILINE
)
_SECTION_RE = re.compile(
    r"^%%\s*(.*)", re.MULTILINE
)


def parse_symbols(text: str) -> list[dict]:
    """Parse function definitions and %% section headers from Octave/Forge code.

    Returns a list of dicts: {"name": str, "line": int, "kind": "function"|"section"}.
    """
    symbols = []
    for m in _FUNCTION_RE.finditer(text):
        line = text[:m.start()].count("\n") + 1
        symbols.append({"name": m.group(1), "line": line, "kind": "function"})
    for m in _SECTION_RE.finditer(text):
        line = text[:m.start()].count("\n") + 1
        title = m.group(1).strip() or "(untitled section)"
        symbols.append({"name": title, "line": line, "kind": "section"})
    symbols.sort(key=lambda s: s["line"])
    return symbols


def symbol_at_line(symbols: list[dict], line: int) -> str | None:
    """Return the name of the symbol that encloses *line*, or None."""
    current = None
    for s in symbols:
        if s["line"] <= line:
            current = s
        else:
            break
    return current["name"] if current else None


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
        # Keywords -- bold blue
        kw_fmt = QTextCharFormat()
        kw_fmt.setForeground(QColor("blue"))
        kw_fmt.setFontWeight(QFont.Bold)
        kw_pattern = r"\b(" + "|".join(self.KEYWORDS) + r")\b"
        self._rules.append((re.compile(kw_pattern), kw_fmt))

        # Numbers -- dark cyan
        num_fmt = QTextCharFormat()
        num_fmt.setForeground(QColor("darkcyan"))
        self._rules.append((re.compile(r"\b\d+\.?\d*([eE][+-]?\d+)?\b"), num_fmt))

        # Strings (single-quoted) -- red
        str_fmt = QTextCharFormat()
        str_fmt.setForeground(QColor("red"))
        self._rules.append((re.compile(r"'[^']*'"), str_fmt))
        # Double-quoted strings
        self._rules.append((re.compile(r'"[^"]*"'), str_fmt))

        # Comments -- green (must come last so it overrides)
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

    cursor_line_changed = Signal(int)   # emits 1-based line number

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
        self.cursorPositionChanged.connect(self._emit_cursor_line)
        self._update_line_area_width(0)

        # Syntax highlighter
        self._highlighter = OctaveSyntaxHighlighter(self.document())

        self._highlight_current_line()
        self._last_line = -1

    # --- cursor line signal -----------------------------------------------

    def _emit_cursor_line(self):
        line = self.textCursor().blockNumber() + 1
        if line != self._last_line:
            self._last_line = line
            self.cursor_line_changed.emit(line)

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
# Clickable breadcrumb label
# ======================================================================

class _BreadcrumbSegment(QLabel):
    """A single clickable segment in the breadcrumb bar."""

    clicked = Signal(str)  # emits the absolute path this segment represents

    def __init__(self, text: str, path: str, is_last: bool = False, parent=None):
        super().__init__(text, parent)
        self._path = path
        self._is_last = is_last
        self.setCursor(Qt.PointingHandCursor)
        weight = "bold" if is_last else "normal"
        color = "#424242" if is_last else "#616161"
        self.setStyleSheet(
            f"QLabel {{ color: {color}; font-weight: {weight}; padding: 0 2px;"
            f"  font-size: 11px; }}"
            f"QLabel:hover {{ color: #1976d2; text-decoration: underline; }}"
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self._path)
        super().mousePressEvent(event)


class BreadcrumbBar(QWidget):
    """A breadcrumb bar showing the path to the current file with clickable segments."""

    directory_requested = Signal(str)  # emitted when a path segment is clicked

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(8, 2, 8, 2)
        self._layout.setSpacing(0)
        self._layout.addStretch()
        self.setFixedHeight(24)
        self.setStyleSheet(
            "BreadcrumbBar { background-color: #f5f5f5;"
            " border-bottom: 1px solid #e0e0e0; }"
        )

    def _clear(self):
        """Remove all widgets from the layout."""
        while self._layout.count():
            item = self._layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _add_separator(self):
        sep = QLabel("\u203a")  # single right-pointing angle quotation mark
        sep.setStyleSheet("color: #bdbdbd; font-size: 11px; padding: 0 1px;")
        self._layout.addWidget(sep)

    def update_path(self, file_path: str | None, symbol_name: str | None = None):
        """Rebuild the breadcrumb for *file_path*, optionally with a symbol suffix."""
        self._clear()

        if not file_path:
            placeholder = QLabel("No file open")
            placeholder.setStyleSheet("color: #9e9e9e; font-size: 11px; padding: 0 2px;")
            self._layout.addWidget(placeholder)
            self._layout.addStretch()
            return

        # Collect ancestor directories
        head = os.path.dirname(file_path)
        fname = os.path.basename(file_path)
        ancestors = []
        cur = head
        for _ in range(8):
            name = os.path.basename(cur)
            if not name:
                break
            ancestors.append((name, cur))
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
        ancestors.reverse()

        # If too many ancestors, truncate with ellipsis
        if len(ancestors) > 4:
            shown = ancestors[-4:]
            ellipsis = QLabel("\u2026")
            ellipsis.setStyleSheet("color: #9e9e9e; font-size: 11px; padding: 0 2px;")
            self._layout.addWidget(ellipsis)
            self._add_separator()
        else:
            shown = ancestors

        for name, path in shown:
            seg = _BreadcrumbSegment(name, path, is_last=False, parent=self)
            seg.clicked.connect(self.directory_requested.emit)
            self._layout.addWidget(seg)
            self._add_separator()

        # File name segment (bold)
        file_seg = _BreadcrumbSegment(fname, file_path, is_last=True, parent=self)
        file_seg.clicked.connect(self.directory_requested.emit)
        self._layout.addWidget(file_seg)

        # Symbol suffix (current function/section)
        if symbol_name:
            self._add_separator()
            sym_label = QLabel(symbol_name)
            sym_label.setStyleSheet(
                "color: #6a1b9a; font-size: 11px; font-style: italic; padding: 0 2px;"
            )
            self._layout.addWidget(sym_label)

        self._layout.addStretch()


# ======================================================================
# Go to Symbol dialog
# ======================================================================

class _SymbolDelegate(QStyledItemDelegate):
    """Custom delegate that renders icon + name + line number for each symbol."""

    def __init__(self, symbols: list[dict], parent=None):
        super().__init__(parent)
        self._symbols = {s["name"]: s for s in symbols}

    def update_symbols(self, symbols: list[dict]):
        self._symbols = {s["name"]: s for s in symbols}

    def paint(self, painter, option, index):
        painter.save()
        text = index.data(Qt.DisplayRole)
        sym = self._symbols.get(text)

        # Selection / hover background
        if option.state & QStyle.State_Selected:
            painter.fillRect(option.rect, QColor("#bbdefb"))
        elif option.state & QStyle.State_MouseOver:
            painter.fillRect(option.rect, QColor("#e3f2fd"))

        rect = option.rect.adjusted(6, 0, 0, 0)

        # Icon: "f" for function, section-sign for section
        icon_rect = rect.adjusted(0, 0, -rect.width() + 20, 0)
        if sym and sym["kind"] == "function":
            painter.setPen(QColor("#6a1b9a"))
            painter.drawText(icon_rect, Qt.AlignCenter, "f")
        elif sym and sym["kind"] == "section":
            painter.setPen(QColor("#00695c"))
            painter.drawText(icon_rect, Qt.AlignCenter, "\u00a7")

        # Name
        name_rect = rect.adjusted(24, 0, -60, 0)
        painter.setPen(QColor("#212121"))
        painter.drawText(name_rect, Qt.AlignVCenter | Qt.AlignLeft, text or "")

        # Line number on the right
        if sym:
            line_rect = rect.adjusted(rect.width() - 60, 0, -6, 0)
            painter.setPen(QColor("#9e9e9e"))
            painter.drawText(
                line_rect, Qt.AlignVCenter | Qt.AlignRight, f":{sym['line']}"
            )

        painter.restore()

    def sizeHint(self, option, index):
        return QSize(option.rect.width(), 26)


class GoToSymbolDialog(QDialog):
    """Dialog listing all symbols in the current file with type-to-filter.

    Similar to VS Code's Ctrl+Shift+O / @ symbol navigation.
    """

    symbol_selected = Signal(int)  # emits 1-based line number

    def __init__(self, symbols: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Go to Symbol")
        self.setMinimumSize(400, 320)
        self.resize(450, 380)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        self._symbols = symbols
        self._filtered = list(symbols)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Search field
        self._search = QLineEdit(self)
        self._search.setPlaceholderText("Type to filter symbols...")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._filter)
        layout.addWidget(self._search)

        # Symbol list
        self._list = QListView(self)
        self._model = QStringListModel(self)
        self._model.setStringList([s["name"] for s in symbols])
        self._list.setModel(self._model)
        self._list.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._delegate = _SymbolDelegate(symbols, self)
        self._list.setItemDelegate(self._delegate)
        self._list.doubleClicked.connect(self._accept_selection)
        self._list.setStyleSheet(
            "QListView { border: 1px solid #e0e0e0; border-radius: 4px; }"
            "QListView::item { padding: 2px 0; }"
        )
        layout.addWidget(self._list)

        # Hint label
        hint = QLabel("Enter to navigate  \u00b7  Esc to cancel")
        hint.setStyleSheet("color: #9e9e9e; font-size: 11px;")
        hint.setAlignment(Qt.AlignCenter)
        layout.addWidget(hint)

        # Select first item
        if symbols:
            self._list.setCurrentIndex(self._model.index(0, 0))

        self._search.setFocus()

    def _filter(self, text: str):
        text_lower = text.lower()
        self._filtered = [
            s for s in self._symbols if text_lower in s["name"].lower()
        ]
        self._model.setStringList([s["name"] for s in self._filtered])
        self._delegate.update_symbols(self._filtered)
        if self._filtered:
            self._list.setCurrentIndex(self._model.index(0, 0))

    def _accept_selection(self, _index=None):
        idx = self._list.currentIndex()
        if idx.isValid() and idx.row() < len(self._filtered):
            line = self._filtered[idx.row()]["line"]
            self.symbol_selected.emit(line)
            self.accept()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._accept_selection()
        elif event.key() == Qt.Key_Down:
            idx = self._list.currentIndex()
            new_row = min(idx.row() + 1, len(self._filtered) - 1)
            self._list.setCurrentIndex(self._model.index(new_row, 0))
        elif event.key() == Qt.Key_Up:
            idx = self._list.currentIndex()
            new_row = max(idx.row() - 1, 0)
            self._list.setCurrentIndex(self._model.index(new_row, 0))
        else:
            super().keyPressEvent(event)


# ======================================================================
# Tabbed editor widget
# ======================================================================

class EditorWidget(QWidget):
    """Tab container for multiple CodeEditor panes."""

    file_run_requested = Signal(str)
    directory_open_requested = Signal(str)  # when a breadcrumb segment is clicked

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Breadcrumb bar (above tabs)
        self._breadcrumb = BreadcrumbBar(self)
        self._breadcrumb.directory_requested.connect(self._on_breadcrumb_click)
        layout.addWidget(self._breadcrumb)

        self.tabs = QTabWidget(self)
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self._close_tab)
        self.tabs.currentChanged.connect(self._update_breadcrumb)
        layout.addWidget(self.tabs)

        # Cache of parsed symbols keyed by editor object id
        self._symbol_cache: dict[int, list[dict]] = {}

        # Start with one empty tab
        self.new_file()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def new_file(self):
        """Open a new untitled editor tab."""
        editor = CodeEditor(self)
        editor.cursor_line_changed.connect(self._on_cursor_line_changed)
        editor.textChanged.connect(lambda ed=editor: self._invalidate_symbols(ed))
        idx = self.tabs.addTab(editor, "untitled")
        self.tabs.setCurrentIndex(idx)
        return editor

    def open_file(self, path: str):
        """Open *path* in a new tab (or focus if already open)."""
        for i in range(self.tabs.count()):
            ed = self.tabs.widget(i)
            if isinstance(ed, CodeEditor) and ed.file_path == path:
                self.tabs.setCurrentIndex(i)
                return ed

        editor = CodeEditor(self)
        editor.cursor_line_changed.connect(self._on_cursor_line_changed)
        editor.textChanged.connect(lambda ed=editor: self._invalidate_symbols(ed))
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

    def go_to_symbol(self):
        """Open the Go to Symbol dialog for the current editor."""
        editor = self.get_current_editor()
        if editor is None:
            return
        symbols = self._get_symbols(editor)
        if not symbols:
            return
        dlg = GoToSymbolDialog(symbols, self)
        dlg.symbol_selected.connect(lambda line: self._goto_line(editor, line))
        dlg.exec()

    # ------------------------------------------------------------------
    # Breadcrumb / symbol helpers
    # ------------------------------------------------------------------

    def _get_symbols(self, editor: CodeEditor) -> list[dict]:
        eid = id(editor)
        if eid not in self._symbol_cache:
            self._symbol_cache[eid] = parse_symbols(editor.toPlainText())
        return self._symbol_cache[eid]

    def _invalidate_symbols(self, editor: CodeEditor):
        self._symbol_cache.pop(id(editor), None)

    def _update_breadcrumb(self, _index=None):
        editor = self.get_current_editor()
        if editor is None:
            self._breadcrumb.update_path(None)
            return
        path = editor.file_path
        sym_name = None
        if path:
            symbols = self._get_symbols(editor)
            line = editor.textCursor().blockNumber() + 1
            sym_name = symbol_at_line(symbols, line)
        self._breadcrumb.update_path(path, sym_name)

    def _on_cursor_line_changed(self, line: int):
        """Re-evaluate which symbol the cursor is inside and update breadcrumb."""
        editor = self.get_current_editor()
        if editor is None or editor.file_path is None:
            return
        symbols = self._get_symbols(editor)
        sym_name = symbol_at_line(symbols, line)
        self._breadcrumb.update_path(editor.file_path, sym_name)

    def _on_breadcrumb_click(self, path: str):
        """Handle a breadcrumb segment click -- open directory in file browser."""
        if os.path.isdir(path):
            self.directory_open_requested.emit(path)
        elif os.path.isfile(path):
            self.directory_open_requested.emit(os.path.dirname(path))

    def _goto_line(self, editor: CodeEditor, line: int):
        """Move the cursor to the given 1-based line number."""
        block = editor.document().findBlockByLineNumber(line - 1)
        cursor = QTextCursor(block)
        editor.setTextCursor(cursor)
        editor.centerCursor()
        editor.setFocus()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _close_tab(self, index: int):
        if self.tabs.count() > 1:
            editor = self.tabs.widget(index)
            if editor:
                self._symbol_cache.pop(id(editor), None)
            self.tabs.removeTab(index)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_F5:
            editor = self.get_current_editor()
            if editor and editor.file_path:
                self.file_run_requested.emit(editor.file_path)
        else:
            super().keyPressEvent(event)
