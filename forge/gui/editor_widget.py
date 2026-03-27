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

        # Enable drag & drop
        self.setAcceptDrops(True)

        # Autocomplete
        self._completer = None
        self._setup_completer()

        # Bracket matching
        self._bracket_selections = []

        self._highlight_current_line()

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
        painter.end()

    # --- current line highlight & bracket matching ---------------------

    def _on_cursor_moved(self):
        self._highlight_current_line()
        self._match_brackets()
        self._line_area.update()  # repaint gutter for active line

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
        """Highlight matching bracket pair."""
        self._bracket_selections = []
        cursor = self.textCursor()
        doc = self.document()
        pos = cursor.position()
        text_at = doc.characterAt(pos)

        OPEN = "([{"
        CLOSE = ")]}"
        PAIRS = dict(zip(OPEN, CLOSE))
        RPAIRS = dict(zip(CLOSE, OPEN))

        match_pos = -1

        if text_at in OPEN:
            target = PAIRS[text_at]
            depth = 0
            for i in range(pos, doc.characterCount()):
                c = doc.characterAt(i)
                if c == text_at:
                    depth += 1
                elif c == target:
                    depth -= 1
                    if depth == 0:
                        match_pos = i
                        break
        elif text_at in CLOSE:
            target = RPAIRS[text_at]
            depth = 0
            for i in range(pos, -1, -1):
                c = doc.characterAt(i)
                if c == text_at:
                    depth += 1
                elif c == target:
                    depth -= 1
                    if depth == 0:
                        match_pos = i
                        break
        # Also check char before cursor
        elif pos > 0:
            text_before = doc.characterAt(pos - 1)
            if text_before in CLOSE:
                target = RPAIRS[text_before]
                depth = 0
                for i in range(pos - 1, -1, -1):
                    c = doc.characterAt(i)
                    if c == text_before:
                        depth += 1
                    elif c == target:
                        depth -= 1
                        if depth == 0:
                            match_pos = i
                            pos = pos - 1  # highlight the close bracket
                            break
            elif text_before in OPEN:
                target = PAIRS[text_before]
                depth = 0
                for i in range(pos - 1, doc.characterCount()):
                    c = doc.characterAt(i)
                    if c == text_before:
                        depth += 1
                    elif c == target:
                        depth -= 1
                        if depth == 0:
                            match_pos = i
                            pos = pos - 1
                            break

        if match_pos >= 0:
            p = get_palette()
            fmt = QTextCharFormat()
            fmt.setBackground(QColor(p.get("keyword", "#cba6f7")).replace(")", ""))
            # Use a subtle underline + bold for matched brackets
            fmt.setFontWeight(QFont.Bold)
            fmt.setForeground(QColor("#ffffff"))
            fmt.setBackground(QColor("#4a4a6a"))

            for bpos in (pos, match_pos):
                sel = QTextEdit.ExtraSelection()
                sel.format = fmt
                c = QTextCursor(doc)
                c.setPosition(bpos)
                c.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor)
                sel.cursor = c
                self._bracket_selections.append(sel)

    # --- prompt stripping on paste (MATLAB-style) --------------------

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

        self.tabs = QTabWidget(self)
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self._close_tab)
        self.tabs.currentChanged.connect(self._on_tab_changed)
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

    def new_file(self):
        """Open a new untitled editor tab."""
        editor = CodeEditor(self)
        editor.cursorPositionChanged.connect(self._update_status)
        editor.help_requested.connect(self.help_requested.emit)
        # Feed engine function names for autocomplete
        if hasattr(self, '_engine_func_names'):
            editor.set_function_names(self._engine_func_names)
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
        editor.cursorPositionChanged.connect(self._update_status)
        editor.help_requested.connect(self.help_requested.emit)
        if hasattr(self, '_engine_func_names'):
            editor.set_function_names(self._engine_func_names)
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
        editor.document().setModified(False)
        return True

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
