"""Forge command widget — single-pane interactive REPL (forge/gui/command_widget.py).

Behaves like MATLAB/Octave command window: a single text area where output and
input live together.  The user types after the ">> " prompt at the bottom.

Supports multi-line control structures:  typing "for", "if", "while", "switch",
"try", or "function" at the >> prompt automatically enters block mode.  Lines
are collected with a ".." continuation prompt until the matching "end" brings
the nesting depth back to zero, at which point the whole block is executed.

The legacy "..." line-continuation syntax is also preserved.
"""

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import (
    QFont, QTextCursor, QColor, QTextCharFormat,
    QSyntaxHighlighter, QTextDocument, QAction,
)
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPlainTextEdit, QMenu, QApplication

import re as _re


# ======================================================================
# Syntax highlighter
# ======================================================================

class _MCodeHighlighter(QSyntaxHighlighter):
    """Rich M-code / Octave syntax highlighting."""

    # -- token sets ---------------------------------------------------
    KEYWORDS = {
        "for", "end", "if", "else", "elseif", "while", "do", "until",
        "switch", "case", "otherwise", "try", "catch", "function",
        "return", "break", "continue", "global", "persistent",
        "endfor", "endif", "endwhile", "endswitch", "endtry",
        "unwind_protect", "unwind_protect_cleanup", "end_unwind_protect",
    }

    CONSTANTS = {
        "pi", "inf", "Inf", "nan", "NaN", "eps", "true", "false",
        "i", "j", "e",
    }

    def __init__(self, parent=None):
        super().__init__(parent)

        # --- formats --------------------------------------------------
        self._kw_fmt = QTextCharFormat()
        self._kw_fmt.setForeground(QColor("#569cd6"))       # blue
        self._kw_fmt.setFontWeight(QFont.Bold)

        self._const_fmt = QTextCharFormat()
        self._const_fmt.setForeground(QColor("#4ec9b0"))     # teal
        self._const_fmt.setFontWeight(QFont.Bold)

        self._str_fmt = QTextCharFormat()
        self._str_fmt.setForeground(QColor("#ce9178"))       # orange

        self._num_fmt = QTextCharFormat()
        self._num_fmt.setForeground(QColor("#b5cea8"))       # green

        self._cmt_fmt = QTextCharFormat()
        self._cmt_fmt.setForeground(QColor("#6a9955"))       # olive-green
        self._cmt_fmt.setFontItalic(True)

        self._op_fmt = QTextCharFormat()
        self._op_fmt.setForeground(QColor("#d4d4d4"))        # light gray

        self._fn_fmt = QTextCharFormat()
        self._fn_fmt.setForeground(QColor("#dcdcaa"))        # yellow

        self._paren_fmt = QTextCharFormat()
        self._paren_fmt.setForeground(QColor("#ffd700"))     # gold

        self._bracket_fmt = QTextCharFormat()
        self._bracket_fmt.setForeground(QColor("#da70d6"))   # orchid

        self._semicolon_fmt = QTextCharFormat()
        self._semicolon_fmt.setForeground(QColor("#808080")) # dim gray

        self._assign_fmt = QTextCharFormat()
        self._assign_fmt.setForeground(QColor("#d4d4d4"))    # light gray
        self._assign_fmt.setFontWeight(QFont.Bold)

        self._comparison_fmt = QTextCharFormat()
        self._comparison_fmt.setForeground(QColor("#c586c0")) # purple

        # --- rule list (applied in order; later wins on overlap) ------
        self._rules = [
            # Multi-char operators first (comparison / element-wise)
            (_re.compile(r"\.[\*\/\^]"),           self._op_fmt),
            (_re.compile(r"[=~<>]="),              self._comparison_fmt),
            (_re.compile(r"&&|\|\|"),              self._comparison_fmt),
            # Single-char operators
            (_re.compile(r"[+\-\*\/\^~]"),         self._op_fmt),
            (_re.compile(r"[<>]"),                 self._comparison_fmt),
            (_re.compile(r"[&|]"),                 self._comparison_fmt),
            # Assignment =  (single = not preceded/followed by another operator char)
            (_re.compile(r"(?<![=~<>!])=(?!=)"),   self._assign_fmt),
            # Semicolons / commas
            (_re.compile(r"[;,]"),                 self._semicolon_fmt),
            # Parentheses
            (_re.compile(r"[()]"),                 self._paren_fmt),
            # Brackets
            (_re.compile(r"[\[\]{}]"),             self._bracket_fmt),
            # Numbers  (integer, float, scientific, hex)
            (_re.compile(r"\b0[xX][0-9a-fA-F]+\b"), self._num_fmt),
            (_re.compile(r"\b\d+\.?\d*(?:[eE][+-]?\d+)?\b"), self._num_fmt),
            (_re.compile(r"\.\d+(?:[eE][+-]?\d+)?\b"), self._num_fmt),
            # Strings  (double-quoted)
            (_re.compile(r'"(?:[^"\\]|\\.)*"'),    self._str_fmt),
            # Strings  (single-quoted — but not the transpose operator)
            (_re.compile(r"(?<![\w\)\]\.])'[^']*'"), self._str_fmt),
            # Comments  (% to end of line)  — applied last so it wins
            (_re.compile(r"%.*$"),                 self._cmt_fmt),
            # Hash-style comments
            (_re.compile(r"#\{|#\}"),              self._cmt_fmt),
            (_re.compile(r"#.*$"),                 self._cmt_fmt),
        ]

    # -----------------------------------------------------------------
    def highlightBlock(self, text: str):
        # 1) Identifiers: keywords, constants, function calls
        for m in _re.finditer(r"\b([a-zA-Z_]\w*)\b", text):
            word = m.group(1)
            start, length = m.start(), m.end() - m.start()
            if word in self.KEYWORDS:
                self.setFormat(start, length, self._kw_fmt)
            elif word in self.CONSTANTS:
                self.setFormat(start, length, self._const_fmt)
            else:
                # Check if it looks like a function call:  name(
                rest = text[m.end():]
                if rest and rest.lstrip().startswith("("):
                    self.setFormat(start, length, self._fn_fmt)

        # 2) Pattern rules (operators, numbers, strings, comments)
        for pattern, fmt in self._rules:
            for m in pattern.finditer(text):
                self.setFormat(m.start(), m.end() - m.start(), fmt)


# ======================================================================
# Block-mode helpers
# ======================================================================

# Tokens that open a new nesting level
_BLOCK_OPENERS = {"for", "if", "while", "switch", "try", "function",
                  "do", "unwind_protect"}

# Tokens that close a nesting level
_BLOCK_CLOSERS = {"end", "endfor", "endif", "endwhile", "endswitch",
                  "endtry", "end_unwind_protect"}

_IDENT_RE = _re.compile(r"\b([a-zA-Z_]\w*)\b")

# If the word directly follows ( or , it is likely used as an index
# (e.g. x(end)) and should not change depth.
_INDEX_CONTEXT_RE = _re.compile(r"[(,]\s*$")


def _strip_strings_and_comments(line: str) -> str:
    """Return *line* with string literals and comments blanked out so
    keyword scanning is not confused by keywords inside strings."""
    out = list(line)
    # blank double-quoted strings
    for m in _re.finditer(r'"(?:[^"\\]|\\.)*"', line):
        for i in range(m.start(), m.end()):
            out[i] = " "
    # blank single-quoted strings (not transpose)
    for m in _re.finditer(r"(?<![\w\)\]\.])'[^']*'", line):
        for i in range(m.start(), m.end()):
            out[i] = " "
    # blank comments
    for m in _re.finditer(r"[%#].*$", line):
        for i in range(m.start(), m.end()):
            out[i] = " "
    return "".join(out)


def _compute_depth_delta(line: str) -> int:
    """Return the net change in nesting depth caused by *line*.

    Positive means more openers than closers; negative means more
    closers; zero means balanced.
    """
    cleaned = _strip_strings_and_comments(line)
    delta = 0
    for m in _IDENT_RE.finditer(cleaned):
        word = m.group(1)
        before = cleaned[:m.start()]
        if word in _BLOCK_CLOSERS:
            if _INDEX_CONTEXT_RE.search(before):
                continue
            delta -= 1
        elif word in _BLOCK_OPENERS:
            delta += 1
    return delta


def _line_starts_block(line: str) -> bool:
    """Does *line* open a block without a matching close on the same line?"""
    return _compute_depth_delta(line) > 0


def _depth_after_line(current_depth: int, line: str) -> int:
    """Return the new nesting depth after processing *line*."""
    return max(0, current_depth + _compute_depth_delta(line))


# ======================================================================
# Command widget
# ======================================================================

class CommandWidget(QWidget):
    """Interactive command window — single-pane terminal style."""

    command_executed = Signal(str)
    help_requested = Signal(str)   # emitted when user right-clicks > Help on 'func'
    edit_requested = Signal(str)   # emitted when edit() command is run

    PROMPT = ">> "
    CONTINUATION = ".. "

    INDENT_WIDTH = 2   # spaces per nesting level in block mode

    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine = None

        # History
        self.history: list[str] = self._load_persistent_history()
        self.history_index: int = -1
        self._history_tmp: str = ""

        # --- Multi-line state -----------------------------------------
        self._accumulator: list[str] = []   # collected lines
        self._block_depth: int = 0          # current nesting depth
        self._in_block_mode: bool = False   # True while collecting a structure
        self._in_continuation: bool = False # True after "..." continuation

        # Tab completion
        self._completion_candidates: list[str] = []
        self._completion_index: int = 0

        # Prompt tracking
        self._prompt_pos: int = 0

        self._build_ui()
        QTimer.singleShot(0, self._show_initial_prompt)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        mono = QFont("Monospace", 10)
        mono.setStyleHint(QFont.Monospace)

        self.console = QPlainTextEdit(self)
        self.console.setFont(mono)
        self.console.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.console.setStyleSheet(
            "QPlainTextEdit { background-color: #1e1e1e; color: #d4d4d4; "
            "selection-background-color: #264f78; border: none; padding: 4px; }"
        )
        self.console.setUndoRedoEnabled(False)
        layout.addWidget(self.console)

        self._highlighter = _MCodeHighlighter(self.console.document())

        self.console.keyPressEvent = self._key_press
        self.console.mousePressEvent = self._mouse_press
        self.console.mouseDoubleClickEvent = self._mouse_double_click
        self.console.contextMenuEvent = self._context_menu_event
        self.console.setMouseTracking(True)
        self.console.event = self._tooltip_event

    def _show_initial_prompt(self):
        self.console.clear()
        # Get function count from engine if available
        func_count = ""
        if self.engine and hasattr(self.engine, '_engine'):
            n = len(self.engine._engine.functions)
            func_count = f" | {n} built-in functions"
        banner = (
            f"  Forge 0.1.0{func_count}\n"
            f"  Octave-Compatible Computing Environment\n"
            f"  \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
            f"  Type \u2018help topic\u2019 for help. "
            f"Press F1 for documentation browser.\n"
            f"  Multi-line blocks collected until matching \u2018end\u2019.\n\n"
        )
        self._append_text(banner)
        self._write_prompt()

    # ------------------------------------------------------------------
    # Prompt management
    # ------------------------------------------------------------------

    def _current_prompt(self) -> str:
        if self._in_block_mode or self._in_continuation:
            return self.CONTINUATION
        return self.PROMPT

    def _start_execution_timer(self):
        """Start timing command execution."""
        import time
        self._exec_start_time = time.perf_counter()

    def _stop_execution_timer(self):
        """Stop timing and optionally display elapsed time."""
        import time
        if hasattr(self, '_exec_start_time'):
            elapsed = time.perf_counter() - self._exec_start_time
            if elapsed > 0.5:  # Only show for slow commands
                self._append_text_colored(f"  [{elapsed:.3f}s]\n", '#6c7086')
            del self._exec_start_time

    def _write_prompt(self):
        """Append a prompt and record where editable text begins."""
        prompt = self._current_prompt()
        cursor = self.console.textCursor()
        cursor.movePosition(QTextCursor.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#569cd6"))
        cursor.insertText(prompt, fmt)
        self._prompt_pos = cursor.position()

        # Auto-indent in block mode
        if self._in_block_mode and self._block_depth > 0:
            indent = " " * (self.INDENT_WIDTH * self._block_depth)
            cursor.insertText(indent)

        self.console.setTextCursor(cursor)
        self.console.ensureCursorVisible()

    def _get_input_text(self) -> str:
        doc = self.console.document()
        full = doc.toPlainText()
        if self._prompt_pos <= len(full):
            return full[self._prompt_pos:]
        return ""

    def _set_input_text(self, text: str):
        cursor = self.console.textCursor()
        cursor.setPosition(self._prompt_pos)
        cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        cursor.removeSelectedText()
        cursor.insertText(text)
        self.console.setTextCursor(cursor)

    def _cursor_in_editable(self) -> bool:
        return self.console.textCursor().position() >= self._prompt_pos

    def _ensure_cursor_editable(self):
        if not self._cursor_in_editable():
            cursor = self.console.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.console.setTextCursor(cursor)

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------

    def _key_press(self, event):
        key = event.key()
        modifiers = event.modifiers()

        # --- History navigation (only when NOT in block mode) ----------
        if key == Qt.Key_Up and not self._in_block_mode:
            self._history_prev()
            return
        if key == Qt.Key_Down and not self._in_block_mode:
            self._history_next()
            return

        # --- Enter / Return ---
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self._on_return()
            return

        # --- Home: go to start of editable area ---
        if key == Qt.Key_Home:
            cursor = self.console.textCursor()
            mode = (QTextCursor.KeepAnchor
                    if modifiers & Qt.ShiftModifier
                    else QTextCursor.MoveAnchor)
            cursor.setPosition(self._prompt_pos, mode)
            self.console.setTextCursor(cursor)
            return

        # --- Backspace: don't delete past prompt ---
        if key == Qt.Key_Backspace:
            if self.console.textCursor().position() <= self._prompt_pos:
                return
            QPlainTextEdit.keyPressEvent(self.console, event)
            return

        # --- Ctrl+C: copy or cancel ---
        if key == Qt.Key_C and modifiers & Qt.ControlModifier:
            if self.console.textCursor().hasSelection():
                self.console.copy()
            else:
                self._cancel_input()
            return

        # --- Ctrl+A: select editable portion ---
        if key == Qt.Key_A and modifiers & Qt.ControlModifier:
            cursor = self.console.textCursor()
            cursor.setPosition(self._prompt_pos)
            cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
            self.console.setTextCursor(cursor)
            return

        # --- Ctrl+V: paste ---
        if key == Qt.Key_V and modifiers & Qt.ControlModifier:
            self._ensure_cursor_editable()
            QPlainTextEdit.keyPressEvent(self.console, event)
            return

        # --- Tab: completion ---
        if key == Qt.Key_Tab:
            self._tab_complete()
            return

        # --- Escape: dismiss completion ---
        if key == Qt.Key_Escape:
            self._completion_candidates = []
            return

        # --- Printable keys ---
        if event.text() and not modifiers & Qt.ControlModifier:
            self._ensure_cursor_editable()
            QPlainTextEdit.keyPressEvent(self.console, event)
            return

        # --- Left arrow: clamp ---
        if key == Qt.Key_Left:
            if self.console.textCursor().position() <= self._prompt_pos:
                return

        QPlainTextEdit.keyPressEvent(self.console, event)

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    def _cancel_input(self):
        """Ctrl+C — abandon whatever is being typed."""
        self._accumulator.clear()
        self._block_depth = 0
        self._in_block_mode = False
        self._in_continuation = False
        self._append_text("\n")
        self._write_prompt()

    # ------------------------------------------------------------------
    # Mouse handling & context menu
    # ------------------------------------------------------------------

    def _mouse_press(self, event):
        QPlainTextEdit.mousePressEvent(self.console, event)

    def _mouse_double_click(self, event):
        QPlainTextEdit.mouseDoubleClickEvent(self.console, event)

    def _tooltip_event(self, event):
        """Show tooltip on hover over function names."""
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.ToolTip:
            pos = event.pos()
            cursor = self.console.cursorForPosition(pos)
            cursor.select(QTextCursor.WordUnderCursor)
            word = cursor.selectedText().strip()
            if word and _re.match(r'^[a-zA-Z_]\w*$', word) and self._is_known_function(word):
                func = self.engine._engine.functions[word]
                doc = getattr(func, '__doc__', None) or 'Built-in function'
                lines = doc.strip().split('\n')[:3]
                tip = f"<b>{word}</b><br><pre>{'<br>'.join(lines)}</pre>"
                from PySide6.QtWidgets import QToolTip
                from PySide6.QtCore import QPoint
                QToolTip.showText(self.console.mapToGlobal(pos), tip)
                return True
            from PySide6.QtWidgets import QToolTip
            QToolTip.hideText()
            return True
        return QPlainTextEdit.event(self.console, event)

    def _word_under_cursor(self):
        """Extract the word (potential function name) under the text cursor."""
        cursor = self.console.textCursor()
        cursor.select(QTextCursor.WordUnderCursor)
        word = cursor.selectedText().strip()
        # Only return if it looks like an identifier
        if word and _re.match(r'^[a-zA-Z_]\w*$', word):
            return word
        return None

    def _is_known_function(self, name):
        """Check if name is a registered function in the engine."""
        if self.engine and hasattr(self.engine, '_engine'):
            return name in self.engine._engine.functions
        return False

    def _context_menu_event(self, event):
        """Custom context menu with Help, Run Selection, Clear, Copy All."""
        menu = self.console.createStandardContextMenu()

        # Add Clear and Copy All at the top
        menu.insertSeparator(menu.actions()[0] if menu.actions() else None)
        act_clear = QAction("Clear Command Window", menu)
        act_clear.triggered.connect(self._clear_output)
        menu.insertAction(menu.actions()[0] if menu.actions() else None, act_clear)

        act_copy_all = QAction("Copy All Output", menu)
        act_copy_all.triggered.connect(self._copy_all_output)
        menu.insertAction(menu.actions()[1] if len(menu.actions()) > 1 else None, act_copy_all)

        word = self._word_under_cursor()

        if word:
            menu.addSeparator()
            if self._is_known_function(word):
                help_act = QAction(f"Help on \u2018{word}\u2019", menu)
                help_act.triggered.connect(lambda: self.help_requested.emit(word))
                menu.addAction(help_act)

                # Also offer "Run 'help word'" shortcut
                run_help_act = QAction(f"Run: help {word}", menu)
                run_help_act.triggered.connect(
                    lambda: self._execute_command(f"help {word}")
                )
                menu.addAction(run_help_act)
            else:
                # Offer to search for it
                search_act = QAction(f"Search help for \u2018{word}\u2019", menu)
                search_act.triggered.connect(lambda: self.help_requested.emit(word))
                menu.addAction(search_act)

        # If there's a selection, offer to run it
        cursor = self.console.textCursor()
        if cursor.hasSelection():
            sel_text = cursor.selectedText().strip()
            if sel_text and len(sel_text) < 200:
                menu.addSeparator()
                eval_act = QAction("Evaluate Selection", menu)
                eval_act.triggered.connect(
                    lambda: self._execute_command(sel_text)
                )
                menu.addAction(eval_act)

        menu.exec(event.globalPos())

    # ------------------------------------------------------------------
    # Diagnostics / Problems tracking
    # ------------------------------------------------------------------

    def get_diagnostics(self):
        """Return list of diagnostic messages from recent execution."""
        if not hasattr(self, '_diagnostics'):
            self._diagnostics = []
        return self._diagnostics

    def _track_diagnostic(self, line, severity='error'):
        """Track a diagnostic message."""
        if not hasattr(self, '_diagnostics'):
            self._diagnostics = []
        import re
        # Try to parse "file:line: message" pattern
        m = re.match(r'(?:(.+):(\d+):\s*)?(.*)', line)
        entry = {
            'severity': severity,
            'message': line,
            'file': m.group(1) if m and m.group(1) else None,
            'line': int(m.group(2)) if m and m.group(2) else None,
            'text': m.group(3) if m else line,
        }
        self._diagnostics.append(entry)
        # Keep only last 100
        if len(self._diagnostics) > 100:
            self._diagnostics = self._diagnostics[-100:]

    def clear_diagnostics(self):
        """Clear all tracked diagnostics."""
        self._diagnostics = []

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def _load_persistent_history(self):
        import json, os
        path = os.path.join(os.path.expanduser('~'), '.forge', 'command_history.json')
        try:
            if os.path.exists(path):
                with open(path) as f:
                    return json.load(f)[-500:]
        except Exception:
            pass
        return []

    def _save_persistent_history(self):
        import json, os
        d = os.path.join(os.path.expanduser('~'), '.forge')
        os.makedirs(d, exist_ok=True)
        try:
            with open(os.path.join(d, 'command_history.json'), 'w') as f:
                json.dump(self.history[-500:], f)
        except Exception:
            pass

    def _history_prev(self):
        if not self.history:
            return
        if self.history_index == -1:
            self._history_tmp = self._get_input_text()
            self.history_index = len(self.history) - 1
        elif self.history_index > 0:
            self.history_index -= 1
        else:
            return
        self._set_input_text(self.history[self.history_index])

    def _history_next(self):
        if self.history_index == -1:
            return
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self._set_input_text(self.history[self.history_index])
        else:
            self.history_index = -1
            self._set_input_text(self._history_tmp)

    # ------------------------------------------------------------------
    # Tab completion
    # ------------------------------------------------------------------

    def _get_completion_prefix(self):
        text = self._get_input_text()
        cursor = self.console.textCursor()
        pos_in_input = cursor.position() - self._prompt_pos
        if pos_in_input < 0:
            return "", 0
        left = text[:pos_in_input]
        i = len(left) - 1
        while i >= 0 and (left[i].isalnum() or left[i] == "_"):
            i -= 1
        prefix = left[i + 1:]
        return prefix, len(prefix)

    def _get_completions(self, prefix):
        if not prefix:
            return []
        matches = []
        if self.engine:
            try:
                ws_names = self.engine._engine.workspace.names()
                matches.extend(
                    n for n in ws_names
                    if n.startswith(prefix) and not n.startswith("_")
                )
            except Exception:
                pass
            try:
                func_names = list(self.engine._engine.functions.keys())
                matches.extend(
                    n for n in func_names
                    if n.startswith(prefix) and n not in matches
                )
            except Exception:
                pass
        keywords = sorted(_MCodeHighlighter.KEYWORDS)
        matches.extend(
            k for k in keywords if k.startswith(prefix) and k not in matches
        )
        constants = sorted(_MCodeHighlighter.CONSTANTS)
        matches.extend(
            c for c in constants if c.startswith(prefix) and c not in matches
        )
        return sorted(set(matches))

    def _tab_complete(self):
        prefix, prefix_len = self._get_completion_prefix()
        if not prefix:
            self._ensure_cursor_editable()
            cursor = self.console.textCursor()
            cursor.insertText("    ")
            return

        if (self._completion_candidates
                and prefix == self._completion_candidates[
                    self._completion_index][:len(prefix)]):
            self._completion_index = (
                (self._completion_index + 1) % len(self._completion_candidates)
            )
        else:
            self._completion_candidates = self._get_completions(prefix)
            self._completion_index = 0

        if not self._completion_candidates:
            return

        if len(self._completion_candidates) == 1:
            completion = self._completion_candidates[0]
            self._replace_prefix(prefix_len, completion)
            self._completion_candidates = []
        else:
            common = self._common_prefix(self._completion_candidates)
            if len(common) > len(prefix):
                self._replace_prefix(prefix_len, common)
            else:
                candidate = self._completion_candidates[
                    self._completion_index
                ]
                self._replace_prefix(prefix_len, candidate)
                if self._completion_index == 0:
                    matches_str = "  ".join(
                        self._completion_candidates[:20]
                    )
                    if len(self._completion_candidates) > 20:
                        matches_str += "  ..."
                    cursor = self.console.textCursor()
                    self._append_text("\n" + matches_str + "\n")
                    self._write_prompt()
                    self._set_input_text(candidate)

    def _replace_prefix(self, prefix_len, replacement):
        cursor = self.console.textCursor()
        for _ in range(prefix_len):
            cursor.deletePreviousChar()
        cursor.insertText(replacement)

    @staticmethod
    def _common_prefix(strings):
        if not strings:
            return ""
        prefix = strings[0]
        for s in strings[1:]:
            while not s.startswith(prefix):
                prefix = prefix[:-1]
                if not prefix:
                    return ""
        return prefix

    # ------------------------------------------------------------------
    # Execution / multi-line logic
    # ------------------------------------------------------------------

    def _on_return(self):
        raw_text = self._get_input_text()
        self.history_index = -1
        self._history_tmp = ""

        # Echo newline
        cursor = self.console.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.console.setTextCursor(cursor)
        self._append_text("\n")

        # ---- Legacy "..." line continuation --------------------------
        if raw_text.rstrip().endswith("..."):
            line_to_store = raw_text.rstrip()[:-3]
            self._accumulator.append(line_to_store)
            self._in_continuation = True
            self._write_prompt()
            return

        # If we were in "..." continuation (not block mode) and this
        # line does NOT end with "...", this is the final piece.
        if self._in_continuation and not self._in_block_mode:
            self._accumulator.append(raw_text)
            full_text = "\n".join(self._accumulator)
            self._accumulator.clear()
            self._in_continuation = False
            self._execute(full_text)
            return

        # ---- Block mode handling -------------------------------------
        if self._in_block_mode:
            # Accumulate the line
            self._accumulator.append(raw_text)
            # Update depth
            self._block_depth = _depth_after_line(
                self._block_depth, raw_text
            )
            if self._block_depth <= 0:
                # Block complete — execute entire accumulated block
                self._block_depth = 0
                self._in_block_mode = False
                full_text = "\n".join(self._accumulator)
                self._accumulator.clear()
                self._execute(full_text)
                return
            else:
                # Still inside block — show continuation prompt
                self._write_prompt()
                return

        # ---- Not in any multi-line mode yet --------------------------
        # Check whether this line opens a block
        if _line_starts_block(raw_text):
            self._accumulator.append(raw_text)
            self._block_depth = _compute_depth_delta(raw_text)
            if self._block_depth <= 0:
                # Block opened and closed on one line (e.g.
                #   "for i=1:3; disp(i); end")
                self._block_depth = 0
                full_text = "\n".join(self._accumulator)
                self._accumulator.clear()
                self._execute(full_text)
                return
            self._in_block_mode = True
            self._write_prompt()
            return

        # Simple single-line command
        self._execute(raw_text)

    # ------------------------------------------------------------------

    def _execute(self, full_text: str):
        """Run *full_text* through the engine and show results."""
        if full_text.strip():
            self.history.append(full_text)
        self._save_persistent_history()

        if self.engine is not None and full_text.strip():
            try:
                result = self.engine.eval(full_text)
                if result is not None and str(result).strip():
                    self._append_text(str(result) + "\n")
            except Exception as exc:
                self._append_error(f"error: {exc}\n")
        elif not self.engine and full_text.strip():
            self._append_error("(no engine connected)\n")

        # Check for clc request
        if self.engine and hasattr(self.engine, '_clc_request') and self.engine._clc_request:
            self.engine._clc_request = False
            self._clear_output()
            return

        # Check for edit request
        if self.engine and hasattr(self.engine, '_edit_request') and self.engine._edit_request:
            path = self.engine._edit_request
            self.engine._edit_request = None
            self.edit_requested.emit(path)

        # Check for doc request
        if self.engine and hasattr(self.engine, '_doc_request') and self.engine._doc_request:
            name = self.engine._doc_request
            self.engine._doc_request = None
            self.help_requested.emit(name)

        self._write_prompt()
        self.command_executed.emit(full_text)

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _append_text(self, text: str):
        cursor = self.console.textCursor()
        cursor.movePosition(QTextCursor.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#d4d4d4"))
        cursor.insertText(text, fmt)
        self.console.setTextCursor(cursor)
        self.console.ensureCursorVisible()

    def _append_error(self, text: str):
        cursor = self.console.textCursor()
        cursor.movePosition(QTextCursor.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#f44747"))
        cursor.insertText(text, fmt)
        self.console.setTextCursor(cursor)
        self.console.ensureCursorVisible()

    def append_output(self, text: str):
        """Public method for external code to write output, coloring errors red and warnings yellow."""
        if not text:
            return
        lines = text.split('\n')
        for line in lines:
            stripped = line.strip().lower()
            if stripped.startswith('error') or stripped.startswith('err:') or ': error' in stripped:
                self._append_text_colored(line + '\n', '#f38ba8')  # red
            elif stripped.startswith('warning') or stripped.startswith('warn:') or ': warning' in stripped:
                self._append_text_colored(line + '\n', '#f9e2af')  # yellow
            else:
                self._append_text(line + '\n')
        self.console.ensureCursorVisible()
