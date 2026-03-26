"""Forge command widget — single-pane interactive REPL (forge/gui/command_widget.py).

Behaves like MATLAB's command window: a single text area where output and
input live together.  The user types after the '>> ' prompt at the bottom.
"""

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QTextCursor, QColor, QTextCharFormat, QSyntaxHighlighter, QTextDocument
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPlainTextEdit


import re as _re


class _MCodeHighlighter(QSyntaxHighlighter):
    """Minimal M-code syntax highlighting for the input line."""

    KEYWORDS = {
        "for", "end", "if", "else", "elseif", "while", "do", "until",
        "switch", "case", "otherwise", "try", "catch", "function",
        "return", "break", "continue", "global", "persistent",
        "endfor", "endif", "endwhile", "endswitch", "endtry",
        "true", "false",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        # Format: keywords
        self._kw_fmt = QTextCharFormat()
        self._kw_fmt.setForeground(QColor("#569cd6"))  # blue
        self._kw_fmt.setFontWeight(QFont.Bold)
        # Format: strings
        self._str_fmt = QTextCharFormat()
        self._str_fmt.setForeground(QColor("#ce9178"))  # orange
        # Format: numbers
        self._num_fmt = QTextCharFormat()
        self._num_fmt.setForeground(QColor("#b5cea8"))  # green
        # Format: comments
        self._cmt_fmt = QTextCharFormat()
        self._cmt_fmt.setForeground(QColor("#6a9955"))  # green
        self._cmt_fmt.setFontItalic(True)
        # Format: operators
        self._op_fmt = QTextCharFormat()
        self._op_fmt.setForeground(QColor("#d4d4d4"))  # light gray
        # Format: functions (built-in calls)
        self._fn_fmt = QTextCharFormat()
        self._fn_fmt.setForeground(QColor("#dcdcaa"))  # yellow

        # Compiled patterns
        self._rules = [
            # Comments (% to end of line)
            (_re.compile(r"%.*$"), self._cmt_fmt),
            # Strings (double-quoted)
            (_re.compile(r'"(?:[^"\\]|\\.)*"'), self._str_fmt),
            # Strings (single-quoted, but not transpose)
            (_re.compile(r"(?<![\w\)\]\.])'[^']*'"), self._str_fmt),
            # Numbers (integer, float, scientific)
            (_re.compile(r"\b\d+\.?\d*(?:[eE][+-]?\d+)?\b"), self._num_fmt),
        ]

    def highlightBlock(self, text):
        # Only highlight the input line (after prompt)
        # The prompt position tracking is in the widget, not accessible here easily
        # So just highlight everything — output lines won't match patterns anyway

        # Keywords
        for match in _re.finditer(r"\b(\w+)\b", text):
            word = match.group(1)
            if word in self.KEYWORDS:
                self.setFormat(match.start(), match.end() - match.start(), self._kw_fmt)
            elif _re.match(r"\w+(?=\s*\()", text[match.start():]):
                # Word followed by ( is a function call
                self.setFormat(match.start(), match.end() - match.start(), self._fn_fmt)

        # Apply pattern rules
        for pattern, fmt in self._rules:
            for match in pattern.finditer(text):
                self.setFormat(match.start(), match.end() - match.start(), fmt)


class CommandWidget(QWidget):
    """Interactive command window — single-pane terminal style."""

    command_executed = Signal(str)

    PROMPT = ">> "
    CONTINUATION = ".. "

    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine = None

        # History
        self.history: list[str] = []
        self.history_index: int = -1
        self._history_tmp: str = ""  # stash current input when browsing history

        # Multi-line accumulation
        self._accumulator: list[str] = []

        # Tab completion
        self._completion_candidates: list[str] = []
        self._completion_index: int = 0

        # Prompt tracking
        self._prompt_pos: int = 0  # character position where editable input starts

        self._build_ui()
        # Show initial prompt after widget is shown
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

        # Syntax highlighting
        self._highlighter = _MCodeHighlighter(self.console.document())

        # Intercept key presses
        self.console.keyPressEvent = self._key_press
        # Prevent mouse clicks from moving cursor into read-only region
        self.console.mousePressEvent = self._mouse_press
        self.console.mouseDoubleClickEvent = self._mouse_double_click

    def _show_initial_prompt(self):
        self.console.clear()
        self._append_text("Forge 0.1 \u2014 Octave-compatible computing environment\n")
        self._append_text("Type commands at the >> prompt. Use up/down for history.\n\n")
        self._write_prompt()

    # ------------------------------------------------------------------
    # Prompt management
    # ------------------------------------------------------------------

    def _current_prompt(self) -> str:
        return self.CONTINUATION if self._accumulator else self.PROMPT

    def _write_prompt(self):
        """Append a prompt and record where editable text begins."""
        prompt = self._current_prompt()
        cursor = self.console.textCursor()
        cursor.movePosition(QTextCursor.End)
        # Insert prompt in a slightly different color (gray-blue)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#569cd6"))
        cursor.insertText(prompt, fmt)
        # Record position — everything after this is editable
        self._prompt_pos = cursor.position()
        self.console.setTextCursor(cursor)
        self.console.ensureCursorVisible()

    def _get_input_text(self) -> str:
        """Return the text the user has typed after the current prompt."""
        doc = self.console.document()
        full = doc.toPlainText()
        if self._prompt_pos <= len(full):
            return full[self._prompt_pos:]
        return ""

    def _set_input_text(self, text: str):
        """Replace the user's input text (after prompt) with *text*."""
        cursor = self.console.textCursor()
        cursor.setPosition(self._prompt_pos)
        cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        cursor.removeSelectedText()
        cursor.insertText(text)
        self.console.setTextCursor(cursor)

    def _cursor_in_editable(self) -> bool:
        return self.console.textCursor().position() >= self._prompt_pos

    def _ensure_cursor_editable(self):
        """Move cursor to end if it's in the read-only region."""
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

        # --- History navigation ---
        if key == Qt.Key_Up:
            self._history_prev()
            return
        if key == Qt.Key_Down:
            self._history_next()
            return

        # --- Enter / Return ---
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self._on_return()
            return

        # --- Home key: go to start of editable area ---
        if key == Qt.Key_Home:
            cursor = self.console.textCursor()
            mode = QTextCursor.KeepAnchor if modifiers & Qt.ShiftModifier else QTextCursor.MoveAnchor
            cursor.setPosition(self._prompt_pos, mode)
            self.console.setTextCursor(cursor)
            return

        # --- Backspace: don't delete past prompt ---
        if key == Qt.Key_Backspace:
            if self.console.textCursor().position() <= self._prompt_pos:
                return  # at or before prompt — ignore
            QPlainTextEdit.keyPressEvent(self.console, event)
            return

        # --- Ctrl+C: copy if selection, else cancel current input ---
        if key == Qt.Key_C and modifiers & Qt.ControlModifier:
            if self.console.textCursor().hasSelection():
                self.console.copy()
            else:
                # Cancel current input
                self._accumulator.clear()
                self._append_text("\n")
                self._write_prompt()
            return

        # --- Ctrl+A: select only the editable portion ---
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

        # --- All other printable keys ---
        if event.text() and not modifiers & Qt.ControlModifier:
            self._ensure_cursor_editable()
            QPlainTextEdit.keyPressEvent(self.console, event)
            return

        # --- Everything else: arrows, etc. (allow but clamp) ---
        # Left arrow: don't go past prompt
        if key == Qt.Key_Left:
            if self.console.textCursor().position() <= self._prompt_pos:
                return
        QPlainTextEdit.keyPressEvent(self.console, event)

    # ------------------------------------------------------------------
    # Mouse handling — keep editable region protected
    # ------------------------------------------------------------------

    def _mouse_press(self, event):
        QPlainTextEdit.mousePressEvent(self.console, event)
        # After click, if cursor is in read-only area, move to end
        # But allow clicks for selection purposes

    def _mouse_double_click(self, event):
        QPlainTextEdit.mouseDoubleClickEvent(self.console, event)

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

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
        """Get the word fragment before cursor for completion."""
        text = self._get_input_text()
        cursor = self.console.textCursor()
        pos_in_input = cursor.position() - self._prompt_pos
        if pos_in_input < 0:
            return "", 0
        left = text[:pos_in_input]
        # Walk backwards to find start of identifier
        i = len(left) - 1
        while i >= 0 and (left[i].isalnum() or left[i] == '_'):
            i -= 1
        prefix = left[i+1:]
        return prefix, len(prefix)

    def _get_completions(self, prefix):
        """Get list of matching completions for prefix."""
        if not prefix:
            return []
        matches = []
        # 1. Workspace variables
        if self.engine:
            try:
                ws_names = self.engine._engine.workspace.names()
                matches.extend(n for n in ws_names if n.startswith(prefix) and not n.startswith('_'))
            except Exception:
                pass
            # 2. Registered functions
            try:
                func_names = list(self.engine._engine.functions.keys())
                matches.extend(n for n in func_names if n.startswith(prefix) and n not in matches)
            except Exception:
                pass
        # 3. Keywords
        keywords = ['break', 'case', 'catch', 'continue', 'do', 'else', 'elseif',
                     'end', 'endfor', 'endif', 'endwhile', 'endswitch', 'endtry',
                     'for', 'function', 'global', 'if', 'otherwise', 'persistent',
                     'return', 'switch', 'try', 'until', 'while']
        matches.extend(k for k in keywords if k.startswith(prefix) and k not in matches)
        return sorted(set(matches))

    def _tab_complete(self):
        """Handle tab key press for completion."""
        prefix, prefix_len = self._get_completion_prefix()
        if not prefix:
            # Just insert spaces if no prefix
            self._ensure_cursor_editable()
            cursor = self.console.textCursor()
            cursor.insertText("    ")
            return

        # If we have ongoing completion candidates, cycle through them
        if self._completion_candidates and prefix == self._completion_candidates[self._completion_index][:len(prefix)]:
            self._completion_index = (self._completion_index + 1) % len(self._completion_candidates)
        else:
            self._completion_candidates = self._get_completions(prefix)
            self._completion_index = 0

        if not self._completion_candidates:
            return

        if len(self._completion_candidates) == 1:
            # Unique match — complete it
            completion = self._completion_candidates[0]
            self._replace_prefix(prefix_len, completion)
            self._completion_candidates = []
        else:
            # Multiple matches
            # First: complete common prefix
            common = self._common_prefix(self._completion_candidates)
            if len(common) > len(prefix):
                self._replace_prefix(prefix_len, common)
            else:
                # Show candidates below
                candidate = self._completion_candidates[self._completion_index]
                self._replace_prefix(prefix_len, candidate)
                # Show all matches as hint
                if self._completion_index == 0:
                    matches_str = "  ".join(self._completion_candidates[:20])
                    if len(self._completion_candidates) > 20:
                        matches_str += "  ..."
                    # Show in output area temporarily
                    cursor = self.console.textCursor()
                    pos = cursor.position()
                    self._append_text("\n" + matches_str + "\n")
                    # Move cursor back to input
                    self._write_prompt()
                    self._set_input_text(candidate)

    def _replace_prefix(self, prefix_len, replacement):
        """Replace the prefix at cursor with replacement text."""
        cursor = self.console.textCursor()
        for _ in range(prefix_len):
            cursor.deletePreviousChar()
        cursor.insertText(replacement)

    def _common_prefix(self, strings):
        """Find longest common prefix of a list of strings."""
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
    # Execution
    # ------------------------------------------------------------------

    def _on_return(self):
        text = self._get_input_text()
        self.history_index = -1
        self._history_tmp = ""

        # Move cursor to end and add newline
        cursor = self.console.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.console.setTextCursor(cursor)
        self._append_text("\n")

        # Multi-line continuation
        if text.rstrip().endswith("..."):
            self._accumulator.append(text.rstrip()[:-3])
            return

        if self._accumulator:
            self._accumulator.append(text)
            full_text = "\n".join(self._accumulator)
            self._accumulator.clear()
        else:
            full_text = text

        # Store in history (skip empty)
        if full_text.strip():
            self.history.append(full_text)

        # Evaluate
        if self.engine is not None and full_text.strip():
            try:
                result = self.engine.eval(full_text)
                if result is not None and str(result).strip():
                    self._append_text(str(result) + "\n")
            except Exception as exc:
                self._append_error(f"error: {exc}\n")
        elif not self.engine and full_text.strip():
            self._append_error("(no engine connected)\n")

        self._write_prompt()
        self.command_executed.emit(full_text)

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _append_text(self, text: str):
        """Append plain text to the console."""
        cursor = self.console.textCursor()
        cursor.movePosition(QTextCursor.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#d4d4d4"))
        cursor.insertText(text, fmt)
        self.console.setTextCursor(cursor)
        self.console.ensureCursorVisible()

    def _append_error(self, text: str):
        """Append error text (red) to the console."""
        cursor = self.console.textCursor()
        cursor.movePosition(QTextCursor.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#f44747"))
        cursor.insertText(text, fmt)
        self.console.setTextCursor(cursor)
        self.console.ensureCursorVisible()

    def append_output(self, text: str):
        """Public method for external code to write output to the console."""
        self._append_text(text + "\n")
        self.console.ensureCursorVisible()
