"""Forge command widget — single-pane interactive REPL (forge/gui/command_widget.py).

Behaves like MATLAB's command window: a single text area where output and
input live together.  The user types after the '>> ' prompt at the bottom.
"""

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QTextCursor, QColor, QTextCharFormat
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPlainTextEdit


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

        # Intercept key presses
        self.console.keyPressEvent = self._key_press
        # Prevent mouse clicks from moving cursor into read-only region
        self.console.mousePressEvent = self._mouse_press
        self.console.mouseDoubleClickEvent = self._mouse_double_click

    def _show_initial_prompt(self):
        self.console.clear()
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
            self._write_prompt()
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
