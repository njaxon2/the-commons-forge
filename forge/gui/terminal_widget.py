"""Simple terminal widget for running shell commands within Forge."""
import os
import subprocess
import threading

from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtGui import QColor, QFont, QTextCursor, QTextCharFormat
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPlainTextEdit


class TerminalOutputSignal(QObject):
    output_ready = Signal(str)
    error_ready = Signal(str)
    finished = Signal(int)


class TerminalWidget(QWidget):
    """A basic terminal/shell widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process = None
        self._history = []
        self._history_idx = -1
        self._cwd = os.path.expanduser("~")
        self._signals = TerminalOutputSignal()
        self._signals.output_ready.connect(self._append_output)
        self._signals.error_ready.connect(self._append_error)
        self._signals.finished.connect(self._on_finished)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._console = QPlainTextEdit()
        self._console.setReadOnly(False)
        self._console.setFont(QFont("Consolas", 11))
        self._console.setStyleSheet(
            "QPlainTextEdit { background: #11111b; color: #a6e3a1; "
            "border: none; selection-background-color: #313244; }"
        )
        self._console.setLineWrapMode(QPlainTextEdit.NoWrap)
        layout.addWidget(self._console)

        # Write initial prompt
        self._write_prompt()

        # Handle key presses
        self._console.installEventFilter(self)

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent
        if obj is self._console and event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self._execute_current_line()
                return True
            # Prevent editing before prompt
            cursor = self._console.textCursor()
            if cursor.position() < getattr(self, '_prompt_pos', 0):
                if event.key() not in (Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right):
                    return True
        return super().eventFilter(obj, event)

    def _write_prompt(self):
        cursor = self._console.textCursor()
        cursor.movePosition(QTextCursor.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#89b4fa"))
        prompt = f"{os.path.basename(self._cwd)}$ "
        cursor.insertText(prompt, fmt)
        self._prompt_pos = cursor.position()
        self._console.setTextCursor(cursor)

    def _get_command(self):
        text = self._console.toPlainText()
        return text[self._prompt_pos - len(self._console.toPlainText()):].strip() if hasattr(self, '_prompt_pos') else ""

    def _execute_current_line(self):
        # Get text after prompt
        text = self._console.toPlainText()
        if not hasattr(self, '_prompt_pos'):
            return

        # Extract command from after prompt position
        full_text = text[self._prompt_pos:]
        cmd = full_text.strip()

        # Move cursor to end and add newline
        cursor = self._console.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText("\n")

        if not cmd:
            self._write_prompt()
            return

        # Handle cd specially
        if cmd.startswith('cd '):
            target = cmd[3:].strip().strip('"').strip("'")
            target = os.path.expanduser(target)
            if not os.path.isabs(target):
                target = os.path.join(self._cwd, target)
            if os.path.isdir(target):
                self._cwd = os.path.abspath(target)
            else:
                self._append_error(f"cd: no such directory: {target}\n")
            self._write_prompt()
            return

        # Run command in background thread
        def run():
            try:
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True,
                    timeout=30, cwd=self._cwd
                )
                if result.stdout:
                    self._signals.output_ready.emit(result.stdout)
                if result.stderr:
                    self._signals.error_ready.emit(result.stderr)
                self._signals.finished.emit(result.returncode)
            except subprocess.TimeoutExpired:
                self._signals.error_ready.emit("Command timed out (30s)\n")
                self._signals.finished.emit(-1)
            except Exception as e:
                self._signals.error_ready.emit(f"Error: {e}\n")
                self._signals.finished.emit(-1)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

    def _append_output(self, text):
        cursor = self._console.textCursor()
        cursor.movePosition(QTextCursor.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#cdd6f4"))
        cursor.insertText(text, fmt)
        self._console.setTextCursor(cursor)
        self._console.ensureCursorVisible()

    def _append_error(self, text):
        cursor = self._console.textCursor()
        cursor.movePosition(QTextCursor.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#f38ba8"))
        cursor.insertText(text, fmt)
        self._console.setTextCursor(cursor)
        self._console.ensureCursorVisible()

    def _on_finished(self, return_code):
        self._write_prompt()
