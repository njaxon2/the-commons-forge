"""Forge command widget — interactive REPL (forge/gui/command_widget.py)."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPlainTextEdit, QLineEdit


class CommandWidget(QWidget):
    """Interactive command window with output display and line input."""

    command_executed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine = None

        # History
        self.history: list[str] = []
        self.history_index: int = -1

        # Multi-line accumulation
        self._accumulator: list[str] = []

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        # Monospace font for the whole command widget
        mono = QFont("Monospace", 10)
        mono.setStyleHint(QFont.Monospace)

        # Output display
        self.output_display = QPlainTextEdit(self)
        self.output_display.setReadOnly(True)
        self.output_display.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.output_display.setFont(mono)
        self.output_display.setStyleSheet(
            "QPlainTextEdit { background-color: #1e1e1e; color: #d4d4d4; "
            "selection-background-color: #264f78; }"
        )
        layout.addWidget(self.output_display)

        # Input line with >> prompt
        self.input_line = QLineEdit(self)
        self.input_line.setFont(mono)
        self.input_line.setPlaceholderText("Type command here...")
        self.input_line.setStyleSheet(
            "QLineEdit { background-color: #1e1e1e; color: #d4d4d4; "
            "border: 1px solid #3c3c3c; padding: 4px; padding-left: 24px; }"
        )
        # We prepend >> in the echo, not the placeholder
        self.input_line.returnPressed.connect(self._on_return)
        self.input_line.installEventFilter(self)
        layout.addWidget(self.input_line)

    # ------------------------------------------------------------------
    # Event filter for history navigation
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):
        if obj is self.input_line and event.type() == event.Type.KeyPress:
            key = event.key()
            if key == Qt.Key_Up:
                self._history_prev()
                return True
            if key == Qt.Key_Down:
                self._history_next()
                return True
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------

    def _history_prev(self):
        if not self.history:
            return
        if self.history_index == -1:
            self.history_index = len(self.history) - 1
        elif self.history_index > 0:
            self.history_index -= 1
        self.input_line.setText(self.history[self.history_index])

    def _history_next(self):
        if not self.history or self.history_index == -1:
            return
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.input_line.setText(self.history[self.history_index])
        else:
            self.history_index = -1
            self.input_line.clear()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _on_return(self):
        text = self.input_line.text()
        self.input_line.clear()
        self.history_index = -1

        # Echo input
        prompt = ">> " if not self._accumulator else ".. "
        self.append_output(f"{prompt}{text}")

        # Multi-line continuation
        if text.rstrip().endswith("..."):
            self._accumulator.append(text.rstrip()[:-3])
            return

        if self._accumulator:
            self._accumulator.append(text)
            text = "\n".join(self._accumulator)
            self._accumulator.clear()

        # Store in history
        self.history.append(text)

        # Evaluate
        if self.engine is not None:
            try:
                result = self.engine.eval(text)
                if result is not None:
                    self.append_output(str(result))
            except Exception as exc:
                self.append_output(f"error: {exc}")
        else:
            self.append_output("(no engine connected)")

        self.command_executed.emit(text)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def append_output(self, text: str):
        """Append a line of text to the output display."""
        self.output_display.appendPlainText(text)
        # Auto-scroll to bottom
        sb = self.output_display.verticalScrollBar()
        sb.setValue(sb.maximum())
