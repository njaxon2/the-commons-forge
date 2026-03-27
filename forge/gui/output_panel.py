"""Structured Output panel -- displays program output with filtering
(forge/gui/output_panel.py)."""

from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCharFormat, QColor, QFont, QTextCursor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit,
)


class OutputPanel(QWidget):
    """Panel that displays colour-coded, timestamped program output.

    Supports four message levels: *info*, *warning*, *error*, and
    *normal*.  Filter buttons let the user show only messages of a
    particular level.
    """

    # Colour map
    _COLOURS = {
        "normal":  "#cdd6f4",
        "info":    "#94e2d5",
        "warning": "#fab387",
        "error":   "#f38ba8",
    }

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, parent=None):
        super().__init__(parent)
        self._messages = []  # list of (level, timestamp_str, text)
        self._filter = "all"

        self.setObjectName("OutputPanel")
        self.setStyleSheet("""
            #OutputPanel {
                background-color: #1e1e2e;
            }
            #OutputPanel QTextEdit {
                background-color: #1e1e2e;
                color: #cdd6f4;
                border: 1px solid #45475a;
                font-family: "Courier New", monospace;
                font-size: 12px;
                selection-background-color: #1e6e5e;
                selection-color: #cdd6f4;
            }
            #OutputPanel QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 12px;
            }
            #OutputPanel QPushButton:hover {
                background-color: #45475a;
            }
            #OutputPanel QPushButton:checked {
                background-color: #1e6e5e;
                border-color: #94e2d5;
                font-weight: bold;
            }
            #OutputPanel QPushButton#ClearBtn {
                background-color: #1e6e5e;
                border: none;
                font-weight: bold;
            }
            #OutputPanel QPushButton#ClearBtn:hover {
                background-color: #2a9d8f;
            }
            #OutputPanel QPushButton#ClearBtn:pressed {
                background-color: #14524a;
            }
        """)

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # --- Filter toolbar ---
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)

        self.btn_all = QPushButton("All")
        self.btn_all.setCheckable(True)
        self.btn_all.setChecked(True)
        self.btn_all.clicked.connect(lambda: self._set_filter("all"))
        btn_layout.addWidget(self.btn_all)

        self.btn_errors = QPushButton("Errors")
        self.btn_errors.setCheckable(True)
        self.btn_errors.clicked.connect(lambda: self._set_filter("error"))
        btn_layout.addWidget(self.btn_errors)

        self.btn_warnings = QPushButton("Warnings")
        self.btn_warnings.setCheckable(True)
        self.btn_warnings.clicked.connect(lambda: self._set_filter("warning"))
        btn_layout.addWidget(self.btn_warnings)

        self.btn_info = QPushButton("Info")
        self.btn_info.setCheckable(True)
        self.btn_info.clicked.connect(lambda: self._set_filter("info"))
        btn_layout.addWidget(self.btn_info)

        btn_layout.addStretch()

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setObjectName("ClearBtn")
        self.btn_clear.setToolTip("Clear all output")
        self.btn_clear.clicked.connect(self.clear)
        btn_layout.addWidget(self.btn_clear)

        layout.addLayout(btn_layout)

        # --- Output text area ---
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Courier New", 11))
        layout.addWidget(self.text_edit)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append_message(self, text, level="normal"):
        """Append a message with a timestamp.

        Parameters
        ----------
        text : str
            The message text.
        level : str
            One of ``"normal"``, ``"info"``, ``"warning"``, ``"error"``.
        """
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self._messages.append((level, ts, text))
        if self._filter == "all" or self._filter == level:
            self._render_line(level, ts, text)

    def append_normal(self, text):
        """Shortcut for append_message(text, 'normal')."""
        self.append_message(text, "normal")

    def append_info(self, text):
        """Shortcut for append_message(text, 'info')."""
        self.append_message(text, "info")

    def append_warning(self, text):
        """Shortcut for append_message(text, 'warning')."""
        self.append_message(text, "warning")

    def append_error(self, text):
        """Shortcut for append_message(text, 'error')."""
        self.append_message(text, "error")

    def clear(self):
        """Clear all output."""
        self._messages.clear()
        self.text_edit.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set_filter(self, level):
        """Change the active filter and rebuild visible output."""
        self._filter = level
        # Update button states
        for btn, lbl in [(self.btn_all, "all"), (self.btn_errors, "error"),
                         (self.btn_warnings, "warning"), (self.btn_info, "info")]:
            btn.setChecked(lbl == level)
        self._rebuild()

    def _rebuild(self):
        """Re-render all messages applying the current filter."""
        self.text_edit.clear()
        for msg_level, ts, text in self._messages:
            if self._filter == "all" or self._filter == msg_level:
                self._render_line(msg_level, ts, text)

    def _render_line(self, level, ts, text):
        """Append a single coloured line to the text area."""
        colour = self._COLOURS.get(level, self._COLOURS["normal"])
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(colour))
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(f"[{ts}] ", fmt)
        # Level tag for non-normal
        if level != "normal":
            tag_fmt = QTextCharFormat()
            tag_fmt.setForeground(QColor(colour))
            tag_fmt.setFontWeight(QFont.Bold)
            cursor.insertText(f"[{level.upper()}] ", tag_fmt)
        cursor.insertText(f"{text}\n", fmt)
        # Auto-scroll to bottom
        self.text_edit.setTextCursor(cursor)
        self.text_edit.ensureCursorVisible()
