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
    _COLOURS = None  # set dynamically

    @staticmethod
    def _get_colours():
        from forge.gui.theme_utils import detect_palette, is_light_theme
        _p = detect_palette()
        if is_light_theme():
            return {
                "normal":  _p.get("fg0", "#1e1e2e"),
                "info":    "#00897B",
                "warning": "#e65100",
                "error":   _p.get("error", "#d32f2f"),
            }
        return {
            "normal":  _p.get("fg0", "#cdd6f4"),
            "info":    _p.get("info", "#94e2d5"),
            "warning": _p.get("warning", "#fab387"),
            "error":   _p.get("error", "#f38ba8"),
        }

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, parent=None):
        super().__init__(parent)
        self._messages = []  # list of (level, timestamp_str, text)
        self._filter = "all"

        self.setObjectName("OutputPanel")
        self.apply_theme()

    def apply_theme(self):
        from forge.gui.theme_utils import detect_palette
        p = detect_palette()
        bg0 = p.get("bg0", "#1e1e2e")
        bg3 = p.get("bg3", "#313145")
        bg5 = p.get("bg5", "#44445a")
        fg0 = p.get("fg0", "#cdd6f4")
        fg3 = p.get("fg3", "#6c7086")
        border1 = p.get("border1", "#44445a")
        accent = p.get("accent", "#00BCD4")
        accent_p = p.get("accent_p", "#0097A7")
        selection = p.get("selection", "#264f78")
        self.setStyleSheet(f"""
            #OutputPanel {{
                background-color: {bg0};
            }}
            #OutputPanel QTextEdit {{
                background-color: {bg0};
                color: {fg0};
                border: 1px solid {border1};
                font-family: "Courier New", monospace;
                font-size: 12px;
                selection-background-color: {selection};
                selection-color: {fg0};
            }}
            #OutputPanel QPushButton {{
                background-color: {bg3};
                color: {fg0};
                border: 1px solid {border1};
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 12px;
            }}
            #OutputPanel QPushButton:hover {{
                background-color: {bg5};
            }}
            #OutputPanel QPushButton:checked {{
                background-color: {accent_p};
                border-color: {accent};
                font-weight: bold;
            }}
            #OutputPanel QPushButton#ClearBtn {{
                background-color: {accent_p};
                border: none;
                font-weight: bold;
                color: #ffffff;
            }}
            #OutputPanel QPushButton#ClearBtn:hover {{
                background-color: {accent};
            }}
            #OutputPanel QPushButton#ClearBtn:pressed {{
                background-color: {accent_p};
            }}
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
        colours = self._get_colours()
        colour = colours.get(level, colours["normal"])
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
