# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Forge Theme Editor -- visual theme customisation dialog (PySide6).

Target location: forge/gui/theme_editor.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from copy import deepcopy
from typing import Dict, Optional

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPainter, QSyntaxHighlighter, QTextCharFormat
from PySide6.QtWidgets import (
    QApplication, QColorDialog, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QFontComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSlider,
    QTabWidget, QVBoxLayout, QWidget,
)

from forge.gui.themes import (
    _THEMES, _EDITOR_COLORS, apply_theme, get_available_themes, get_editor_colors,
)

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
THEMES_DIR = Path.home() / ".forge" / "themes"


def _ensure_themes_dir() -> Path:
    THEMES_DIR.mkdir(parents=True, exist_ok=True)
    return THEMES_DIR


# ------------------------------------------------------------------
# Default palette snapshots (for "Reset to Default")
# ------------------------------------------------------------------
_DEFAULT_EDITOR_COLORS: Dict[str, Dict[str, str]] = {
    "dark": {
        "keyword":      "#4fc3f7",
        "builtin":      "#ffb74d",
        "string":       "#81c784",
        "number":       "#ff8a65",
        "comment":      "#757575",
        "operator":     "#e0e0e0",
        "function":     "#ce93d8",
        "class":        "#80cbc4",
        "decorator":    "#f48fb1",
        "self":         "#64b5f6",
        "error":        "#ef5350",
        "line_number":  "#616161",
        "current_line": "#333333",
        "brace_match":  "#0d47a1",
        "selection":    "#0d47a1",
        "background":   "#2b2b2b",
        "foreground":   "#e0e0e0",
    },
    "light": {
        "keyword":      "#0000ff",
        "builtin":      "#795548",
        "string":       "#388e3c",
        "number":       "#e65100",
        "comment":      "#9e9e9e",
        "operator":     "#212121",
        "function":     "#6a1b9a",
        "class":        "#00695c",
        "decorator":    "#ad1457",
        "self":         "#1565c0",
        "error":        "#d32f2f",
        "line_number":  "#bdbdbd",
        "current_line": "#f5f5f5",
        "brace_match":  "#bbdefb",
        "selection":    "#bbdefb",
        "background":   "#ffffff",
        "foreground":   "#212121",
    },
    "midnight": {
        "keyword":      "#ff79c6",
        "builtin":      "#f1fa8c",
        "string":       "#50fa7b",
        "number":       "#bd93f9",
        "comment":      "#6272a4",
        "operator":     "#f8f8f2",
        "function":     "#8be9fd",
        "class":        "#ffb86c",
        "decorator":    "#ff5555",
        "self":         "#ff79c6",
        "error":        "#ff5555",
        "line_number":  "#44475a",
        "current_line": "#44475a",
        "brace_match":  "#6272a4",
        "selection":    "#44475a",
        "background":   "#282a36",
        "foreground":   "#f8f8f2",
    },
}


# ==================================================================
# Colour-swatch widget
# ==================================================================

class ColorSwatchButton(QPushButton):
    """A clickable button displaying a colour rectangle, name, and hex."""

    color_changed = Signal(str, str)  # (color_key, new_hex)

    def __init__(self, key: str, hex_color: str, parent=None):
        super().__init__(parent)
        self._key = key
        self._hex = hex_color
        self.setFixedHeight(36)
        self.setMinimumWidth(200)
        self._update_display()
        self.clicked.connect(self._pick_color)

    # -- public API --------------------------------------------------

    @property
    def hex_color(self) -> str:
        return self._hex

    def set_hex_color(self, hex_color: str):
        self._hex = hex_color
        self._update_display()

    # -- internals ---------------------------------------------------

    def _update_display(self):
        self.setText(f"  {self._key}:  {self._hex}")
        self.setStyleSheet(
            f"QPushButton {{ text-align: left; padding-left: 8px; "
            f"border: 1px solid #555; border-radius: 3px; "
            f"background-color: #383838; color: #e0e0e0; }}"
            f"QPushButton::before {{ content: ''; }}"
        )

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        color = QColor(self._hex)
        painter.setBrush(color)
        painter.setPen(QColor("#888888"))
        rect_x = self.width() - 44
        rect_y = 6
        painter.drawRoundedRect(rect_x, rect_y, 36, 24, 3, 3)
        painter.end()

    def _pick_color(self):
        initial = QColor(self._hex)
        color = QColorDialog.getColor(
            initial, self, f"Pick colour for '{self._key}'",
            QColorDialog.ShowAlphaChannel,
        )
        if color.isValid():
            self._hex = color.name()
            self._update_display()
            self.color_changed.emit(self._key, self._hex)


# ==================================================================
# Preview highlighter (simple token colouring for sample code)
# ==================================================================

_SAMPLE_CODE = """\
% Forge sample code
function result = fibonacci(n)
    if n <= 1
        result = n;
        return;
    end
    a = 0; b = 1;
    for i = 2:n
        temp = a + b;  % accumulate
        a = b;
        b = temp;
    end
    result = b;
end

x = fibonacci(10);
disp(x);
"""


class _PreviewHighlighter(QSyntaxHighlighter):
    """Minimal highlighter that applies palette colours to sample code."""

    def __init__(self, parent, palette: Dict[str, str]):
        super().__init__(parent)
        self._palette = palette

    def set_palette(self, palette: Dict[str, str]):
        self._palette = palette
        self.rehighlight()

    def _fmt(self, key: str) -> QTextCharFormat:
        fmt = QTextCharFormat()
        if key in self._palette:
            fmt.setForeground(QColor(self._palette[key]))
        return fmt

    def highlightBlock(self, text: str):
        import re as _re
        rules = [
            ("comment",   r"%.*$"),
            ("string",    r"'[^']*'"),
            ("number",    r"\b\d+(\.\d*)?\b"),
            ("keyword",   r"\b(?:function|end|if|else|elseif|for|while|return|switch|case|otherwise|try|catch|break|continue)\b"),
            ("builtin",   r"\b(?:disp|fprintf|sprintf|zeros|ones|linspace|size|length|plot|figure)\b"),
            ("function",  r"\b[a-zA-Z_]\w*(?=\s*\()"),
        ]
        for key, pattern in rules:
            fmt = self._fmt(key)
            for m in _re.finditer(pattern, text):
                self.setFormat(m.start(), m.end() - m.start(), fmt)


# ==================================================================
# Main dialog
# ==================================================================

class ThemeEditorDialog(QDialog):
    """Visual theme customisation dialog for Forge."""

    theme_applied = Signal()  # emitted when user clicks Apply/OK

    def __init__(self, current_theme: str = "dark", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Customize Theme")
        self.setMinimumSize(720, 560)
        self.resize(800, 620)

        self._current_theme = current_theme
        # Work on a copy so cancel discards changes
        self._working_palette: Dict[str, str] = deepcopy(
            _EDITOR_COLORS.get(current_theme, _EDITOR_COLORS.get("dark", {}))
        )
        self._font_family = "Consolas"
        self._font_size = 12
        self._swatch_buttons: Dict[str, ColorSwatchButton] = {}

        self._build_ui()
        self._apply_dark_dialog_style()

    # ----------------------------------------------------------
    # UI construction
    # ----------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Theme selector row
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Base theme:"))
        self._theme_combo = QComboBox()
        for t in get_available_themes():
            self._theme_combo.addItem(t.capitalize(), t)
        idx = self._theme_combo.findData(self._current_theme)
        if idx >= 0:
            self._theme_combo.setCurrentIndex(idx)
        self._theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        top_row.addWidget(self._theme_combo)
        top_row.addStretch()

        btn_reset = QPushButton("Reset to Default")
        btn_reset.clicked.connect(self._reset_to_default)
        top_row.addWidget(btn_reset)

        layout.addLayout(top_row)

        # Tabs
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_colors_tab(), "Colors")
        self._tabs.addTab(self._build_font_tab(), "Font")
        self._tabs.addTab(self._build_preview_tab(), "Preview")
        layout.addWidget(self._tabs)

        # Bottom buttons
        btn_box = QHBoxLayout()
        btn_export = QPushButton("Export JSON...")
        btn_export.clicked.connect(self._export_json)
        btn_load = QPushButton("Load Theme...")
        btn_load.clicked.connect(self._load_theme)
        btn_save = QPushButton("Save Theme...")
        btn_save.clicked.connect(self._save_theme)
        btn_box.addWidget(btn_export)
        btn_box.addWidget(btn_load)
        btn_box.addWidget(btn_save)
        btn_box.addStretch()

        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(self._apply_and_accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_apply = QPushButton("Apply")
        btn_apply.clicked.connect(self._apply_changes)
        btn_box.addWidget(btn_apply)
        btn_box.addWidget(btn_ok)
        btn_box.addWidget(btn_cancel)

        layout.addLayout(btn_box)

    # -- Colors tab -------------------------------------------------

    def _build_colors_tab(self) -> QWidget:
        container = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        inner = QWidget()
        grid = QGridLayout(inner)
        grid.setSpacing(6)

        row = 0
        col = 0
        for key, hex_val in self._working_palette.items():
            btn = ColorSwatchButton(key, hex_val)
            btn.color_changed.connect(self._on_color_changed)
            grid.addWidget(btn, row, col)
            self._swatch_buttons[key] = btn
            col += 1
            if col >= 2:
                col = 0
                row += 1

        grid.setRowStretch(row + 1, 1)
        scroll.setWidget(inner)

        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        return container

    # -- Font tab ---------------------------------------------------

    def _build_font_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Font family
        fam_group = QGroupBox("Font Family (monospace)")
        fam_layout = QHBoxLayout(fam_group)
        self._font_combo = QComboBox()
        mono_families = self._get_monospace_fonts()
        self._font_combo.addItems(mono_families)
        idx = self._font_combo.findText(self._font_family)
        if idx >= 0:
            self._font_combo.setCurrentIndex(idx)
        self._font_combo.currentTextChanged.connect(self._on_font_family_changed)
        fam_layout.addWidget(self._font_combo)
        layout.addWidget(fam_group)

        # Font size
        size_group = QGroupBox("Font Size")
        size_layout = QHBoxLayout(size_group)
        self._size_slider = QSlider(Qt.Horizontal)
        self._size_slider.setRange(8, 24)
        self._size_slider.setValue(self._font_size)
        self._size_slider.setTickInterval(1)
        self._size_slider.setTickPosition(QSlider.TicksBelow)
        self._size_label = QLabel(f"{self._font_size} pt")
        self._size_slider.valueChanged.connect(self._on_font_size_changed)
        size_layout.addWidget(self._size_slider)
        size_layout.addWidget(self._size_label)
        layout.addWidget(size_group)

        # Font preview
        self._font_preview = QLabel("AaBbCcDdEeFf  0123456789  ()[]{}  <>=+-*/")
        self._font_preview.setFont(QFont(self._font_family, self._font_size))
        self._font_preview.setStyleSheet(
            "padding: 12px; background: #2b2b2b; color: #e0e0e0; "
            "border: 1px solid #555; border-radius: 4px;"
        )
        layout.addWidget(self._font_preview)

        layout.addStretch()
        return widget

    @staticmethod
    def _get_monospace_fonts():
        db = QFontDatabase()
        mono = []
        for family in db.families():
            if db.isFixedPitch(family):
                mono.append(family)
        # Ensure some common ones appear
        for fallback in ("Consolas", "Courier New", "DejaVu Sans Mono",
                         "Fira Code", "JetBrains Mono", "Source Code Pro",
                         "Ubuntu Mono", "Liberation Mono", "Monospace"):
            if fallback not in mono:
                mono.append(fallback)
        mono.sort()
        return mono

    # -- Preview tab ------------------------------------------------

    def _build_preview_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self._preview_editor = QPlainTextEdit()
        self._preview_editor.setReadOnly(True)
        self._preview_editor.setPlainText(_SAMPLE_CODE)
        self._preview_editor.setFont(QFont(self._font_family, self._font_size))

        self._highlighter = _PreviewHighlighter(
            self._preview_editor.document(), self._working_palette
        )

        self._update_preview_colors()
        layout.addWidget(self._preview_editor)
        return widget

    def _update_preview_colors(self):
        bg = self._working_palette.get("background", "#2b2b2b")
        fg = self._working_palette.get("foreground", "#e0e0e0")
        self._preview_editor.setStyleSheet(
            f"QPlainTextEdit {{ background-color: {bg}; color: {fg}; "
            f"border: 1px solid #555; border-radius: 4px; padding: 8px; }}"
        )
        self._preview_editor.setFont(QFont(self._font_family, self._font_size))
        self._highlighter.set_palette(self._working_palette)

    # ----------------------------------------------------------
    # Slots
    # ----------------------------------------------------------

    def _on_theme_changed(self, index):
        theme = self._theme_combo.currentData()
        self._current_theme = theme
        if theme in _EDITOR_COLORS:
            self._working_palette = deepcopy(_EDITOR_COLORS[theme])
        self._refresh_swatches()
        self._update_preview_colors()

    def _on_color_changed(self, key: str, new_hex: str):
        self._working_palette[key] = new_hex
        self._update_preview_colors()

    def _on_font_family_changed(self, family: str):
        self._font_family = family
        self._font_preview.setFont(QFont(self._font_family, self._font_size))
        self._update_preview_colors()

    def _on_font_size_changed(self, value: int):
        self._font_size = value
        self._size_label.setText(f"{value} pt")
        self._font_preview.setFont(QFont(self._font_family, self._font_size))
        self._update_preview_colors()

    def _refresh_swatches(self):
        for key, btn in self._swatch_buttons.items():
            if key in self._working_palette:
                btn.set_hex_color(self._working_palette[key])

    def _reset_to_default(self):
        theme = self._current_theme
        if theme in _DEFAULT_EDITOR_COLORS:
            self._working_palette = deepcopy(_DEFAULT_EDITOR_COLORS[theme])
        elif theme in _EDITOR_COLORS:
            # Fallback: reload from module-level dict
            self._working_palette = deepcopy(_EDITOR_COLORS[theme])
        self._refresh_swatches()
        self._update_preview_colors()

    # ----------------------------------------------------------
    # Apply / persist
    # ----------------------------------------------------------

    def _apply_changes(self):
        """Push working palette into the live themes module and re-apply."""
        theme = self._current_theme
        _EDITOR_COLORS[theme] = deepcopy(self._working_palette)
        # Re-apply QSS theme to the application
        app = QApplication.instance()
        if app and theme in _THEMES:
            app.setStyleSheet(_THEMES[theme])
        self.theme_applied.emit()

    def _apply_and_accept(self):
        self._apply_changes()
        self.accept()

    # ----------------------------------------------------------
    # Save / Load / Export
    # ----------------------------------------------------------

    def _theme_to_dict(self) -> dict:
        return {
            "name": self._current_theme,
            "editor_colors": deepcopy(self._working_palette),
            "font_family": self._font_family,
            "font_size": self._font_size,
        }

    def _export_json(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Theme as JSON", str(THEMES_DIR / "custom_theme.json"),
            "JSON files (*.json)",
        )
        if path:
            _ensure_themes_dir()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._theme_to_dict(), f, indent=2)
            QMessageBox.information(self, "Exported", f"Theme saved to:\n{path}")

    def _save_theme(self):
        _ensure_themes_dir()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Custom Theme", str(THEMES_DIR / "my_theme.json"),
            "JSON files (*.json)",
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._theme_to_dict(), f, indent=2)
            QMessageBox.information(self, "Saved", f"Theme saved to:\n{path}")

    def _load_theme(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Custom Theme", str(THEMES_DIR),
            "JSON files (*.json)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "editor_colors" in data:
                self._working_palette = data["editor_colors"]
            if "font_family" in data:
                self._font_family = data["font_family"]
                idx = self._font_combo.findText(self._font_family)
                if idx >= 0:
                    self._font_combo.setCurrentIndex(idx)
            if "font_size" in data:
                self._font_size = data["font_size"]
                self._size_slider.setValue(self._font_size)
            self._refresh_swatches()
            self._update_preview_colors()
        except Exception as exc:
            QMessageBox.warning(self, "Load Error", f"Failed to load theme:\n{exc}")

    # ----------------------------------------------------------
    # Dialog styling
    # ----------------------------------------------------------

    def _apply_dark_dialog_style(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: #e0e0e0;
            }
            QLabel {
                color: #e0e0e0;
            }
            QGroupBox {
                color: #4fc3f7;
                border: 1px solid #555;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 16px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QComboBox {
                background-color: #383838;
                color: #e0e0e0;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 4px 8px;
            }
            QComboBox:hover {
                border-color: #4fc3f7;
            }
            QComboBox QAbstractItemView {
                background-color: #383838;
                color: #e0e0e0;
                selection-background-color: #0d47a1;
            }
            QSlider::groove:horizontal {
                background: #555;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4fc3f7;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QPushButton {
                background-color: #4fc3f7;
                color: #212121;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #29b6f6;
            }
            QPushButton:pressed {
                background-color: #0288d1;
            }
            QTabWidget::pane {
                border: 1px solid #444;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #333;
                color: #9e9e9e;
                padding: 8px 16px;
                border: 1px solid #444;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #2b2b2b;
                color: #4fc3f7;
                border-bottom: 2px solid #4fc3f7;
            }
            QScrollArea {
                border: none;
                background-color: #2b2b2b;
            }
        """)
