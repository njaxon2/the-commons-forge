"""Forge Preferences Dialog (forge/gui/preferences_dialog.py).

Provides a tabbed dialog for IDE appearance and editor settings.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QLabel, QComboBox, QSpinBox, QPushButton, QGroupBox,
    QFormLayout, QColorDialog, QDialogButtonBox, QCheckBox,
    QFontComboBox, QTextEdit, QLineEdit, QFrame,
)

from forge.gui.themes import (
    get_preferences, save_preferences, THEMES, apply_theme,
    get_theme_palette,
)


class ColorButton(QPushButton):
    """Small button that shows / picks a colour."""
    color_changed = Signal(str)

    def __init__(self, color="#1e1e2e", parent=None):
        super().__init__(parent)
        self._color = color
        self.setFixedSize(60, 28)
        self._update_style()
        self.clicked.connect(self._pick)

    def _update_style(self):
        self.setStyleSheet(
            f"background-color: {self._color}; border: 1px solid #888; border-radius: 4px;"
        )

    def _pick(self):
        c = QColorDialog.getColor(QColor(self._color), self, "Choose Colour")
        if c.isValid():
            self._color = c.name()
            self._update_style()
            self.color_changed.emit(self._color)

    def color(self):
        return self._color

    def set_color(self, c):
        self._color = c
        self._update_style()


class PreferencesDialog(QDialog):
    """IDE Preferences with Appearance, Editor, and Advanced tabs."""

    preferences_changed = Signal(dict)   # emitted when user clicks Apply/OK

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumSize(520, 440)
        self._prefs = get_preferences()
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_appearance_tab(), "Appearance")
        self.tabs.addTab(self._build_editor_tab(), "Editor")
        self.tabs.addTab(self._build_advanced_tab(), "Advanced")
        layout.addWidget(self.tabs)

        # Buttons
        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        btn_box.accepted.connect(self._on_ok)
        btn_box.rejected.connect(self.reject)
        btn_box.button(QDialogButtonBox.Apply).clicked.connect(self._on_apply)
        layout.addWidget(btn_box)

    # ── Appearance ────────────────────────────────────────────────────
    def _build_appearance_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        # Theme
        grp_theme = QGroupBox("Theme")
        fl = QFormLayout(grp_theme)
        self.combo_theme = QComboBox()
        self.combo_theme.addItems(list(THEMES.keys()))
        cur = self._prefs.get("default_theme", "dark")
        idx = self.combo_theme.findText(cur)
        if idx >= 0:
            self.combo_theme.setCurrentIndex(idx)
        fl.addRow("Theme:", self.combo_theme)

        # Accent colour
        self.btn_accent = ColorButton(self._prefs.get("accent_color", "#89b4fa"))
        fl.addRow("Accent colour:", self.btn_accent)
        lay.addWidget(grp_theme)

        # Fonts
        grp_font = QGroupBox("Font")
        fl2 = QFormLayout(grp_font)
        self.font_combo = QFontComboBox()
        self.font_combo.setCurrentFont(
            QFont(self._prefs.get("font_family", "Consolas"))
        )
        fl2.addRow("Family:", self.font_combo)
        self.spin_font_size = QSpinBox()
        self.spin_font_size.setRange(6, 36)
        self.spin_font_size.setValue(self._prefs.get("font_size", 10))
        fl2.addRow("Size:", self.spin_font_size)
        lay.addWidget(grp_font)

        # Preview
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setMaximumHeight(100)
        self.preview.setPlainText(
            ">> x = linspace(0, 2*pi, 100);\n"
            ">> y = sin(x);\n"
            ">> plot(x, y, 'LineWidth', 2);\n"
            ">> title('Sine Wave');\n"
        )
        lay.addWidget(QLabel("Preview:"))
        lay.addWidget(self.preview)

        lay.addStretch()
        return w

    # ── Editor ────────────────────────────────────────────────────────
    def _build_editor_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        grp = QGroupBox("Editor Behaviour")
        fl = QFormLayout(grp)

        self.spin_tab_width = QSpinBox()
        self.spin_tab_width.setRange(1, 16)
        self.spin_tab_width.setValue(self._prefs.get("tab_width", 4))
        fl.addRow("Tab width:", self.spin_tab_width)

        self.chk_line_numbers = QCheckBox("Show line numbers")
        self.chk_line_numbers.setChecked(self._prefs.get("show_line_numbers", True))
        fl.addRow(self.chk_line_numbers)

        self.chk_highlight_line = QCheckBox("Highlight current line")
        self.chk_highlight_line.setChecked(self._prefs.get("highlight_current_line", True))
        fl.addRow(self.chk_highlight_line)

        self.chk_auto_indent = QCheckBox("Auto-indent")
        self.chk_auto_indent.setChecked(self._prefs.get("auto_indent", True))
        fl.addRow(self.chk_auto_indent)

        self.chk_bracket_match = QCheckBox("Bracket matching")
        self.chk_bracket_match.setChecked(self._prefs.get("bracket_matching", True))
        fl.addRow(self.chk_bracket_match)

        self.chk_strip_prompts = QCheckBox("Strip >> prompts on paste")
        self.chk_strip_prompts.setChecked(self._prefs.get("strip_prompts_on_paste", True))
        fl.addRow(self.chk_strip_prompts)

        lay.addWidget(grp)
        lay.addStretch()
        return w

    # ── Advanced ──────────────────────────────────────────────────────
    def _build_advanced_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        grp = QGroupBox("Custom CSS (appended to theme)")
        vl = QVBoxLayout(grp)
        self.txt_custom_css = QTextEdit()
        self.txt_custom_css.setPlainText(self._prefs.get("custom_css", ""))
        self.txt_custom_css.setMaximumHeight(200)
        vl.addWidget(self.txt_custom_css)
        lay.addWidget(grp)

        lay.addStretch()
        return w

    # ------------------------------------------------------------------
    def _collect(self) -> dict:
        return {
            "default_theme": self.combo_theme.currentText(),
            "accent_color": self.btn_accent.color(),
            "font_family": self.font_combo.currentFont().family(),
            "font_size": self.spin_font_size.value(),
            "tab_width": self.spin_tab_width.value(),
            "show_line_numbers": self.chk_line_numbers.isChecked(),
            "highlight_current_line": self.chk_highlight_line.isChecked(),
            "auto_indent": self.chk_auto_indent.isChecked(),
            "bracket_matching": self.chk_bracket_match.isChecked(),
            "strip_prompts_on_paste": self.chk_strip_prompts.isChecked(),
            "custom_css": self.txt_custom_css.toPlainText(),
        }

    def _on_apply(self):
        p = self._collect()
        save_preferences(p)
        self._prefs = p
        self.preferences_changed.emit(p)

    def _on_ok(self):
        self._on_apply()
        self.accept()
