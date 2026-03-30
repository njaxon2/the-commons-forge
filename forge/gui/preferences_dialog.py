"""Comprehensive Preferences dialog for Forge IDE."""
import json
import os
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QLabel, QComboBox, QSpinBox, QCheckBox, QPushButton,
    QFontComboBox, QGroupBox, QFormLayout, QLineEdit,
    QColorDialog, QDialogButtonBox, QListWidget, QPlainTextEdit,
)


PREFS_FILE = os.path.expanduser("~/.forge/preferences.json")

DEFAULT_PREFS = {
    "editor.font_family": "Fira Code",
    "editor.font_size": 13,
    "editor.tab_size": 4,
    "editor.insert_spaces": True,
    "editor.word_wrap": False,
    "editor.show_line_numbers": True,
    "editor.show_minimap": True,
    "editor.show_indent_guides": True,
    "editor.highlight_current_line": True,
    "editor.bracket_matching": True,
    "editor.auto_close_brackets": True,
    "editor.auto_indent": True,
    "editor.show_whitespace": False,
    "editor.rulers": [80, 120],
    "editor.auto_save": True,
    "editor.auto_save_delay": 30,
    "appearance.theme": "dark",
    "appearance.icon_theme": "default",
    "appearance.ui_font_size": 10,
    "terminal.shell": "/bin/bash",
    "terminal.font_size": 12,
    "terminal.scrollback": 10000,
    "terminal.cursor_style": "block",
    "command.history_size": 1000,
    "command.auto_complete": True,
    "files.exclude_patterns": ["*.pyc", "__pycache__", ".git"],
    "files.auto_detect_encoding": True,
    "files.default_encoding": "utf-8",
    "files.eol": "auto",
}


def load_prefs():
    """Load preferences from disk, merging with defaults."""
    prefs = dict(DEFAULT_PREFS)
    if os.path.exists(PREFS_FILE):
        try:
            with open(PREFS_FILE, 'r') as f:
                saved = json.load(f)
            prefs.update(saved)
        except Exception:
            pass
    return prefs


def save_prefs(prefs):
    """Save preferences to disk."""
    os.makedirs(os.path.dirname(PREFS_FILE), exist_ok=True)
    with open(PREFS_FILE, 'w') as f:
        json.dump(prefs, f, indent=2)


class PreferencesDialog(QDialog):
    """Full IDE preferences dialog."""

    preferences_changed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumSize(650, 520)
        self._prefs = load_prefs()
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        tabs = QTabWidget()
        tabs.addTab(self._create_editor_tab(), "Editor")
        tabs.addTab(self._create_appearance_tab(), "Appearance")
        tabs.addTab(self._create_terminal_tab(), "Terminal")
        tabs.addTab(self._create_files_tab(), "Files")
        tabs.addTab(self._create_shortcuts_tab(), "Shortcuts")
        layout.addWidget(tabs)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self._apply)
        layout.addWidget(buttons)

    def _create_editor_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        # Font group
        font_group = QGroupBox("Font")
        font_layout = QFormLayout(font_group)

        self._font_combo = QFontComboBox()
        self._font_combo.setCurrentFont(QFont(self._prefs["editor.font_family"]))
        font_layout.addRow("Family:", self._font_combo)

        self._font_size = QSpinBox()
        self._font_size.setRange(8, 36)
        self._font_size.setValue(self._prefs["editor.font_size"])
        font_layout.addRow("Size:", self._font_size)

        self._tab_size = QSpinBox()
        self._tab_size.setRange(1, 8)
        self._tab_size.setValue(self._prefs["editor.tab_size"])
        font_layout.addRow("Tab Size:", self._tab_size)

        self._insert_spaces = QCheckBox("Insert spaces instead of tabs")
        self._insert_spaces.setChecked(self._prefs["editor.insert_spaces"])
        font_layout.addRow(self._insert_spaces)

        layout.addWidget(font_group)

        # Display group
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)

        self._show_line_numbers = QCheckBox("Show line numbers")
        self._show_line_numbers.setChecked(self._prefs["editor.show_line_numbers"])
        display_layout.addWidget(self._show_line_numbers)

        self._show_minimap = QCheckBox("Show minimap")
        self._show_minimap.setChecked(self._prefs["editor.show_minimap"])
        display_layout.addWidget(self._show_minimap)

        self._show_indent_guides = QCheckBox("Show indentation guides")
        self._show_indent_guides.setChecked(self._prefs["editor.show_indent_guides"])
        display_layout.addWidget(self._show_indent_guides)

        self._highlight_line = QCheckBox("Highlight current line")
        self._highlight_line.setChecked(self._prefs["editor.highlight_current_line"])
        display_layout.addWidget(self._highlight_line)

        self._bracket_matching = QCheckBox("Bracket matching")
        self._bracket_matching.setChecked(self._prefs["editor.bracket_matching"])
        display_layout.addWidget(self._bracket_matching)

        self._word_wrap = QCheckBox("Word wrap")
        self._word_wrap.setChecked(self._prefs["editor.word_wrap"])
        display_layout.addWidget(self._word_wrap)

        self._show_whitespace = QCheckBox("Show whitespace characters")
        self._show_whitespace.setChecked(self._prefs["editor.show_whitespace"])
        display_layout.addWidget(self._show_whitespace)

        layout.addWidget(display_group)

        # Behavior group
        behavior_group = QGroupBox("Behavior")
        behavior_layout = QVBoxLayout(behavior_group)

        self._auto_close_brackets = QCheckBox("Auto-close brackets and quotes")
        self._auto_close_brackets.setChecked(self._prefs["editor.auto_close_brackets"])
        behavior_layout.addWidget(self._auto_close_brackets)

        self._auto_indent = QCheckBox("Auto-indent on new line")
        self._auto_indent.setChecked(self._prefs["editor.auto_indent"])
        behavior_layout.addWidget(self._auto_indent)

        self._auto_save = QCheckBox("Auto-save files")
        self._auto_save.setChecked(self._prefs["editor.auto_save"])
        behavior_layout.addWidget(self._auto_save)

        layout.addWidget(behavior_group)
        layout.addStretch()
        return w

    def _create_appearance_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        theme_group = QGroupBox("Theme")
        theme_layout = QFormLayout(theme_group)

        self._theme_combo = QComboBox()
        self._theme_combo.addItems(["dark", "light", "midnight"])
        self._theme_combo.setCurrentText(self._prefs["appearance.theme"])
        theme_layout.addRow("Color Theme:", self._theme_combo)

        self._ui_font_size = QSpinBox()
        self._ui_font_size.setRange(8, 20)
        self._ui_font_size.setValue(self._prefs["appearance.ui_font_size"])
        theme_layout.addRow("UI Font Size:", self._ui_font_size)

        layout.addWidget(theme_group)

        # Preview
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        self._theme_preview = QLabel(
            "The quick brown fox jumps over the lazy dog.\n"
            "function result = compute(x, y)\n"
            "    result = x .* y + sin(x);\n"
            "end"
        )
        self._theme_preview.setFont(QFont("Fira Code", 11))
        from forge.gui.theme_utils import detect_palette
        _p = detect_palette()
        self._theme_preview.setStyleSheet(
            f"padding: 16px; border-radius: 8px; "
            f"background: {_p.get('bg0', '#1e1e2e')}; color: {_p.get('fg0', '#cdd6f4')};"
        )
        preview_layout.addWidget(self._theme_preview)
        layout.addWidget(preview_group)

        self._theme_combo.currentTextChanged.connect(self._update_theme_preview)

        layout.addStretch()
        return w

    def _update_theme_preview(self, theme_name):
        """Update the theme preview swatch."""
        try:
            from forge.gui.themes import get_theme_palette
            p = get_theme_palette(theme_name)
            style = f"background: {p.get('bg0', '#1e1e2e')}; color: {p.get('fg0', '#cdd6f4')};"
        except Exception:
            style = "background: #1e1e2e; color: #cdd6f4;"
        self._theme_preview.setStyleSheet(
            f"padding: 16px; border-radius: 8px; {style}"
        )

    def _create_terminal_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        shell_group = QGroupBox("Shell")
        shell_layout = QFormLayout(shell_group)

        self._shell_path = QLineEdit(self._prefs["terminal.shell"])
        shell_layout.addRow("Shell Path:", self._shell_path)

        self._term_font_size = QSpinBox()
        self._term_font_size.setRange(8, 24)
        self._term_font_size.setValue(self._prefs["terminal.font_size"])
        shell_layout.addRow("Font Size:", self._term_font_size)

        self._scrollback = QSpinBox()
        self._scrollback.setRange(100, 100000)
        self._scrollback.setSingleStep(1000)
        self._scrollback.setValue(self._prefs["terminal.scrollback"])
        shell_layout.addRow("Scrollback Lines:", self._scrollback)

        self._cursor_style = QComboBox()
        self._cursor_style.addItems(["block", "underline", "line"])
        self._cursor_style.setCurrentText(self._prefs["terminal.cursor_style"])
        shell_layout.addRow("Cursor Style:", self._cursor_style)

        layout.addWidget(shell_group)

        # Command window settings
        cmd_group = QGroupBox("Command Window")
        cmd_layout = QFormLayout(cmd_group)

        self._history_size = QSpinBox()
        self._history_size.setRange(100, 50000)
        self._history_size.setSingleStep(100)
        self._history_size.setValue(self._prefs["command.history_size"])
        cmd_layout.addRow("History Size:", self._history_size)

        self._auto_complete = QCheckBox("Enable auto-completion")
        self._auto_complete.setChecked(self._prefs["command.auto_complete"])
        cmd_layout.addRow(self._auto_complete)

        layout.addWidget(cmd_group)
        layout.addStretch()
        return w

    def _create_files_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        encoding_group = QGroupBox("Encoding")
        enc_layout = QFormLayout(encoding_group)

        self._encoding = QComboBox()
        self._encoding.addItems(["utf-8", "ascii", "latin-1", "utf-16", "cp1252"])
        self._encoding.setCurrentText(self._prefs["files.default_encoding"])
        enc_layout.addRow("Default Encoding:", self._encoding)

        self._eol = QComboBox()
        self._eol.addItems(["auto", "LF (Unix)", "CRLF (Windows)", "CR (Mac)"])
        eol_val = self._prefs["files.eol"]
        self._eol.setCurrentText(eol_val if eol_val in ["auto"] else "auto")
        enc_layout.addRow("Line Endings:", self._eol)

        self._auto_detect_encoding = QCheckBox("Auto-detect file encoding")
        self._auto_detect_encoding.setChecked(self._prefs["files.auto_detect_encoding"])
        enc_layout.addRow(self._auto_detect_encoding)

        layout.addWidget(encoding_group)

        # Exclude patterns
        exclude_group = QGroupBox("Exclude Patterns")
        exclude_layout = QVBoxLayout(exclude_group)
        self._exclude_list = QPlainTextEdit()
        self._exclude_list.setPlainText("\n".join(self._prefs["files.exclude_patterns"]))
        self._exclude_list.setMaximumHeight(100)
        exclude_layout.addWidget(self._exclude_list)
        exclude_layout.addWidget(QLabel("One pattern per line"))
        layout.addWidget(exclude_group)

        layout.addStretch()
        return w

    def _create_shortcuts_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        layout.addWidget(QLabel("Keyboard Shortcuts"))

        shortcuts = [
            ("New File", "Ctrl+N"),
            ("Open File", "Ctrl+O"),
            ("Save", "Ctrl+S"),
            ("Save As", "Ctrl+Shift+S"),
            ("Close Tab", "Ctrl+W"),
            ("Quick Open", "Ctrl+P"),
            ("Command Palette", "Ctrl+Shift+P"),
            ("Find & Replace", "Ctrl+F"),
            ("Go to Line", "Ctrl+G"),
            ("Search in Files", "Ctrl+Shift+F"),
            ("Run File", "F5"),
            ("Run Selection", "F9"),
            ("Toggle Comment", "Ctrl+/"),
            ("Duplicate Line", "Ctrl+D"),
            ("Delete Line", "Ctrl+Shift+K"),
            ("Move Line Up", "Alt+Up"),
            ("Move Line Down", "Alt+Down"),
            ("Toggle Bookmark", "Ctrl+F2"),
            ("Next Bookmark", "F2"),
            ("Previous Bookmark", "Shift+F2"),
            ("Cycle Theme", "Ctrl+Shift+T"),
            ("Preferences", "Ctrl+,"),
            ("Focus Command Window", "Ctrl+0"),
            ("Focus Editor", "Ctrl+1"),
        ]

        self._shortcut_list = QListWidget()
        for name, keys in shortcuts:
            self._shortcut_list.addItem(f"{name:<30} {keys}")
        self._shortcut_list.setFont(QFont("Consolas", 11))
        layout.addWidget(self._shortcut_list)

        layout.addWidget(QLabel("Shortcut customization coming in a future update."))
        layout.addStretch()
        return w

    def _apply(self):
        """Apply current settings."""
        self._prefs["editor.font_family"] = self._font_combo.currentFont().family()
        self._prefs["editor.font_size"] = self._font_size.value()
        self._prefs["editor.tab_size"] = self._tab_size.value()
        self._prefs["editor.insert_spaces"] = self._insert_spaces.isChecked()
        self._prefs["editor.show_line_numbers"] = self._show_line_numbers.isChecked()
        self._prefs["editor.show_minimap"] = self._show_minimap.isChecked()
        self._prefs["editor.show_indent_guides"] = self._show_indent_guides.isChecked()
        self._prefs["editor.highlight_current_line"] = self._highlight_line.isChecked()
        self._prefs["editor.bracket_matching"] = self._bracket_matching.isChecked()
        self._prefs["editor.word_wrap"] = self._word_wrap.isChecked()
        self._prefs["editor.show_whitespace"] = self._show_whitespace.isChecked()
        self._prefs["editor.auto_close_brackets"] = self._auto_close_brackets.isChecked()
        self._prefs["editor.auto_indent"] = self._auto_indent.isChecked()
        self._prefs["editor.auto_save"] = self._auto_save.isChecked()
        self._prefs["appearance.theme"] = self._theme_combo.currentText()
        self._prefs["appearance.ui_font_size"] = self._ui_font_size.value()
        self._prefs["terminal.shell"] = self._shell_path.text()
        self._prefs["terminal.font_size"] = self._term_font_size.value()
        self._prefs["terminal.scrollback"] = self._scrollback.value()
        self._prefs["terminal.cursor_style"] = self._cursor_style.currentText()
        self._prefs["command.history_size"] = self._history_size.value()
        self._prefs["command.auto_complete"] = self._auto_complete.isChecked()
        self._prefs["files.default_encoding"] = self._encoding.currentText()
        self._prefs["files.eol"] = self._eol.currentText()
        self._prefs["files.auto_detect_encoding"] = self._auto_detect_encoding.isChecked()
        self._prefs["files.exclude_patterns"] = [
            p.strip() for p in self._exclude_list.toPlainText().split("\n") if p.strip()
        ]

        save_prefs(self._prefs)
        self.preferences_changed.emit(self._prefs)

    def _on_ok(self):
        self._apply()
        self.accept()
