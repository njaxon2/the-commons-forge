"""Forge GUI Theme System.

Target location: forge/gui/themes.py

Provides QSS stylesheets for light and dark themes, theme application,
and syntax-highlight colour palettes for the code editor.
"""

from __future__ import annotations

from typing import Any, Dict, List


# =====================================================================
# QSS Stylesheets
# =====================================================================

LIGHT_THEME = """
/* ── Light Theme ────────────────────────────────────────────────── */
QMainWindow, QDialog, QWidget {
    background-color: #ffffff;
    color: #212121;
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
}

QMenuBar {
    background-color: #f5f5f5;
    color: #212121;
    border-bottom: 1px solid #e0e0e0;
}

QMenuBar::item:selected {
    background-color: #1976d2;
    color: #ffffff;
}

QMenu {
    background-color: #ffffff;
    color: #212121;
    border: 1px solid #e0e0e0;
}

QMenu::item:selected {
    background-color: #bbdefb;
    color: #212121;
}

QToolBar {
    background-color: #fafafa;
    border-bottom: 1px solid #e0e0e0;
    spacing: 4px;
    padding: 2px;
}

QToolButton:hover {
    background-color: #e3f2fd;
    border-radius: 3px;
}

QPushButton {
    background-color: #1976d2;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 6px 16px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #1565c0;
}

QPushButton:pressed {
    background-color: #0d47a1;
}

QPushButton:disabled {
    background-color: #bdbdbd;
    color: #757575;
}

QLineEdit, QTextEdit, QPlainTextEdit {
    background-color: #ffffff;
    color: #212121;
    border: 1px solid #bdbdbd;
    border-radius: 4px;
    padding: 4px 8px;
    selection-background-color: #bbdefb;
    selection-color: #212121;
}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
    border: 2px solid #1976d2;
}

QTabWidget::pane {
    border: 1px solid #e0e0e0;
    background-color: #ffffff;
}

QTabBar::tab {
    background-color: #f5f5f5;
    color: #616161;
    padding: 8px 16px;
    border: 1px solid #e0e0e0;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}

QTabBar::tab:selected {
    background-color: #ffffff;
    color: #1976d2;
    border-bottom: 2px solid #1976d2;
}

QTreeView, QListView, QTableView {
    background-color: #ffffff;
    color: #212121;
    alternate-background-color: #fafafa;
    border: 1px solid #e0e0e0;
    selection-background-color: #bbdefb;
    selection-color: #212121;
}

QTreeView::item:hover, QListView::item:hover {
    background-color: #e3f2fd;
}

QHeaderView::section {
    background-color: #f5f5f5;
    color: #424242;
    padding: 4px 8px;
    border: 1px solid #e0e0e0;
    font-weight: bold;
}

QScrollBar:vertical {
    background-color: #fafafa;
    width: 12px;
    border: none;
}

QScrollBar::handle:vertical {
    background-color: #bdbdbd;
    border-radius: 6px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #9e9e9e;
}

QScrollBar:horizontal {
    background-color: #fafafa;
    height: 12px;
    border: none;
}

QScrollBar::handle:horizontal {
    background-color: #bdbdbd;
    border-radius: 6px;
    min-width: 20px;
}

QStatusBar {
    background-color: #f5f5f5;
    color: #616161;
    border-top: 1px solid #e0e0e0;
}

QSplitter::handle {
    background-color: #e0e0e0;
}

QDockWidget::title {
    background-color: #f5f5f5;
    color: #424242;
    padding: 6px;
    border-bottom: 1px solid #e0e0e0;
}

QDockWidget::close-button, QDockWidget::float-button {
    background: transparent;
    border: none;
    padding: 2px;
}

QDockWidget::close-button:hover, QDockWidget::float-button:hover {
    background-color: #e0e0e0;
    border-radius: 2px;
}

QComboBox {
    background-color: #ffffff;
    color: #212121;
    border: 1px solid #bdbdbd;
    border-radius: 4px;
    padding: 4px 8px;
}

QComboBox:hover {
    border-color: #1976d2;
}

QProgressBar {
    background-color: #e0e0e0;
    border-radius: 4px;
    text-align: center;
    color: #212121;
}

QProgressBar::chunk {
    background-color: #1976d2;
    border-radius: 4px;
}

QToolTip {
    background-color: #424242;
    color: #ffffff;
    border: 1px solid #616161;
    padding: 4px;
    border-radius: 2px;
}
"""

DARK_THEME = """
/* ── Dark Theme ─────────────────────────────────────────────────── */
QMainWindow, QDialog, QWidget {
    background-color: #2b2b2b;
    color: #e0e0e0;
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
}

QMenuBar {
    background-color: #333333;
    color: #e0e0e0;
    border-bottom: 1px solid #444444;
}

QMenuBar::item:selected {
    background-color: #4fc3f7;
    color: #212121;
}

QMenu {
    background-color: #383838;
    color: #e0e0e0;
    border: 1px solid #555555;
}

QMenu::item:selected {
    background-color: #0d47a1;
    color: #e0e0e0;
}

QToolBar {
    background-color: #303030;
    border-bottom: 1px solid #444444;
    spacing: 4px;
    padding: 2px;
}

QToolButton:hover {
    background-color: #424242;
    border-radius: 3px;
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

QPushButton:disabled {
    background-color: #555555;
    color: #888888;
}

QLineEdit, QTextEdit, QPlainTextEdit {
    background-color: #383838;
    color: #e0e0e0;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 4px 8px;
    selection-background-color: #0d47a1;
    selection-color: #e0e0e0;
}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
    border: 2px solid #4fc3f7;
}

QTabWidget::pane {
    border: 1px solid #444444;
    background-color: #2b2b2b;
}

QTabBar::tab {
    background-color: #333333;
    color: #9e9e9e;
    padding: 8px 16px;
    border: 1px solid #444444;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}

QTabBar::tab:selected {
    background-color: #2b2b2b;
    color: #4fc3f7;
    border-bottom: 2px solid #4fc3f7;
}

QTreeView, QListView, QTableView {
    background-color: #2b2b2b;
    color: #e0e0e0;
    alternate-background-color: #333333;
    border: 1px solid #444444;
    selection-background-color: #0d47a1;
    selection-color: #e0e0e0;
}

QTreeView::item:hover, QListView::item:hover {
    background-color: #383838;
}

QHeaderView::section {
    background-color: #383838;
    color: #b0bec5;
    padding: 4px 8px;
    border: 1px solid #444444;
    font-weight: bold;
}

QScrollBar:vertical {
    background-color: #2b2b2b;
    width: 12px;
    border: none;
}

QScrollBar::handle:vertical {
    background-color: #555555;
    border-radius: 6px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #757575;
}

QScrollBar:horizontal {
    background-color: #2b2b2b;
    height: 12px;
    border: none;
}

QScrollBar::handle:horizontal {
    background-color: #555555;
    border-radius: 6px;
    min-width: 20px;
}

QStatusBar {
    background-color: #303030;
    color: #9e9e9e;
    border-top: 1px solid #444444;
}

QSplitter::handle {
    background-color: #444444;
}

QDockWidget::title {
    background-color: #333333;
    color: #b0bec5;
    padding: 6px;
    border-bottom: 1px solid #444444;
}

QDockWidget::close-button, QDockWidget::float-button {
    background: transparent;
    border: none;
    padding: 2px;
}

QDockWidget::close-button:hover, QDockWidget::float-button:hover {
    background-color: #555555;
    border-radius: 2px;
}

QComboBox {
    background-color: #383838;
    color: #e0e0e0;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 4px 8px;
}

QComboBox:hover {
    border-color: #4fc3f7;
}

QProgressBar {
    background-color: #444444;
    border-radius: 4px;
    text-align: center;
    color: #e0e0e0;
}

QProgressBar::chunk {
    background-color: #4fc3f7;
    border-radius: 4px;
}

QToolTip {
    background-color: #555555;
    color: #e0e0e0;
    border: 1px solid #757575;
    padding: 4px;
    border-radius: 2px;
}
"""


# =====================================================================
# Theme registry
# =====================================================================

_THEMES: Dict[str, str] = {
    'light': LIGHT_THEME,
    'dark': DARK_THEME,
}


# =====================================================================
# Syntax highlight colour palettes
# =====================================================================

_EDITOR_COLORS: Dict[str, Dict[str, str]] = {
    'light': {
        'keyword':      '#0000ff',   # blue
        'builtin':      '#795548',   # brown
        'string':       '#388e3c',   # green
        'number':       '#e65100',   # deep orange
        'comment':      '#9e9e9e',   # grey
        'operator':     '#212121',   # dark
        'function':     '#6a1b9a',   # purple
        'class':        '#00695c',   # teal
        'decorator':    '#ad1457',   # pink
        'self':         '#1565c0',   # dark blue
        'error':        '#d32f2f',   # red
        'line_number':  '#bdbdbd',   # light grey
        'current_line': '#f5f5f5',   # near white
        'brace_match':  '#bbdefb',   # light blue
        'selection':    '#bbdefb',   # light blue
        'background':   '#ffffff',   # white
        'foreground':   '#212121',   # dark
    },
    'dark': {
        'keyword':      '#4fc3f7',   # light blue (accent)
        'builtin':      '#ffb74d',   # amber
        'string':       '#81c784',   # light green
        'number':       '#ff8a65',   # deep orange light
        'comment':      '#757575',   # grey
        'operator':     '#e0e0e0',   # light
        'function':     '#ce93d8',   # light purple
        'class':        '#80cbc4',   # light teal
        'decorator':    '#f48fb1',   # light pink
        'self':         '#64b5f6',   # blue
        'error':        '#ef5350',   # red
        'line_number':  '#616161',   # dark grey
        'current_line': '#333333',   # slightly lighter bg
        'brace_match':  '#0d47a1',   # dark blue
        'selection':    '#0d47a1',   # dark blue
        'background':   '#2b2b2b',   # dark bg
        'foreground':   '#e0e0e0',   # light text
    },
}


# =====================================================================
# Public API
# =====================================================================

def get_available_themes() -> List[str]:
    """Return list of available theme names.

    Returns
    -------
    list of str
        Currently ``['light', 'dark']``.
    """
    return list(_THEMES.keys())


def apply_theme(app: Any, theme_name: str) -> None:
    """Apply a named theme to a QApplication instance.

    Parameters
    ----------
    app : QApplication
        The running Qt application instance.
    theme_name : str
        One of the names returned by :func:`get_available_themes`.

    Raises
    ------
    ValueError
        If *theme_name* is not recognised.
    """
    theme_name = theme_name.lower()
    if theme_name not in _THEMES:
        raise ValueError(
            f"Unknown theme '{theme_name}'. "
            f"Available: {get_available_themes()}"
        )
    app.setStyleSheet(_THEMES[theme_name])


def get_editor_colors(theme: str) -> Dict[str, str]:
    """Return syntax-highlight colour palette for a theme.

    Parameters
    ----------
    theme : str
        Theme name (``'light'`` or ``'dark'``).

    Returns
    -------
    dict
        Mapping of token type -> hex colour string.

    Raises
    ------
    ValueError
        If *theme* is not recognised.
    """
    theme = theme.lower()
    if theme not in _EDITOR_COLORS:
        raise ValueError(
            f"Unknown theme '{theme}'. "
            f"Available: {get_available_themes()}"
        )
    return dict(_EDITOR_COLORS[theme])
