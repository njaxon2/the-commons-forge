"""Keyboard shortcuts cheat-sheet overlay for Forge IDE."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout,
    QScrollArea, QWidget, QFrame,
)


# -- Shortcut data by category ----------------------------------------

SHORTCUT_DATA = {
    "File": [
        ("Ctrl+N", "New file"),
        ("Ctrl+O", "Open file"),
        ("Ctrl+S", "Save"),
        ("Ctrl+Shift+S", "Save as"),
        ("Ctrl+Q", "Quit"),
    ],
    "Edit": [
        ("Ctrl+Z", "Undo"),
        ("Ctrl+Y", "Redo"),
        ("Ctrl+X", "Cut"),
        ("Ctrl+C", "Copy"),
        ("Ctrl+V", "Paste"),
        ("Ctrl+F", "Find / Replace"),
    ],
    "View": [
        ("Ctrl+1", "Command Window"),
        ("Ctrl+2", "Editor"),
        ("Ctrl+3", "File Browser"),
        ("Ctrl+4", "Workspace"),
    ],
    "Navigation": [
        ("Ctrl+G", "Go to line"),
        ("Ctrl+P", "Quick open file"),
        ("Ctrl+Tab", "Next editor tab"),
        ("Ctrl+Shift+Tab", "Previous editor tab"),
    ],
    "Code": [
        ("F5", "Run file"),
        ("Shift+F5", "Stop execution"),
        ("Ctrl+/", "Toggle comment"),
        ("Tab", "Indent"),
        ("Shift+Tab", "Outdent"),
        ("Ctrl+D", "Duplicate line"),
    ],
    "Debug": [
        ("F5", "Run / Continue"),
        ("F10", "Step over"),
        ("F11", "Step into"),
        ("Shift+F11", "Step out"),
        ("F9", "Toggle breakpoint"),
    ],
    "Search": [
        ("Ctrl+F", "Find in file"),
        ("Ctrl+H", "Find and replace"),
        ("Ctrl+Shift+F", "Find in project"),
        ("F3", "Find next"),
        ("Shift+F3", "Find previous"),
    ],
}

# -- Stylesheet fragments ---------------------------------------------

def _get_overlay_colors():
    """Get overlay colors based on current theme."""
    from forge.gui.theme_utils import detect_palette, is_light_theme
    _p = detect_palette()
    if is_light_theme():
        return {
            "overlay_bg": "rgba(240, 242, 248, 230)",
            "card_bg": "#ffffff",
            "accent": _p.get("accent", "#00897B"),
            "text": _p.get("fg0", "#1e1e2e"),
            "dim_text": _p.get("fg3", "#6c6f85"),
            "key_bg": _p.get("bg2", "#eef0f5"),
            "key_border": _p.get("border1", "#bcc0cc"),
            "title_clr": _p.get("accent", "#00897B"),
            "card_border": _p.get("border0", "#dce0e8"),
            "scrollbar_bg": _p.get("bg2", "#eef0f5"),
            "scrollbar_handle": _p.get("border1", "#bcc0cc"),
        }
    return {
        "overlay_bg": "rgba(20, 20, 36, 225)",
        "card_bg": _p.get("bg0", "#1e1e2e"),
        "accent": _p.get("accent", "#00BCD4"),
        "text": _p.get("fg0", "#cdd6f4"),
        "dim_text": _p.get("fg2", "#a6adc8"),
        "key_bg": _p.get("bg2", "#2a2a3e"),
        "key_border": _p.get("border1", "#45475a"),
        "title_clr": _p.get("accent", "#00BCD4"),
        "card_border": _p.get("border0", "#313244"),
        "scrollbar_bg": _p.get("bg0", "#1e1e2e"),
        "scrollbar_handle": _p.get("border1", "#45475a"),
    }

# Legacy aliases (used by _overlay_qss / _key_badge_qss at import time — refreshed in __init__)
_c = _get_overlay_colors()
_OVERLAY_BG = _c["overlay_bg"]
_CARD_BG    = _c["card_bg"]
_ACCENT     = _c["accent"]
_TEXT        = _c["text"]
_DIM_TEXT   = _c["dim_text"]
_KEY_BG     = _c["key_bg"]
_KEY_BORDER = _c["key_border"]
_TITLE_CLR  = _c["title_clr"]


def _overlay_qss(c=None):
    if c is None:
        c = _get_overlay_colors()
    return f"""
    ShortcutsOverlay {{
        background-color: {c['overlay_bg']};
    }}
    #shortcuts-card {{
        background-color: {c['card_bg']};
        border-radius: 12px;
        border: 1px solid {c['card_border']};
    }}
    #category-title {{
        color: {c['title_clr']};
        font-weight: bold;
        font-size: 13px;
        padding: 2px 0;
    }}
    #shortcut-desc {{
        color: {c['text']};
        font-size: 12px;
    }}
    #overlay-title {{
        color: {c['text']};
        font-size: 18px;
        font-weight: bold;
    }}
    #overlay-hint {{
        color: {c['dim_text']};
        font-size: 11px;
    }}
    """


def _key_badge_qss(c=None):
    """QSS for a single key badge label."""
    if c is None:
        c = _get_overlay_colors()
    return (
        f"background-color: {c['key_bg']};"
        f"color: {c['accent']};"
        f"border: 1px solid {c['key_border']};"
        "border-radius: 4px;"
        "padding: 2px 6px;"
        "font-family: 'Consolas', 'Fira Code', 'Courier New', monospace;"
        "font-size: 11px;"
        "font-weight: bold;"
    )


# -- Overlay dialog ----------------------------------------------------

class ShortcutsOverlay(QDialog):
    """Full-window semi-transparent overlay showing keyboard shortcuts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.Dialog
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._colors = _get_overlay_colors()
        c = self._colors
        self.setStyleSheet(_overlay_qss(c))

        # Fill the parent window
        if parent is not None:
            self.setGeometry(parent.rect())

        self._build_ui()

    # -- UI construction -----------------------------------------------

    def _build_ui(self):
        c = self._colors
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # Translucent click-catcher fills the whole overlay
        bg = QWidget(self)
        bg.setObjectName("shortcuts-bg")
        bg.setStyleSheet(f"background-color: {c['overlay_bg']};")
        outer.addWidget(bg)

        bg_layout = QVBoxLayout(bg)
        bg_layout.setContentsMargins(40, 30, 40, 30)

        # -- Title row --
        title_row = QHBoxLayout()
        title = QLabel("Keyboard Shortcuts")
        title.setObjectName("overlay-title")
        title_row.addWidget(title)
        title_row.addStretch()
        hint = QLabel("Press  Esc  or click anywhere to close")
        hint.setObjectName("overlay-hint")
        title_row.addWidget(hint)
        bg_layout.addLayout(title_row)
        bg_layout.addSpacing(10)

        # -- Scrollable card area --
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }"
                             f"QScrollBar:vertical {{ background: {c['card_bg']}; width: 8px; }}"
                             f"QScrollBar::handle:vertical {{ background: {c['key_border']}; border-radius: 4px; }}"
                             "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }")

        card = QWidget()
        card.setObjectName("shortcuts-card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 20, 24, 20)
        card_layout.setSpacing(18)

        # Build a two-column grid of categories
        columns_layout = QHBoxLayout()
        columns_layout.setSpacing(32)

        categories = list(SHORTCUT_DATA.items())
        mid = (len(categories) + 1) // 2
        left_cats = categories[:mid]
        right_cats = categories[mid:]

        for col_cats in (left_cats, right_cats):
            col_widget = QWidget()
            col_layout = QVBoxLayout(col_widget)
            col_layout.setContentsMargins(0, 0, 0, 0)
            col_layout.setSpacing(16)

            for cat_name, shortcuts in col_cats:
                col_layout.addWidget(self._make_category(cat_name, shortcuts))

            col_layout.addStretch()
            columns_layout.addWidget(col_widget)

        card_layout.addLayout(columns_layout)
        scroll.setWidget(card)
        bg_layout.addWidget(scroll)

    def _make_category(self, name, shortcuts):
        c = self._colors
        """Build a single category block with title + shortcut grid."""
        frame = QWidget()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title = QLabel(name)
        title.setObjectName("category-title")
        layout.addWidget(title)

        # Horizontal separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {c['card_border']};")
        sep.setFixedHeight(1)
        layout.addWidget(sep)

        grid = QGridLayout()
        grid.setContentsMargins(0, 4, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)

        for row, (key, desc) in enumerate(shortcuts):
            # Key badge(s) -- split on + for multi-key display
            key_container = QHBoxLayout()
            key_container.setSpacing(3)
            key_container.setContentsMargins(0, 0, 0, 0)
            parts = key.split("+")
            for i, part in enumerate(parts):
                badge = QLabel(part.strip())
                badge.setStyleSheet(_key_badge_qss(c))
                badge.setFixedHeight(22)
                key_container.addWidget(badge)
                if i < len(parts) - 1:
                    plus = QLabel("+")
                    plus.setStyleSheet(f"color: {c['dim_text']}; font-size: 11px;")
                    key_container.addWidget(plus)
            key_container.addStretch()

            key_widget = QWidget()
            key_widget.setLayout(key_container)
            grid.addWidget(key_widget, row, 0)

            desc_label = QLabel(desc)
            desc_label.setObjectName("shortcut-desc")
            grid.addWidget(desc_label, row, 1)

        grid.setColumnMinimumWidth(0, 150)
        layout.addLayout(grid)
        return frame

    # -- Event handling ------------------------------------------------

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        self.close()

    def showEvent(self, event):
        super().showEvent(event)
        # Re-fit to parent when shown
        if self.parent() is not None:
            self.setGeometry(self.parent().rect())
