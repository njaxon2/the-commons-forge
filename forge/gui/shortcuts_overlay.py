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

_OVERLAY_BG = "rgba(20, 20, 36, 225)"        # semi-transparent dark
_CARD_BG    = "#1e1e2e"
_ACCENT     = "#00BCD4"
_TEXT        = "#cdd6f4"
_DIM_TEXT   = "#a6adc8"
_KEY_BG     = "#2a2a3e"
_KEY_BORDER = "#45475a"
_TITLE_CLR  = "#00BCD4"


def _overlay_qss():
    return f"""
    ShortcutsOverlay {{
        background-color: {_OVERLAY_BG};
    }}
    #shortcuts-card {{
        background-color: {_CARD_BG};
        border-radius: 12px;
        border: 1px solid #313244;
    }}
    #category-title {{
        color: {_TITLE_CLR};
        font-weight: bold;
        font-size: 13px;
        padding: 2px 0;
    }}
    #shortcut-desc {{
        color: {_TEXT};
        font-size: 12px;
    }}
    #overlay-title {{
        color: {_TEXT};
        font-size: 18px;
        font-weight: bold;
    }}
    #overlay-hint {{
        color: {_DIM_TEXT};
        font-size: 11px;
    }}
    """


def _key_badge_qss():
    """QSS for a single key badge label."""
    return (
        f"background-color: {_KEY_BG};"
        f"color: {_ACCENT};"
        f"border: 1px solid {_KEY_BORDER};"
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
        self.setStyleSheet(_overlay_qss())

        # Fill the parent window
        if parent is not None:
            self.setGeometry(parent.rect())

        self._build_ui()

    # -- UI construction -----------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # Translucent click-catcher fills the whole overlay
        bg = QWidget(self)
        bg.setObjectName("shortcuts-bg")
        bg.setStyleSheet(f"background-color: {_OVERLAY_BG};")
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
                             "QScrollBar:vertical { background: #1e1e2e; width: 8px; }"
                             "QScrollBar::handle:vertical { background: #45475a; border-radius: 4px; }"
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
        sep.setStyleSheet(f"color: #313244;")
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
                badge.setStyleSheet(_key_badge_qss())
                badge.setFixedHeight(22)
                key_container.addWidget(badge)
                if i < len(parts) - 1:
                    plus = QLabel("+")
                    plus.setStyleSheet(f"color: {_DIM_TEXT}; font-size: 11px;")
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
