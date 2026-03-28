"""Bookmarks panel -- lists all bookmarks across open editor tabs
(forge/gui/bookmarks_panel.py)."""

import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTreeWidget, QTreeWidgetItem, QAbstractItemView,
)


class BookmarksPanel(QWidget):
    """Panel that displays all bookmarks across open files.

    Signals
    -------
    navigate_requested(str, int)
        Emitted when the user double-clicks a bookmark.
        Arguments are (file_path_or_tab_title, line_number).
    """

    navigate_requested = Signal(str, int)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, parent=None):
        super().__init__(parent)
        self._editor_widget = None  # set via set_editor_widget()

        self.setObjectName("BookmarksPanel")
        self.apply_theme()

    def apply_theme(self):
        """Apply current theme colors to the bookmarks panel."""
        try:
            from forge.gui.themes import get_theme_palette
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            qss = app.styleSheet() if app else ""
            palette = None
            # Check light/midnight first since dark's bg0 also appears as fg0 in light QSS
            for name in ("light", "midnight", "dark"):
                try:
                    p = get_theme_palette(name)
                    bg0 = p.get("bg0", "")
                    bg1 = p.get("bg1", "")
                    # Both bg0 and bg1 must appear to confirm this is the active theme
                    if bg0 and bg1 and bg0 in qss and bg1 in qss:
                        palette = p
                        break
                except Exception:
                    pass
            if palette is None:
                palette = get_theme_palette("dark")
        except Exception:
            palette = {
                "bg0": "#1e1e2e", "bg1": "#252536", "bg2": "#2a2a3c",
                "fg0": "#cdd6f4", "fg3": "#6c7086", "border1": "#44445a",
                "accent": "#00BCD4", "accent_p": "#0097A7", "alt_row": "#282840",
                "selection": "#264f78", "bg5": "#44445a",
            }

        bg0 = palette.get("bg0", "#1e1e2e")
        bg1 = palette.get("bg1", "#252536")
        bg3 = palette.get("bg3", "#313145")
        bg5 = palette.get("bg5", "#44445a")
        fg0 = palette.get("fg0", "#cdd6f4")
        fg3 = palette.get("fg3", "#6c7086")
        border1 = palette.get("border1", "#44445a")
        accent = palette.get("accent", "#00BCD4")
        accent_p = palette.get("accent_p", "#0097A7")
        selection = palette.get("selection", "#264f78")
        alt_row = palette.get("alt_row", "#282840")

        self.setStyleSheet(f"""
            #BookmarksPanel {{
                background-color: {bg0};
            }}
            #BookmarksPanel QTreeWidget {{
                background-color: {bg0};
                color: {fg0};
                border: 1px solid {border1};
                font-family: "Courier New", monospace;
                font-size: 12px;
                selection-background-color: {selection};
                selection-color: {fg0};
                alternate-background-color: {alt_row};
            }}
            #BookmarksPanel QTreeWidget::item:hover {{
                background-color: {bg1};
            }}
            #BookmarksPanel QTreeWidget::item:selected {{
                background-color: {selection};
            }}
            #BookmarksPanel QHeaderView::section {{
                background-color: {bg3};
                color: {accent};
                border: 1px solid {border1};
                padding: 4px 8px;
                font-weight: bold;
            }}
            #BookmarksPanel QPushButton {{
                background-color: {accent_p};
                color: {fg0};
                border: none;
                border-radius: 4px;
                padding: 5px 12px;
                font-weight: bold;
                font-size: 12px;
            }}
            #BookmarksPanel QPushButton:hover {{
                background-color: {accent};
            }}
            #BookmarksPanel QPushButton:pressed {{
                background-color: {accent_p};
            }}
            #BookmarksPanel QPushButton:disabled {{
                background-color: {bg5};
                color: {fg3};
            }}
        """)

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # --- Toolbar row ---
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)

        self.btn_toggle = QPushButton("Toggle  (Ctrl+F2)")
        self.btn_toggle.setToolTip("Toggle bookmark on the current editor line")
        self.btn_toggle.clicked.connect(self._on_toggle)
        btn_layout.addWidget(self.btn_toggle)

        self.btn_prev = QPushButton("Previous  (F2)")
        self.btn_prev.setToolTip("Jump to the previous bookmark")
        self.btn_prev.clicked.connect(self._on_previous)
        btn_layout.addWidget(self.btn_prev)

        self.btn_next = QPushButton("Next  (Shift+F2)")
        self.btn_next.setToolTip("Jump to the next bookmark")
        self.btn_next.clicked.connect(self._on_next)
        btn_layout.addWidget(self.btn_next)

        self.btn_remove = QPushButton("Remove")
        self.btn_remove.setToolTip("Remove the selected bookmark")
        self.btn_remove.clicked.connect(self._on_remove)
        btn_layout.addWidget(self.btn_remove)

        self.btn_remove_all = QPushButton("Remove All")
        self.btn_remove_all.setToolTip("Clear every bookmark in every file")
        self.btn_remove_all.clicked.connect(self._on_remove_all)
        btn_layout.addWidget(self.btn_remove_all)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # --- Bookmark tree ---
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["File", "Line", "Preview"])
        self.tree.setColumnCount(3)
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.header().setStretchLastSection(True)
        self.tree.setColumnWidth(0, 180)
        self.tree.setColumnWidth(1, 55)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.tree)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_editor_widget(self, editor_widget):
        """Store a reference to the EditorWidget so we can query tabs."""
        self._editor_widget = editor_widget

    def refresh(self):
        """Re-scan all editor tabs and rebuild the tree."""
        self.tree.clear()
        ew = self._editor_widget
        if ew is None:
            return
        for idx in range(ew.tabs.count()):
            editor = ew.tabs.widget(idx)
            if not hasattr(editor, '_bookmarks'):
                continue
            tab_title = ew.tabs.tabText(idx)
            file_path = getattr(editor, 'file_path', None) or tab_title
            display_name = os.path.basename(file_path) if file_path else tab_title
            for line_no in sorted(editor._bookmarks):
                block = editor.document().findBlockByNumber(line_no - 1)
                preview = block.text().strip() if block.isValid() else ""
                item = QTreeWidgetItem([
                    display_name,
                    str(line_no),
                    preview[:120],
                ])
                # Store the real path and line for navigation
                item.setData(0, Qt.UserRole, file_path)
                item.setData(1, Qt.UserRole, line_no)
                self.tree.addTopLevelItem(item)

    # ------------------------------------------------------------------
    # Button / shortcut handlers
    # ------------------------------------------------------------------

    def _current_editor(self):
        if self._editor_widget is None:
            return None
        return self._editor_widget.get_current_editor()

    def _on_toggle(self):
        ed = self._current_editor()
        if ed is None:
            return
        line = ed.textCursor().blockNumber() + 1
        if line in ed._bookmarks:
            ed._bookmarks.discard(line)
        else:
            ed._bookmarks.add(line)
        ed.viewport().update()  # repaint gutter
        self.refresh()

    def _on_previous(self):
        ed = self._current_editor()
        if ed is None or not ed._bookmarks:
            return
        cur = ed.textCursor().blockNumber() + 1
        lower = sorted(b for b in ed._bookmarks if b < cur)
        target = lower[-1] if lower else max(ed._bookmarks)
        self._goto_line(ed, target)

    def _on_next(self):
        ed = self._current_editor()
        if ed is None or not ed._bookmarks:
            return
        cur = ed.textCursor().blockNumber() + 1
        upper = sorted(b for b in ed._bookmarks if b > cur)
        target = upper[0] if upper else min(ed._bookmarks)
        self._goto_line(ed, target)

    def _on_remove(self):
        item = self.tree.currentItem()
        if item is None:
            return
        file_path = item.data(0, Qt.UserRole)
        line_no = item.data(1, Qt.UserRole)
        ew = self._editor_widget
        if ew is None:
            return
        for idx in range(ew.tabs.count()):
            editor = ew.tabs.widget(idx)
            ed_path = getattr(editor, 'file_path', None) or ew.tabs.tabText(idx)
            if ed_path == file_path and hasattr(editor, '_bookmarks'):
                editor._bookmarks.discard(line_no)
                editor.viewport().update()
                break
        self.refresh()

    def _on_remove_all(self):
        ew = self._editor_widget
        if ew is None:
            return
        for idx in range(ew.tabs.count()):
            editor = ew.tabs.widget(idx)
            if hasattr(editor, '_bookmarks'):
                editor._bookmarks.clear()
                editor.viewport().update()
        self.refresh()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _on_item_double_clicked(self, item, _column):
        file_path = item.data(0, Qt.UserRole)
        line_no = item.data(1, Qt.UserRole)
        self.navigate_requested.emit(file_path, line_no)

    @staticmethod
    def _goto_line(editor, line_no):
        """Move the editor cursor to *line_no* (1-based)."""
        block = editor.document().findBlockByNumber(line_no - 1)
        if not block.isValid():
            return
        cursor = editor.textCursor()
        cursor.setPosition(block.position())
        editor.setTextCursor(cursor)
        editor.centerCursor()
