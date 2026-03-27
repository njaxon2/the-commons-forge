"""
Notification Center - slide-in panel for Forge IDE.

Stores a history of toast notifications with timestamps, icons, and dismiss
functionality. Toggled via the bell icon in the status bar.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from PySide6.QtCore import (
    Qt, QPropertyAnimation, QEasingCurve, Signal, QPoint,
)
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QWidget, QGraphicsDropShadowEffect,
)


MAX_NOTIFICATIONS = 50


@dataclass
class NotificationEntry:
    """Single notification record."""
    level: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    read: bool = False
    uid: int = 0

class NotificationCard(QFrame):
    """Visual card for one notification inside the panel."""

    dismissed = Signal(int)
    navigate_requested = Signal(str, int)

    LEVEL_ICONS = {
        "info": "ℹ️",
        "warning": "⚠️",
        "error": "❌",
    }
    LEVEL_COLORS = {
        "info": "#3498db",
        "warning": "#f39c12",
        "error": "#e74c3c",
    }

    def __init__(self, entry: NotificationEntry, parent=None):
        super().__init__(parent)
        self._entry = entry
        self.setObjectName("NotificationCard")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setCursor(
            Qt.CursorShape.PointingHandCursor
            if entry.file_path
            else Qt.CursorShape.ArrowCursor
        )
        self._build_ui()

    def _build_ui(self):
        e = self._entry
        accent = self.LEVEL_COLORS.get(e.level, "#888")

        self.setStyleSheet(f"""
            #NotificationCard {{
                background: #2b2b2b;
                border-left: 3px solid {accent};
                border-radius: 4px;
                margin: 2px 4px;
                padding: 6px 8px;
            }}
            #NotificationCard:hover {{
                background: #333333;
            }}
        """)

        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        icon_lbl = QLabel(self.LEVEL_ICONS.get(e.level, ""))
        icon_lbl.setFixedWidth(22)
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignTop)
        root.addWidget(icon_lbl)

        text_col = QVBoxLayout()
        text_col.setSpacing(2)

        msg_lbl = QLabel(e.message)
        msg_lbl.setWordWrap(True)
        msg_lbl.setStyleSheet("color: #dcdcdc; font-size: 12px;")
        text_col.addWidget(msg_lbl)

        meta_parts = [e.timestamp.strftime("%H:%M:%S")]
        if e.file_path:
            short = os.path.basename(e.file_path)
            if e.line_number is not None:
                short += f":{e.line_number}"
            meta_parts.append(short)

        meta_lbl = QLabel("  ·  ".join(meta_parts))
        meta_lbl.setStyleSheet("color: #777; font-size: 10px;")
        text_col.addWidget(meta_lbl)

        root.addLayout(text_col, stretch=1)

        dismiss_btn = QPushButton("✕")
        dismiss_btn.setFixedSize(20, 20)
        dismiss_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #777;
                border: none;
                font-size: 13px;
            }
            QPushButton:hover { color: #e74c3c; }
        """)
        dismiss_btn.clicked.connect(lambda: self.dismissed.emit(self._entry.uid))
        root.addWidget(dismiss_btn, alignment=Qt.AlignmentFlag.AlignTop)

    def mousePressEvent(self, event):
        if self._entry.file_path and event.button() == Qt.MouseButton.LeftButton:
            self.navigate_requested.emit(
                self._entry.file_path,
                self._entry.line_number or 0,
            )
        super().mousePressEvent(event)

class NotificationCenter(QFrame):
    """Slide-in panel showing notification history."""

    navigate_to_file = Signal(str, int)
    PANEL_WIDTH = 340

    def __init__(self, parent=None):
        super().__init__(parent)
        self._entries: List[NotificationEntry] = []
        self._uid_counter = 0
        self._cards: dict[int, NotificationCard] = {}
        self._unread_count = 0
        self.setObjectName("NotificationCenter")
        self.setFixedWidth(self.PANEL_WIDTH)
        self.setMinimumHeight(200)
        self.hide()
        self._build_ui()
        self._apply_style()
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setOffset(-4, 0)
        shadow.setColor(QColor(0, 0, 0, 120))
        self.setGraphicsEffect(shadow)
        self._anim = QPropertyAnimation(self, b"pos")
        self._anim.setDuration(220)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        header = QFrame()
        header.setObjectName("NCHeader")
        header_lay = QHBoxLayout(header)
        header_lay.setContentsMargins(10, 8, 10, 8)
        title = QLabel("🔔 Notifications")
        title.setStyleSheet("color: #eee; font-size: 14px; font-weight: bold;")
        header_lay.addWidget(title)
        header_lay.addStretch()
        self._clear_btn = QPushButton("Clear All")
        self._clear_btn.setStyleSheet("""
            QPushButton {
                background: transparent; color: #e74c3c;
                border: 1px solid #e74c3c; border-radius: 3px;
                padding: 2px 8px; font-size: 11px;
            }
            QPushButton:hover { background: #e74c3c; color: #fff; }
        """)
        self._clear_btn.clicked.connect(self.clear_all)
        header_lay.addWidget(self._clear_btn)
        outer.addWidget(header)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet("""
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical { width: 6px; background: #1e1e1e; }
            QScrollBar::handle:vertical { background: #555; border-radius: 3px; }
        """)
        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setContentsMargins(4, 4, 4, 4)
        self._list_layout.setSpacing(4)
        self._list_layout.addStretch()
        self._scroll.setWidget(self._list_widget)
        outer.addWidget(self._scroll, stretch=1)
        self._empty_label = QLabel("No notifications")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color: #666; font-size: 12px; padding: 30px;")
        self._list_layout.insertWidget(0, self._empty_label)

    def _apply_style(self):
        self.setStyleSheet("""
            #NotificationCenter { background: #1e1e1e; border-left: 1px solid #444; }
            #NCHeader { background: #252525; border-bottom: 1px solid #444; }
        """)
    @property
    def unread_count(self) -> int:
        return self._unread_count

    def add_notification(self, message: str, level: str = "info",
                         file_path=None, line_number=None):
        """Append a notification. Trims to MAX_NOTIFICATIONS."""
        self._uid_counter += 1
        entry = NotificationEntry(
            level=level, message=message, file_path=file_path,
            line_number=line_number, uid=self._uid_counter,
        )
        self._entries.insert(0, entry)
        self._unread_count += 1
        while len(self._entries) > MAX_NOTIFICATIONS:
            removed = self._entries.pop()
            self._remove_card(removed.uid)
        self._add_card(entry, position=0)
        self._update_empty_state()

    def mark_all_read(self):
        for e in self._entries:
            e.read = True
        self._unread_count = 0

    def clear_all(self):
        for uid in list(self._cards.keys()):
            self._remove_card(uid)
        self._entries.clear()
        self._unread_count = 0
        self._update_empty_state()

    def _add_card(self, entry, position=0):
        card = NotificationCard(entry, self)
        card.dismissed.connect(self._on_dismiss)
        card.navigate_requested.connect(self.navigate_to_file.emit)
        self._cards[entry.uid] = card
        self._list_layout.insertWidget(position, card)

    def _remove_card(self, uid):
        card = self._cards.pop(uid, None)
        if card:
            self._list_layout.removeWidget(card)
            card.deleteLater()
        self._entries = [e for e in self._entries if e.uid != uid]

    def _on_dismiss(self, uid):
        entry = next((e for e in self._entries if e.uid == uid), None)
        if entry and not entry.read:
            self._unread_count = max(0, self._unread_count - 1)
        self._remove_card(uid)
        self._update_empty_state()

    def _update_empty_state(self):
        self._empty_label.setVisible(len(self._entries) == 0)

    def toggle(self):
        if self.isVisible():
            self.slide_out()
        else:
            self.slide_in()

    def slide_in(self):
        parent = self.parentWidget()
        if not parent:
            return
        self.setFixedHeight(parent.height())
        self.move(parent.width(), 0)
        self.show()
        self.raise_()
        self._anim.stop()
        self._anim.setStartValue(self.pos())
        self._anim.setEndValue(QPoint(parent.width() - self.PANEL_WIDTH, 0))
        self._anim.start()
        self.mark_all_read()

    def slide_out(self):
        parent = self.parentWidget()
        if not parent:
            self.hide()
            return
        self._anim.stop()
        self._anim.setStartValue(self.pos())
        self._anim.setEndValue(QPoint(parent.width(), 0))
        self._anim.finished.connect(self._on_slide_out_done)
        self._anim.start()

    def _on_slide_out_done(self):
        self.hide()
        try:
            self._anim.finished.disconnect(self._on_slide_out_done)
        except RuntimeError:
            pass

    def reposition(self):
        """Keep panel anchored to right edge on resize."""
        parent = self.parentWidget()
        if not parent or not self.isVisible():
            return
        self.setFixedHeight(parent.height())
        self.move(parent.width() - self.PANEL_WIDTH, 0)