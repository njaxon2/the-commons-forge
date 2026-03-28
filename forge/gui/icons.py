"""
Custom SVG-based toolbar icons for the Forge IDE.

Every public function returns a QIcon built by painting simple geometric
shapes onto a QPixmap with QPainter.  Colour palette loosely follows
Catppuccin Mocha so the icons feel at home in a dark-themed editor.
"""

from PySide6.QtCore import Qt, QRect, QPoint, QPointF
from PySide6.QtGui import (
    QColor, QIcon, QPainter, QPen, QPixmap, QBrush, QPolygonF, QFont,
)

_SIZE = 32  # default icon size in pixels


def _make_pixmap(size: int = _SIZE) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    return pm


# -- colours --------------------------------------------------------------
# Dark theme defaults
_DARK_ICON_COLORS = {
    "teal":   "#94e2d5",
    "green":  "#a6e3a1",
    "red":    "#f38ba8",
    "orange": "#fab387",
    "mauve":  "#cba6f7",
    "text":   "#cdd6f4",
}

# Light theme: darker/more saturated for visibility on light backgrounds
_LIGHT_ICON_COLORS = {
    "teal":   "#00796B",
    "green":  "#2e7d32",
    "red":    "#c62828",
    "orange": "#e65100",
    "mauve":  "#6A1B9A",
    "text":   "#37474F",
}


def _is_light_theme() -> bool:
    """Detect if the current Qt application theme is light."""
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app:
            ss = app.styleSheet()
            # The light theme uses bg0: #f8f9fc
            if "#f8f9fc" in ss or "#eef0f5" in ss:
                return True
    except Exception:
        pass
    return False


def _icon_colors():
    """Return the appropriate icon color dict for the current theme."""
    return _LIGHT_ICON_COLORS if _is_light_theme() else _DARK_ICON_COLORS


def _get_teal():
    return QColor(_icon_colors()["teal"])

def _get_green():
    return QColor(_icon_colors()["green"])

def _get_red():
    return QColor(_icon_colors()["red"])

def _get_orange():
    return QColor(_icon_colors()["orange"])

def _get_mauve():
    return QColor(_icon_colors()["mauve"])

def _get_text():
    return QColor(_icon_colors()["text"])


# Keep module-level names for backward compatibility but they are now
# only used as fallback defaults. Each icon function fetches fresh colors.
TEAL    = QColor("#94e2d5")
GREEN   = QColor("#a6e3a1")
RED     = QColor("#f38ba8")
ORANGE  = QColor("#fab387")
MAUVE   = QColor("#cba6f7")
TEXT_FG = QColor("#cdd6f4")


# ========================================================================
# File operations
# ========================================================================

def icon_new_file() -> QIcon:
    """Document with a + sign, teal."""
    pm = _make_pixmap()
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(_get_teal(), 2)
    p.setPen(pen)
    # document outline
    p.drawRect(QRect(6, 2, 18, 26))
    # folded corner
    p.drawLine(18, 2, 24, 8)
    p.drawLine(24, 8, 18, 8)
    # plus sign
    pen.setWidth(2)
    p.setPen(pen)
    p.drawLine(12, 16, 20, 16)
    p.drawLine(16, 12, 16, 20)
    p.end()
    return QIcon(pm)


def icon_open() -> QIcon:
    """Open folder, teal."""
    pm = _make_pixmap()
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(_get_teal(), 2)
    p.setPen(pen)
    p.setBrush(QBrush(_get_teal().darker(200)))
    # back of folder
    pts_back = QPolygonF([
        QPointF(3, 10), QPointF(3, 27), QPointF(29, 27),
        QPointF(29, 10), QPointF(18, 10), QPointF(15, 6),
        QPointF(3, 6),
    ])
    p.drawPolygon(pts_back)
    # front flap (open)
    p.setBrush(QBrush(_get_teal().darker(150)))
    pts_front = QPolygonF([
        QPointF(1, 14), QPointF(7, 27), QPointF(29, 27),
        QPointF(31, 14),
    ])
    p.drawPolygon(pts_front)
    p.end()
    return QIcon(pm)


def icon_save() -> QIcon:
    """Floppy disk, teal."""
    pm = _make_pixmap()
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(_get_teal(), 2)
    p.setPen(pen)
    p.setBrush(QBrush(_get_teal().darker(250)))
    # outer body
    p.drawRect(QRect(4, 4, 24, 24))
    # label area (white-ish)
    p.setBrush(QBrush(_get_teal().darker(150)))
    p.drawRect(QRect(9, 4, 14, 10))
    # metal shutter
    p.setBrush(QBrush(_get_teal()))
    p.drawRect(QRect(8, 18, 16, 10))
    p.end()
    return QIcon(pm)


# ========================================================================
# Run / Stop
# ========================================================================

def icon_run() -> QIcon:
    """Play triangle, green."""
    pm = _make_pixmap()
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(_get_green()))
    tri = QPolygonF([
        QPointF(8, 4), QPointF(8, 28), QPointF(28, 16),
    ])
    p.drawPolygon(tri)
    p.end()
    return QIcon(pm)


def icon_stop() -> QIcon:
    """Stop square, red."""
    pm = _make_pixmap()
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(_get_red()))
    p.drawRect(QRect(6, 6, 20, 20))
    p.end()
    return QIcon(pm)


# ========================================================================
# Debugging
# ========================================================================

def icon_debug() -> QIcon:
    """Bug icon, orange."""
    pm = _make_pixmap()
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(_get_orange(), 2)
    p.setPen(pen)
    # body (ellipse)
    p.setBrush(QBrush(_get_orange().darker(180)))
    p.drawEllipse(QRect(9, 10, 14, 18))
    # head
    p.drawEllipse(QRect(12, 4, 8, 8))
    # legs left
    p.drawLine(9, 14, 3, 10)
    p.drawLine(9, 20, 3, 20)
    p.drawLine(9, 26, 3, 28)
    # legs right
    p.drawLine(23, 14, 29, 10)
    p.drawLine(23, 20, 29, 20)
    p.drawLine(23, 26, 29, 28)
    p.end()
    return QIcon(pm)


def icon_step_in() -> QIcon:
    """Arrow pointing down into a bracket (step into)."""
    pm = _make_pixmap()
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(_get_mauve(), 2.5)
    p.setPen(pen)
    # vertical arrow shaft
    p.drawLine(16, 4, 16, 22)
    # arrowhead
    p.drawLine(16, 22, 11, 17)
    p.drawLine(16, 22, 21, 17)
    # bracket at bottom
    p.drawLine(8, 26, 8, 29)
    p.drawLine(8, 29, 24, 29)
    p.drawLine(24, 29, 24, 26)
    p.end()
    return QIcon(pm)


def icon_step_over() -> QIcon:
    """Arrow curving over an obstacle (step over)."""
    pm = _make_pixmap()
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(_get_mauve(), 2.5)
    p.setPen(pen)
    # obstacle dot
    p.setBrush(QBrush(_get_mauve().darker(150)))
    p.drawEllipse(QRect(13, 20, 6, 6))
    # arc over
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawArc(QRect(4, 4, 24, 24), 30 * 16, 120 * 16)
    # arrowhead on the right end of arc
    p.drawLine(25, 14, 28, 10)
    p.drawLine(25, 14, 21, 10)
    p.end()
    return QIcon(pm)


def icon_step_out() -> QIcon:
    """Arrow going up out of a bracket (step out)."""
    pm = _make_pixmap()
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(_get_mauve(), 2.5)
    p.setPen(pen)
    # bracket at bottom
    p.drawLine(8, 24, 8, 29)
    p.drawLine(8, 29, 24, 29)
    p.drawLine(24, 29, 24, 24)
    # vertical arrow shaft going up
    p.drawLine(16, 22, 16, 4)
    # arrowhead
    p.drawLine(16, 4, 11, 9)
    p.drawLine(16, 4, 21, 9)
    p.end()
    return QIcon(pm)


# ========================================================================
# Edit operations
# ========================================================================

def icon_undo() -> QIcon:
    """Curved arrow pointing left (undo)."""
    pm = _make_pixmap()
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(_get_text(), 2.5)
    p.setPen(pen)
    # arc
    p.drawArc(QRect(6, 8, 20, 18), 90 * 16, 180 * 16)
    # arrowhead at top-left of arc
    p.drawLine(6, 17, 2, 12)
    p.drawLine(6, 17, 11, 13)
    p.end()
    return QIcon(pm)


def icon_redo() -> QIcon:
    """Curved arrow pointing right (redo)."""
    pm = _make_pixmap()
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(_get_text(), 2.5)
    p.setPen(pen)
    # arc
    p.drawArc(QRect(6, 8, 20, 18), -90 * 16, 180 * 16)
    # arrowhead at top-right of arc
    p.drawLine(26, 17, 30, 12)
    p.drawLine(26, 17, 21, 13)
    p.end()
    return QIcon(pm)


def icon_search() -> QIcon:
    """Magnifying glass."""
    pm = _make_pixmap()
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(_get_text(), 2.5)
    p.setPen(pen)
    # lens circle
    p.drawEllipse(QRect(4, 4, 18, 18))
    # handle
    p.drawLine(20, 20, 28, 28)
    p.end()
    return QIcon(pm)
