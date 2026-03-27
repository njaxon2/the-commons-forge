"""
Professional splash screen for Forge.
"""

from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QFont, QLinearGradient, QColor, QPainter, QBrush
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QProgressBar, QApplication, QGraphicsOpacityEffect,
)


class ForgeSplashScreen(QWidget):
    """Frameless splash screen with gradient background, progress bar, and fade-in."""

    _TEAL = "#00BCD4"
    _BG_TOP = "#1e1e2e"
    _BG_BOTTOM = "#0d0d18"
    _VERSION = "1.0.0"
    _DURATION_MS = 2000  # auto-close after 2 s

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.SplashScreen)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setFixedSize(520, 320)
        self._center_on_screen()

        # --- opacity effect for fade-in ---
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity_effect)

        self._fade_anim = QPropertyAnimation(self._opacity_effect, b"opacity")
        self._fade_anim.setDuration(600)
        self._fade_anim.setStartValue(0.0)
        self._fade_anim.setEndValue(1.0)
        self._fade_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        # --- layout ---
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 30)
        layout.setSpacing(6)

        # Title
        self._title = QLabel("FORGE")
        title_font = QFont("Segoe UI", 42, QFont.Weight.Bold)
        self._title.setFont(title_font)
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title.setStyleSheet(f"color: {self._TEAL}; background: transparent;")
        layout.addWidget(self._title)

        # Subtitle
        self._subtitle = QLabel("Octave-Compatible Computing Environment")
        sub_font = QFont("Segoe UI", 12)
        self._subtitle.setFont(sub_font)
        self._subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._subtitle.setStyleSheet("color: #aaaacc; background: transparent;")
        layout.addWidget(self._subtitle)

        layout.addStretch()

        # Status message
        self._status = QLabel("Initializing...")
        self._status.setFont(QFont("Segoe UI", 9))
        self._status.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._status.setStyleSheet("color: #888899; background: transparent;")
        layout.addWidget(self._status)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setFixedHeight(6)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._progress.setStyleSheet(f"""
            QProgressBar {{
                background: #2a2a3d;
                border: none;
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background: {self._TEAL};
                border-radius: 3px;
            }}
        """)
        layout.addWidget(self._progress)

        # Version label
        self._version_label = QLabel(f"v{self._VERSION}")
        self._version_label.setFont(QFont("Segoe UI", 8))
        self._version_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._version_label.setStyleSheet("color: #555566; background: transparent;")
        layout.addWidget(self._version_label)

        # --- progress timer ---
        self._progress_value = 0
        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(40)
        self._tick_timer.timeout.connect(self._tick_progress)

        # --- auto-close timer ---
        self._close_timer = QTimer(self)
        self._close_timer.setSingleShot(True)
        self._close_timer.setInterval(self._DURATION_MS)
        self._close_timer.timeout.connect(self.finish)

        # Status messages shown at various progress %
        self._messages = [
            (0, "Initializing..."),
            (20, "Loading modules..."),
            (45, "Preparing workspace..."),
            (70, "Starting GUI engine..."),
            (90, "Almost ready..."),
        ]

    # --- painting -----------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        grad = QLinearGradient(0, 0, 0, self.height())
        grad.setColorAt(0.0, QColor(self._BG_TOP))
        grad.setColorAt(1.0, QColor(self._BG_BOTTOM))
        painter.setBrush(QBrush(grad))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 12, 12)
        painter.end()

    # --- helpers -------------------------------------------------------------

    def _center_on_screen(self):
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            x = (geo.width() - self.width()) // 2 + geo.x()
            y = (geo.height() - self.height()) // 2 + geo.y()
            self.move(x, y)

    def _tick_progress(self):
        self._progress_value = min(self._progress_value + 2, 100)
        self._progress.setValue(self._progress_value)
        for threshold, msg in self._messages:
            if self._progress_value >= threshold:
                self._status.setText(msg)
        if self._progress_value >= 100:
            self._tick_timer.stop()

    # --- public API ----------------------------------------------------------

    def start(self):
        """Show the splash and begin animations."""
        self.show()
        self._fade_anim.start()
        self._tick_timer.start()
        self._close_timer.start()
        QApplication.processEvents()

    def set_status(self, text: str):
        """Update the status message from outside."""
        self._status.setText(text)
        QApplication.processEvents()

    def finish(self):
        """Close the splash screen."""
        self._tick_timer.stop()
        self._close_timer.stop()
        self._progress.setValue(100)
        QApplication.processEvents()
        self.close()
