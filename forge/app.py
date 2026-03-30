"""Forge application entry point."""
import logging
logging.basicConfig(level=logging.WARNING)
from forge import __version__
import sys
import os


def main():
    """Launch the Forge IDE."""
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFont, QIcon
    from PySide6.QtCore import Qt

    # High-DPI support (must be set before QApplication creation)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName('Forge')
    app.setOrganizationName('Forge')
    app.setApplicationVersion(__version__)

    # Show splash screen
    from PySide6.QtWidgets import QSplashScreen
    from PySide6.QtGui import QPixmap, QPainter, QColor, QFont as QF, QLinearGradient
    from PySide6.QtCore import QRect

    splash_pix = QPixmap(480, 300)
    painter = QPainter(splash_pix)
    # Gradient background
    grad = QLinearGradient(0, 0, 0, 300)
    grad.setColorAt(0, QColor("#1e1e2e"))
    grad.setColorAt(1, QColor("#11111b"))
    painter.fillRect(0, 0, 480, 300, grad)
    # Title
    painter.setPen(QColor("#89b4fa"))
    title_font = QF("Consolas", 36, QF.Bold)
    painter.setFont(title_font)
    painter.drawText(QRect(0, 60, 480, 60), Qt.AlignCenter, "Forge")
    # Subtitle
    painter.setPen(QColor("#cdd6f4"))
    sub_font = QF("Consolas", 12)
    painter.setFont(sub_font)
    painter.drawText(QRect(0, 130, 480, 30), Qt.AlignCenter,
                     "Octave-Compatible Computing Environment")
    # Version
    painter.setPen(QColor("#6c7086"))
    ver_font = QF("Consolas", 10)
    painter.setFont(ver_font)
    painter.drawText(QRect(0, 165, 480, 25), Qt.AlignCenter, f"v{__version__}")
    # Loading text
    painter.setPen(QColor("#585b70"))
    painter.drawText(QRect(0, 250, 480, 25), Qt.AlignCenter,
                     "Loading engine...")
    # Border
    painter.setPen(QColor("#313244"))
    painter.drawRect(0, 0, 479, 299)
    painter.end()

    splash = QSplashScreen(splash_pix)
    splash.show()
    app.processEvents()

    # Load user preferences
    from forge.gui.themes import get_preferences, apply_theme
    from forge.gui.splash_screen import ForgeSplashScreen
    prefs = get_preferences()
    theme_name = prefs.get('default_theme', 'dark')
    font_family = prefs.get('font_family', 'Consolas')
    font_size = prefs.get('font_size', 10)

    # Default monospace font for the whole application
    mono = QFont(font_family, font_size)
    mono.setStyleHint(QFont.StyleHint.Monospace)
    app.setFont(mono)

    # Apply theme on startup
    apply_theme(app, theme_name)
    from forge.gui.editor_widget import set_editor_palette
    set_editor_palette(theme_name)


    # Create engine session
    from forge.engine.session import ForgeSession
    session = ForgeSession()

    # Create main window
    from forge.gui.main_window import ForgeMainWindow
    # --- Splash screen ---
    splash = ForgeSplashScreen()
    splash.start()
    app.processEvents()

    window = ForgeMainWindow()
    window.setup_engine(session)
    window.setWindowTitle('Forge \u2014 Octave-Compatible Computing Environment')
    window.resize(1200, 800)

    # Centre on screen
    screen_geo = app.primaryScreen().availableGeometry()
    frame_geo = window.frameGeometry()
    frame_geo.moveCenter(screen_geo.center())
    window.move(frame_geo.topLeft())

    window.show()

    # Close splash now that main window is visible
    splash.finish()

    # Close splash
    try:
        splash.finish(window)
    except Exception:
        pass

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
