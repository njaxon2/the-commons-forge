"""Forge application entry point."""
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
    app.setApplicationVersion('0.1.0')

    # Default monospace font for the whole application
    mono = QFont('Consolas', 10)
    mono.setStyleHint(QFont.StyleHint.Monospace)
    app.setFont(mono)

    # C. Apply theme on startup
    from forge.gui.themes import apply_theme
    apply_theme(app, 'light')

    # Create engine session
    from forge.engine.session import ForgeSession
    session = ForgeSession()

    # Create main window
    from forge.gui.main_window import ForgeMainWindow
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

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
