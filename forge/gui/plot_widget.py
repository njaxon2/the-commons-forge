"""Forge plot widget — matplotlib embedded in PySide6
(forge/gui/plot_widget.py)."""

from PySide6.QtWidgets import QWidget, QVBoxLayout

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure


class PlotWidget(QWidget):
    """Embeddable matplotlib figure with navigation toolbar."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.ax = self.figure.add_subplot(111)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot(self, x, y=None, **kwargs):
        """Plot data on the current axes."""
        if y is None:
            self.ax.plot(x, **kwargs)
        else:
            self.ax.plot(x, y, **kwargs)
        self.canvas.draw()

    def clear(self):
        """Clear the axes."""
        self.ax.cla()
        self.canvas.draw()

    def set_title(self, t: str):
        self.ax.set_title(t)
        self.canvas.draw()

    def set_xlabel(self, label: str):
        self.ax.set_xlabel(label)
        self.canvas.draw()

    def set_ylabel(self, label: str):
        self.ax.set_ylabel(label)
        self.canvas.draw()

    @staticmethod
    def create_new():
        """Factory: return a fresh PlotWidget for a new figure window."""
        return PlotWidget()
