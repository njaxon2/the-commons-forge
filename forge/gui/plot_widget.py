"""Forge plot widget — rich matplotlib figure with interactive tools
(forge/gui/plot_widget.py).

Features:
- Embedded matplotlib with navigation toolbar
- Data cursor (click to read coordinates)
- Multiple subplot support
- Export to PNG/SVG/PDF/EPS
- Figure properties panel
"""

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QToolBar, QLabel,
    QComboBox, QPushButton, QMenu, QFileDialog, QSplitter,
    QDockWidget, QFrame, QSpinBox, QCheckBox, QColorDialog,
    QStatusBar, QSizePolicy, QGroupBox, QFormLayout, QLineEdit,
    QDoubleSpinBox,
)

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
import matplotlib
import matplotlib.pyplot as plt


class DataCursorWidget(QLabel):
    """Status label showing data coordinates on hover/click."""

    def __init__(self, parent=None):
        super().__init__("Ready", parent)
        self.setStyleSheet(
            "font-size: 11px; padding: 2px 8px; "
            "color: #cdd6f4; background: transparent;"
        )

    def update_coords(self, x, y, label=""):
        if label:
            self.setText(f"{label}: ({x:.6g}, {y:.6g})")
        else:
            self.setText(f"({x:.6g}, {y:.6g})")

    def clear_coords(self):
        self.setText("Ready")


class FigurePropertiesPanel(QFrame):
    """Side panel for editing figure/axes properties."""

    properties_changed = Signal()

    def __init__(self, plot_widget, parent=None):
        super().__init__(parent)
        self._pw = plot_widget
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumWidth(200)
        self.setMaximumWidth(280)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # Title group
        grp_title = QGroupBox("Labels")
        fl = QFormLayout(grp_title)
        self.edit_title = QLineEdit()
        self.edit_title.setPlaceholderText("Title")
        self.edit_title.editingFinished.connect(self._apply_labels)
        fl.addRow("Title:", self.edit_title)

        self.edit_xlabel = QLineEdit()
        self.edit_xlabel.setPlaceholderText("X Label")
        self.edit_xlabel.editingFinished.connect(self._apply_labels)
        fl.addRow("X:", self.edit_xlabel)

        self.edit_ylabel = QLineEdit()
        self.edit_ylabel.setPlaceholderText("Y Label")
        self.edit_ylabel.editingFinished.connect(self._apply_labels)
        fl.addRow("Y:", self.edit_ylabel)
        layout.addWidget(grp_title)

        # Axes limits
        grp_limits = QGroupBox("Axis Limits")
        fl2 = QFormLayout(grp_limits)

        self.chk_auto_x = QCheckBox("Auto X")
        self.chk_auto_x.setChecked(True)
        fl2.addRow(self.chk_auto_x)

        h_xlim = QHBoxLayout()
        self.spin_xmin = QDoubleSpinBox()
        self.spin_xmin.setRange(-1e15, 1e15)
        self.spin_xmin.setDecimals(4)
        self.spin_xmax = QDoubleSpinBox()
        self.spin_xmax.setRange(-1e15, 1e15)
        self.spin_xmax.setDecimals(4)
        h_xlim.addWidget(self.spin_xmin)
        h_xlim.addWidget(QLabel("to"))
        h_xlim.addWidget(self.spin_xmax)
        fl2.addRow("X:", h_xlim)

        self.chk_auto_y = QCheckBox("Auto Y")
        self.chk_auto_y.setChecked(True)
        fl2.addRow(self.chk_auto_y)

        h_ylim = QHBoxLayout()
        self.spin_ymin = QDoubleSpinBox()
        self.spin_ymin.setRange(-1e15, 1e15)
        self.spin_ymin.setDecimals(4)
        self.spin_ymax = QDoubleSpinBox()
        self.spin_ymax.setRange(-1e15, 1e15)
        self.spin_ymax.setDecimals(4)
        h_ylim.addWidget(self.spin_ymin)
        h_ylim.addWidget(QLabel("to"))
        h_ylim.addWidget(self.spin_ymax)
        fl2.addRow("Y:", h_ylim)
        layout.addWidget(grp_limits)

        # Grid / legend
        grp_disp = QGroupBox("Display")
        fl3 = QFormLayout(grp_disp)
        self.chk_grid = QCheckBox("Show Grid")
        self.chk_grid.toggled.connect(self._toggle_grid)
        fl3.addRow(self.chk_grid)

        self.chk_legend = QCheckBox("Show Legend")
        self.chk_legend.toggled.connect(self._toggle_legend)
        fl3.addRow(self.chk_legend)

        self.combo_scale_x = QComboBox()
        self.combo_scale_x.addItems(["linear", "log"])
        self.combo_scale_x.currentTextChanged.connect(self._apply_scale)
        fl3.addRow("X Scale:", self.combo_scale_x)

        self.combo_scale_y = QComboBox()
        self.combo_scale_y.addItems(["linear", "log"])
        self.combo_scale_y.currentTextChanged.connect(self._apply_scale)
        fl3.addRow("Y Scale:", self.combo_scale_y)
        layout.addWidget(grp_disp)

        # Apply button
        btn_apply = QPushButton("Apply Limits")
        btn_apply.clicked.connect(self._apply_limits)
        layout.addWidget(btn_apply)

        layout.addStretch()

    def _apply_labels(self):
        ax = self._pw.ax
        if self.edit_title.text():
            ax.set_title(self.edit_title.text())
        if self.edit_xlabel.text():
            ax.set_xlabel(self.edit_xlabel.text())
        if self.edit_ylabel.text():
            ax.set_ylabel(self.edit_ylabel.text())
        self._pw.canvas.draw()

    def _apply_limits(self):
        ax = self._pw.ax
        if not self.chk_auto_x.isChecked():
            ax.set_xlim(self.spin_xmin.value(), self.spin_xmax.value())
        else:
            ax.autoscale(axis='x')
        if not self.chk_auto_y.isChecked():
            ax.set_ylim(self.spin_ymin.value(), self.spin_ymax.value())
        else:
            ax.autoscale(axis='y')
        self._pw.canvas.draw()

    def _toggle_grid(self, on):
        self._pw.ax.grid(on)
        self._pw.canvas.draw()

    def _toggle_legend(self, on):
        if on:
            self._pw.ax.legend()
        else:
            leg = self._pw.ax.get_legend()
            if leg:
                leg.remove()
        self._pw.canvas.draw()

    def _apply_scale(self):
        self._pw.ax.set_xscale(self.combo_scale_x.currentText())
        self._pw.ax.set_yscale(self.combo_scale_y.currentText())
        self._pw.canvas.draw()

    def sync_from_axes(self):
        """Read current axes state into the controls."""
        ax = self._pw.ax
        self.edit_title.setText(ax.get_title())
        self.edit_xlabel.setText(ax.get_xlabel())
        self.edit_ylabel.setText(ax.get_ylabel())
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        self.spin_xmin.setValue(xlim[0])
        self.spin_xmax.setValue(xlim[1])
        self.spin_ymin.setValue(ylim[0])
        self.spin_ymax.setValue(ylim[1])


class PlotWidget(QWidget):
    """Rich embeddable matplotlib figure with interactive tools and properties panel."""

    def __init__(self, parent=None, show_properties=True):
        super().__init__(parent)

        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # Navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Data cursor label
        self.data_cursor = DataCursorWidget(self)

        # Build layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left: figure area
        fig_layout = QVBoxLayout()
        fig_layout.setContentsMargins(0, 0, 0, 0)

        # Custom toolbar with extra buttons
        tool_layout = QHBoxLayout()
        tool_layout.setContentsMargins(0, 0, 0, 0)
        tool_layout.addWidget(self.toolbar)

        # Extra toolbar buttons
        _btn_ss = """
            QPushButton {
                background: transparent; border: 1px solid #555;
                border-radius: 3px; padding: 2px 6px; font-size: 11px;
                color: #cdd6f4;
            }
            QPushButton:hover { background: #313244; }
        """

        self.btn_grid = QPushButton("Grid")
        self.btn_grid.setCheckable(True)
        self.btn_grid.setStyleSheet(_btn_ss)
        self.btn_grid.toggled.connect(self._toggle_grid)
        tool_layout.addWidget(self.btn_grid)

        self.btn_legend = QPushButton("Legend")
        self.btn_legend.setCheckable(True)
        self.btn_legend.setStyleSheet(_btn_ss)
        self.btn_legend.toggled.connect(self._toggle_legend)
        tool_layout.addWidget(self.btn_legend)

        self.btn_export = QPushButton("Export")
        self.btn_export.setStyleSheet(_btn_ss)
        export_menu = QMenu(self)
        for fmt_name, ext in [("PNG", "png"), ("SVG", "svg"), ("PDF", "pdf"), ("EPS", "eps")]:
            act = QAction(f"Save as {fmt_name}...", self)
            act.triggered.connect(lambda checked=False, e=ext: self._export(e))
            export_menu.addAction(act)
        self.btn_export.setMenu(export_menu)
        tool_layout.addWidget(self.btn_export)

        self.btn_props = QPushButton("Properties")
        self.btn_props.setCheckable(True)
        self.btn_props.setStyleSheet(_btn_ss)
        self.btn_props.toggled.connect(self._toggle_properties)
        tool_layout.addWidget(self.btn_props)

        tool_layout.addStretch()
        tool_layout.addWidget(self.data_cursor)
        fig_layout.addLayout(tool_layout)

        fig_layout.addWidget(self.canvas, 1)

        fig_widget = QWidget()
        fig_widget.setLayout(fig_layout)
        main_layout.addWidget(fig_widget, 1)

        # Right: properties panel (hidden by default)
        self.props_panel = FigurePropertiesPanel(self)
        self.props_panel.setVisible(False)
        main_layout.addWidget(self.props_panel)

        # Connect mouse events for data cursor
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas.mpl_connect('button_press_event', self._on_mouse_click)

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

    # ------------------------------------------------------------------
    # Interactive features
    # ------------------------------------------------------------------

    def _on_mouse_move(self, event):
        if event.inaxes:
            self.data_cursor.update_coords(event.xdata, event.ydata)
        else:
            self.data_cursor.clear_coords()

    def _on_mouse_click(self, event):
        if event.inaxes and event.button == 1:
            self.data_cursor.update_coords(event.xdata, event.ydata, "Click")

    def _toggle_grid(self, on):
        self.ax.grid(on)
        self.canvas.draw()

    def _toggle_legend(self, on):
        if on:
            self.ax.legend()
        else:
            leg = self.ax.get_legend()
            if leg:
                leg.remove()
        self.canvas.draw()

    def _toggle_properties(self, on):
        self.props_panel.setVisible(on)
        if on:
            self.props_panel.sync_from_axes()

    def _export(self, ext):
        path, _ = QFileDialog.getSaveFileName(
            self, f"Export as {ext.upper()}",
            f"figure.{ext}", f"{ext.upper()} files (*.{ext})"
        )
        if path:
            self.figure.savefig(path, dpi=150, bbox_inches='tight')
