"""Code Profiler / Timer panel -- shows execution timing information
(forge/gui/profiler_panel.py)."""

import csv
import io
import time

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QFileDialog,
)


class ProfilerPanel(QWidget):
    """Panel that displays execution timing / profiling information.

    Signals
    -------
    profile_requested()
        Emitted when the user clicks *Profile* so the main window can
        wrap the current editor code execution with timing.
    """

    profile_requested = Signal()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, parent=None):
        super().__init__(parent)
        self._engine = None
        self._editor_widget = None
        self._entries = []  # list of dicts {function, time_ms, calls, avg_ms}

        self.setObjectName("ProfilerPanel")
        self.setStyleSheet("""
            #ProfilerPanel {
                background-color: #1e1e2e;
            }
            #ProfilerPanel QLabel {
                color: #cdd6f4;
                font-size: 12px;
            }
            #ProfilerPanel QLabel#TotalTimeLabel {
                color: #94e2d5;
                font-size: 16px;
                font-weight: bold;
                padding: 4px 0;
            }
            #ProfilerPanel QTableWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                border: 1px solid #45475a;
                font-family: "Courier New", monospace;
                font-size: 12px;
                selection-background-color: #1e6e5e;
                selection-color: #cdd6f4;
                alternate-background-color: #232336;
                gridline-color: #45475a;
            }
            #ProfilerPanel QTableWidget::item:hover {
                background-color: #2a2a3d;
            }
            #ProfilerPanel QTableWidget::item:selected {
                background-color: #1e6e5e;
            }
            #ProfilerPanel QHeaderView::section {
                background-color: #181825;
                color: #94e2d5;
                border: 1px solid #45475a;
                padding: 4px 8px;
                font-weight: bold;
            }
            #ProfilerPanel QPushButton {
                background-color: #1e6e5e;
                color: #cdd6f4;
                border: none;
                border-radius: 4px;
                padding: 5px 12px;
                font-weight: bold;
                font-size: 12px;
            }
            #ProfilerPanel QPushButton:hover {
                background-color: #2a9d8f;
            }
            #ProfilerPanel QPushButton:pressed {
                background-color: #14524a;
            }
            #ProfilerPanel QPushButton:disabled {
                background-color: #45475a;
                color: #6c7086;
            }
        """)

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # --- Total time label ---
        self.total_label = QLabel("Total: --")
        self.total_label.setObjectName("TotalTimeLabel")
        layout.addWidget(self.total_label)

        # --- Toolbar row ---
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)

        self.btn_profile = QPushButton("Profile")
        self.btn_profile.setToolTip("Run the current editor code with profiling")
        self.btn_profile.clicked.connect(self._on_profile)
        btn_layout.addWidget(self.btn_profile)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setToolTip("Clear all profiling results")
        self.btn_clear.clicked.connect(self.clear)
        btn_layout.addWidget(self.btn_clear)

        self.btn_export = QPushButton("Export CSV")
        self.btn_export.setToolTip("Export profiling results to a CSV file")
        self.btn_export.clicked.connect(self._on_export_csv)
        btn_layout.addWidget(self.btn_export)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # --- Results table ---
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Function", "Time (ms)", "Calls", "Avg (ms)"])
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(True)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        layout.addWidget(self.table)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_engine(self, engine):
        """Store a reference to the Octave engine."""
        self._engine = engine

    def set_editor_widget(self, editor_widget):
        """Store a reference to the EditorWidget."""
        self._editor_widget = editor_widget

    def clear(self):
        """Remove all profiling entries."""
        self._entries.clear()
        self.table.setRowCount(0)
        self.total_label.setText("Total: --")

    def add_entry(self, function_name, time_ms, calls=1):
        """Add a profiling entry to the table."""
        avg = time_ms / calls if calls > 0 else 0.0
        entry = {
            "function": function_name,
            "time_ms": round(time_ms, 3),
            "calls": calls,
            "avg_ms": round(avg, 3),
        }
        self._entries.append(entry)
        row = self.table.rowCount()
        self.table.insertRow(row)

        item_fn = QTableWidgetItem(function_name)
        item_time = QTableWidgetItem()
        item_time.setData(Qt.DisplayRole, round(time_ms, 3))
        item_calls = QTableWidgetItem()
        item_calls.setData(Qt.DisplayRole, calls)
        item_avg = QTableWidgetItem()
        item_avg.setData(Qt.DisplayRole, round(avg, 3))

        self.table.setItem(row, 0, item_fn)
        self.table.setItem(row, 1, item_time)
        self.table.setItem(row, 2, item_calls)
        self.table.setItem(row, 3, item_avg)

    def set_total_time(self, total_ms):
        """Update the total execution time label."""
        self.total_label.setText(f"Total: {total_ms:.3f} ms")

    def run_profile(self):
        """Profile the current editor code through the engine.

        This reads the current editor text, sends it to the Octave engine,
        and measures how long it takes.
        """
        if self._editor_widget is None:
            return
        editor = self._editor_widget.get_current_editor()
        if editor is None:
            return

        code = editor.toPlainText().strip()
        if not code:
            return

        # Determine a display name for the profiled code
        file_path = getattr(editor, 'file_path', None)
        if file_path:
            import os
            display_name = os.path.basename(file_path)
        else:
            # Use first non-empty line truncated
            first_line = code.split("\n")[0][:60]
            display_name = first_line if first_line else "<script>"

        if self._engine is None:
            # No engine -- just measure a dummy pass
            self.add_entry(display_name, 0.0, 1)
            self.set_total_time(0.0)
            return

        # Measure execution time
        t_start = time.perf_counter()
        try:
            self._engine.run(code)
        except Exception:
            pass  # errors are shown in command window / output panel
        t_end = time.perf_counter()

        elapsed_ms = (t_end - t_start) * 1000.0
        self.add_entry(display_name, elapsed_ms, 1)
        self.set_total_time(elapsed_ms)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_profile(self):
        """Handle the Profile button click."""
        self.run_profile()
        self.profile_requested.emit()

    def _on_export_csv(self):
        """Export profiling results to a CSV file."""
        if not self._entries:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Profiling Results", "profile_results.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Function", "Time (ms)", "Calls", "Avg (ms)"])
            for e in self._entries:
                writer.writerow([e["function"], e["time_ms"],
                                 e["calls"], e["avg_ms"]])
