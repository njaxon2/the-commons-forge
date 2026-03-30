# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Add-ons configuration dialog for Forge IDE."""

import shlex

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QLabel, QPushButton, QGroupBox, QFrame, QLineEdit, QMessageBox,
)


class AddonsDialog(QDialog):
    """Dialog for managing Forge toolboxes and Octave packages.

    Matches Octave's pkg interface: list, load/unload, install from URL.
    """

    addons_changed = Signal()

    def __init__(self, addon_manager, parent=None):
        super().__init__(parent)
        self._mgr = addon_manager
        self.setWindowTitle("Manage Add-ons  (pkg)")
        self.setMinimumSize(700, 520)
        self._build_ui()
        self._populate()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Top: Install bar (like Octave's pkg install)
        install_bar = QHBoxLayout()
        install_bar.addWidget(QLabel("Install package:"))
        self._install_input = QLineEdit()
        self._install_input.setPlaceholderText(
            "URL or local path (e.g. https://example.com/pkg.tar.gz)")
        self._install_input.returnPressed.connect(self._do_install)
        install_bar.addWidget(self._install_input, 1)
        btn_install = QPushButton("Install")
        btn_install.setToolTip("pkg install <url-or-path>")
        btn_install.clicked.connect(self._do_install)
        install_bar.addWidget(btn_install)
        layout.addLayout(install_bar)

        # Main area: tree + details
        main = QHBoxLayout()

        # Left: tree + action buttons
        left = QVBoxLayout()
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Add-on", "Functions", "Status"])
        self._tree.setColumnWidth(0, 300)
        self._tree.setColumnWidth(1, 80)
        self._tree.setColumnWidth(2, 80)
        self._tree.setRootIsDecorated(True)
        self._tree.itemChanged.connect(self._on_item_changed)
        self._tree.currentItemChanged.connect(self._on_selection_changed)
        left.addWidget(self._tree)

        # Action buttons row
        btn_row = QHBoxLayout()
        btn_enable_all = QPushButton("Enable All")
        btn_enable_all.clicked.connect(self._enable_all)
        btn_row.addWidget(btn_enable_all)
        btn_disable_all = QPushButton("Disable All")
        btn_disable_all.clicked.connect(self._disable_all)
        btn_row.addWidget(btn_disable_all)
        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self._refresh)
        btn_row.addWidget(btn_refresh)
        btn_row.addStretch()
        left.addLayout(btn_row)
        main.addLayout(left, 3)

        # Right: details panel
        right = QVBoxLayout()
        self._detail_box = QGroupBox("Details")
        detail_layout = QVBoxLayout(self._detail_box)
        self._lbl_name = QLabel()
        self._lbl_name.setFont(QFont("", 12, QFont.Bold))
        self._lbl_backend = QLabel()
        self._lbl_count = QLabel()
        self._lbl_status = QLabel()
        self._lbl_conflicts = QLabel()
        self._lbl_conflicts.setWordWrap(True)
        for lbl in (self._lbl_name, self._lbl_backend, self._lbl_count,
                    self._lbl_status, self._lbl_conflicts):
            detail_layout.addWidget(lbl)

        # Command hint
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        detail_layout.addWidget(sep)
        cmd_hint = QLabel(
            "<small><b>Command line equivalent:</b><br>"
            "<code>pkg list</code><br>"
            "<code>pkg load &lt;name&gt;</code><br>"
            "<code>pkg unload &lt;name&gt;</code><br>"
            "<code>pkg describe &lt;name&gt;</code></small>"
        )
        cmd_hint.setWordWrap(True)
        detail_layout.addWidget(cmd_hint)

        detail_layout.addStretch()
        right.addWidget(self._detail_box)

        # Summary
        self._lbl_summary = QLabel()
        right.addWidget(self._lbl_summary)

        # Close button
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        right.addWidget(btn_close, alignment=Qt.AlignRight)
        main.addLayout(right, 2)

        layout.addLayout(main)

    def _populate(self):
        self._tree.blockSignals(True)
        self._tree.clear()

        total_functions = 0
        loaded_count = 0

        # Forge section
        forge_root = QTreeWidgetItem(self._tree, ["Forge Engine", "", ""])
        forge_root.setFlags(forge_root.flags() & ~Qt.ItemIsUserCheckable)
        forge_root.setExpanded(True)
        font = forge_root.font(0)
        font.setBold(True)
        forge_root.setFont(0, font)

        for name, display, count, enabled in self._mgr.forge_toolboxes():
            status = "loaded" if enabled else "unloaded"
            item = QTreeWidgetItem(forge_root, [display, str(count), status])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Checked if enabled else Qt.Unchecked)
            item.setData(0, Qt.UserRole, ("forge", name))
            if enabled:
                total_functions += count
                loaded_count += 1

        # Octave section
        octave_root = QTreeWidgetItem(self._tree, ["GNU Octave", "", ""])
        octave_root.setFlags(octave_root.flags() & ~Qt.ItemIsUserCheckable)
        octave_root.setExpanded(True)
        octave_root.setFont(0, font)

        if not self._mgr.octave_available:
            hint = QTreeWidgetItem(octave_root, ["(octave-cli not found)", "", ""])
            hint.setFlags(hint.flags() & ~Qt.ItemIsSelectable)
            hint.setDisabled(True)
        else:
            for info in self._mgr.octave_packages():
                if isinstance(info, dict):
                    name = info["name"]
                    version = info.get("version", "?")
                    enabled = self._mgr._octave_state.get(name, False)
                else:
                    name, version, enabled = info
                status = "loaded" if enabled else "unloaded"
                label = f"{name}  (v{version})"
                item = QTreeWidgetItem(octave_root, [label, "", status])
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(0, Qt.Checked if enabled else Qt.Unchecked)
                item.setData(0, Qt.UserRole, ("octave", name))
                if enabled:
                    loaded_count += 1

        self._tree.blockSignals(False)
        self._lbl_summary.setText(
            f"{loaded_count} packages loaded | {total_functions} functions available")

    def _on_item_changed(self, item, column):
        data = item.data(0, Qt.UserRole)
        if data is None:
            return
        backend, name = data
        enabled = item.checkState(0) == Qt.Checked
        if backend == "forge":
            self._mgr.set_forge_enabled(name, enabled)
        else:
            self._mgr.set_octave_enabled(name, enabled)
        # Update status column
        item.setText(2, "loaded" if enabled else "unloaded")
        self.addons_changed.emit()
        self._update_detail(item)
        self._update_summary()

    def _on_selection_changed(self, current, _previous):
        if current is not None:
            self._update_detail(current)

    def _update_detail(self, item):
        data = item.data(0, Qt.UserRole)
        if data is None:
            self._lbl_name.setText(item.text(0))
            self._lbl_backend.setText("")
            self._lbl_count.setText("")
            self._lbl_status.setText("")
            self._lbl_conflicts.setText("")
            return

        backend, name = data
        self._lbl_name.setText(item.text(0))
        self._lbl_backend.setText(f"Backend: {backend.capitalize()}")

        if backend == "forge":
            enabled = self._mgr.is_forge_enabled(name)
            count = item.text(1)
            self._lbl_count.setText(f"Functions: {count}")
        else:
            enabled = self._mgr.is_octave_enabled(name)
            self._lbl_count.setText("")

        self._lbl_status.setText(f"Status: {'Loaded' if enabled else 'Unloaded'}")

        conflicts = self._mgr.get_conflicts()
        if conflicts:
            self._lbl_conflicts.setText(
                f"Conflicts ({len(conflicts)}): {', '.join(sorted(conflicts)[:10])}"
                + (" ..." if len(conflicts) > 10 else "")
                + "\nForge versions take priority."
            )
        else:
            self._lbl_conflicts.setText("")

    def _update_summary(self):
        total = 0
        loaded = 0
        for name, display, count, enabled in self._mgr.forge_toolboxes():
            if enabled:
                total += count
                loaded += 1
        self._lbl_summary.setText(
            f"{loaded} packages loaded | {total} functions available")

    def _enable_all(self):
        """Enable all packages (pkg load all)."""
        for name in list(self._mgr._forge_state):
            self._mgr.set_forge_enabled(name, True)
        for name in list(self._mgr._octave_state):
            self._mgr.set_octave_enabled(name, True)
        self._populate()
        self.addons_changed.emit()

    def _disable_all(self):
        """Disable all packages (pkg unload all)."""
        for name in list(self._mgr._forge_state):
            self._mgr.set_forge_enabled(name, False)
        for name in list(self._mgr._octave_state):
            self._mgr.set_octave_enabled(name, False)
        self._populate()
        self.addons_changed.emit()

    def _refresh(self):
        """Re-scan for packages."""
        self._mgr._init_octave()
        self._populate()

    def _do_install(self):
        """Install package from URL or file path (pkg install)."""
        url = self._install_input.text().strip()
        if not url:
            return
        if not self._mgr._octave_bridge or not self._mgr._octave_bridge.available:
            QMessageBox.warning(self, "Install",
                "Package installation requires octave-cli.\n"
                "Forge toolboxes are built-in and always available.")
            return
        # Delegate to octave's pkg install
        import subprocess, sys
        try:
            result = subprocess.run(
                ["octave-cli", "--no-gui", "--silent", "--eval",
                 f"pkg install {shlex.quote(url)}"],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                QMessageBox.information(self, "Install",
                    f"Package installed successfully.\n{result.stdout.strip()}")
                self._refresh()
                self.addons_changed.emit()
            else:
                QMessageBox.warning(self, "Install Failed",
                    f"Installation failed:\n{result.stderr.strip()}")
        except Exception as e:
            QMessageBox.warning(self, "Install Error", str(e))
        self._install_input.clear()
