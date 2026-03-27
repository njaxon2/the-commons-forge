"""Forge main window — PySide6 GUI shell (forge/gui/main_window.py).

Provides the top-level QMainWindow with dockable panels for:
  - Command Window (REPL)
  - Code Editor (tabbed)
  - File Browser (tree)
  - Workspace Browser (variable table)

Menu bar: File, Edit, View (theme/docks), Debug, Window, Help
"""

import os

from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QAction, QKeySequence, QFont, QIcon
from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QToolBar, QMenuBar, QStatusBar, QWidget,
    QApplication, QLabel,
)


class ForgeMainWindow(QMainWindow):
    """Top-level application window for the Forge IDE."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Forge")
        self.resize(1280, 800)

        self.session = None

        # Enable tabbed docking — dragging one panel onto another
        # creates a tab group instead of hiding it.
        self.setDockNestingEnabled(True)

        self._create_actions()
        self._create_docks()
        self._create_menus()
        self._create_toolbar()
        self._create_status_bar()
        self._set_default_layout()
        self._restore_state()

    # ==================================================================
    # Engine integration
    # ==================================================================

    def setup_engine(self, session):
        """Connect the evaluator engine so widgets can call engine.eval()."""
        self.session = session
        if self.command_widget is not None:
            self.command_widget.engine = session
        self._connect_signals()
        if hasattr(session, 'path'):
            self.file_browser_widget.set_search_paths(session.path)
        self._update_workspace()
        # Connect help viewer
        if hasattr(self, 'help_widget'):
            self.help_widget.set_session(session)
        # Show function count in toolbar
        if hasattr(self, '_func_count_label'):
            n = len(session._engine.functions) if hasattr(session, '_engine') else 0
            self._func_count_label.setText(f"{n} functions")

    def _connect_signals(self):
        self.command_widget.command_executed.connect(self._update_workspace)
        self.editor_widget.file_run_requested.connect(self._run_file)
        self.file_browser_widget.file_open_requested.connect(
            self.editor_widget.open_file
        )
        self.workspace_widget.variable_delete_requested.connect(
            self._delete_variable
        )
        self.workspace_widget.variable_inspect_requested.connect(
            self._inspect_variable
        )
        self.file_browser_widget.path_changed.connect(self._on_path_changed)

    # ==================================================================
    # Workspace helpers
    # ==================================================================

    def _update_workspace(self, _text=None):
        if self.session:
            ws_dict = self.session.get_workspace_dict()
            self.workspace_widget.update_workspace(ws_dict)

    def _delete_variable(self, name):
        if self.session:
            ws = self.session.workspace
            if ws.has(name):
                ws.delete(name)
            self._update_workspace()

    def _inspect_variable(self, name):
        try:
            if self.session is None:
                return
            ws = self.session.workspace
            if ws.has(name):
                value = ws.get(name)
                from forge.gui.variable_editor import VariableEditorDialog
                dlg = VariableEditorDialog(name, value, parent=self)
                dlg.value_changed.connect(self._on_variable_edited)
                dlg.show()
        except Exception as e:
            self.command_widget.append_output(f"Error inspecting {name}: {e}")

    def _on_variable_edited(self, name, value):
        try:
            self.session.workspace.set(name, value)
            self._update_workspace()
        except Exception as e:
            self.command_widget.append_output(f"Error updating {name}: {e}")

    def _run_file(self, path):
        if self.session is None:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                code = fh.read()
            result = self.session.eval(code)
            if result:
                self.command_widget.append_output(result)
        except Exception as exc:
            self.command_widget.append_output(f"error: {exc}")
        self._update_workspace()

    def _on_path_changed(self, new_paths):
        if self.session and hasattr(self.session, 'path'):
            self.session.path.clear()
            self.session.path.extend(new_paths)

    # ==================================================================
    # Actions
    # ==================================================================

    def _create_actions(self):
        # File
        self.act_new = QAction("&New", self, shortcut=QKeySequence.New)
        self.act_new.triggered.connect(self._on_new_file)
        self.act_open = QAction("&Open...", self, shortcut=QKeySequence.Open)
        self.act_open.triggered.connect(self._on_open_file)
        self.act_save = QAction("&Save", self, shortcut=QKeySequence.Save)
        self.act_save.triggered.connect(self._on_save_file)
        self.act_save_as = QAction("Save &As...", self, shortcut=QKeySequence.SaveAs)
        self.act_save_as.triggered.connect(self._on_save_as)
        self.act_exit = QAction("E&xit", self, shortcut=QKeySequence.Quit)
        self.act_exit.triggered.connect(self.close)

        # Edit
        self.act_undo = QAction("&Undo", self, shortcut=QKeySequence.Undo)
        self.act_redo = QAction("&Redo", self, shortcut=QKeySequence.Redo)
        self.act_cut = QAction("Cu&t", self, shortcut=QKeySequence.Cut)
        self.act_copy = QAction("&Copy", self, shortcut=QKeySequence.Copy)
        self.act_paste = QAction("&Paste", self, shortcut=QKeySequence.Paste)
        self.act_find = QAction("&Find...", self, shortcut=QKeySequence.Find)
        self.act_preferences = QAction("Preferences...", self, shortcut="Ctrl+,")
        self.act_preferences.triggered.connect(self._open_preferences)

        # Debug
        self.act_run = QAction("&Run File", self, shortcut="F5")
        self.act_run.triggered.connect(self._on_run_file)
        self.act_stop = QAction("&Stop", self, shortcut="Shift+F5")
        self.act_step = QAction("S&tep", self, shortcut="F10")
        self.act_continue = QAction("&Continue", self, shortcut="F5")

        # Window
        self.act_focus_cmd = QAction("Focus &Command", self, shortcut="Ctrl+0")
        self.act_focus_cmd.triggered.connect(self._focus_command_input)
        self.addAction(self.act_focus_cmd)

        self.act_reset_layout = QAction("Reset Layout", self)
        self.act_reset_layout.triggered.connect(self._reset_layout)

        # Help
        self.act_about = QAction("&About Forge", self)
        self.act_about.triggered.connect(self._show_about)
        self.act_docs = QAction("&Documentation", self, shortcut="F1")
        self.act_docs.triggered.connect(self._show_docs)

    # ==================================================================
    # Menus
    # ==================================================================

    def _create_menus(self):
        mb = self.menuBar()

        # ── File ──
        file_menu = mb.addMenu("&File")
        file_menu.addAction(self.act_new)
        file_menu.addAction(self.act_open)
        file_menu.addSeparator()
        file_menu.addAction(self.act_save)
        file_menu.addAction(self.act_save_as)
        file_menu.addSeparator()
        file_menu.addAction(self.act_exit)

        # ── Edit ──
        edit_menu = mb.addMenu("&Edit")
        edit_menu.addAction(self.act_undo)
        edit_menu.addAction(self.act_redo)
        edit_menu.addSeparator()
        edit_menu.addAction(self.act_cut)
        edit_menu.addAction(self.act_copy)
        edit_menu.addAction(self.act_paste)
        edit_menu.addSeparator()
        edit_menu.addAction(self.act_find)
        edit_menu.addSeparator()
        edit_menu.addAction(self.act_preferences)

        # ── View ──
        view_menu = mb.addMenu("&View")

        # Dock panel toggles
        panels_menu = view_menu.addMenu("Panels")
        panels_menu.addAction(self.command_dock.toggleViewAction())
        panels_menu.addAction(self.editor_dock.toggleViewAction())
        panels_menu.addAction(self.file_browser_dock.toggleViewAction())
        panels_menu.addAction(self.workspace_dock.toggleViewAction())
        panels_menu.addAction(self.help_dock.toggleViewAction())

        # Theme submenu
        theme_menu = view_menu.addMenu("Theme")
        for tname in ("dark", "light", "midnight"):
            act = QAction(tname.capitalize(), self)
            act.triggered.connect(
                lambda checked=False, t=tname: self._switch_theme(t)
            )
            theme_menu.addAction(act)

        view_menu.addSeparator()
        view_menu.addAction(self.act_reset_layout)

        # ── Debug ──
        debug_menu = mb.addMenu("&Debug")
        debug_menu.addAction(self.act_run)
        debug_menu.addAction(self.act_stop)
        debug_menu.addSeparator()
        debug_menu.addAction(self.act_step)
        debug_menu.addAction(self.act_continue)

        # ── Window ──
        window_menu = mb.addMenu("&Window")
        window_menu.addAction(self.act_focus_cmd)
        window_menu.addSeparator()
        window_menu.addAction(self.act_reset_layout)

        # ── Help ──
        help_menu = mb.addMenu("&Help")
        help_menu.addAction(self.act_about)
        help_menu.addAction(self.act_docs)

    # ==================================================================
    # Toolbar
    # ==================================================================

    def _create_toolbar(self):
        from PySide6.QtCore import QSize
        from PySide6.QtWidgets import QStyle

        tb = QToolBar("Main Toolbar", self)
        tb.setObjectName("MainToolbar")
        tb.setIconSize(QSize(18, 18))
        self.addToolBar(tb)

        style = self.style()
        self.act_new.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_FileIcon))
        self.act_open.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton))
        self.act_save.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.act_run.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.act_stop.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaStop))

        tb.addAction(self.act_new)
        tb.addAction(self.act_open)
        tb.addAction(self.act_save)
        tb.addSeparator()
        tb.addAction(self.act_run)
        tb.addAction(self.act_stop)

        # Spacer to push function count to the right
        spacer = QWidget()
        spacer.setSizePolicy(
            __import__('PySide6.QtWidgets', fromlist=['QSizePolicy']).QSizePolicy.Expanding,
            __import__('PySide6.QtWidgets', fromlist=['QSizePolicy']).QSizePolicy.Preferred
        )
        tb.addWidget(spacer)

        self._func_count_label = QLabel("")
        self._func_count_label.setStyleSheet(
            "color: #89b4fa; font-size: 11px; padding: 0 12px;"
        )
        tb.addWidget(self._func_count_label)

    # ==================================================================
    # Dock widgets
    # ==================================================================

    def _create_docks(self):
        from forge.gui.command_widget import CommandWidget
        from forge.gui.editor_widget import EditorWidget
        from forge.gui.file_browser import FileBrowserWidget
        from forge.gui.workspace_browser import WorkspaceBrowserWidget
        from forge.gui.help_viewer import HelpViewerWidget

        # Command Window (bottom)
        self.command_widget = CommandWidget(self)
        self.command_widget.setMinimumHeight(180)
        self.command_dock = self._make_dock(
            "Command Window", "CommandDock", self.command_widget
        )
        self.addDockWidget(Qt.BottomDockWidgetArea, self.command_dock)

        # File Browser (left)
        self.file_browser_widget = FileBrowserWidget(parent=self)
        self.file_browser_dock = self._make_dock(
            "File Browser", "FileBrowserDock", self.file_browser_widget
        )
        self.addDockWidget(Qt.LeftDockWidgetArea, self.file_browser_dock)

        # Workspace Browser (left, tabbed with file browser)
        self.workspace_widget = WorkspaceBrowserWidget(self)
        self.workspace_dock = self._make_dock(
            "Workspace", "WorkspaceDock", self.workspace_widget
        )
        self.addDockWidget(Qt.LeftDockWidgetArea, self.workspace_dock)
        # Tab workspace with file browser so they share space
        self.tabifyDockWidget(self.file_browser_dock, self.workspace_dock)
        self.file_browser_dock.raise_()  # File browser on top initially

        # Help Viewer (left, tabbed with file browser and workspace)
        self.help_widget = HelpViewerWidget(parent=self)
        self.help_dock = self._make_dock(
            "Documentation", "HelpDock", self.help_widget
        )
        self.addDockWidget(Qt.LeftDockWidgetArea, self.help_dock)
        self.tabifyDockWidget(self.workspace_dock, self.help_dock)
        self.file_browser_dock.raise_()

        # Editor (right — takes majority of horizontal space)
        self.editor_widget = EditorWidget(self)
        self.editor_dock = self._make_dock(
            "Editor", "EditorDock", self.editor_widget
        )
        self.addDockWidget(Qt.RightDockWidgetArea, self.editor_dock)

    def _make_dock(self, title, object_name, widget):
        dock = QDockWidget(title, self)
        dock.setObjectName(object_name)
        dock.setWidget(widget)
        dock.setFeatures(
            QDockWidget.DockWidgetClosable
            | QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
        )
        return dock

    def _set_default_layout(self):
        self.resizeDocks([self.command_dock], [250], Qt.Vertical)
        self.resizeDocks([self.file_browser_dock], [300], Qt.Horizontal)

    # ==================================================================
    # Status bar
    # ==================================================================

    def _create_status_bar(self):
        sb = QStatusBar(self)
        self.setStatusBar(sb)
        self._status_msg = QLabel("Ready")
        self._status_msg.setStyleSheet("padding: 0 8px;")
        sb.addWidget(self._status_msg, 1)

        # Function count badge
        try:
            from forge.engine.session import ForgeSession
            s = ForgeSession.__new__(ForgeSession)
            # We'll update this after engine setup
        except Exception:
            pass

    def set_status(self, msg: str):
        self._status_msg.setText(msg)

    # ==================================================================
    # State save / restore
    # ==================================================================

    def _settings(self):
        return QSettings("Forge", "ForgeIDE")

    def _restore_state(self):
        s = self._settings()
        geom = s.value("geometry")
        if geom is not None:
            self.restoreGeometry(geom)
        state = s.value("windowState")
        if state is not None:
            self.restoreState(state)

    def closeEvent(self, event):
        s = self._settings()
        s.setValue("geometry", self.saveGeometry())
        s.setValue("windowState", self.saveState())
        super().closeEvent(event)

    # ==================================================================
    # Preferences & Themes
    # ==================================================================

    def _open_preferences(self):
        from forge.gui.preferences_dialog import PreferencesDialog
        dlg = PreferencesDialog(self)
        dlg.preferences_changed.connect(self._apply_preferences)
        dlg.exec()

    def _apply_preferences(self, prefs):
        from forge.gui.themes import apply_theme
        app = QApplication.instance()
        apply_theme(app, prefs.get('default_theme', 'dark'))
        font = QFont(prefs.get('font_family', 'Consolas'), prefs.get('font_size', 10))
        font.setStyleHint(QFont.StyleHint.Monospace)
        app.setFont(font)
        # Update editor palette
        from forge.gui.editor_widget import set_editor_palette
        set_editor_palette(prefs.get('default_theme', 'dark'))

    def _switch_theme(self, theme_name):
        from forge.gui.themes import apply_theme, get_preferences, save_preferences
        app = QApplication.instance()
        apply_theme(app, theme_name)
        prefs = get_preferences()
        prefs['default_theme'] = theme_name
        save_preferences(prefs)
        # Update editor colours
        from forge.gui.editor_widget import set_editor_palette
        set_editor_palette(theme_name)

    # ==================================================================
    # Helpers
    # ==================================================================

    # ==================================================================
    # File actions
    # ==================================================================

    def _on_new_file(self):
        self.editor_widget.new_file()
        self.editor_dock.raise_()

    def _on_open_file(self):
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Open File", "",
            "M-files (*.m);;All Files (*)"
        )
        if path:
            self.editor_widget.open_file(path)
            self.editor_dock.raise_()

    def _on_save_file(self):
        editor = self.editor_widget.get_current_editor()
        if editor is None:
            return
        if editor.file_path:
            self.editor_widget.save_file()
            self.set_status(f"Saved {os.path.basename(editor.file_path)}")
        else:
            self._on_save_as()

    def _on_save_as(self):
        from PySide6.QtWidgets import QFileDialog
        editor = self.editor_widget.get_current_editor()
        if editor is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save As", "",
            "M-files (*.m);;All Files (*)"
        )
        if path:
            editor.file_path = path
            self.editor_widget.save_file()
            idx = self.editor_widget.tabs.currentIndex()
            self.editor_widget.tabs.setTabText(idx, os.path.basename(path))
            self.set_status(f"Saved {os.path.basename(path)}")

    def _on_run_file(self):
        editor = self.editor_widget.get_current_editor()
        if editor and editor.file_path:
            self._run_file(editor.file_path)

    def open_file_in_editor(self, path):
        """Public method to open a file in the editor (used by 'edit' command)."""
        self.editor_widget.open_file(path)
        self.editor_dock.raise_()

    def _focus_command_input(self):
        if hasattr(self, 'command_widget') and self.command_widget:
            self.command_widget.console.setFocus()

    def _reset_layout(self):
        """Clear saved state and reset docks."""
        s = self._settings()
        s.remove("geometry")
        s.remove("windowState")
        # Re-dock all panels
        for dock in [self.command_dock, self.editor_dock,
                     self.file_browser_dock, self.workspace_dock]:
            dock.setFloating(False)
            dock.setVisible(True)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.command_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.editor_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.file_browser_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.workspace_dock)
        self._set_default_layout()
        self.set_status("Layout reset")

    def _show_docs(self):
        """Show the documentation panel."""
        self.help_dock.setVisible(True)
        self.help_dock.raise_()
        self.help_widget.search_edit.setFocus()

    def _show_about(self):
        from PySide6.QtWidgets import QMessageBox
        func_count = 0
        if self.session and hasattr(self.session, '_engine'):
            func_count = len(self.session._engine.functions)
        QMessageBox.about(
            self,
            "About Forge",
            "<div style='text-align:center;'>"
            "<h2 style='color:#89b4fa;'>Forge IDE</h2>"
            "<p><b>Octave-Compatible Computing Environment</b></p>"
            "<hr>"
            f"<p>{func_count} built-in functions</p>"
            "<p>Built with PySide6, NumPy, SciPy, and Matplotlib</p>"
            "<p>Version 0.1.0</p>"
            "<p style='color:#6c7086;'>Licensed under MIT</p>"
            "</div>"
        )
