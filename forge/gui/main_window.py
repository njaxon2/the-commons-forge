"""Forge main window -- PySide6 GUI shell (forge/gui/main_window.py)."""

from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QToolBar, QMenuBar, QStatusBar, QWidget,
)


class ForgeMainWindow(QMainWindow):
    """Top-level application window for the Forge IDE."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Forge")
        self.resize(1280, 800)

        self.session = None

        # Enable tabbed docking so dragging one panel onto another
        # creates a tab group instead of making the panel vanish.
        self.setDockNestingEnabled(True)

        self._create_actions()
        self._create_menus()
        self._create_toolbar()
        self._create_docks()
        self._create_status_bar()
        self._set_default_layout()
        self._restore_state()

    # ------------------------------------------------------------------
    # Engine integration
    # ------------------------------------------------------------------

    def setup_engine(self, session):
        """Connect the evaluator engine so widgets can call engine.eval()."""
        self.session = session
        if self.command_widget is not None:
            self.command_widget.engine = session
        self._connect_signals()
        # Sync file browser with session's search path
        if hasattr(session, 'path'):
            self.file_browser_widget.set_search_paths(session.path)
        self._update_workspace()

    def _connect_signals(self):
        """Wire inter-widget signals after engine is set."""
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
        self.workspace_widget.variable_inspect_requested.connect(
            self._inspect_variable
        )
        # When search path changes in file browser, sync to session
        self.file_browser_widget.path_changed.connect(self._on_path_changed)

    # ------------------------------------------------------------------
    # Workspace helpers
    # ------------------------------------------------------------------

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
        """Open variable editor dialog for the named variable."""
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
        """Handle variable edit from variable editor."""
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
        """Sync file browser path changes back to the session."""
        if self.session and hasattr(self.session, 'path'):
            self.session.path.clear()
            self.session.path.extend(new_paths)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _create_actions(self):
        self.action_new = QAction("&New", self, shortcut=QKeySequence.New)
        self.action_open = QAction("&Open...", self, shortcut=QKeySequence.Open)
        self.action_save = QAction("&Save", self, shortcut=QKeySequence.Save)
        self.action_save_as = QAction("Save &As...", self, shortcut=QKeySequence.SaveAs)
        self.action_exit = QAction("E&xit", self, shortcut=QKeySequence.Quit)
        self.action_exit.triggered.connect(self.close)

        self.action_undo = QAction("&Undo", self, shortcut=QKeySequence.Undo)
        self.action_redo = QAction("&Redo", self, shortcut=QKeySequence.Redo)
        self.action_cut = QAction("Cu&t", self, shortcut=QKeySequence.Cut)
        self.action_copy = QAction("&Copy", self, shortcut=QKeySequence.Copy)
        self.action_paste = QAction("&Paste", self, shortcut=QKeySequence.Paste)
        self.action_find = QAction("&Find...", self, shortcut=QKeySequence.Find)

        self.action_toggle_command = QAction("Command Window", self, checkable=True, checked=True)
        self.action_toggle_editor = QAction("Editor", self, checkable=True, checked=True)
        self.action_toggle_file_browser = QAction("File Browser", self, checkable=True, checked=True)
        self.action_toggle_workspace = QAction("Workspace", self, checkable=True, checked=True)

        self.action_run = QAction("&Run File", self, shortcut=QKeySequence("F5"))
        self.action_focus_command = QAction("Focus &Command", self, shortcut=QKeySequence("Ctrl+0"))
        self.action_focus_command.triggered.connect(self._focus_command_input)
        self.action_stop = QAction("&Stop", self, shortcut=QKeySequence("Shift+F5"))
        self.action_step = QAction("S&tep", self, shortcut=QKeySequence("F10"))
        self.action_continue = QAction("&Continue", self, shortcut=QKeySequence("F5"))

        self.action_reset_layout = QAction("Reset Layout", self)
        self.action_reset_layout.triggered.connect(self._reset_layout)

        self.addAction(self.action_focus_command)  # Make shortcut global
        self.action_about = QAction("&About", self)
        self.action_docs = QAction("&Documentation", self)

    # ------------------------------------------------------------------
    # Menus
    # ------------------------------------------------------------------

    def _create_menus(self):
        mb = self.menuBar()

        file_menu = mb.addMenu("&File")
        file_menu.addAction(self.action_new)
        file_menu.addAction(self.action_open)
        file_menu.addSeparator()
        file_menu.addAction(self.action_save)
        file_menu.addAction(self.action_save_as)
        file_menu.addSeparator()
        file_menu.addAction(self.action_exit)

        edit_menu = mb.addMenu("&Edit")
        edit_menu.addAction(self.action_undo)
        edit_menu.addAction(self.action_redo)
        edit_menu.addSeparator()
        edit_menu.addAction(self.action_cut)
        edit_menu.addAction(self.action_copy)
        edit_menu.addAction(self.action_paste)
        edit_menu.addSeparator()
        edit_menu.addAction(self.action_find)

        view_menu = mb.addMenu("&View")
        view_menu.addAction(self.action_toggle_command)
        view_menu.addAction(self.action_toggle_editor)
        view_menu.addAction(self.action_toggle_file_browser)
        view_menu.addAction(self.action_toggle_workspace)

        debug_menu = mb.addMenu("&Debug")
        debug_menu.addAction(self.action_run)
        debug_menu.addAction(self.action_stop)
        debug_menu.addAction(self.action_step)
        debug_menu.addAction(self.action_continue)

        window_menu = mb.addMenu("&Window")
        window_menu.addAction(self.action_reset_layout)

        help_menu = mb.addMenu("&Help")
        help_menu.addAction(self.action_about)
        help_menu.addAction(self.action_docs)

    # ------------------------------------------------------------------
    # Toolbar
    # ------------------------------------------------------------------

    def _create_toolbar(self):
        tb = QToolBar("Main Toolbar", self)
        tb.setObjectName("MainToolbar")
        self.addToolBar(tb)
        tb.addAction(self.action_new)
        tb.addAction(self.action_open)
        tb.addAction(self.action_save)
        tb.addSeparator()
        tb.addAction(self.action_run)
        tb.addAction(self.action_stop)

    # ------------------------------------------------------------------
    def _set_default_layout(self):
        """Set initial dock sizes: command window gets ~40% of vertical space."""
        # resizeDocks takes a list of docks and a list of sizes
        # Give command window 320px (40% of 800), file browser 280px width
        self.resizeDocks(
            [self.command_dock],
            [320],
            Qt.Vertical,
        )
        self.resizeDocks(
            [self.file_browser_dock],
            [280],
            Qt.Horizontal,
        )

    # Dock widgets
    # ------------------------------------------------------------------

    def _create_docks(self):
        from forge.gui.command_widget import CommandWidget
        from forge.gui.editor_widget import EditorWidget
        from forge.gui.file_browser import FileBrowserWidget
        from forge.gui.workspace_browser import WorkspaceBrowserWidget

        # --- Command Window (bottom) ---
        self.command_widget = CommandWidget(self)
        self.command_widget.setMinimumHeight(200)
        self.command_dock = self._make_dock("Command Window", "CommandDock",
                                           self.command_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.command_dock)
        self.action_toggle_command.toggled.connect(self.command_dock.setVisible)
        self.command_dock.visibilityChanged.connect(self.action_toggle_command.setChecked)

        # --- Editor (centre-right) ---
        self.editor_widget = EditorWidget(self)
        self.editor_dock = self._make_dock("Editor", "EditorDock",
                                           self.editor_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.editor_dock)
        self.action_toggle_editor.toggled.connect(self.editor_dock.setVisible)
        self.editor_dock.visibilityChanged.connect(self.action_toggle_editor.setChecked)

        # --- File Browser (left) ---
        self.file_browser_widget = FileBrowserWidget(parent=self)
        self.file_browser_dock = self._make_dock("File Browser", "FileBrowserDock",
                                                  self.file_browser_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.file_browser_dock)
        self.action_toggle_file_browser.toggled.connect(self.file_browser_dock.setVisible)
        self.file_browser_dock.visibilityChanged.connect(self.action_toggle_file_browser.setChecked)

        # --- Workspace (right) ---
        self.workspace_widget = WorkspaceBrowserWidget(self)
        self.workspace_dock = self._make_dock("Workspace", "WorkspaceDock",
                                              self.workspace_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.workspace_dock)
        self.action_toggle_workspace.toggled.connect(self.workspace_dock.setVisible)
        self.workspace_dock.visibilityChanged.connect(self.action_toggle_workspace.setChecked)

    def _make_dock(self, title, object_name, widget):
        """Create a QDockWidget with a close button in the title bar.

        DockWidgetClosable  — adds an X button to the title bar
        DockWidgetMovable   — allows dragging
        DockWidgetFloatable — allows floating as a separate window

        When dragged onto another dock, Qt will automatically create
        a tabbed group (because setDockNestingEnabled is True).
        Closing the dock just hides it; View menu can re-show it.
        """
        dock = QDockWidget(title, self)
        dock.setObjectName(object_name)
        dock.setWidget(widget)
        dock.setFeatures(
            QDockWidget.DockWidgetClosable
            | QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
        )
        return dock

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _create_status_bar(self):
        sb = QStatusBar(self)
        self.setStatusBar(sb)
        sb.showMessage("Ready")

    # ------------------------------------------------------------------
    # State save / restore
    # ------------------------------------------------------------------

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

    def _focus_command_input(self):
        """Focus the command input line."""
        if hasattr(self, "command_widget") and self.command_widget:
            self.command_widget.console.setFocus()
            pass  # single-pane console

    def _reset_layout(self):
        """Remove saved state so next launch uses defaults."""
        s = self._settings()
        s.remove("geometry")
        s.remove("windowState")
