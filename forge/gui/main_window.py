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

from forge.gui.commons_integration import (
    UpdateChecker, AMSConnector, FeatureRequestDialog,
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

        self._recent_files = self._load_recent_files()
        self._create_actions()
        self._create_docks()
        self._create_menus()
        self._create_toolbar()
        self._create_status_bar()
        self._set_default_layout()
        self._restore_state()

        # TheCommons integration
        self._setup_commons()

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

        # Feed function names to editor autocomplete
        if hasattr(session, '_engine'):
            func_names = list(session._engine.functions.keys())
            for i in range(self.editor_widget.tabs.count()):
                editor = self.editor_widget.tabs.widget(i)
                if hasattr(editor, 'set_function_names'):
                    editor.set_function_names(func_names)
            # Store for new tabs
            self.editor_widget._engine_func_names = func_names

    def _connect_signals(self):
        self.command_widget.command_executed.connect(self._update_workspace)
        self.command_widget.command_executed.connect(lambda _: self._refresh_problems())
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

        # Update status bar from editor cursor position
        self.editor_widget.tabs.currentChanged.connect(self._update_status_bar)
        # Also connect first editor
        editor = self.editor_widget.get_current_editor()
        if editor:
            editor.cursorPositionChanged.connect(self._update_status_bar)

        # Help-on-function: right-click context menus in command & editor
        self.command_widget.help_requested.connect(self._show_help_for)
        self.editor_widget.help_requested.connect(self._show_help_for)
        self.editor_widget.eval_requested.connect(self._eval_in_command)

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
        self.act_find.triggered.connect(self._on_find)
        self.act_preferences = QAction("Preferences...", self, shortcut="Ctrl+,")
        self.act_preferences.triggered.connect(self._open_preferences)

        # Debug
        self.act_run = QAction("&Run File", self, shortcut="F5")
        self.act_run.triggered.connect(self._on_run_file)
        
        self.act_run_selection = QAction("Run &Selection", self, shortcut="F9")
        self.act_run_selection.triggered.connect(self._run_selection)
        self.act_stop = QAction("&Stop", self, shortcut="Shift+F5")
        
        self.act_step = QAction("Step &Over", self, shortcut="F10")
        self.act_step.setEnabled(False)
        self.act_step_in = QAction("Step &In", self, shortcut="F11")
        self.act_step_in.setEnabled(False)
        self.act_step_out = QAction("Step O&ut", self, shortcut="Shift+F11")
        self.act_step_out.setEnabled(False)
        self.act_continue = QAction("&Continue", self, shortcut="F8")
        self.act_continue.setEnabled(False)
        self.act_toggle_bp = QAction("Toggle &Breakpoint", self, shortcut="F12")
        self.act_toggle_bp.triggered.connect(lambda: self._editor_action('toggle_bookmark'))
        self.act_profile = QAction("&Profile Code", self)
        self.act_profile.triggered.connect(self._profile_current_file)

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

        # TheCommons actions
        self.act_check_updates = QAction("Check for &Updates...", self)
        self.act_check_updates.triggered.connect(self._check_for_updates)
        self.act_feature_request = QAction("Submit &Feature Request...", self)
        self.act_feature_request.triggered.connect(self._open_feature_request)
        self.act_ams_toggle = QAction("AMS &Telemetry...", self)
        self.act_ams_toggle.triggered.connect(self._toggle_ams)

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
        self.recent_menu = file_menu.addMenu("Recent Files")
        self._update_recent_menu()
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

        # Editor display
        self.act_word_wrap = QAction("Word Wrap", self, checkable=True)
        self.act_word_wrap.setChecked(False)
        self.act_word_wrap.triggered.connect(self._toggle_word_wrap)
        view_menu.addAction(self.act_word_wrap)

        self.act_zoom_in = QAction("Zoom In", self, shortcut="Ctrl+=")
        self.act_zoom_in.triggered.connect(lambda: self._zoom(1))
        view_menu.addAction(self.act_zoom_in)

        self.act_zoom_out = QAction("Zoom Out", self, shortcut="Ctrl+-")
        self.act_zoom_out.triggered.connect(lambda: self._zoom(-1))
        view_menu.addAction(self.act_zoom_out)

        self.act_zoom_reset = QAction("Reset Zoom", self, shortcut="Ctrl+0")
        # Don't add shortcut since Ctrl+0 is focus command — use menu only
        self.act_zoom_reset.triggered.connect(lambda: self._zoom(0))
        view_menu.addAction(self.act_zoom_reset)

        view_menu.addSeparator()
        view_menu.addAction(self.act_reset_layout)

        # ── Debug ──
        debug_menu = mb.addMenu("&Debug")
        debug_menu.addAction(self.act_run)
        debug_menu.addAction(self.act_run_selection)
        debug_menu.addAction(self.act_stop)
        debug_menu.addSeparator()
        debug_menu.addAction(self.act_toggle_bp)
        debug_menu.addSeparator()
        debug_menu.addAction(self.act_step_in)
        debug_menu.addAction(self.act_step)
        debug_menu.addAction(self.act_step_out)
        debug_menu.addAction(self.act_continue)
        debug_menu.addSeparator()
        debug_menu.addAction(self.act_profile)

        # ── Window ──
        window_menu = mb.addMenu("&Window")
        window_menu.addAction(self.act_focus_cmd)
        window_menu.addSeparator()
        window_menu.addAction(self.act_reset_layout)

        # ── Help ──
        help_menu = mb.addMenu("&Help")
        help_menu.addAction(self.act_about)
        help_menu.addAction(self.act_docs)

        act_shortcuts = QAction("Keyboard Shortcuts", self)
        act_shortcuts.triggered.connect(self._show_shortcuts)
        help_menu.addAction(act_shortcuts)
        help_menu.addSeparator()
        help_menu.addAction(self.act_check_updates)
        help_menu.addAction(self.act_feature_request)
        help_menu.addSeparator()
        help_menu.addAction(self.act_ams_toggle)

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

        # Debug toolbar
        debug_tb = QToolBar("Debug", self)
        debug_tb.setObjectName("DebugToolbar")
        debug_tb.setIconSize(QSize(18, 18))
        debug_tb.setMovable(False)
        debug_tb.addAction(self.act_run)
        debug_tb.addAction(self.act_run_selection)
        debug_tb.addAction(self.act_stop)
        debug_tb.addSeparator()
        debug_tb.addAction(self.act_step_in)
        debug_tb.addAction(self.act_step)
        debug_tb.addAction(self.act_step_out)
        debug_tb.addAction(self.act_continue)
        self.addToolBarBreak()
        self.addToolBar(debug_tb)

    # ==================================================================
    # Dock widgets
    # ==================================================================

    def _create_problems_panel(self):
        """Create the Problems/Diagnostics panel."""
        from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem, QHeaderView
        self._problems_tree = QTreeWidget()
        self._problems_tree.setHeaderLabels(["Severity", "Message", "File", "Line"])
        self._problems_tree.setAlternatingRowColors(True)
        self._problems_tree.setRootIsDecorated(False)
        header = self._problems_tree.header()
        header.setStretchLastSection(True)
        header.resizeSection(0, 60)
        header.resizeSection(1, 400)
        header.resizeSection(2, 150)
        header.resizeSection(3, 50)
        self._problems_tree.itemDoubleClicked.connect(self._on_problem_clicked)
        return self._problems_tree

    def _on_problem_clicked(self, item, column):
        """Navigate to the file/line of a diagnostic."""
        file_path = item.text(2)
        line_str = item.text(3)
        if file_path and os.path.isfile(file_path):
            self.open_file_in_editor(file_path)
            if line_str:
                try:
                    editor = self.editor_widget.get_current_editor()
                    if editor:
                        editor._goto_line(int(line_str))
                except (ValueError, AttributeError):
                    pass

    def _refresh_problems(self):
        """Refresh the problems panel from command widget diagnostics."""
        if not hasattr(self, '_problems_tree'):
            return
        from PySide6.QtWidgets import QTreeWidgetItem
        self._problems_tree.clear()
        diagnostics = self.command_widget.get_diagnostics()
        for d in diagnostics:
            item = QTreeWidgetItem([
                d['severity'].upper(),
                d['text'][:200],
                d.get('file', '') or '',
                str(d.get('line', '')) if d.get('line') else '',
            ])
            # Color based on severity
            if d['severity'] == 'error':
                item.setForeground(0, QColor("#f38ba8"))
            else:
                item.setForeground(0, QColor("#f9e2af"))
            self._problems_tree.addTopLevelItem(item)

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

        # Editor position indicator
        self._sb_position = QLabel("Ln 1, Col 1")
        self._sb_position.setStyleSheet("color: #a6adc8; font-size: 11px; padding: 0 8px;")
        sb.addPermanentWidget(self._sb_position)

        # Language mode
        self._sb_lang = QLabel("M-code")
        self._sb_lang.setStyleSheet("color: #a6adc8; font-size: 11px; padding: 0 8px;")
        sb.addPermanentWidget(self._sb_lang)

        # Encoding
        self._sb_encoding = QLabel("UTF-8")
        self._sb_encoding.setStyleSheet("color: #a6adc8; font-size: 11px; padding: 0 8px;")
        sb.addPermanentWidget(self._sb_encoding)

        # Theme indicator
        self._sb_theme = QLabel("dark")
        self._sb_theme.setStyleSheet("color: #6c7086; font-size: 11px; padding: 0 8px;")
        sb.addPermanentWidget(self._sb_theme)

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
        if hasattr(self, '_sb_theme'):
            self._sb_theme.setText(theme_name)
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
            self._add_recent_file(path)

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

    def _run_selection(self):
        """Run the selected text in the editor."""
        editor = self.editor_widget.get_current_editor()
        if editor:
            cursor = editor.textCursor()
            selected = cursor.selectedText()
            if selected:
                selected = selected.replace('\u2029', '\n')
                for line in selected.split('\n'):
                    line = line.strip()
                    if line:
                        self.command_widget._execute_command(line)

    def _profile_current_file(self):
        """Run the current file with basic profiling."""
        editor = self.editor_widget.get_current_editor()
        if not editor or not editor.file_path:
            return
        path = editor.file_path
        self.command_widget.append_output(f"Profiling {os.path.basename(path)}...")
        import time
        start = time.perf_counter()
        if hasattr(self, 'session') and self.session:
            try:
                self.session.eval(f"source('{path}')")
            except Exception as e:
                self.command_widget.append_output(f"Error: {e}")
        elapsed = time.perf_counter() - start
        self.command_widget.append_output(f"Profile: {elapsed:.3f}s elapsed")

    def _on_run_file(self):
        editor = self.editor_widget.get_current_editor()
        if editor and editor.file_path:
            self._run_file(editor.file_path)

    def open_file_in_editor(self, path):
        """Public method to open a file in the editor (used by 'edit' command)."""
        self.editor_widget.open_file(path)
        self.editor_dock.raise_()

    # ------------------------------------------------------------------
    # Recent files
    # ------------------------------------------------------------------

    def _load_recent_files(self):
        s = self._settings()
        files = s.value("recentFiles", [])
        if isinstance(files, str):
            files = [files] if files else []
        return files[:10]

    def _save_recent_files(self):
        s = self._settings()
        s.setValue("recentFiles", self._recent_files[:10])

    def _add_recent_file(self, path):
        if path in self._recent_files:
            self._recent_files.remove(path)
        self._recent_files.insert(0, path)
        self._recent_files = self._recent_files[:10]
        self._save_recent_files()
        self._update_recent_menu()

    def _update_recent_menu(self):
        if not hasattr(self, 'recent_menu'):
            return
        self.recent_menu.clear()
        for path in self._recent_files:
            name = os.path.basename(path)
            act = QAction(f"{name}  ({path})", self)
            act.triggered.connect(lambda checked=False, p=path: self._open_recent(p))
            self.recent_menu.addAction(act)
        if not self._recent_files:
            act = QAction("(No recent files)", self)
            act.setEnabled(False)
            self.recent_menu.addAction(act)

    def _open_recent(self, path):
        if os.path.exists(path):
            self.editor_widget.open_file(path)
            self.editor_dock.raise_()
        else:
            self._recent_files.remove(path)
            self._save_recent_files()
            self._update_recent_menu()

    def _toggle_word_wrap(self, checked):
        from PySide6.QtWidgets import QPlainTextEdit
        mode = QPlainTextEdit.WidgetWidth if checked else QPlainTextEdit.NoWrap
        for i in range(self.editor_widget.tabs.count()):
            editor = self.editor_widget.tabs.widget(i)
            if hasattr(editor, 'setLineWrapMode'):
                editor.setLineWrapMode(mode)

    def _zoom(self, direction):
        app = QApplication.instance()
        font = app.font()
        if direction == 0:
            font.setPointSize(10)
        else:
            font.setPointSize(max(6, font.pointSize() + direction))
        app.setFont(font)

    def _show_shortcuts(self):
        from PySide6.QtWidgets import QMessageBox
        shortcuts = (
            "<table cellpadding='4' style='font-size:11px;'>"
            "<tr><th>Shortcut</th><th>Action</th></tr>"
            "<tr><td><code>Ctrl+N</code></td><td>New File</td></tr>"
            "<tr><td><code>Ctrl+O</code></td><td>Open File</td></tr>"
            "<tr><td><code>Ctrl+S</code></td><td>Save</td></tr>"
            "<tr><td><code>Ctrl+F</code></td><td>Find/Replace</td></tr>"
            "<tr><td><code>F5</code></td><td>Run File</td></tr>"
            "<tr><td><code>F1</code></td><td>Documentation</td></tr>"
            "<tr><td><code>Ctrl+,</code></td><td>Preferences</td></tr>"
            "<tr><td><code>Ctrl+=/-</code></td><td>Zoom In/Out</td></tr>"
            "<tr><td><code>Tab</code></td><td>Complete/Indent</td></tr>"
            "<tr><td><code>Up/Down</code></td><td>History</td></tr>"
            "</table>"
        )
        QMessageBox.information(self, "Keyboard Shortcuts", shortcuts)

    def _goto_line_dialog(self):
        """Show Go to Line dialog."""
        from PySide6.QtWidgets import QInputDialog
        editor = self.editor_widget.get_current_editor()
        if editor:
            max_line = editor.document().blockCount()
            line, ok = QInputDialog.getInt(
                self, "Go to Line", f"Line number (1-{max_line}):",
                editor.textCursor().blockNumber() + 1, 1, max_line
            )
            if ok:
                editor._goto_line(line)

    def _editor_action(self, action_name):
        """Dispatch an action to the current editor."""
        editor = self.editor_widget.get_current_editor()
        if editor and hasattr(editor, action_name):
            getattr(editor, action_name)()

    def _on_find(self):
        """Open find/replace bar in the editor."""
        self.editor_widget.show_find()
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

    def _update_status_bar(self, _=None):
        """Update status bar with editor cursor position."""
        editor = self.editor_widget.get_current_editor()
        if editor and hasattr(self, '_sb_position'):
            cursor = editor.textCursor()
            line = cursor.blockNumber() + 1
            col = cursor.columnNumber() + 1
            self._sb_position.setText(f"Ln {line}, Col {col}")
            # Update language mode based on file extension
            if editor.file_path:
                ext = os.path.splitext(editor.file_path)[1].lower()
                lang_map = {'.m': 'M-code', '.py': 'Python', '.txt': 'Text', '.json': 'JSON'}
                self._sb_lang.setText(lang_map.get(ext, 'Text'))

    def _show_help_for(self, func_name):
        """Open help viewer and show docs for the given function."""
        self.help_dock.setVisible(True)
        self.help_dock.raise_()
        self.help_widget.search_edit.setText(func_name)
        self.help_widget.show_help(func_name)

    def _eval_in_command(self, code):
        """Evaluate code from editor selection in the command window."""
        if self.session:
            result = self.session.eval(code)
            if result:
                self.command_widget.append_output(result)
            self._update_workspace()

    def _show_docs(self):
        """Show the documentation panel."""
        self.help_dock.setVisible(True)
        self.help_dock.raise_()
        self.help_widget.search_edit.setFocus()

    # ==================================================================
    # TheCommons integration
    # ==================================================================

    def _setup_commons(self):
        """Initialize TheCommons services (update checker, AMS)."""
        import logging
        logging.basicConfig(level=logging.DEBUG)

        # Update checker
        self._update_checker = UpdateChecker(self)
        self._update_checker.update_available.connect(self._on_update_available)
        self._update_checker.start()

        # AMS connector
        self._ams = AMSConnector(self)

    def _on_update_available(self, version, url):
        """Show non-intrusive update notification in status bar."""
        self.statusBar().showMessage(
            f"Forge update available: v{version}  —  Visit thecommons.earth to download",
            0,  # permanent until cleared
        )

    def _check_for_updates(self):
        """Manual update check triggered from Help menu."""
        self.set_status("Checking for updates...")
        self._update_checker.check_now()
        from PySide6.QtCore import QTimer
        QTimer.singleShot(3000, lambda: self.set_status("Ready"))

    def _open_feature_request(self):
        """Open the feature request dialog."""
        dlg = FeatureRequestDialog(self)
        dlg.exec()

    def _toggle_ams(self):
        """Toggle AMS telemetry via opt-in dialog."""
        if self._ams.enabled:
            from PySide6.QtWidgets import QMessageBox
            result = QMessageBox.question(
                self, "AMS Telemetry",
                "Anonymized telemetry is currently <b>enabled</b>.<br>"
                "Would you like to disable it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if result == QMessageBox.StandardButton.Yes:
                self._ams.disconnect()
                self.set_status("AMS telemetry disabled")
            else:
                self.set_status("AMS telemetry remains enabled")
        else:
            opted_in = self._ams.show_opt_in_dialog(self)
            if opted_in:
                self.set_status("AMS telemetry enabled")
            else:
                self.set_status("AMS telemetry remains disabled")

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
