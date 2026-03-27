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
        self.command_widget.edit_requested.connect(self.open_file_in_editor)
        if hasattr(self.command_widget, 'plot_requested'):
            self.command_widget.plot_requested.connect(self._handle_plot_requests)
        self.editor_widget.help_requested.connect(self._show_help_for)
        self.editor_widget.eval_requested.connect(self._eval_in_command)

        # Wire editor cursor to status bar updates
        def _connect_editor_cursor():
            editor = self.editor_widget.get_current_editor() if hasattr(self.editor_widget, 'get_current_editor') else None
            if editor:
                try:
                    editor.cursorPositionChanged.disconnect(self._update_cursor_status)
                except (TypeError, RuntimeError):
                    pass
                editor.cursorPositionChanged.connect(self._update_cursor_status)
                self._update_cursor_status()
        self.editor_widget.tabs.currentChanged.connect(lambda _: _connect_editor_cursor())
        self.editor_widget.tabs.currentChanged.connect(lambda _: self._update_window_title())
        self.editor_widget.tabs.currentChanged.connect(lambda _: self._update_file_info_status())
        _connect_editor_cursor()

    # ==================================================================
    # Workspace helpers
    # ==================================================================

        # Update outline when editor content changes
        self.editor_widget.tabs.currentChanged.connect(self._update_outline)

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
        self.act_close_tab = QAction("Close Tab", self, shortcut="Ctrl+W")
        self.act_close_tab.triggered.connect(lambda: self.editor_widget._close_tab(self.editor_widget.tabs.currentIndex()))
        self.addAction(self.act_close_tab)

        self.act_cmd_palette = QAction("Command &Palette...", self, shortcut="Ctrl+Shift+P")
        self.act_cmd_palette.triggered.connect(self._show_command_palette)

        self.act_quick_open = QAction("Quick &Open...", self, shortcut="Ctrl+P")
        self.act_quick_open.triggered.connect(self._show_quick_open)

        self.act_search_files = QAction("Search in &Files...", self, shortcut="Ctrl+Shift+F")
        self.act_search_files.triggered.connect(self._show_search_in_files)

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
        self.act_focus_editor = QAction("Focus &Editor", self, shortcut="Ctrl+1")
        self.act_focus_editor.triggered.connect(lambda: self.editor_widget.tabs.currentWidget().setFocus() if self.editor_widget.tabs.currentWidget() else None)
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
        file_menu.addAction(self.act_quick_open)
        file_menu.addSeparator()
        file_menu.addAction(self.act_save)
        file_menu.addAction(self.act_save_as)
        file_menu.addSeparator()
        self.recent_menu = file_menu.addMenu("Recent Files")
        self._update_recent_menu()
        file_menu.addSeparator()
        act_compare = file_menu.addAction("Compare Files...")
        act_compare.triggered.connect(self._compare_files)
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
        edit_menu.addAction(self.act_cmd_palette)
        edit_menu.addSeparator()
        edit_menu.addAction(self.act_find)
        act_snippet = edit_menu.addAction("Insert Snippet...")
        act_snippet.setShortcut("Ctrl+Shift+S")
        act_snippet.triggered.connect(self._insert_snippet)
        edit_menu.addAction(self.act_search_files)
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
        act_cycle_theme = view_menu.addAction("Cycle Theme")
        act_cycle_theme.setShortcut(QKeySequence("Ctrl+Shift+T"))
        act_cycle_theme.triggered.connect(self._cycle_theme)

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
        help_menu.addSeparator()

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

        # Terminal (bottom, tabbed with command window)
        from forge.gui.terminal_widget import TerminalWidget
        self._terminal_widget = TerminalWidget(self)
        self._terminal_dock = self._make_dock(
            "Terminal", "TerminalDock", self._terminal_widget
        )
        self.addDockWidget(Qt.BottomDockWidgetArea, self._terminal_dock)
        self.tabifyDockWidget(self.command_dock, self._terminal_dock)
        self.command_dock.raise_()  # Command window on top initially

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

        # Selection info
        self._sb_selection = QLabel("")
        self._sb_selection.setStyleSheet("color: #a6adc8; font-size: 11px; padding: 0 8px;")
        sb.addPermanentWidget(self._sb_selection)

        # EOL indicator
        self._sb_eol = QLabel("LF")
        self._sb_eol.setStyleSheet("color: #a6adc8; font-size: 11px; padding: 0 8px;")
        sb.addPermanentWidget(self._sb_eol)

        # Engine status indicator
        self._sb_status = QLabel("● Idle")
        self._sb_status.setStyleSheet("color: #a6e3a1; font-size: 11px; padding: 0 8px;")
        sb.addPermanentWidget(self._sb_status)

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


    # ------------------------------------------------------------------
    # Keyboard shortcuts overlay
    # ------------------------------------------------------------------

    def _show_shortcuts_overlay(self):
        """Show the keyboard shortcuts cheat-sheet overlay."""
        from forge.gui.shortcuts_overlay import ShortcutsOverlay
        overlay = ShortcutsOverlay(self)
        overlay.exec()

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

    def _cycle_theme(self):
        """Cycle through available themes."""
        from forge.gui.themes import THEMES
        theme_names = list(THEMES.keys())
        current = getattr(self, '_current_theme', 'dark')
        idx = theme_names.index(current) if current in theme_names else 0
        next_theme = theme_names[(idx + 1) % len(theme_names)]
        self._switch_theme(next_theme)

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
            self._show_toast(f"Saved {os.path.basename(editor.file_path)}", 2000, "success")
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

    def _split_editor(self, direction):
        """Split the editor view."""
        from PySide6.QtWidgets import QSplitter
        from PySide6.QtCore import Qt as QtCore_Qt

        # Get the editor dock widget content
        editor_container = self.editor_dock.widget()
        if isinstance(editor_container, QSplitter):
            # Already split, don't split again
            return

        # Create a new editor widget for the split
        from forge.gui.editor_widget import EditorWidget
        self._split_editor_widget = EditorWidget(self)

        # Create splitter
        orientation = QtCore_Qt.Horizontal if direction == 'right' else QtCore_Qt.Vertical
        splitter = QSplitter(orientation)
        splitter.addWidget(self.editor_widget)
        splitter.addWidget(self._split_editor_widget)
        splitter.setSizes([500, 500])

        self.editor_dock.setWidget(splitter)
        self._editor_splitter = splitter

    def _unsplit_editor(self):
        """Remove editor split."""
        if hasattr(self, '_editor_splitter') and self._editor_splitter:
            # Remove the split editor
            if hasattr(self, '_split_editor_widget'):
                self._split_editor_widget.setParent(None)
                self._split_editor_widget.deleteLater()
                del self._split_editor_widget

            self.editor_dock.setWidget(self.editor_widget)
            self._editor_splitter = None

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

    def _show_snippets(self):
        """Show available code snippets."""
        from PySide6.QtWidgets import QMessageBox
        snippets_html = (
            "<h3>Code Snippets</h3>"
            "<p>Type a trigger word and press <b>Tab</b> to expand:</p>"
            "<table cellpadding='4' style='font-size:11px;'>"
            "<tr><th>Trigger</th><th>Expansion</th></tr>"
            "<tr><td><code>fori</code></td><td>for i = 1:n ... end</td></tr>"
            "<tr><td><code>forj</code></td><td>for j = 1:n ... end</td></tr>"
            "<tr><td><code>ife</code></td><td>if ... else ... end</td></tr>"
            "<tr><td><code>ifel</code></td><td>if ... elseif ... else ... end</td></tr>"
            "<tr><td><code>whi</code></td><td>while ... end</td></tr>"
            "<tr><td><code>swi</code></td><td>switch ... case ... end</td></tr>"
            "<tr><td><code>tryc</code></td><td>try ... catch ... end</td></tr>"
            "<tr><td><code>func</code></td><td>function definition</td></tr>"
            "<tr><td><code>cls</code></td><td>classdef template</td></tr>"
            "<tr><td><code>plt</code></td><td>figure + plot + labels</td></tr>"
            "<tr><td><code>fprintf</code></td><td>fprintf template</td></tr>"
            "<tr><td><code>fopen</code></td><td>fopen/fclose template</td></tr>"
            "</table>"
        )
        QMessageBox.information(self, "Code Snippets", snippets_html)

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
            "<tr><td><code>Ctrl+D</code></td><td>Duplicate Line</td></tr>"
            "<tr><td><code>Ctrl+/</code></td><td>Toggle Comment</td></tr>"
            "<tr><td><code>Alt+Up/Down</code></td><td>Move Line</td></tr>"
            "<tr><td><code>Ctrl+Shift+K</code></td><td>Delete Line</td></tr>"
            "<tr><td><code>Ctrl+G</code></td><td>Go to Line</td></tr>"
            "<tr><td><code>Ctrl+F2</code></td><td>Toggle Bookmark</td></tr>"
            "<tr><td><code>F2/Shift+F2</code></td><td>Next/Prev Bookmark</td></tr>"
            "<tr><td><code>F9</code></td><td>Run Selection</td></tr>"
            "<tr><td><code>F12</code></td><td>Toggle Breakpoint</td></tr>"
            "</table>"
        )
        QMessageBox.information(self, "Keyboard Shortcuts", shortcuts)

    def _goto_line_dialog(self):
        """Show Go to Line dialog with current position info."""
        from PySide6.QtWidgets import QInputDialog
        editor = self.editor_widget.get_current_editor() if hasattr(self.editor_widget, 'get_current_editor') else None
        if editor is None:
            return
        current_line = editor.textCursor().blockNumber() + 1
        total_lines = editor.document().blockCount()
        line, ok = QInputDialog.getInt(
            self, "Go to Line",
            f"Current: {current_line} / {total_lines}\nLine number (1-{total_lines}):",
            current_line, 1, total_lines
        )
        if ok:
            from PySide6.QtGui import QTextCursor
            block = editor.document().findBlockByLineNumber(line - 1)
            cursor = QTextCursor(block)
            editor.setTextCursor(cursor)
            editor.centerCursor()
            editor.setFocus()


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
        """Update status bar with editor cursor position and document stats."""
        editor = self.editor_widget.get_current_editor()
        if editor and hasattr(self, '_sb_position'):
            cursor = editor.textCursor()
            line = cursor.blockNumber() + 1
            col = cursor.columnNumber() + 1
            self._sb_position.setText(f"Ln {line}, Col {col}")

            # Update language mode based on file extension
            if hasattr(editor, 'file_path') and editor.file_path:
                ext = os.path.splitext(editor.file_path)[1].lower()
                lang_map = {'.m': 'M-code', '.py': 'Python', '.txt': 'Text', '.json': 'JSON', '.csv': 'CSV'}
                self._sb_lang.setText(lang_map.get(ext, 'Text'))

            # Selection info
            if cursor.hasSelection():
                selected = cursor.selectedText()
                sel_lines = selected.count('\u2029') + 1
                sel_chars = len(selected)
                self._status_msg.setText(f"{sel_lines} lines, {sel_chars} chars selected")
            else:
                total_lines = editor.document().blockCount()
                total_chars = editor.document().characterCount()
                self._status_msg.setText(f"{total_lines} lines, {total_chars} chars")


    def _show_help_for(self, func_name):
        """Open help viewer and show docs for the given function."""
        self.help_dock.setVisible(True)
        self.help_dock.raise_()
        self.help_widget.search_edit.setText(func_name)
        self.help_widget.show_help(func_name)

    def _show_search_in_files(self):
        """Show and focus the Search in Files panel."""
        if hasattr(self, '_bottom_tabs') and hasattr(self, '_search_panel'):
            self._bottom_tabs.setCurrentWidget(self._search_panel)
            self._search_panel._search_input.setFocus()
            self._search_panel._search_input.selectAll()

    def _update_outline(self, _=None):
        """Update the outline panel for the current editor."""
        if not hasattr(self, '_outline'):
            return
        editor = self.editor_widget.get_current_editor()
        if editor:
            text = editor.toPlainText()
            file_path = getattr(editor, 'file_path', None)
            self._outline.update_outline(text, file_path)

    def _goto_outline_line(self, line_num):
        """Navigate editor to line from outline panel."""
        editor = self.editor_widget.get_current_editor()
        if editor and hasattr(editor, '_goto_line'):
            editor._goto_line(line_num)

    def _open_search_result(self, file_path, line_num):
        """Open a file from search results at a specific line."""
        self.open_file_in_editor(file_path)
        editor = self.editor_widget.get_current_editor()
        if editor and hasattr(editor, '_goto_line'):
            editor._goto_line(line_num)

    def _handle_plot_requests(self, requests):
        """Process plot requests from the engine."""
        from forge.gui.plot_widget import PlotWidget
        pw = PlotWidget.create_new()

        for req in requests:
            cmd = req[0]
            if cmd == 'plot' and len(req) >= 3:
                pw.ax.plot(req[1], req[2])
            elif cmd == 'xlabel':
                pw.ax.set_xlabel(req[1])
            elif cmd == 'ylabel':
                pw.ax.set_ylabel(req[1])
            elif cmd == 'title':
                pw.ax.set_title(req[1])
            elif cmd == 'grid':
                pw.ax.grid(req[1] == 'on')

        pw.canvas.draw()
        pw.show()

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

    def _show_command_palette(self):
        """Show a command palette with all available actions."""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLineEdit, QListWidget, QListWidgetItem
        from PySide6.QtCore import Qt

        dialog = QDialog(self)
        dialog.setWindowTitle("Command Palette")
        dialog.setMinimumSize(500, 400)
        dialog.setStyleSheet("background: #1e1e2e; border-radius: 8px;")

        layout = QVBoxLayout(dialog)

        search_input = QLineEdit()
        search_input.setPlaceholderText("> Type a command...")
        search_input.setStyleSheet(
            "padding: 10px; font-size: 14px; background: #313244; "
            "color: #cdd6f4; border: 1px solid #45475a; border-radius: 6px;"
        )
        layout.addWidget(search_input)

        cmd_list = QListWidget()
        cmd_list.setStyleSheet(
            "QListWidget { background: #1e1e2e; color: #cdd6f4; border: none; font-size: 13px; }"
            "QListWidget::item { padding: 6px 8px; }"
            "QListWidget::item:selected { background: #313244; color: #cba6f7; }"
        )
        layout.addWidget(cmd_list)

        # Build command list from all available actions
        commands = [
            ("New File", "Ctrl+N", lambda: self.editor_widget.new_file()),
            ("Open File", "Ctrl+O", lambda: self._on_open()),
            ("Save", "Ctrl+S", lambda: self._on_save()),
            ("Quick Open", "Ctrl+P", self._show_quick_open),
            ("Find & Replace", "Ctrl+F", self._on_find),
            ("Go to Line", "Ctrl+G", self._goto_line_dialog),
            ("Run File", "F5", self._on_run_file),
            ("Run Selection", "F9", self._run_selection),
            ("Toggle Comment", "Ctrl+/", lambda: self._editor_action('toggle_comment')),
            ("Duplicate Line", "Ctrl+D", lambda: self._editor_action('_duplicate_line')),
            ("Delete Line", "Ctrl+Shift+K", lambda: self._editor_action('_delete_line')),
            ("Toggle Bookmark", "Ctrl+F2", lambda: self._editor_action('toggle_bookmark')),
            ("Next Bookmark", "F2", lambda: self._editor_action('next_bookmark')),
            ("Clear Bookmarks", "", lambda: self._editor_action('clear_bookmarks')),
            ("Toggle Minimap", "", lambda: self._editor_action('toggle_minimap')),
            ("Cycle Theme", "Ctrl+Shift+T", self._cycle_theme),
            ("Preferences", "Ctrl+,", self._open_preferences),
            ("Search in Files", "Ctrl+Shift+F", self._show_search_in_files),
            ("Split Editor Right", "", lambda: self._split_editor('right')),
            ("Split Editor Down", "", lambda: self._split_editor('down')),
            ("Remove Split", "", self._unsplit_editor),
            ("Clear Command Window", "", lambda: self.command_widget._clear_output()),
            ("Profile Code", "", self._profile_current_file),
            ("Code Snippets", "", self._show_snippets),
            ("Join Lines", "Ctrl+J", lambda: self._editor_action('_join_lines')),
            ("Sort Lines", "", lambda: self._editor_action('_sort_lines')),
            ("Transform to Upper Case", "", lambda: self._editor_action('_to_upper')),
            ("Transform to Lower Case", "", lambda: self._editor_action('_to_lower')),
            ("Indent Selection", "Ctrl+]", lambda: self._editor_action('_indent_selection')),
            ("Outdent Selection", "Ctrl+[", lambda: self._editor_action('_outdent_selection')),
            ("Select Line", "Ctrl+L", lambda: self._editor_action('_select_line')),
            ("Jump to Matching Bracket", "Ctrl+M", lambda: self._editor_action('_move_to_matching_bracket')),
            ("Fold All", "", lambda: self._editor_action('_fold_all')),
            ("Unfold All", "", lambda: self._editor_action('_unfold_all')),
            ("Keyboard Shortcuts", "", self._show_shortcuts),
            ("About Forge", "", self._show_about),
            ("Focus Command Window", "Ctrl+0", self._focus_command_input),
            ("Reset Layout", "", self._reset_layout),
        ]

        def update_list(text):
            cmd_list.clear()
            text = text.lower()
            for name, shortcut, action in commands:
                if not text or text in name.lower():
                    display = f"{name}"
                    if shortcut:
                        display += f"  ({shortcut})"
                    item = QListWidgetItem(display)
                    item.setData(Qt.UserRole, action)
                    cmd_list.addItem(item)

        search_input.textChanged.connect(update_list)
        update_list("")

        def on_select(item):
            action = item.data(Qt.UserRole)
            dialog.accept()
            if callable(action):
                action()

        cmd_list.itemDoubleClicked.connect(on_select)
        cmd_list.itemActivated.connect(on_select)

        def on_return():
            item = cmd_list.currentItem()
            if item:
                on_select(item)

        search_input.returnPressed.connect(on_return)
        dialog.exec()

    def _show_quick_open(self):
        """Show a quick-open dialog for files."""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLineEdit, QListWidget, QListWidgetItem
        from PySide6.QtCore import Qt

        dialog = QDialog(self)
        dialog.setWindowTitle("Quick Open")
        dialog.setMinimumSize(500, 400)
        dialog.setStyleSheet("background: #1e1e2e; border-radius: 8px;")

        layout = QVBoxLayout(dialog)

        search_input = QLineEdit()
        search_input.setPlaceholderText("Type to search files...")
        search_input.setStyleSheet(
            "padding: 10px; font-size: 14px; background: #313244; "
            "color: #cdd6f4; border: 1px solid #45475a; border-radius: 6px;"
        )
        layout.addWidget(search_input)

        file_list = QListWidget()
        file_list.setStyleSheet(
            "QListWidget { background: #1e1e2e; color: #cdd6f4; border: none; font-size: 13px; }"
            "QListWidget::item { padding: 6px 8px; }"
            "QListWidget::item:selected { background: #313244; color: #cba6f7; }"
        )
        layout.addWidget(file_list)

        # Collect files: recent files + .m files in current directory
        all_files = []
        # Recent files
        for path in self._recent_files:
            if os.path.exists(path):
                all_files.append(path)
        # .m files in working directory
        cwd = os.path.expanduser("~")
        if hasattr(self, 'session') and self.session and hasattr(self.session, '_workspace'):
            cwd = getattr(self.session, '_cwd', cwd)
        for root, dirs, files in os.walk(cwd):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('venv', '__pycache__')]
            for f in files:
                if f.endswith('.m') or f.endswith('.py'):
                    path = os.path.join(root, f)
                    if path not in all_files:
                        all_files.append(path)
            if len(all_files) > 200:
                break

        def update_list(text):
            file_list.clear()
            text = text.lower()
            for path in all_files:
                name = os.path.basename(path)
                if not text or text in name.lower() or text in path.lower():
                    rel = os.path.relpath(path, cwd)
                    item = QListWidgetItem(f"{name}  —  {rel}")
                    item.setData(Qt.UserRole, path)
                    file_list.addItem(item)
                    if file_list.count() >= 50:
                        break

        search_input.textChanged.connect(update_list)
        update_list("")

        def on_select(item):
            path = item.data(Qt.UserRole)
            if path:
                self.open_file_in_editor(path)
                dialog.accept()

        file_list.itemDoubleClicked.connect(on_select)
        file_list.itemActivated.connect(on_select)

        # Enter key opens selected item
        def on_return():
            item = file_list.currentItem()
            if item:
                on_select(item)

        search_input.returnPressed.connect(on_return)

        dialog.exec()




    def _compare_files(self):
        """Open a diff viewer to compare two files."""
        from PySide6.QtWidgets import QFileDialog
        path_a, _ = QFileDialog.getOpenFileName(self, "Select First File", "", "All Files (*)")
        if not path_a:
            return
        path_b, _ = QFileDialog.getOpenFileName(self, "Select Second File", "", "All Files (*)")
        if not path_b:
            return
        try:
            with open(path_a, 'r') as f:
                text_a = f.read()
            with open(path_b, 'r') as f:
                text_b = f.read()
            from forge.gui.diff_viewer import DiffViewer
            viewer = DiffViewer(text_a, text_b,
                              os.path.basename(path_a),
                              os.path.basename(path_b),
                              self)
            viewer.exec()
        except Exception as e:
            self._show_toast(f"Compare failed: {e}", 3000, "error")

    def _insert_snippet(self):
        """Open snippet dialog for current editor."""
        editor = self.editor_widget.get_current_editor()
        if editor and hasattr(editor, "open_snippet_dialog"):
            editor.open_snippet_dialog()

    def _show_toast(self, message, duration=3000, level="info"):
        """Show a brief toast notification at the bottom of the window."""
        from PySide6.QtWidgets import QLabel
        from PySide6.QtCore import QTimer, Qt, QPropertyAnimation
        from PySide6.QtGui import QColor

        colors = {
            "info": ("#89b4fa", "#1e1e2e"),
            "success": ("#a6e3a1", "#1e1e2e"),
            "warning": ("#f9e2af", "#1e1e2e"),
            "error": ("#f38ba8", "#1e1e2e"),
        }
        fg, bg = colors.get(level, colors["info"])

        toast = QLabel(f"  {message}  ", self)
        toast.setStyleSheet(f"""
            background: {bg};
            color: {fg};
            border: 1px solid {fg};
            border-radius: 6px;
            padding: 8px 16px;
            font-size: 12px;
            font-weight: bold;
        """)
        toast.setAlignment(Qt.AlignCenter)
        toast.adjustSize()

        # Position at bottom center
        x = (self.width() - toast.width()) // 2
        y = self.height() - toast.height() - 40
        toast.move(x, y)
        toast.show()
        toast.raise_()

        QTimer.singleShot(duration, toast.deleteLater)

    def _update_window_title(self):
        """Update window title with current file name."""
        title = "Forge IDE"
        if hasattr(self, 'editor_widget'):
            idx = self.editor_widget.tabs.currentIndex()
            if idx >= 0:
                tab_text = self.editor_widget.tabs.tabText(idx).rstrip(' \u25cf')
                if tab_text and tab_text != "Welcome":
                    title = f"{tab_text} \u2014 Forge IDE"
        self.setWindowTitle(title)


    def _update_file_info_status(self):
        """Update encoding and line ending info in status bar."""
        editor = self.editor_widget.get_current_editor() if hasattr(self.editor_widget, 'get_current_editor') else None
        if editor is None:
            return

        # Detect line endings
        text = editor.toPlainText()
        if hasattr(self, '_sb_eol'):
            if '\r\n' in text:
                self._sb_eol.setText("CRLF")
            elif '\r' in text:
                self._sb_eol.setText("CR")
            else:
                self._sb_eol.setText("LF")

        # Show file size
        path = getattr(editor, '_file_path', None) or getattr(editor, 'file_path', None)
        if path and os.path.exists(path):
            size = os.path.getsize(path)
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024 * 1024:
                size_str = f"{size/1024:.1f} KB"
            else:
                size_str = f"{size/1024/1024:.1f} MB"
            if hasattr(self, '_sb_encoding'):
                self._sb_encoding.setText(f"UTF-8 ({size_str})")

    def _update_cursor_status(self):
        """Update status bar from editor cursor position."""
        editor = self.editor_widget.get_current_editor() if hasattr(self.editor_widget, 'get_current_editor') else None
        if editor is None:
            return
        cursor = editor.textCursor()
        line = cursor.blockNumber() + 1
        col = cursor.columnNumber() + 1

        if hasattr(self, '_sb_position'):
            self._sb_position.setText(f"Ln {line}, Col {col}")

        if hasattr(self, '_sb_selection'):
            if cursor.hasSelection():
                sel_text = cursor.selectedText()
                nlines = sel_text.count('\u2029') + 1
                nchars = len(sel_text)
                self._sb_selection.setText(f"({nchars} selected, {nlines} lines)")
            else:
                self._sb_selection.setText("")

        # Update language indicator based on current tab
        if hasattr(self, '_sb_lang'):
            idx = self.editor_widget.tabs.currentIndex()
            if idx >= 0:
                name = self.editor_widget.tabs.tabText(idx).rstrip(' *')
                ext_map = {
                    '.py': 'Python', '.m': 'M-code', '.json': 'JSON',
                    '.csv': 'CSV', '.txt': 'Text', '.log': 'Log',
                    '.html': 'HTML', '.xml': 'XML', '.md': 'Markdown',
                    '.sh': 'Shell', '.c': 'C', '.cpp': 'C++',
                }
                lang = 'M-code'
                for ext, lname in ext_map.items():
                    if name.endswith(ext):
                        lang = lname
                        break
                self._sb_lang.setText(lang)

    def _show_about(self):
        """Show the About Forge dialog."""
        import platform
        import sys
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QDialogButtonBox
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QFont

        dialog = QDialog(self)
        dialog.setWindowTitle("About Forge")
        dialog.setFixedSize(420, 380)
        layout = QVBoxLayout(dialog)
        layout.setSpacing(12)

        # Logo/title
        title = QLabel("Forge")
        title.setFont(QFont("Fira Code", 28, QFont.Bold))
        title.setStyleSheet("color: #cba6f7;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Octave-Compatible Computing Environment")
        subtitle.setFont(QFont("Fira Code", 11))
        subtitle.setStyleSheet("color: #a6adc8;")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        version = QLabel("Version 0.1.0")
        version.setStyleSheet("color: #89b4fa; font-size: 13px;")
        version.setAlignment(Qt.AlignCenter)
        layout.addWidget(version)

        # System info
        func_count = len(self.session._engine.functions) if self.session else 0
        info_text = f"""
        <table style='color: #a6adc8; font-size: 11px;' cellpadding='4'>
        <tr><td><b>Functions:</b></td><td>{func_count} registered</td></tr>
        <tr><td><b>Python:</b></td><td>{sys.version.split()[0]}</td></tr>
        <tr><td><b>Qt:</b></td><td>{__import__('PySide6').__version__}</td></tr>
        <tr><td><b>NumPy:</b></td><td>{__import__('numpy').__version__}</td></tr>
        <tr><td><b>SciPy:</b></td><td>{__import__('scipy').__version__}</td></tr>
        <tr><td><b>Platform:</b></td><td>{platform.platform()}</td></tr>
        </table>
        """
        info = QLabel(info_text)
        info.setAlignment(Qt.AlignCenter)
        layout.addWidget(info)

        # Credits
        credits_text = QLabel("Built with \u2764 for engineers and scientists")
        credits_text.setStyleSheet("color: #6c7086; font-size: 10px;")
        credits_text.setAlignment(Qt.AlignCenter)
        layout.addWidget(credits_text)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)

        dialog.exec()

