"""Forge file browser widget (forge/gui/file_browser.py).

Shows the current directory tree. Folders on the Forge search path
appear with solid icons; folders NOT on the path appear grayed-out.
Right-click context menu allows adding/removing folders from the path.
"""

import os

from PySide6.QtCore import QFileInfo, Qt, Signal, QDir, QModelIndex, QSortFilterProxyModel, QMimeData, QUrl
from PySide6.QtGui import QAction, QColor, QBrush, QKeySequence, QIcon, QDrag, QShortcut
from PySide6.QtWidgets import QFileIconProvider
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QTreeView,
    QPushButton, QFileSystemModel, QMenu, QInputDialog, QMessageBox,
    QStyle, QComboBox, QApplication,
)


class PathAwareFSModel(QFileSystemModel):
    """QFileSystemModel subclass that grays-out folders NOT on the search path."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._search_paths = set()

    def set_search_paths(self, paths):
        """Update the set of active search-path directories."""
        self._search_paths = {os.path.normpath(p) for p in paths}
        self.layoutChanged.emit()

    def is_on_path(self, file_path):
        return os.path.normpath(file_path) in self._search_paths

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.ForegroundRole:
            path = self.filePath(index)
            if self.isDir(index) and path:
                if not self.is_on_path(path):
                    return QBrush(QColor(160, 160, 160))
        return super().data(index, role)



class FileIconProvider(QFileIconProvider):
    """Provides coloured icons based on file extension."""

    # Colour-coded SVG circle with a letter overlay
    _ICON_CACHE: dict = {}

    _EXT_MAP = {
        # (letter, colour)
        ".m":    ("M", "#FF8C00"),   # orange
        ".mat":  ("M", "#FF8C00"),
        ".py":   ("Py", "#3572A5"),  # blue
        ".pyw":  ("Py", "#3572A5"),
        ".txt":  ("T", "#888888"),   # gray
        ".md":   ("T", "#888888"),
        ".rst":  ("T", "#888888"),
        ".log":  ("T", "#888888"),
        ".json": ("J", "#2E7D32"),   # green
        ".yaml": ("Y", "#2E7D32"),
        ".yml":  ("Y", "#2E7D32"),
        ".toml": ("C", "#2E7D32"),
        ".csv":  ("C", "#00897B"),   # teal
        ".xlsx": ("X", "#00897B"),
        ".xls":  ("X", "#00897B"),
        ".tsv":  ("C", "#00897B"),
    }

    @classmethod
    def _make_icon(cls, letter: str, colour: str) -> "QIcon":
        key = (letter, colour)
        if key in cls._ICON_CACHE:
            return cls._ICON_CACHE[key]


        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32">'
            f'<rect x="2" y="2" width="28" height="28" rx="6" '
            f'fill="{colour}" opacity="0.85"/>'
            f'<text x="16" y="22" text-anchor="middle" '
            f'font-family="sans-serif" font-size="14" font-weight="bold" '
            f'fill="white">{letter}</text></svg>'
        )
        renderer = QSvgRenderer(QByteArray(svg.encode()))
        pix = QPixmap(32, 32)
        pix.fill(Qt.transparent)
        painter = QPainter(pix)
        renderer.render(painter)
        painter.end()
        icon = QIcon(pix)
        cls._ICON_CACHE[key] = icon
        return icon

    # QFileIconProvider overrides ----------------------------------------
    def icon(self, info_or_type):
        if isinstance(info_or_type, QFileInfo):
            if info_or_type.isDir():
                return super().icon(info_or_type)
            ext = os.path.splitext(info_or_type.fileName())[1].lower()
            if ext in self._EXT_MAP:
                letter, colour = self._EXT_MAP[ext]
                return self._make_icon(letter, colour)
        return super().icon(info_or_type)



class GlobFilterProxyModel(QSortFilterProxyModel):
    """Filters QFileSystemModel rows by glob pattern (e.g. *.m, *.py)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._glob = ""
        self._filter_type = "All Files"
        self.setRecursiveFilteringEnabled(True)

    # --- public API ---------------------------------------------------
    def set_glob(self, pattern: str):
        self._glob = pattern.strip()
        self.invalidateFilter()

    def set_filter_type(self, label: str):
        self._filter_type = label
        self.invalidateFilter()

    # --- override -----------------------------------------------------
    def filterAcceptsRow(self, source_row, source_parent):
        model = self.sourceModel()
        idx = model.index(source_row, 0, source_parent)
        if not idx.isValid():
            return False
        # Always show directories so the tree stays navigable
        if model.isDir(idx):
            return True
        name = model.fileName(idx)
        # Apply filter-type preset
        if self._filter_type == "M-Files" and not name.lower().endswith(".m"):
            return False
        elif self._filter_type == "Python" and not name.lower().endswith((".py", ".pyw")):
            return False
        elif self._filter_type == "Text" and not name.lower().endswith(
            (".txt", ".md", ".rst", ".log")
        ):
            return False
        # Apply glob
        if self._glob:
            import fnmatch
            if not fnmatch.fnmatch(name, self._glob):
                return False
        return True


class FileBrowserWidget(QWidget):
    """Tree-based file browser with path-aware folder highlighting."""

    file_open_requested = Signal(str)
    open_terminal_requested = Signal(str)
    path_changed = Signal(list)

    def __init__(self, root_path=None, parent=None):
        super().__init__(parent)
        self._root = root_path or os.path.expanduser("~")
        self._search_paths = []
        self._nav_history = [self._root]
        self._nav_index = 0
        self._nav_recording = True
        self._build_ui()

    def set_search_paths(self, paths):
        """Set the Forge search path (called by main window after engine init)."""
        self._search_paths = list(paths)
        self.fs_model.set_search_paths(paths)

    def get_search_paths(self):
        return list(self._search_paths)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(2)

        # Use standard platform icons instead of text for the nav buttons
        style = self.style()
        home_icon = style.standardIcon(QStyle.StandardPixmap.SP_DirHomeIcon)
        up_icon = style.standardIcon(QStyle.StandardPixmap.SP_ArrowUp)

        # Override the global QPushButton theme for these small nav buttons
        # so they don't appear as solid blue rectangles.
        _nav_btn_style = """
            QPushButton {
                background-color: transparent;
                border: 1px solid #ccc;
                border-radius: 3px;
                padding: 2px;
                font-weight: normal;
            }
            QPushButton:hover {
                background-color: #e3f2fd;
            }
            QPushButton:pressed {
                background-color: #bbdefb;
            }
        """

        self.btn_home = QPushButton(self)
        self.btn_home.setIcon(home_icon)
        self.btn_home.setToolTip("Home directory")
        self.btn_home.setFixedSize(28, 28)
        self.btn_home.setStyleSheet(_nav_btn_style)
        self.btn_home.clicked.connect(self._go_home)
        nav_layout.addWidget(self.btn_home)

        self.btn_up = QPushButton(self)
        self.btn_up.setIcon(up_icon)
        self.btn_up.setToolTip("Parent directory")
        self.btn_up.setFixedSize(28, 28)
        self.btn_up.setStyleSheet(_nav_btn_style)
        self.btn_up.clicked.connect(self._go_up)
        nav_layout.addWidget(self.btn_up)

        self.path_edit = QLineEdit(self._root, self)
        self.path_edit.returnPressed.connect(self._navigate_to_path)
        nav_layout.addWidget(self.path_edit)
        layout.addLayout(nav_layout)

        # Path-aware file-system model
        self.fs_model = PathAwareFSModel(self)
        self.fs_model.setRootPath(self._root)
        self.model = self.fs_model  # alias for context menu code
        self.fs_model.setFilter(QDir.AllDirs | QDir.Files | QDir.NoDotAndDotDot)

        self._proxy = None  # proxy disabled for stability

        self.tree = QTreeView(self)
        self.tree.setDragEnabled(True)

        self._rename_shortcut = QShortcut(QKeySequence("F2"), self.tree)
        self._rename_shortcut.activated.connect(self._on_rename_shortcut)

        self.tree.setDragDropMode(QTreeView.DragOnly)
        self.tree.setModel(self.fs_model)
        self.tree.setRootIndex(self.fs_model.index(self._root))
        self.tree.setHeaderHidden(False)
        self.tree.setColumnWidth(0, 220)
        self.tree.doubleClicked.connect(self._on_double_click)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)

        # --- filter / search bar ---
        self._filter_row = QHBoxLayout()
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Filter (glob): e.g. *.m, *.py")
        self._search_edit.setClearButtonEnabled(True)
        self._search_edit.textChanged.connect(self._on_filter_glob_changed)

        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["All Files", "M-Files", "Python", "Text"])
        self._filter_combo.currentTextChanged.connect(self._on_filter_type_changed)
        self._filter_combo.setFixedWidth(100)

        self._filter_row.addWidget(self._search_edit)
        self._filter_row.addWidget(self._filter_combo)

        layout.addLayout(self._filter_row)
        layout.addWidget(self.tree)

        # Keyboard navigation shortcuts
        self._shortcut_up = QShortcut(QKeySequence('Alt+Up'), self)
        self._shortcut_up.activated.connect(self._go_up)
        self._shortcut_back = QShortcut(QKeySequence('Alt+Left'), self)
        self._shortcut_back.activated.connect(self._go_back)
        self._shortcut_fwd = QShortcut(QKeySequence('Alt+Right'), self)
        self._shortcut_fwd.activated.connect(self._go_forward)

    def _on_filter_glob_changed(self, text):
        if hasattr(self, '_proxy'):
            self._proxy.invalidateFilter()

    def _on_filter_type_changed(self, label):
        if hasattr(self, '_proxy'):
            self._proxy.set_filter_type(label)
            self._proxy.invalidateFilter()

    def _resolve_index(self, proxy_index):
        """Map a proxy QModelIndex back to the source QFileSystemModel index."""
        if hasattr(self, "_proxy") and self._proxy is not None:
            return self._proxy.mapToSource(proxy_index)
        return proxy_index

    def _resolve_source_to_proxy(self, source_index):
        """Map a source QModelIndex to the proxy index (for tree selection)."""
        if hasattr(self, "_proxy") and self._proxy is not None:
            return self._proxy.mapFromSource(source_index)
        return source_index

    # --- enhanced context menu -----------------------------------------
    def _show_context_menu(self, pos):
        """Right-click menu with file operations."""
        index = self.tree.indexAt(pos)
        menu = QMenu(self)

        # Map proxy index -> source index for path resolution
        if hasattr(self, "_proxy") and self._proxy is not None:
            source_index = self._proxy.mapToSource(index)
        else:
            source_index = index

        model = self.model
        path = model.filePath(source_index) if source_index.isValid() else ""
        is_dir = model.isDir(source_index) if source_index.isValid() else False
        parent_dir = path if is_dir else os.path.dirname(path) if path else ""

        # -- New File / Folder --
        if parent_dir:
            new_file_act = menu.addAction("New File...")
            new_file_act.triggered.connect(lambda: self._action_new_file(parent_dir))
            new_folder_act = menu.addAction("New Folder...")
            new_folder_act.triggered.connect(lambda: self._action_new_folder(parent_dir))
            menu.addSeparator()

        # -- Rename / Delete --
        if path and source_index.isValid():
            rename_act = menu.addAction("Rename...\tF2")
            rename_act.triggered.connect(lambda: self._action_rename(path))
            delete_act = menu.addAction("Delete")
            delete_act.triggered.connect(lambda: self._action_delete(path))
            menu.addSeparator()

        # -- Copy Path --
        if path:
            copy_act = menu.addAction("Copy Path")
            copy_act.triggered.connect(lambda: self._action_copy_path(path))

        # -- Open in Terminal --
        if parent_dir:
            term_act = menu.addAction("Open in Terminal")
            term_act.triggered.connect(lambda: self._action_open_terminal(parent_dir))

        # -- Reveal in file manager --
        if path:
            reveal_act = menu.addAction("Reveal in System File Manager")
            reveal_act.triggered.connect(lambda: self._action_reveal(path))

        # -- legacy root-path actions (add / remove watched path) --
        menu.addSeparator()
        add_act = menu.addAction("Add Path...")
        add_act.triggered.connect(self._add_root_path)
        if path:
            remove_act = menu.addAction("Remove Path")
            remove_act.triggered.connect(lambda: self._remove_root_path_by_selection(source_index))

        if not menu.isEmpty():
            menu.exec(self.tree.viewport().mapToGlobal(pos))

    # --- action helpers ------------------------------------------------
    def _action_new_file(self, directory):
        name, ok = QInputDialog.getText(self, "New File", "File name:")
        if ok and name:
            fpath = os.path.join(directory, name)
            try:
                open(fpath, "w").close()
            except OSError as e:
                QMessageBox.warning(self, "Error", str(e))

    def _action_new_folder(self, directory):
        name, ok = QInputDialog.getText(self, "New Folder", "Folder name:")
        if ok and name:
            fpath = os.path.join(directory, name)
            try:
                os.makedirs(fpath, exist_ok=True)
            except OSError as e:
                QMessageBox.warning(self, "Error", str(e))

    def _on_rename_shortcut(self):
        """F2 shortcut: rename the currently selected item."""
        index = self.tree.currentIndex()
        if index.isValid():
            source_index = self._resolve_index(index)
            path = self.fs_model.filePath(source_index)
            if path:
                self._action_rename(path)

    def _action_rename(self, path):
        old_name = os.path.basename(path)
        new_name, ok = QInputDialog.getText(self, "Rename", "New name:", text=old_name)
        if ok and new_name and new_name != old_name:
            new_path = os.path.join(os.path.dirname(path), new_name)
            try:
                os.rename(path, new_path)
            except OSError as e:
                QMessageBox.warning(self, "Error", str(e))

    def _action_delete(self, path):
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete '{os.path.basename(path)}'?\n\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except OSError as e:
                QMessageBox.warning(self, "Error", str(e))

    def _action_copy_path(self, path):
        QApplication.clipboard().setText(path)

    def _action_open_terminal(self, directory):
        """Emit signal so the main window can open a terminal widget."""
        if hasattr(self, "open_terminal_requested"):
            self.open_terminal_requested.emit(directory)

    def _action_reveal(self, path):
        import subprocess, sys
        target = path if os.path.isdir(path) else os.path.dirname(path)
        if sys.platform == "darwin":
            subprocess.Popen(["open", target])
        elif sys.platform == "win32":
            subprocess.Popen(["explorer", target])
        else:
            subprocess.Popen(["xdg-open", target])



    # --- drag support (file URLs for editor drop) ----------------------
    def _start_file_drag(self):
        """Called internally; creates a QDrag with file URL mime data."""
        index = self.tree.currentIndex()
        if hasattr(self, "_proxy") and self._proxy is not None:
            index = self._proxy.mapToSource(index)
        if not index.isValid():
            return
        path = self.model.filePath(index)
        if not path or self.model.isDir(index):
            return
        drag = QDrag(self)
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(path)])
        mime.setText(path)
        drag.setMimeData(mime)
        drag.exec(Qt.CopyAction)

    # -- Navigation --

    def _go_back(self):
        if self._nav_index > 0:
            self._nav_index -= 1
            self._nav_recording = False
            self._set_root(self._nav_history[self._nav_index])
            self._nav_recording = True

    def _go_forward(self):
        if self._nav_index < len(self._nav_history) - 1:
            self._nav_index += 1
            self._nav_recording = False
            self._set_root(self._nav_history[self._nav_index])
            self._nav_recording = True

    def _go_home(self):
        self._set_root(os.path.expanduser("~"))

    def _go_up(self):
        parent = os.path.dirname(self._root)
        if parent and parent != self._root:
            self._set_root(parent)

    def _navigate_to_path(self):
        path = self.path_edit.text().strip()
        if os.path.isdir(path):
            self._set_root(path)

    def _set_root(self, path):
        self._root = path
        self.path_edit.setText(path)
        self.fs_model.setRootPath(path)
        self.tree.setRootIndex(self.fs_model.index(path))
        # Record in navigation history
        if self._nav_recording:
            # Trim any forward history beyond current index
            self._nav_history = self._nav_history[:self._nav_index + 1]
            # Avoid duplicates at the tip
            if not self._nav_history or self._nav_history[-1] != path:
                self._nav_history.append(path)
                self._nav_index = len(self._nav_history) - 1

    # -- Double-click --

    def _on_double_click(self, index):
        path = self.fs_model.filePath(index)
        if os.path.isfile(path) and path.endswith(".m"):
            self.file_open_requested.emit(path)
        elif os.path.isdir(path):
            self._set_root(path)

    # -- Context menu with path management --

    def _context_menu(self, pos):
        index = self.tree.indexAt(pos)
        menu = QMenu(self)

        act_new_mfile = QAction("New M-File...", self)
        act_new_mfile.triggered.connect(self._new_mfile)
        menu.addAction(act_new_mfile)

        act_new_func = QAction("New Function...", self)
        act_new_func.triggered.connect(self._new_function_file)
        menu.addAction(act_new_func)

        act_new_file = QAction("New File", self)
        act_new_file.triggered.connect(lambda: self._new_item(is_dir=False))
        menu.addAction(act_new_file)

        act_new_folder = QAction("New Folder", self)
        act_new_folder.triggered.connect(lambda: self._new_item(is_dir=True))
        menu.addAction(act_new_folder)

        if index.isValid():
            path = self.fs_model.filePath(index)

            if os.path.isdir(path):
                menu.addSeparator()
                norm_path = os.path.normpath(path)
                on_path = norm_path in {os.path.normpath(p) for p in self._search_paths}

                if not on_path:
                    act_add = QAction("Add to Search Path", self)
                    act_add.triggered.connect(lambda checked=False, p=path: self._add_to_path(p))
                    menu.addAction(act_add)

                    act_add_sub = QAction("Add with Subfolders to Search Path", self)
                    act_add_sub.triggered.connect(lambda checked=False, p=path: self._add_to_path_recursive(p))
                    menu.addAction(act_add_sub)
                else:
                    act_rm = QAction("Remove from Search Path", self)
                    act_rm.triggered.connect(lambda checked=False, p=path: self._remove_from_path(p))
                    menu.addAction(act_rm)

                    act_rm_sub = QAction("Remove with Subfolders from Search Path", self)
                    act_rm_sub.triggered.connect(lambda checked=False, p=path: self._remove_from_path_recursive(p))
                    menu.addAction(act_rm_sub)

            menu.addSeparator()
            act_rename = QAction("Rename", self)
            act_rename.triggered.connect(lambda checked=False, i=index: self._rename(i))
            menu.addAction(act_rename)

            act_delete = QAction("Delete", self)
            act_delete.triggered.connect(lambda checked=False, i=index: self._delete(i))
            menu.addAction(act_delete)

        menu.exec(self.tree.viewport().mapToGlobal(pos))

    # -- Search path management --

    def _add_to_path(self, path):
        norm = os.path.normpath(path)
        if norm not in self._search_paths:
            self._search_paths.insert(0, norm)
        self.fs_model.set_search_paths(self._search_paths)
        self.path_changed.emit(self._search_paths)

    def _add_to_path_recursive(self, path):
        norm = os.path.normpath(path)
        if norm not in self._search_paths:
            self._search_paths.insert(0, norm)
        for root, dirs, _ in os.walk(path):
            for d in dirs:
                if d.startswith('.') or d.startswith('+') or d.startswith('@'):
                    continue
                sub = os.path.normpath(os.path.join(root, d))
                if sub not in self._search_paths:
                    self._search_paths.append(sub)
        self.fs_model.set_search_paths(self._search_paths)
        self.path_changed.emit(self._search_paths)

    def _remove_from_path(self, path):
        norm = os.path.normpath(path)
        self._search_paths = [p for p in self._search_paths if os.path.normpath(p) != norm]
        self.fs_model.set_search_paths(self._search_paths)
        self.path_changed.emit(self._search_paths)

    def _remove_from_path_recursive(self, path):
        norm = os.path.normpath(path)
        self._search_paths = [
            p for p in self._search_paths
            if not os.path.normpath(p).startswith(norm)
        ]
        self.fs_model.set_search_paths(self._search_paths)
        self.path_changed.emit(self._search_paths)

    # -- File operations --

    def _new_item(self, is_dir):
        label = "folder" if is_dir else "file"
        name, ok = QInputDialog.getText(self, f"New {label}", f"{label.capitalize()} name:")
        if not ok or not name:
            return
        full = os.path.join(self._root, name)
        if is_dir:
            os.makedirs(full, exist_ok=True)
        else:
            with open(full, "w") as fh:
                fh.write("")

    def _rename(self, index):
        old_path = self.fs_model.filePath(index)
        old_name = os.path.basename(old_path)
        new_name, ok = QInputDialog.getText(self, "Rename", "New name:", text=old_name)
        if ok and new_name and new_name != old_name:
            new_path = os.path.join(os.path.dirname(old_path), new_name)
            os.rename(old_path, new_path)

    def _delete(self, index):
        path = self.fs_model.filePath(index)
        ans = QMessageBox.question(
            self, "Delete", f"Delete '{os.path.basename(path)}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if ans == QMessageBox.Yes:
            if os.path.isdir(path):
                import shutil
                shutil.rmtree(path)
            else:
                os.remove(path)

    def _new_mfile(self):
        """Create a new .m script file from template."""
        name, ok = QInputDialog.getText(self, "New M-File", "File name (without .m):")
        if ok and name:
            path = os.path.join(self._current_path(), f"{name}.m")
            template = (
                f"% {name}.m \u2014 Description\n"
                f"% Created: {__import__('datetime').datetime.now().strftime('%Y-%m-%d')}\n"
                f"\n"
                f"% Your code here\n"
            )
            with open(path, 'w') as f:
                f.write(template)
            self.file_open_requested.emit(path)

    def _new_function_file(self):
        """Create a new function .m file from template."""
        name, ok = QInputDialog.getText(self, "New Function", "Function name:")
        if ok and name:
            path = os.path.join(self._current_path(), f"{name}.m")
            template = (
                f"function result = {name}(varargin)\n"
                f"% {name.upper()} - Description\n"
                f"%\n"
                f"%   result = {name}(arg1, arg2, ...)\n"
                f"%\n"
                f"% See also: \n"
                f"\n"
                f"    % Your code here\n"
                f"    result = [];\n"
                f"\n"
                f"end\n"
            )
            with open(path, 'w') as f:
                f.write(template)
            self.file_open_requested.emit(path)

    def _current_path(self):
        """Get the current directory from the file browser."""
        idx = self.tree.currentIndex()
        if idx.isValid():
            path = self.model.filePath(idx)
            if os.path.isfile(path):
                return os.path.dirname(path)
            return path
        return os.path.expanduser("~")

