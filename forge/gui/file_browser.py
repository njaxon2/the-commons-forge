"""Forge file browser widget (forge/gui/file_browser.py).

Shows the current directory tree. Folders on the Forge search path
appear with solid icons; folders NOT on the path appear grayed-out.
Right-click context menu allows adding/removing folders from the path.
"""

import os

from PySide6.QtCore import QFileInfo, Qt, Signal, QDir, QModelIndex
from PySide6.QtGui import QAction, QColor, QBrush
from PySide6.QtWidgets import QFileIconProvider
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QTreeView,
    QPushButton, QFileSystemModel, QMenu, QInputDialog, QMessageBox,
    QStyle,
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


class FileBrowserWidget(QWidget):
    """Tree-based file browser with path-aware folder highlighting."""

    file_open_requested = Signal(str)
    path_changed = Signal(list)

    def __init__(self, root_path=None, parent=None):
        super().__init__(parent)
        self._root = root_path or os.path.expanduser("~")
        self._search_paths = []
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
        self.fs_model.setFilter(QDir.AllDirs | QDir.Files | QDir.NoDotAndDotDot)

        self.tree = QTreeView(self)
        self.tree.setModel(self.fs_model)
        self.tree.setRootIndex(self.fs_model.index(self._root))
        self.tree.setHeaderHidden(False)
        self.tree.setColumnWidth(0, 220)
        self.tree.doubleClicked.connect(self._on_double_click)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)
        layout.addWidget(self.tree)

    # -- Navigation --

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
        from PySide6.QtWidgets import QInputDialog
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
        from PySide6.QtWidgets import QInputDialog
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

