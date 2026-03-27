"""Search in Files panel for Forge IDE."""
import os
import re

from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QColor, QFont, QTextCursor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
    QTreeWidget, QTreeWidgetItem, QLabel, QCheckBox, QComboBox,
    QProgressBar,
)


class SearchWorker(QThread):
    """Background thread for searching files."""
    result_found = Signal(str, int, str)  # file, line_num, line_text
    finished = Signal(int)  # total matches

    def __init__(self, directory, pattern, case_sensitive=False, regex=False, file_filter="*.m"):
        super().__init__()
        self.directory = directory
        self.pattern = pattern
        self.case_sensitive = case_sensitive
        self.regex = regex
        self.file_filter = file_filter
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        import fnmatch
        total = 0
        extensions = [f.strip() for f in self.file_filter.split(',')]

        for root, dirs, files in os.walk(self.directory):
            if self._stop:
                break
            # Skip hidden directories and common non-code dirs
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('venv', '__pycache__', 'node_modules')]

            for fname in files:
                if self._stop:
                    break
                # Check file filter
                if not any(fnmatch.fnmatch(fname, ext) for ext in extensions):
                    continue

                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, 'r', errors='replace') as f:
                        for line_num, line in enumerate(f, 1):
                            if self._stop:
                                break
                            match = False
                            if self.regex:
                                flags = 0 if self.case_sensitive else re.IGNORECASE
                                match = bool(re.search(self.pattern, line, flags))
                            elif self.case_sensitive:
                                match = self.pattern in line
                            else:
                                match = self.pattern.lower() in line.lower()

                            if match:
                                self.result_found.emit(fpath, line_num, line.rstrip())
                                total += 1
                                if total >= 1000:  # Limit results
                                    self.finished.emit(total)
                                    return
                except (OSError, UnicodeDecodeError):
                    continue

        self.finished.emit(total)


class SearchInFilesPanel(QWidget):
    """Panel for searching across files in the project."""
    file_open_requested = Signal(str, int)  # file_path, line_number

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Search input
        search_row = QHBoxLayout()
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Search in files...")
        self._search_input.returnPressed.connect(self._start_search)
        search_row.addWidget(self._search_input)

        self._search_btn = QPushButton("Search")
        self._search_btn.clicked.connect(self._start_search)
        search_row.addWidget(self._search_btn)
        layout.addLayout(search_row)

        # Options row
        opts_row = QHBoxLayout()
        self._case_check = QCheckBox("Cc")
        self._case_check.setToolTip("Case Sensitive")
        opts_row.addWidget(self._case_check)

        self._regex_check = QCheckBox(".*")
        self._regex_check.setToolTip("Regular Expression")
        opts_row.addWidget(self._regex_check)

        self._filter_combo = QComboBox()
        self._filter_combo.setEditable(True)
        self._filter_combo.addItems(["*.m", "*.m,*.py", "*.m,*.py,*.txt", "*.*"])
        self._filter_combo.setToolTip("File filter")
        opts_row.addWidget(self._filter_combo)

        opts_row.addStretch()
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #6c7086; font-size: 11px;")
        opts_row.addWidget(self._status_label)
        layout.addLayout(opts_row)

        # Results tree
        self._results = QTreeWidget()
        self._results.setHeaderLabels(["File", "Line", "Text"])
        self._results.setAlternatingRowColors(True)
        self._results.setRootIsDecorated(True)
        self._results.itemDoubleClicked.connect(self._on_result_clicked)
        header = self._results.header()
        header.resizeSection(0, 200)
        header.resizeSection(1, 50)
        header.setStretchLastSection(True)
        layout.addWidget(self._results)

    def set_directory(self, path):
        """Set the search root directory."""
        self._directory = path

    def _start_search(self):
        pattern = self._search_input.text().strip()
        if not pattern:
            return

        # Stop previous search
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait()

        self._results.clear()
        self._file_items = {}  # cache file-level tree items
        self._status_label.setText("Searching...")

        directory = getattr(self, '_directory', os.path.expanduser("~"))

        self._worker = SearchWorker(
            directory, pattern,
            case_sensitive=self._case_check.isChecked(),
            regex=self._regex_check.isChecked(),
            file_filter=self._filter_combo.currentText(),
        )
        self._worker.result_found.connect(self._on_result)
        self._worker.finished.connect(self._on_search_done)
        self._worker.start()

    def _on_result(self, file_path, line_num, line_text):
        # Group by file
        if file_path not in self._file_items:
            rel = os.path.relpath(file_path, getattr(self, '_directory', os.path.expanduser("~")))
            file_item = QTreeWidgetItem(self._results, [rel, "", ""])
            file_item.setForeground(0, QColor("#89b4fa"))
            font = file_item.font(0)
            font.setBold(True)
            file_item.setFont(0, font)
            file_item.setExpanded(True)
            file_item._file_path = file_path
            self._file_items[file_path] = file_item

        parent = self._file_items[file_path]
        item = QTreeWidgetItem(parent, ["", str(line_num), line_text[:200]])
        item._file_path = file_path
        item._line_num = line_num

    def _on_search_done(self, total):
        file_count = len(self._file_items)
        self._status_label.setText(f"{total} results in {file_count} files")

    def _on_result_clicked(self, item, column):
        if hasattr(item, '_file_path') and hasattr(item, '_line_num'):
            self.file_open_requested.emit(item._file_path, item._line_num)
        elif hasattr(item, '_file_path'):
            self.file_open_requested.emit(item._file_path, 1)
