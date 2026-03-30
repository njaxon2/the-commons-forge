# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Search (and Replace) in Files panel with background worker."""

import os
import re
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
    QTreeWidget, QTreeWidgetItem, QCheckBox, QLabel, QProgressBar,
    QComboBox,
)
from PySide6.QtGui import QFont, QColor


class SearchWorker(QThread):
    """Background thread for searching files."""
    result_found = Signal(str, int, str)  # file, line_num, line_text
    finished = Signal(int)  # total matches
    progress = Signal(int, int)  # current, total

    def __init__(self, root_dir, pattern, file_filter="*.*",
                 case_sensitive=False, use_regex=False, whole_word=False):
        super().__init__()
        self.root_dir = root_dir
        self.pattern = pattern
        self.file_filter = file_filter
        self.case_sensitive = case_sensitive
        self.use_regex = use_regex
        self.whole_word = whole_word
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        import fnmatch
        total_matches = 0

        # Collect files
        files = []
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            # Skip hidden and common non-code dirs
            dirnames[:] = [d for d in dirnames if not d.startswith('.')
                          and d not in ('__pycache__', 'node_modules', '.git', 'venv')]
            for fn in filenames:
                if fnmatch.fnmatch(fn, self.file_filter):
                    files.append(os.path.join(dirpath, fn))

        for i, fpath in enumerate(files):
            if self._cancelled:
                break
            self.progress.emit(i + 1, len(files))
            try:
                with open(fpath, 'r', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        if self._cancelled:
                            break
                        if self._match(line):
                            self.result_found.emit(fpath, line_num, line.rstrip())
                            total_matches += 1
            except (OSError, PermissionError):
                pass

        self.finished.emit(total_matches)

    def _match(self, line):
        flags = 0 if self.case_sensitive else re.IGNORECASE
        pattern = self.pattern
        if not self.use_regex:
            pattern = re.escape(pattern)
        if self.whole_word:
            pattern = r'\b' + pattern + r'\b'
        try:
            return bool(re.search(pattern, line, flags))
        except re.error:
            return False


class SearchInFilesPanel(QWidget):
    """Panel for searching (and replacing) across files."""

    file_open_requested = Signal(str, int)  # filepath, line number

    def __init__(self, root_dir=None, parent=None):
        super().__init__(parent)
        self.root_dir = root_dir or os.path.expanduser("~")
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Search input row
        search_row = QHBoxLayout()
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Search in files...")
        self._search_input.returnPressed.connect(self._do_search)
        search_row.addWidget(self._search_input)

        self._btn_search = QPushButton("Search")
        self._btn_search.setMinimumWidth(70)
        self._btn_search.clicked.connect(self._do_search)
        search_row.addWidget(self._btn_search)
        layout.addLayout(search_row)

        # Replace row
        replace_row = QHBoxLayout()
        self._replace_input = QLineEdit()
        self._replace_input.setPlaceholderText("Replace with...")
        replace_row.addWidget(self._replace_input)

        self._btn_replace_all = QPushButton("Replace All")
        self._btn_replace_all.setMinimumWidth(100)
        self._btn_replace_all.clicked.connect(self._do_replace_all)
        self._btn_replace_all.setEnabled(False)
        replace_row.addWidget(self._btn_replace_all)
        layout.addLayout(replace_row)

        # Options row
        opts_row = QHBoxLayout()
        self._case_check = QCheckBox("Aa")
        self._case_check.setToolTip("Case sensitive")
        opts_row.addWidget(self._case_check)

        self._regex_check = QCheckBox(".*")
        self._regex_check.setToolTip("Use regex")
        opts_row.addWidget(self._regex_check)

        self._word_check = QCheckBox("W")
        self._word_check.setToolTip("Whole word")
        opts_row.addWidget(self._word_check)

        self._filter_combo = QComboBox()
        self._filter_combo.setEditable(True)
        self._filter_combo.addItems(["*.*", "*.m", "*.py", "*.json", "*.txt"])
        self._filter_combo.setToolTip("File filter pattern")
        self._filter_combo.setFixedWidth(80)
        opts_row.addWidget(self._filter_combo)

        opts_row.addStretch()

        self._status_label = QLabel("")
        from forge.gui.theme_utils import detect_palette
        _p = detect_palette()
        self._status_label.setStyleSheet(f"font-size: 11px; color: {_p.get('fg3', '#6c7086')};")
        opts_row.addWidget(self._status_label)
        layout.addLayout(opts_row)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setMaximumHeight(3)
        self._progress.setTextVisible(False)
        self._progress.hide()
        layout.addWidget(self._progress)

        # Results tree
        self._results = QTreeWidget()
        self._results.setHeaderLabels(["Results"])
        self._results.setRootIsDecorated(True)
        self._results.setFont(QFont("Consolas", 10))
        self._results.itemDoubleClicked.connect(self._on_result_clicked)
        layout.addWidget(self._results)

    def set_root(self, path):
        self.root_dir = path

    def _do_search(self):
        pattern = self._search_input.text().strip()
        if not pattern:
            return

        # Cancel any running search
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait()

        self._results.clear()
        self._file_items = {}
        self._progress.setValue(0)
        self._progress.show()
        self._status_label.setText("Searching...")
        self._btn_replace_all.setEnabled(False)

        self._worker = SearchWorker(
            self.root_dir, pattern,
            file_filter=self._filter_combo.currentText(),
            case_sensitive=self._case_check.isChecked(),
            use_regex=self._regex_check.isChecked(),
            whole_word=self._word_check.isChecked(),
        )
        self._worker.result_found.connect(self._on_result)
        self._worker.finished.connect(self._on_search_done)
        self._worker.progress.connect(self._on_progress)
        self._worker.start()

    def _on_result(self, filepath, line_num, line_text):
        # Group by file
        if filepath not in self._file_items:
            rel = os.path.relpath(filepath, self.root_dir)
            file_item = QTreeWidgetItem([rel])
            file_item.setForeground(0, QColor("#89b4fa"))
            file_item.setData(0, Qt.UserRole, filepath)
            self._results.addTopLevelItem(file_item)
            self._file_items[filepath] = file_item

        parent = self._file_items[filepath]
        # Truncate long lines
        display = line_text[:120] + "..." if len(line_text) > 120 else line_text
        item = QTreeWidgetItem([f"  {line_num}: {display}"])
        item.setData(0, Qt.UserRole, filepath)
        item.setData(0, Qt.UserRole + 1, line_num)
        parent.addChild(item)

        # Auto-expand as results come in
        parent.setExpanded(True)

    def _on_progress(self, current, total):
        self._progress.setMaximum(total)
        self._progress.setValue(current)

    def _on_search_done(self, total_matches):
        file_count = self._results.topLevelItemCount()
        self._status_label.setText(f"{total_matches} matches in {file_count} files")
        self._progress.hide()
        if total_matches > 0:
            self._btn_replace_all.setEnabled(True)

    def _on_result_clicked(self, item, column):
        filepath = item.data(0, Qt.UserRole)
        line_num = item.data(0, Qt.UserRole + 1)
        if filepath and line_num:
            self.file_open_requested.emit(filepath, line_num)

    def _do_replace_all(self):
        """Replace all occurrences in found files."""
        pattern = self._search_input.text().strip()
        replacement = self._replace_input.text()
        if not pattern:
            return

        flags = 0 if self._case_check.isChecked() else re.IGNORECASE
        if not self._regex_check.isChecked():
            pattern = re.escape(pattern)
        if self._word_check.isChecked():
            pattern = r'\b' + pattern + r'\b'

        replaced_count = 0
        for filepath in self._file_items:
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                new_content, n = re.subn(pattern, replacement, content, flags=flags)
                if n > 0:
                    with open(filepath, 'w') as f:
                        f.write(new_content)
                    replaced_count += n
            except (OSError, PermissionError):
                pass

        self._status_label.setText(f"Replaced {replaced_count} occurrences")
        self._btn_replace_all.setEnabled(False)
