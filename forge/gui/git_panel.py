"""Git integration panel – shows status, staged files, diff preview."""

import subprocess
from PySide6.QtCore import Qt, Signal, QProcess
from PySide6.QtGui import QColor, QFont, QIcon
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTreeWidget, QTreeWidgetItem, QTextEdit, QLineEdit,
    QSplitter, QHeaderView, QMessageBox, QFrame,
)


_STATUS_MAP = {
    "M": ("Modified", QColor("#e8a317")),   # orange
    "A": ("Added",    QColor("#2ecc71")),    # green
    "D": ("Deleted",  QColor("#e74c3c")),    # red
    "?": ("Untracked", QColor("#95a5a6")),   # gray
    "R": ("Renamed",  QColor("#3498db")),    # blue
    "C": ("Copied",   QColor("#3498db")),    # blue
    "U": ("Unmerged", QColor("#9b59b6")),    # purple
}


def _run_git(args: list[str], cwd: str | None = None) -> tuple[bool, str]:
    """Run a git command and return (success, stdout/stderr)."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, (result.stderr or result.stdout).strip()
    except FileNotFoundError:
        return False, "git is not installed or not in PATH."
    except subprocess.TimeoutExpired:
        return False, "git command timed out."
    except Exception as exc:
        return False, str(exc)


class GitPanel(QWidget):
    """Dock-able Git integration panel."""

    # Emitted after a successful commit so the editor can react.
    committed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cwd: str | None = None
        self._files: list[dict] = []  # {status, path, staged}
        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # -- Info bar --
        info_frame = QFrame()
        info_layout = QHBoxLayout(info_frame)
        info_layout.setContentsMargins(4, 2, 4, 2)

        self._branch_label = QLabel("Branch: –")
        self._branch_label.setStyleSheet("font-weight: bold;")
        info_layout.addWidget(self._branch_label)

        self._status_label = QLabel("")
        info_layout.addWidget(self._status_label)

        info_layout.addStretch()

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setFixedWidth(70)
        self._refresh_btn.clicked.connect(self.refresh)
        info_layout.addWidget(self._refresh_btn)

        root.addWidget(info_frame)

        # -- Splitter: file tree + diff preview --
        splitter = QSplitter(Qt.Vertical)

        # File tree
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["", "Status", "File"])
        self._tree.setRootIsDecorated(False)
        self._tree.setSelectionMode(QTreeWidget.SingleSelection)
        header = self._tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        self._tree.itemClicked.connect(self._on_item_clicked)
        self._tree.itemChanged.connect(self._on_item_changed)
        splitter.addWidget(self._tree)

        # Diff preview
        self._diff_view = QTextEdit()
        self._diff_view.setReadOnly(True)
        self._diff_view.setFont(QFont("Courier New", 9))
        self._diff_view.setPlaceholderText("Select a file to view its diff…")
        splitter.addWidget(self._diff_view)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)

        # -- Commit bar --
        commit_frame = QFrame()
        commit_layout = QHBoxLayout(commit_frame)
        commit_layout.setContentsMargins(0, 0, 0, 0)

        self._msg_input = QLineEdit()
        self._msg_input.setPlaceholderText("Commit message…")
        self._msg_input.returnPressed.connect(self._do_commit)
        commit_layout.addWidget(self._msg_input, 1)

        self._commit_btn = QPushButton("Commit")
        self._commit_btn.setFixedWidth(70)
        self._commit_btn.clicked.connect(self._do_commit)
        commit_layout.addWidget(self._commit_btn)

        root.addWidget(commit_frame)

    # --------------------------------------------------------- Public API
    def set_cwd(self, path: str | None):
        """Set the working directory to query git in."""
        self._cwd = path
        self.refresh()

    def refresh(self):
        """Re-read git state and update the UI."""
        self._tree.blockSignals(True)
        self._tree.clear()
        self._files.clear()
        self._diff_view.clear()

        if not self._cwd:
            self._branch_label.setText("Branch: –")
            self._status_label.setText("(no directory)")
            self._tree.blockSignals(False)
            return

        # Branch name
        ok, branch = _run_git(["branch", "--show-current"], self._cwd)
        if not ok:
            self._branch_label.setText("Branch: –")
            self._status_label.setText("Not a git repo")
            self._tree.blockSignals(False)
            return
        self._branch_label.setText(f"Branch: {branch or 'HEAD detached'}")

        # Ahead / behind
        ahead_behind = ""
        ok_ab, ab_text = _run_git(
            ["rev-list", "--left-right", "--count", f"{branch}...@{{u}}"],
            self._cwd,
        )
        if ok_ab:
            parts = ab_text.split()
            if len(parts) == 2:
                ahead, behind = int(parts[0]), int(parts[1])
                segments = []
                if ahead:
                    segments.append(f"↑{ahead}")
                if behind:
                    segments.append(f"↓{behind}")
                if segments:
                    ahead_behind = "  " + " ".join(segments)

        # Last commit
        ok_log, log_line = _run_git(["log", "--oneline", "-1"], self._cwd)
        last_commit = log_line if ok_log else ""

        # Porcelain status
        ok_st, st_text = _run_git(["status", "--porcelain"], self._cwd)
        if not ok_st:
            self._status_label.setText(st_text)
            self._tree.blockSignals(False)
            return

        lines = [l for l in st_text.splitlines() if l.strip()] if st_text else []
        self._status_label.setText(
            f"{len(lines)} changed{ahead_behind}"
        )

        for line in lines:
            if len(line) < 4:
                continue
            index_status = line[0]
            work_status = line[1]
            filepath = line[3:]

            # Determine display status
            if index_status == "?" and work_status == "?":
                code = "?"
                staged = False
            elif index_status != " " and index_status != "?":
                code = index_status
                staged = True
            else:
                code = work_status if work_status != " " else index_status
                staged = False

            entry = {"status": code, "path": filepath, "staged": staged}
            self._files.append(entry)

            label, color = _STATUS_MAP.get(code, (code, QColor("#cccccc")))

            item = QTreeWidgetItem()
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Checked if staged else Qt.Unchecked)
            item.setText(1, f"[{code}] {label}")
            item.setForeground(1, color)
            item.setText(2, filepath)
            item.setForeground(2, color)
            item.setData(0, Qt.UserRole, len(self._files) - 1)  # index
            self._tree.addTopLevelItem(item)

        self._tree.blockSignals(False)

    # ----------------------------------------------------------- Slots
    def _on_item_clicked(self, item: QTreeWidgetItem, column: int):
        idx = item.data(0, Qt.UserRole)
        if idx is None or idx >= len(self._files):
            return
        entry = self._files[idx]
        self._show_diff(entry)

    def _on_item_changed(self, item: QTreeWidgetItem, column: int):
        if column != 0:
            return
        idx = item.data(0, Qt.UserRole)
        if idx is None or idx >= len(self._files):
            return
        self._files[idx]["staged"] = item.checkState(0) == Qt.Checked

    def _show_diff(self, entry: dict):
        path = entry["path"]
        if entry["status"] == "?":
            # Untracked – show file contents (limited)
            ok, content = _run_git([], self._cwd)  # won't use this
            try:
                import os
                full = os.path.join(self._cwd, path)
                with open(full, "r", errors="replace") as f:
                    text = f.read(50_000)
                self._diff_view.setPlainText(f"(untracked file)\n\n{text}")
            except Exception as exc:
                self._diff_view.setPlainText(f"Cannot read file: {exc}")
            return

        ok, diff_text = _run_git(["diff", "--", path], self._cwd)
        if not ok or not diff_text:
            # Try staged diff
            ok, diff_text = _run_git(["diff", "--cached", "--", path], self._cwd)
        if ok and diff_text:
            self._diff_view.setPlainText(diff_text)
        else:
            self._diff_view.setPlainText("(no diff available)")

    def _do_commit(self):
        msg = self._msg_input.text().strip()
        if not msg:
            QMessageBox.warning(self, "Commit", "Please enter a commit message.")
            return
        if not self._cwd:
            return

        # Collect staged files
        staged = [e["path"] for e in self._files if e["staged"]]
        if not staged:
            QMessageBox.warning(self, "Commit", "No files staged for commit.")
            return

        # Reset index first, then add only the checked files
        _run_git(["reset", "HEAD"], self._cwd)

        for fp in staged:
            ok, err = _run_git(["add", "--", fp], self._cwd)
            if not ok:
                QMessageBox.critical(
                    self, "Git Error", f"Failed to stage {fp}:\n{err}"
                )
                return

        ok, err = _run_git(["commit", "-m", msg], self._cwd)
        if not ok:
            QMessageBox.critical(self, "Git Error", f"Commit failed:\n{err}")
            return

        self._msg_input.clear()
        self.refresh()
        self.committed.emit()
