# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Git integration panel – shows status, staged files, diff preview,
visual commit history, repo init, remote management, pull/push."""

import subprocess
import os
from PySide6.QtCore import Qt, Signal, QProcess
from PySide6.QtGui import QColor, QFont, QIcon
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTreeWidget, QTreeWidgetItem, QTextEdit, QLineEdit,
    QSplitter, QHeaderView, QMessageBox, QFrame, QPlainTextEdit,
    QDialog, QFormLayout, QDialogButtonBox, QListWidget,
    QTabWidget, QStackedWidget,
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


def _is_git_repo(cwd: str | None) -> bool:
    """Check if cwd is inside a git repository."""
    if not cwd:
        return False
    ok, _ = _run_git(["rev-parse", "--is-inside-work-tree"], cwd)
    return ok


class RemoteDialog(QDialog):
    """Dialog for viewing/adding/removing git remotes."""

    def __init__(self, cwd: str, parent=None):
        super().__init__(parent)
        self._cwd = cwd
        self.setWindowTitle("Git Remotes")
        self.setMinimumWidth(480)
        self.setMinimumHeight(300)
        self._build_ui()
        self._refresh_remotes()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Remote list
        layout.addWidget(QLabel("Current remotes:"))
        self._remote_list = QListWidget()
        self._remote_list.setFont(QFont("Courier New", 9))
        layout.addWidget(self._remote_list, 1)

        # Remove button
        rm_layout = QHBoxLayout()
        rm_layout.addStretch()
        self._remove_btn = QPushButton("Remove Selected")
        self._remove_btn.clicked.connect(self._remove_remote)
        rm_layout.addWidget(self._remove_btn)
        layout.addLayout(rm_layout)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        layout.addWidget(sep)

        # Add remote form
        layout.addWidget(QLabel("Add remote:"))
        form = QFormLayout()
        self._name_input = QLineEdit()
        self._name_input.setPlaceholderText("e.g. origin")
        form.addRow("Name:", self._name_input)
        self._url_input = QLineEdit()
        self._url_input.setPlaceholderText("e.g. git@github.com:user/repo.git")
        form.addRow("URL:", self._url_input)
        layout.addLayout(form)

        self._add_btn = QPushButton("Add Remote")
        self._add_btn.clicked.connect(self._add_remote)
        layout.addWidget(self._add_btn)

        # Close button
        btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _refresh_remotes(self):
        self._remote_list.clear()
        ok, text = _run_git(["remote", "-v"], self._cwd)
        if ok and text:
            for line in text.splitlines():
                self._remote_list.addItem(line)
        else:
            self._remote_list.addItem("(no remotes configured)")

    def _add_remote(self):
        name = self._name_input.text().strip()
        url = self._url_input.text().strip()
        if not name or not url:
            QMessageBox.warning(self, "Add Remote", "Both name and URL are required.")
            return
        ok, err = _run_git(["remote", "add", name, url], self._cwd)
        if ok:
            self._name_input.clear()
            self._url_input.clear()
            self._refresh_remotes()
        else:
            QMessageBox.critical(self, "Git Error", f"Failed to add remote:\n{err}")

    def _remove_remote(self):
        item = self._remote_list.currentItem()
        if not item:
            QMessageBox.warning(self, "Remove Remote", "Select a remote first.")
            return
        text = item.text()
        # Parse remote name from "origin\thttps://... (fetch)" format
        name = text.split()[0] if text else ""
        if not name or name.startswith("("):
            return
        reply = QMessageBox.question(
            self, "Remove Remote",
            f"Remove remote '{name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            ok, err = _run_git(["remote", "remove", name], self._cwd)
            if ok:
                self._refresh_remotes()
            else:
                QMessageBox.critical(
                    self, "Git Error", f"Failed to remove remote:\n{err}"
                )


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

        # -- Stacked widget: no-repo view vs repo view --
        self._stack = QStackedWidget()
        root.addWidget(self._stack, 1)

        # Page 0: No git repo
        self._no_repo_page = QWidget()
        self._build_no_repo_page()
        self._stack.addWidget(self._no_repo_page)

        # Page 1: Active repo
        self._repo_page = QWidget()
        self._build_repo_page()
        self._stack.addWidget(self._repo_page)

    def _build_no_repo_page(self):
        layout = QVBoxLayout(self._no_repo_page)
        layout.setContentsMargins(20, 40, 20, 40)

        lbl = QLabel("No Git Repository")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #95a5a6;")
        layout.addWidget(lbl)

        desc = QLabel("Initialize a new repository in the current directory.")
        desc.setAlignment(Qt.AlignCenter)
        desc.setStyleSheet("color: #7f8c8d;")
        layout.addWidget(desc)

        layout.addSpacing(16)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self._init_btn_large = QPushButton("  Init Repository  ")
        self._init_btn_large.setMinimumHeight(36)
        self._init_btn_large.setStyleSheet(
            "QPushButton { font-size: 14px; font-weight: bold;"
            " padding: 8px 24px; }"
        )
        self._init_btn_large.clicked.connect(self._do_init)
        btn_layout.addWidget(self._init_btn_large)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        layout.addStretch()

    def _build_repo_page(self):
        repo_layout = QVBoxLayout(self._repo_page)
        repo_layout.setContentsMargins(0, 0, 0, 0)
        repo_layout.setSpacing(4)

        # -- Info bar --
        info_frame = QFrame()
        info_layout = QHBoxLayout(info_frame)
        info_layout.setContentsMargins(4, 2, 4, 2)

        self._branch_label = QLabel("Branch: -")
        self._branch_label.setStyleSheet("font-weight: bold;")
        info_layout.addWidget(self._branch_label)

        self._status_label = QLabel("")
        info_layout.addWidget(self._status_label)

        info_layout.addStretch()

        # Action buttons
        self._init_btn = QPushButton("Init")
        self._init_btn.setToolTip("Initialize a git repository here")
        self._init_btn.setMinimumWidth(50)
        self._init_btn.clicked.connect(self._do_init)
        info_layout.addWidget(self._init_btn)

        self._remote_btn = QPushButton("Remote")
        self._remote_btn.setToolTip("Manage remotes")
        self._remote_btn.setMinimumWidth(60)
        self._remote_btn.clicked.connect(self._do_remote)
        info_layout.addWidget(self._remote_btn)

        self._pull_btn = QPushButton("Pull")
        self._pull_btn.setToolTip("Pull from remote")
        self._pull_btn.setMinimumWidth(50)
        self._pull_btn.clicked.connect(self._do_pull)
        info_layout.addWidget(self._pull_btn)

        self._push_btn = QPushButton("Push")
        self._push_btn.setToolTip("Push to remote")
        self._push_btn.setMinimumWidth(50)
        self._push_btn.clicked.connect(self._do_push)
        info_layout.addWidget(self._push_btn)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setMinimumWidth(70)
        self._refresh_btn.clicked.connect(self.refresh)
        info_layout.addWidget(self._refresh_btn)

        repo_layout.addWidget(info_frame)

        # -- Tabs: Changes | History --
        self._tabs = QTabWidget()
        repo_layout.addWidget(self._tabs, 1)

        # Tab 0: Changes (file tree + diff)
        changes_widget = QWidget()
        changes_layout = QVBoxLayout(changes_widget)
        changes_layout.setContentsMargins(0, 0, 0, 0)

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
        self._diff_view.setPlaceholderText("Select a file to view its diff...")
        splitter.addWidget(self._diff_view)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        changes_layout.addWidget(splitter, 1)

        self._tabs.addTab(changes_widget, "Changes")

        # Tab 1: History (visual commit log)
        history_widget = QWidget()
        history_layout = QVBoxLayout(history_widget)
        history_layout.setContentsMargins(0, 0, 0, 0)

        self._history_view = QPlainTextEdit()
        self._history_view.setReadOnly(True)
        self._history_view.setFont(QFont("Courier New", 9))
        self._history_view.setPlaceholderText("Commit history will appear here...")
        self._history_view.setLineWrapMode(QPlainTextEdit.NoWrap)
        history_layout.addWidget(self._history_view, 1)

        self._tabs.addTab(history_widget, "History")

        # -- Commit bar --
        commit_frame = QFrame()
        commit_layout = QHBoxLayout(commit_frame)
        commit_layout.setContentsMargins(0, 0, 0, 0)

        self._msg_input = QLineEdit()
        self._msg_input.setPlaceholderText("Commit message...")
        self._msg_input.returnPressed.connect(self._do_commit)
        commit_layout.addWidget(self._msg_input, 1)

        self._commit_btn = QPushButton("Commit")
        self._commit_btn.setMinimumWidth(90)
        self._commit_btn.clicked.connect(self._do_commit)
        commit_layout.addWidget(self._commit_btn)

        repo_layout.addWidget(commit_frame)

    # --------------------------------------------------------- Theming
    def apply_theme(self):
        """Re-apply theme colors from the current palette."""
        from forge.gui.theme_utils import detect_palette
        p = detect_palette()
        fg0 = p.get('fg0', '#cdd6f4')
        fg2 = p.get('fg2', '#a6adc8')
        fg3 = p.get('fg3', '#6c7086')
        bg1 = p.get('bg1', '#252536')
        accent = p.get('accent', '#00BCD4')

        # No-repo page labels
        for child in self._no_repo_page.findChildren(type(self._no_repo_page)):
            pass
        # Update hardcoded label colors
        if hasattr(self, '_branch_label'):
            self._branch_label.setStyleSheet(f"font-weight: bold; color: {accent};")
        if hasattr(self, '_status_label'):
            self._status_label.setStyleSheet(f"color: {fg2};")
        if hasattr(self, '_diff_view'):
            self._diff_view.setStyleSheet(
                f"background: {bg1}; color: {fg0}; border: none;"
            )
        if hasattr(self, '_history_view'):
            self._history_view.setStyleSheet(
                f"background: {bg1}; color: {fg0}; border: none;"
            )

    # --------------------------------------------------------- Public API
    def set_cwd(self, path: str | None):
        """Set the working directory to query git in."""
        self._cwd = path
        self.refresh()

    def refresh(self):
        """Re-read git state and update the UI."""
        if not self._cwd:
            self._stack.setCurrentIndex(0)
            return

        if not _is_git_repo(self._cwd):
            self._stack.setCurrentIndex(0)
            return

        self._stack.setCurrentIndex(1)
        self._refresh_changes()
        self._refresh_history()

    def _refresh_changes(self):
        """Refresh the Changes tab."""
        self._tree.blockSignals(True)
        self._tree.clear()
        self._files.clear()
        self._diff_view.clear()

        # Branch name
        ok, branch = _run_git(["branch", "--show-current"], self._cwd)
        if not ok:
            self._branch_label.setText("Branch: -")
            self._status_label.setText("Not a git repo")
            self._tree.blockSignals(False)
            return
        self._branch_label.setText(
            "Branch: " + (branch or "HEAD detached")
        )

        # Ahead / behind
        ahead_behind = ""
        ok_ab, ab_text = _run_git(
            ["rev-list", "--left-right", "--count",
             branch + "...@{u}"],
            self._cwd,
        )
        if ok_ab:
            parts = ab_text.split()
            if len(parts) == 2:
                ahead, behind = int(parts[0]), int(parts[1])
                segments = []
                if ahead:
                    segments.append("\u2191" + str(ahead))
                if behind:
                    segments.append("\u2193" + str(behind))
                if segments:
                    ahead_behind = "  " + " ".join(segments)

        # Last commit
        ok_log, log_line = _run_git(["log", "--oneline", "-1"], self._cwd)

        # Porcelain status
        ok_st, st_text = _run_git(["status", "--porcelain"], self._cwd)
        if not ok_st:
            self._status_label.setText(st_text)
            self._tree.blockSignals(False)
            return

        lines = [l for l in st_text.splitlines() if l.strip()] if st_text else []
        self._status_label.setText(
            str(len(lines)) + " changed" + ahead_behind
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
            item.setText(1, "[" + code + "] " + label)
            item.setForeground(1, color)
            item.setText(2, filepath)
            item.setForeground(2, color)
            item.setData(0, Qt.UserRole, len(self._files) - 1)  # index
            self._tree.addTopLevelItem(item)

        self._tree.blockSignals(False)

    def _refresh_history(self):
        """Refresh the History tab with color-coded visual branch graph."""
        self._history_view.clear()

        ok, log_text = _run_git(
            ["log", "--oneline", "--graph", "--all", "--decorate", "-80"],
            self._cwd,
        )
        if not ok or not log_text:
            if not ok and "does not have any commits" in (log_text or ""):
                self._history_view.setPlainText("(no commits yet)")
            elif not log_text:
                self._history_view.setPlainText("(no commits yet)")
            else:
                self._history_view.setPlainText(log_text)
            return

        # Apply color-coded formatting
        import re
        from PySide6.QtGui import QTextCharFormat, QTextCursor

        cursor = self._history_view.textCursor()
        cursor.beginEditBlock()

        graph_color = QColor("#6c7086")     # gray for graph lines
        hash_color = QColor("#f9e2af")      # yellow for commit hash
        branch_color = QColor("#a6e3a1")    # green for branch names
        head_color = QColor("#f38ba8")      # red for HEAD
        tag_color = QColor("#fab387")       # orange for tags
        msg_color = QColor("#cdd6f4")       # light for message text

        for line in log_text.split("\n"):
            # Split into graph part and content part
            # Graph chars: * | / \ _ space
            graph_end = 0
            for i, ch in enumerate(line):
                if ch in ("*", "|", "/", "\\", "_", " "):
                    graph_end = i + 1
                else:
                    break

            graph_part = line[:graph_end]
            content_part = line[graph_end:]

            # Write graph part in gray
            fmt = QTextCharFormat()
            fmt.setForeground(graph_color)
            # Star (*) in accent color
            for ch in graph_part:
                if ch == "*":
                    star_fmt = QTextCharFormat()
                    star_fmt.setForeground(QColor("#cba6f7"))  # purple
                    star_fmt.setFontWeight(QFont.Bold)
                    cursor.insertText(ch, star_fmt)
                else:
                    cursor.insertText(ch, fmt)

            # Parse content: hash, decorations, message
            # Pattern: <hash> (decorations) message
            m = re.match(r"([0-9a-f]{7,12})\s*(\(.*?\))?\s*(.*)", content_part)
            if m:
                commit_hash, decorations, message = m.groups()

                # Hash in yellow
                hash_fmt = QTextCharFormat()
                hash_fmt.setForeground(hash_color)
                cursor.insertText(commit_hash + " ", hash_fmt)

                # Decorations
                if decorations:
                    # Parse individual decorations
                    dec_text = decorations[1:-1]  # strip parens
                    cursor.insertText("(", fmt)
                    parts = dec_text.split(", ")
                    for pi, part in enumerate(parts):
                        dec_fmt = QTextCharFormat()
                        if "HEAD" in part:
                            dec_fmt.setForeground(head_color)
                            dec_fmt.setFontWeight(QFont.Bold)
                        elif "tag:" in part:
                            dec_fmt.setForeground(tag_color)
                        else:
                            dec_fmt.setForeground(branch_color)
                        cursor.insertText(part, dec_fmt)
                        if pi < len(parts) - 1:
                            cursor.insertText(", ", fmt)
                    cursor.insertText(") ", fmt)

                # Message
                msg_fmt = QTextCharFormat()
                msg_fmt.setForeground(msg_color)
                cursor.insertText(message, msg_fmt)
            else:
                # No match — just write as-is
                msg_fmt = QTextCharFormat()
                msg_fmt.setForeground(msg_color)
                cursor.insertText(content_part, msg_fmt)

            cursor.insertText("\n", fmt)

        cursor.endEditBlock()

    # ------------------------------------------------- Action slots
    def _do_init(self):
        """Initialize a new git repository."""
        if not self._cwd:
            QMessageBox.warning(
                self, "Init", "No working directory set."
            )
            return

        if _is_git_repo(self._cwd):
            QMessageBox.information(
                self, "Init", "Already a git repository."
            )
            return

        ok, msg = _run_git(["init"], self._cwd)
        if ok:
            QMessageBox.information(
                self, "Init",
                "Initialized git repository in:\n" + self._cwd
            )
            self.refresh()
        else:
            QMessageBox.critical(
                self, "Git Error", "Failed to init:\n" + msg
            )

    def _do_remote(self):
        """Open the remote management dialog."""
        if not self._cwd or not _is_git_repo(self._cwd):
            QMessageBox.warning(
                self, "Remote", "Not a git repository."
            )
            return
        dlg = RemoteDialog(self._cwd, parent=self)
        dlg.exec()
        self.refresh()

    def _do_pull(self):
        """Pull from the tracking remote."""
        if not self._cwd:
            return

        # Check for a configured upstream
        ok_branch, branch = _run_git(["branch", "--show-current"], self._cwd)
        if not ok_branch or not branch:
            QMessageBox.warning(self, "Pull", "Cannot determine current branch.")
            return

        ok_remote, remote = _run_git(
            ["config", "branch." + branch + ".remote"], self._cwd
        )
        if not ok_remote or not remote:
            QMessageBox.warning(
                self, "Pull",
                "No tracking remote configured for this branch.\n"
                "Use Remote dialog to add a remote, then set upstream with:\n"
                "  git push -u <remote> <branch>"
            )
            return

        ok, msg = _run_git(["pull", "--ff-only"], self._cwd)
        if ok:
            self.refresh()
            QMessageBox.information(self, "Pull", msg or "Already up to date.")
        else:
            QMessageBox.critical(self, "Pull Failed", msg)

    def _do_push(self):
        """Push to the tracking remote."""
        if not self._cwd:
            return

        ok_branch, branch = _run_git(["branch", "--show-current"], self._cwd)
        if not ok_branch or not branch:
            QMessageBox.warning(self, "Push", "Cannot determine current branch.")
            return

        ok_remote, remote = _run_git(
            ["config", "branch." + branch + ".remote"], self._cwd
        )
        if not ok_remote or not remote:
            # Offer to set upstream
            reply = QMessageBox.question(
                self, "Push",
                "No upstream configured. Push with --set-upstream to origin?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                ok, msg = _run_git(
                    ["push", "--set-upstream", "origin", branch], self._cwd
                )
                if ok:
                    self.refresh()
                    QMessageBox.information(self, "Push", msg or "Pushed.")
                else:
                    QMessageBox.critical(self, "Push Failed", msg)
            return

        ok, msg = _run_git(["push"], self._cwd)
        if ok:
            self.refresh()
            QMessageBox.information(self, "Push", msg or "Pushed.")
        else:
            QMessageBox.critical(self, "Push Failed", msg)

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
            # Untracked - show file contents (limited)
            try:
                full = os.path.join(self._cwd, path)
                with open(full, "r", errors="replace") as f:
                    text = f.read(50_000)
                self._diff_view.setPlainText("(untracked file)\n\n" + text)
            except Exception as exc:
                self._diff_view.setPlainText("Cannot read file: " + str(exc))
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
                    self, "Git Error", "Failed to stage " + fp + ":\n" + err
                )
                return

        ok, err = _run_git(["commit", "-m", msg], self._cwd)
        if not ok:
            QMessageBox.critical(self, "Git Error", "Commit failed:\n" + err)
            return

        self._msg_input.clear()
        self.refresh()
        self.committed.emit()
