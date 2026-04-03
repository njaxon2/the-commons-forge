"""Forge Git Panel widget (forge/gui/git_panel.py).

Provides repository initialization, remote management, branch
visualization, commit history, and basic git operations (stage, commit,
push, pull) within the Forge IDE.
"""

from __future__ import annotations

import os
import re
import subprocess
from datetime import datetime, timezone

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QAction, QColor, QBrush, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QTreeWidget, QTreeWidgetItem, QHeaderView, QTextEdit,
    QMenu, QInputDialog, QMessageBox, QGroupBox, QFrame,
    QTabWidget, QApplication,
)


# ── Helpers ───────────────────────────────────────────────────────────

def _run_git(args: list[str], cwd: str | None = None,
             timeout: int = 15) -> tuple[bool, str]:
    """Run a git command and return (success, output)."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            err = result.stderr.strip()
            return False, err or output
        return True, output
    except FileNotFoundError:
        return False, "git is not installed or not on PATH"
    except subprocess.TimeoutExpired:
        return False, "git command timed out"
    except Exception as exc:
        return False, str(exc)


def _relative_time(iso_date: str) -> str:
    """Convert ISO 8601 date to a human-readable relative time string."""
    try:
        dt = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = now - dt
        secs = int(delta.total_seconds())
        if secs < 60:
            return "just now"
        elif secs < 3600:
            m = secs // 60
            return f"{m} min{'s' if m > 1 else ''} ago"
        elif secs < 86400:
            h = secs // 3600
            return f"{h} hour{'s' if h > 1 else ''} ago"
        elif secs < 604800:
            d = secs // 86400
            return f"{d} day{'s' if d > 1 else ''} ago"
        elif secs < 2592000:
            w = secs // 604800
            return f"{w} week{'s' if w > 1 else ''} ago"
        else:
            return dt.strftime("%Y-%m-%d")
    except Exception:
        return iso_date


# ── Colour constants for branch labels ────────────────────────────────

_BRANCH_COLORS = {
    "main": QColor("#4caf50"),
    "master": QColor("#4caf50"),
    "develop": QColor("#ff9800"),
    "HEAD": QColor("#e91e63"),
}

_TAG_COLOR = QColor("#9c27b0")
_REMOTE_COLOR = QColor("#2196f3")
_DEFAULT_BRANCH_COLOR = QColor("#00bcd4")


def _branch_color(name: str) -> QColor:
    """Return a colour for a branch/ref label."""
    if name.startswith("tag:"):
        return _TAG_COLOR
    for prefix in ("origin/", "upstream/"):
        if name.startswith(prefix):
            return _REMOTE_COLOR
    return _BRANCH_COLORS.get(name, _DEFAULT_BRANCH_COLOR)


# =====================================================================
# Git Panel Widget
# =====================================================================

class GitPanelWidget(QWidget):
    """Full-featured Git panel for the Forge IDE.

    Signals
    -------
    status_message : str
        Emitted to relay messages to the status bar.
    """

    status_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._repo_path: str | None = None
        self._git_available = False
        self._has_repo = False
        self._build_ui()
        self._check_git_available()

        # Auto-refresh timer (every 5 seconds when visible)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(5000)
        self._refresh_timer.timeout.connect(self._auto_refresh)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_repo_path(self, path: str):
        """Set the working directory for git operations."""
        self._repo_path = path
        self._check_repo_state()
        self.refresh()

    def refresh(self):
        """Refresh all git information."""
        if not self._repo_path or not self._git_available:
            return
        self._check_repo_state()
        if self._has_repo:
            self._show_repo_view()
            self._refresh_branch_info()
            self._refresh_commit_history()
            self._refresh_status()
            self._refresh_remotes()
            self._refresh_branches()
        else:
            self._show_init_view()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(4, 4, 4, 4)
        self._main_layout.setSpacing(4)

        # -- No-git warning (hidden by default) --
        self._no_git_label = QLabel(
            "Git is not installed or not found on PATH.\n"
            "Install Git to enable version control features."
        )
        self._no_git_label.setAlignment(Qt.AlignCenter)
        self._no_git_label.setStyleSheet("color: #f44336; padding: 20px;")
        self._no_git_label.setWordWrap(True)
        self._no_git_label.hide()
        self._main_layout.addWidget(self._no_git_label)

        # -- Init view (shown when no .git) --
        self._init_widget = QWidget()
        init_layout = QVBoxLayout(self._init_widget)
        init_layout.setAlignment(Qt.AlignCenter)

        init_label = QLabel("No Git repository found in the current project.")
        init_label.setAlignment(Qt.AlignCenter)
        init_label.setWordWrap(True)
        init_layout.addWidget(init_label)

        self._btn_init = QPushButton("Initialize Repository")
        self._btn_init.setFixedWidth(200)
        self._btn_init.clicked.connect(self._init_repo)
        init_layout.addWidget(self._btn_init, alignment=Qt.AlignCenter)

        self._init_widget.hide()
        self._main_layout.addWidget(self._init_widget)

        # -- Repo view (main working view) --
        self._repo_widget = QWidget()
        repo_layout = QVBoxLayout(self._repo_widget)
        repo_layout.setContentsMargins(0, 0, 0, 0)
        repo_layout.setSpacing(4)

        # Branch header
        branch_frame = QFrame()
        branch_layout = QHBoxLayout(branch_frame)
        branch_layout.setContentsMargins(4, 2, 4, 2)
        branch_icon_label = QLabel("Branch:")
        branch_icon_label.setStyleSheet("font-weight: bold;")
        branch_layout.addWidget(branch_icon_label)
        self._branch_label = QLabel("(none)")
        self._branch_label.setStyleSheet("font-weight: bold; color: #4caf50;")
        branch_layout.addWidget(self._branch_label)
        branch_layout.addStretch()

        self._btn_refresh = QPushButton("Refresh")
        self._btn_refresh.setFixedSize(70, 26)
        self._btn_refresh.setStyleSheet(self._small_btn_style())
        self._btn_refresh.clicked.connect(self.refresh)
        branch_layout.addWidget(self._btn_refresh)

        repo_layout.addWidget(branch_frame)

        # Tabs: History | Changes | Branches | Remotes
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)

        # --- History tab ---
        history_widget = QWidget()
        history_layout = QVBoxLayout(history_widget)
        history_layout.setContentsMargins(2, 2, 2, 2)

        self._commit_tree = QTreeWidget()
        self._commit_tree.setHeaderLabels(
            ["Graph", "Hash", "Message", "Author", "Date", "Refs"]
        )
        self._commit_tree.setRootIsDecorated(False)
        self._commit_tree.setAlternatingRowColors(True)
        self._commit_tree.setSelectionMode(QTreeWidget.SingleSelection)
        header = self._commit_tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self._commit_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self._commit_tree.customContextMenuRequested.connect(
            self._commit_context_menu
        )
        history_layout.addWidget(self._commit_tree)
        self._tabs.addTab(history_widget, "History")

        # --- Changes tab (staging area) ---
        changes_widget = QWidget()
        changes_layout = QVBoxLayout(changes_widget)
        changes_layout.setContentsMargins(2, 2, 2, 2)

        self._status_tree = QTreeWidget()
        self._status_tree.setHeaderLabels(["Status", "File"])
        self._status_tree.setRootIsDecorated(False)
        self._status_tree.setAlternatingRowColors(True)
        self._status_tree.header().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self._status_tree.header().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        changes_layout.addWidget(self._status_tree)

        # Commit controls
        commit_group = QGroupBox("Commit")
        commit_layout = QVBoxLayout(commit_group)

        self._commit_msg = QTextEdit()
        self._commit_msg.setPlaceholderText("Commit message...")
        self._commit_msg.setMaximumHeight(80)
        commit_layout.addWidget(self._commit_msg)

        btn_row = QHBoxLayout()
        self._btn_stage_all = QPushButton("Stage All")
        self._btn_stage_all.setStyleSheet(self._small_btn_style())
        self._btn_stage_all.clicked.connect(self._stage_all)
        btn_row.addWidget(self._btn_stage_all)

        self._btn_commit = QPushButton("Commit")
        self._btn_commit.setStyleSheet(self._small_btn_style())
        self._btn_commit.clicked.connect(self._commit)
        btn_row.addWidget(self._btn_commit)

        self._btn_push = QPushButton("Push")
        self._btn_push.setStyleSheet(self._small_btn_style())
        self._btn_push.clicked.connect(self._push)
        btn_row.addWidget(self._btn_push)

        self._btn_pull = QPushButton("Pull")
        self._btn_pull.setStyleSheet(self._small_btn_style())
        self._btn_pull.clicked.connect(self._pull)
        btn_row.addWidget(self._btn_pull)

        commit_layout.addLayout(btn_row)
        changes_layout.addWidget(commit_group)
        self._tabs.addTab(changes_widget, "Changes")

        # --- Branches tab ---
        branches_widget = QWidget()
        branches_layout = QVBoxLayout(branches_widget)
        branches_layout.setContentsMargins(2, 2, 2, 2)

        self._branch_list = QTreeWidget()
        self._branch_list.setHeaderLabels(
            ["Branch", "Tracking", "Last Commit"]
        )
        self._branch_list.setRootIsDecorated(False)
        self._branch_list.setAlternatingRowColors(True)
        self._branch_list.header().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self._branch_list.header().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self._branch_list.header().setSectionResizeMode(
            2, QHeaderView.Stretch
        )
        self._branch_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._branch_list.customContextMenuRequested.connect(
            self._branch_context_menu
        )
        self._branch_list.itemDoubleClicked.connect(self._checkout_branch)
        branches_layout.addWidget(self._branch_list)

        branch_btn_row = QHBoxLayout()
        self._btn_new_branch = QPushButton("New Branch")
        self._btn_new_branch.setStyleSheet(self._small_btn_style())
        self._btn_new_branch.clicked.connect(self._create_branch)
        branch_btn_row.addWidget(self._btn_new_branch)
        branch_btn_row.addStretch()
        branches_layout.addLayout(branch_btn_row)

        self._tabs.addTab(branches_widget, "Branches")

        # --- Remotes tab ---
        remotes_widget = QWidget()
        remotes_layout = QVBoxLayout(remotes_widget)
        remotes_layout.setContentsMargins(2, 2, 2, 2)

        self._remote_list = QTreeWidget()
        self._remote_list.setHeaderLabels(["Name", "URL"])
        self._remote_list.setRootIsDecorated(False)
        self._remote_list.setAlternatingRowColors(True)
        self._remote_list.header().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self._remote_list.header().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self._remote_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._remote_list.customContextMenuRequested.connect(
            self._remote_context_menu
        )
        remotes_layout.addWidget(self._remote_list)

        # Add remote controls
        add_remote_layout = QHBoxLayout()
        self._remote_name_edit = QLineEdit()
        self._remote_name_edit.setPlaceholderText(
            "Remote name (e.g. origin)"
        )
        self._remote_name_edit.setFixedWidth(140)
        add_remote_layout.addWidget(self._remote_name_edit)

        self._remote_url_edit = QLineEdit()
        self._remote_url_edit.setPlaceholderText("Remote URL")
        add_remote_layout.addWidget(self._remote_url_edit)

        self._btn_add_remote = QPushButton("Add Remote")
        self._btn_add_remote.setStyleSheet(self._small_btn_style())
        self._btn_add_remote.clicked.connect(self._add_remote)
        add_remote_layout.addWidget(self._btn_add_remote)

        remotes_layout.addLayout(add_remote_layout)
        self._tabs.addTab(remotes_widget, "Remotes")

        repo_layout.addWidget(self._tabs)
        self._repo_widget.hide()
        self._main_layout.addWidget(self._repo_widget)

    # ------------------------------------------------------------------
    # Styling helper
    # ------------------------------------------------------------------

    @staticmethod
    def _small_btn_style() -> str:
        return """
            QPushButton {
                background-color: transparent;
                border: 1px solid #ccc;
                border-radius: 3px;
                padding: 3px 10px;
                font-weight: normal;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #e3f2fd;
            }
            QPushButton:pressed {
                background-color: #bbdefb;
            }
        """

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _check_git_available(self):
        ok, _ = _run_git(["--version"])
        self._git_available = ok
        if not ok:
            self._no_git_label.show()
            self._init_widget.hide()
            self._repo_widget.hide()

    def _check_repo_state(self):
        if not self._repo_path:
            self._has_repo = False
            return
        git_dir = os.path.join(self._repo_path, ".git")
        self._has_repo = os.path.isdir(git_dir)

    def _show_init_view(self):
        self._no_git_label.hide()
        self._init_widget.show()
        self._repo_widget.hide()

    def _show_repo_view(self):
        self._no_git_label.hide()
        self._init_widget.hide()
        self._repo_widget.show()

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_timer.start()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._refresh_timer.stop()

    def _auto_refresh(self):
        """Lightweight status refresh on timer."""
        if self._has_repo and self._repo_path:
            self._refresh_status()
            self._refresh_branch_info()

    # ------------------------------------------------------------------
    # Init repository
    # ------------------------------------------------------------------

    def _init_repo(self):
        if not self._repo_path:
            self._msg("No project directory set.")
            return
        ok, out = _run_git(["init"], cwd=self._repo_path)
        if ok:
            self._msg(f"Initialized repository in {self._repo_path}")
            self.refresh()
        else:
            self._msg(f"git init failed: {out}")

    # ------------------------------------------------------------------
    # Branch info
    # ------------------------------------------------------------------

    def _refresh_branch_info(self):
        ok, branch = _run_git(
            ["rev-parse", "--abbrev-ref", "HEAD"], cwd=self._repo_path
        )
        if ok:
            self._branch_label.setText(branch)
        else:
            self._branch_label.setText("(detached/empty)")

    # ------------------------------------------------------------------
    # Commit history
    # ------------------------------------------------------------------

    def _refresh_commit_history(self):
        self._commit_tree.clear()

        # Get graph + structured log
        fmt = "%H%x00%h%x00%s%x00%an%x00%aI%x00%D"
        ok, out = _run_git(
            ["log", "--all", "--graph", "--format=" + fmt, "-100"],
            cwd=self._repo_path,
        )
        if not ok or not out:
            return

        for line in out.split("\n"):
            # The graph characters come before the format fields
            parts = line.split("\x00")
            if len(parts) >= 6:
                graph_and_hash = parts[0]
                full_hash = parts[0].split()[-1] if parts[0].strip() else ""
                short_hash = parts[1]
                message = parts[2]
                author = parts[3]
                date_str = parts[4]
                refs = parts[5]

                # Extract graph characters (everything before the full hash)
                graph = graph_and_hash
                if full_hash and full_hash in graph:
                    graph = graph[:graph.index(full_hash)].rstrip()

                item = QTreeWidgetItem([
                    graph,
                    short_hash,
                    message,
                    author,
                    _relative_time(date_str),
                    refs,
                ])

                # Style the hash column in monospace
                mono_font = QFont("Consolas", 11)
                mono_font.setStyleHint(QFont.Monospace)
                item.setFont(0, mono_font)
                item.setFont(1, mono_font)

                # Colour ref labels
                if refs:
                    item.setForeground(5, QBrush(_branch_color(
                        refs.split(",")[0].strip().replace(
                            "HEAD -> ", ""
                        )
                    )))

                item.setData(0, Qt.UserRole, full_hash)
                self._commit_tree.addTopLevelItem(item)
            else:
                # Graph-only line (no commit data)
                item = QTreeWidgetItem([line.rstrip(), "", "", "", "", ""])
                mono_font = QFont("Consolas", 11)
                mono_font.setStyleHint(QFont.Monospace)
                item.setFont(0, mono_font)
                item.setForeground(0, QBrush(QColor("#9e9e9e")))
                self._commit_tree.addTopLevelItem(item)

    # ------------------------------------------------------------------
    # Working tree status (Changes tab)
    # ------------------------------------------------------------------

    def _refresh_status(self):
        self._status_tree.clear()
        ok, out = _run_git(
            ["status", "--porcelain=v1"], cwd=self._repo_path
        )
        if not ok or not out:
            return

        _status_labels = {
            "M": ("Modified", QColor("#ff9800")),
            "A": ("Added", QColor("#4caf50")),
            "D": ("Deleted", QColor("#f44336")),
            "R": ("Renamed", QColor("#2196f3")),
            "C": ("Copied", QColor("#2196f3")),
            "U": ("Unmerged", QColor("#e91e63")),
            "?": ("Untracked", QColor("#9e9e9e")),
            "!": ("Ignored", QColor("#757575")),
        }

        for line in out.split("\n"):
            if len(line) < 4:
                continue
            idx = line[0]   # index status
            wt = line[1]    # working tree status
            filepath = line[3:]

            # Prefer working tree status, fall back to index
            code = wt if wt.strip() else idx
            label, color = _status_labels.get(
                code, (code, QColor("#9e9e9e"))
            )

            item = QTreeWidgetItem([label, filepath])
            item.setForeground(0, QBrush(color))
            item.setData(0, Qt.UserRole, filepath)
            self._status_tree.addTopLevelItem(item)

    # ------------------------------------------------------------------
    # Remotes
    # ------------------------------------------------------------------

    def _refresh_remotes(self):
        self._remote_list.clear()
        ok, out = _run_git(["remote", "-v"], cwd=self._repo_path)
        if not ok or not out:
            return

        seen = set()
        for line in out.split("\n"):
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0]
                url = parts[1]
                key = f"{name}\t{url}"
                if key not in seen:
                    seen.add(key)
                    item = QTreeWidgetItem([name, url])
                    self._remote_list.addTopLevelItem(item)

    def _add_remote(self):
        name = self._remote_name_edit.text().strip()
        url = self._remote_url_edit.text().strip()
        if not name or not url:
            self._msg("Enter both a remote name and URL.")
            return
        ok, out = _run_git(
            ["remote", "add", name, url], cwd=self._repo_path
        )
        if ok:
            self._msg(f"Added remote '{name}'")
            self._remote_name_edit.clear()
            self._remote_url_edit.clear()
            self._refresh_remotes()
        else:
            self._msg(f"Failed to add remote: {out}")

    def _remote_context_menu(self, pos):
        item = self._remote_list.itemAt(pos)
        if not item:
            return
        menu = QMenu(self)
        name = item.text(0)

        act_remove = QAction(f"Remove '{name}'", self)
        act_remove.triggered.connect(lambda: self._remove_remote(name))
        menu.addAction(act_remove)

        act_fetch = QAction(f"Fetch '{name}'", self)
        act_fetch.triggered.connect(lambda: self._fetch_remote(name))
        menu.addAction(act_fetch)

        menu.exec(self._remote_list.viewport().mapToGlobal(pos))

    def _remove_remote(self, name):
        ans = QMessageBox.question(
            self, "Remove Remote",
            f"Remove remote '{name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if ans == QMessageBox.Yes:
            ok, out = _run_git(
                ["remote", "remove", name], cwd=self._repo_path
            )
            if ok:
                self._msg(f"Removed remote '{name}'")
                self._refresh_remotes()
            else:
                self._msg(f"Failed: {out}")

    def _fetch_remote(self, name):
        ok, out = _run_git(
            ["fetch", name], cwd=self._repo_path, timeout=30
        )
        if ok:
            self._msg(f"Fetched '{name}'")
            self.refresh()
        else:
            self._msg(f"Fetch failed: {out}")

    # ------------------------------------------------------------------
    # Branches
    # ------------------------------------------------------------------

    def _refresh_branches(self):
        self._branch_list.clear()

        # Local branches with upstream info
        ok, out = _run_git(
            ["branch", "-vv", "--no-color"], cwd=self._repo_path
        )
        if not ok or not out:
            return

        for line in out.split("\n"):
            if not line.strip():
                continue
            is_current = line.startswith("*")
            line = line[2:]  # strip "* " or "  "

            # Parse: branch_name  hash [tracking] message
            parts = line.split(None, 2)
            if len(parts) < 2:
                continue
            branch_name = parts[0]
            rest = parts[2] if len(parts) > 2 else ""

            # Extract tracking info [origin/main] or [origin/main: ahead 1]
            tracking = ""
            message = rest
            track_match = re.match(r"\[([^\]]+)\]\s*(.*)", rest)
            if track_match:
                tracking = track_match.group(1)
                message = track_match.group(2)

            item = QTreeWidgetItem([branch_name, tracking, message])
            item.setData(0, Qt.UserRole, branch_name)

            if is_current:
                bold = QFont()
                bold.setBold(True)
                item.setFont(0, bold)
                item.setForeground(0, QBrush(QColor("#4caf50")))

            self._branch_list.addTopLevelItem(item)

        # Also show remote branches
        ok, out = _run_git(
            ["branch", "-r", "--no-color"], cwd=self._repo_path
        )
        if ok and out:
            for line in out.split("\n"):
                name = line.strip()
                if not name or " -> " in name:
                    continue
                item = QTreeWidgetItem([name, "(remote)", ""])
                item.setForeground(0, QBrush(_REMOTE_COLOR))
                item.setData(0, Qt.UserRole, name)
                self._branch_list.addTopLevelItem(item)

    def _checkout_branch(self, item, _column):
        """Double-click a branch to check it out."""
        branch = item.data(0, Qt.UserRole)
        if not branch:
            return
        # For remote branches, create local tracking branch
        if branch.startswith("origin/"):
            local_name = branch[len("origin/"):]
            ok, out = _run_git(
                ["checkout", "-b", local_name, "--track", branch],
                cwd=self._repo_path,
            )
        else:
            ok, out = _run_git(
                ["checkout", branch], cwd=self._repo_path
            )
        if ok:
            self._msg(f"Switched to '{branch}'")
            self.refresh()
        else:
            self._msg(f"Checkout failed: {out}")

    def _branch_context_menu(self, pos):
        item = self._branch_list.itemAt(pos)
        menu = QMenu(self)

        if item:
            branch = item.data(0, Qt.UserRole)
            act_checkout = QAction(f"Checkout '{branch}'", self)
            act_checkout.triggered.connect(
                lambda: self._checkout_branch(item, 0)
            )
            menu.addAction(act_checkout)

            # Only allow deleting non-current local branches
            ok, current = _run_git(
                ["rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self._repo_path,
            )
            if (ok and branch != current
                    and not branch.startswith("origin/")):
                act_delete = QAction(f"Delete '{branch}'", self)
                act_delete.triggered.connect(
                    lambda: self._delete_branch(branch)
                )
                menu.addAction(act_delete)

        act_new = QAction("New Branch...", self)
        act_new.triggered.connect(self._create_branch)
        menu.addAction(act_new)

        menu.exec(self._branch_list.viewport().mapToGlobal(pos))

    def _create_branch(self):
        name, ok = QInputDialog.getText(
            self, "New Branch", "Branch name:"
        )
        if not ok or not name:
            return
        ok, out = _run_git(
            ["checkout", "-b", name.strip()], cwd=self._repo_path
        )
        if ok:
            self._msg(f"Created and switched to '{name.strip()}'")
            self.refresh()
        else:
            self._msg(f"Failed: {out}")

    def _delete_branch(self, name):
        ans = QMessageBox.question(
            self, "Delete Branch",
            f"Delete branch '{name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if ans == QMessageBox.Yes:
            ok, out = _run_git(
                ["branch", "-d", name], cwd=self._repo_path
            )
            if ok:
                self._msg(f"Deleted branch '{name}'")
                self._refresh_branches()
            else:
                self._msg(f"Failed (use -D for force): {out}")

    # ------------------------------------------------------------------
    # Commit context menu
    # ------------------------------------------------------------------

    def _commit_context_menu(self, pos):
        item = self._commit_tree.itemAt(pos)
        if not item:
            return
        full_hash = item.data(0, Qt.UserRole)
        if not full_hash:
            return
        menu = QMenu(self)

        act_copy = QAction("Copy Hash", self)
        act_copy.triggered.connect(
            lambda: QApplication.clipboard().setText(full_hash)
        )
        menu.addAction(act_copy)

        act_checkout = QAction("Checkout This Commit", self)
        act_checkout.triggered.connect(
            lambda: self._checkout_commit(full_hash)
        )
        menu.addAction(act_checkout)

        act_tag = QAction("Tag This Commit...", self)
        act_tag.triggered.connect(lambda: self._tag_commit(full_hash))
        menu.addAction(act_tag)

        menu.exec(self._commit_tree.viewport().mapToGlobal(pos))

    def _checkout_commit(self, commit_hash):
        ok, out = _run_git(
            ["checkout", commit_hash], cwd=self._repo_path
        )
        if ok:
            self._msg(f"Checked out {commit_hash[:7]}")
            self.refresh()
        else:
            self._msg(f"Checkout failed: {out}")

    def _tag_commit(self, commit_hash):
        name, ok = QInputDialog.getText(
            self, "Create Tag", "Tag name:"
        )
        if not ok or not name:
            return
        ok, out = _run_git(
            ["tag", name.strip(), commit_hash], cwd=self._repo_path
        )
        if ok:
            self._msg(f"Created tag '{name.strip()}'")
            self._refresh_commit_history()
        else:
            self._msg(f"Failed: {out}")

    # ------------------------------------------------------------------
    # Basic operations: stage, commit, push, pull
    # ------------------------------------------------------------------

    def _stage_all(self):
        ok, out = _run_git(["add", "-A"], cwd=self._repo_path)
        if ok:
            self._msg("Staged all changes")
            self._refresh_status()
        else:
            self._msg(f"Stage failed: {out}")

    def _commit(self):
        msg = self._commit_msg.toPlainText().strip()
        if not msg:
            self._msg("Commit message cannot be empty.")
            return
        ok, out = _run_git(
            ["commit", "-m", msg], cwd=self._repo_path
        )
        if ok:
            self._msg("Committed successfully")
            self._commit_msg.clear()
            self.refresh()
        else:
            self._msg(f"Commit failed: {out}")

    def _push(self):
        ok, out = _run_git(
            ["push"], cwd=self._repo_path, timeout=30
        )
        if ok:
            self._msg("Pushed successfully")
            self.refresh()
        else:
            # Try push with upstream set
            if ("no upstream" in out.lower()
                    or "set-upstream" in out.lower()):
                ok2, branch = _run_git(
                    ["rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=self._repo_path,
                )
                if ok2:
                    ok3, out3 = _run_git(
                        ["push", "-u", "origin", branch],
                        cwd=self._repo_path, timeout=30,
                    )
                    if ok3:
                        self._msg(
                            f"Pushed and set upstream for '{branch}'"
                        )
                        self.refresh()
                        return
                    out = out3
            self._msg(f"Push failed: {out}")

    def _pull(self):
        ok, out = _run_git(
            ["pull"], cwd=self._repo_path, timeout=30
        )
        if ok:
            self._msg("Pulled successfully")
            self.refresh()
        else:
            self._msg(f"Pull failed: {out}")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _msg(self, text: str):
        """Show message in the status bar (or print for headless testing)."""
        self.status_message.emit(text)
