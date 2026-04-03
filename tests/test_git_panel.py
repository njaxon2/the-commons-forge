"""Tests for forge.gui.git_panel -- headless (no Qt display required).

These tests exercise the helper functions and the _run_git wrapper
without needing a running X server or PySide6 display.  The Git
operations are tested against real temporary repositories.
"""

import os
import subprocess
import tempfile

import pytest

# Import the module-level helpers that do NOT require Qt
from forge.gui.git_panel import _run_git, _relative_time


# ===========================================================================
# _run_git tests
# ===========================================================================

class TestRunGit:
    """Tests for the _run_git subprocess wrapper."""

    def test_git_version(self):
        ok, out = _run_git(["--version"])
        assert ok is True
        assert "git version" in out

    def test_git_invalid_command(self):
        ok, out = _run_git(["not-a-real-subcommand"])
        assert ok is False

    def test_git_with_cwd(self, tmp_path):
        """git init in a temp directory should succeed."""
        ok, out = _run_git(["init"], cwd=str(tmp_path))
        assert ok is True
        assert os.path.isdir(os.path.join(str(tmp_path), ".git"))

    def test_git_timeout(self):
        """Passing a very short timeout should not crash."""
        ok, out = _run_git(["--version"], timeout=1)
        assert ok is True

    def test_git_nonexistent_cwd(self):
        """Running git in a nonexistent directory should fail gracefully."""
        ok, out = _run_git(["status"], cwd="/nonexistent_dir_12345")
        assert ok is False


# ===========================================================================
# _relative_time tests
# ===========================================================================

class TestRelativeTime:
    """Tests for the ISO date -> relative time converter."""

    def test_recent(self):
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        iso = now.isoformat()
        result = _relative_time(iso)
        assert result == "just now"

    def test_minutes_ago(self):
        from datetime import datetime, timezone, timedelta
        dt = datetime.now(timezone.utc) - timedelta(minutes=5)
        result = _relative_time(dt.isoformat())
        assert "min" in result

    def test_hours_ago(self):
        from datetime import datetime, timezone, timedelta
        dt = datetime.now(timezone.utc) - timedelta(hours=3)
        result = _relative_time(dt.isoformat())
        assert "hour" in result

    def test_days_ago(self):
        from datetime import datetime, timezone, timedelta
        dt = datetime.now(timezone.utc) - timedelta(days=2)
        result = _relative_time(dt.isoformat())
        assert "day" in result

    def test_weeks_ago(self):
        from datetime import datetime, timezone, timedelta
        dt = datetime.now(timezone.utc) - timedelta(weeks=2)
        result = _relative_time(dt.isoformat())
        assert "week" in result

    def test_old_date_shows_ymd(self):
        result = _relative_time("2020-01-15T00:00:00+00:00")
        assert "2020-01-15" in result

    def test_invalid_date_returns_input(self):
        result = _relative_time("not-a-date")
        assert result == "not-a-date"


# ===========================================================================
# Git repository operation tests (using real temp repos)
# ===========================================================================

class TestGitRepoOperations:
    """Integration tests using real temporary git repositories."""

    @pytest.fixture
    def git_repo(self, tmp_path):
        """Create a temporary git repository with one commit."""
        repo = str(tmp_path / "test_repo")
        os.makedirs(repo)
        _run_git(["init"], cwd=repo)
        _run_git(["config", "user.email", "test@test.com"], cwd=repo)
        _run_git(["config", "user.name", "Test User"], cwd=repo)

        # Create initial commit
        readme = os.path.join(repo, "README.md")
        with open(readme, "w") as f:
            f.write("# Test Project\n")
        _run_git(["add", "README.md"], cwd=repo)
        _run_git(["commit", "-m", "Initial commit"], cwd=repo)
        return repo

    def test_status_clean(self, git_repo):
        ok, out = _run_git(["status", "--porcelain=v1"], cwd=git_repo)
        assert ok is True
        assert out == ""

    def test_status_with_changes(self, git_repo):
        with open(os.path.join(git_repo, "new_file.txt"), "w") as f:
            f.write("hello\n")
        ok, out = _run_git(["status", "--porcelain=v1"], cwd=git_repo)
        assert ok is True
        assert "new_file.txt" in out
        assert "??" in out  # untracked

    def test_stage_and_commit(self, git_repo):
        with open(os.path.join(git_repo, "file.txt"), "w") as f:
            f.write("content\n")
        ok, _ = _run_git(["add", "-A"], cwd=git_repo)
        assert ok is True
        ok, _ = _run_git(["commit", "-m", "Add file"], cwd=git_repo)
        assert ok is True
        ok, out = _run_git(["log", "--oneline"], cwd=git_repo)
        assert ok is True
        assert "Add file" in out

    def test_branch_create_and_list(self, git_repo):
        ok, _ = _run_git(["checkout", "-b", "feature-x"], cwd=git_repo)
        assert ok is True
        ok, out = _run_git(
            ["rev-parse", "--abbrev-ref", "HEAD"], cwd=git_repo
        )
        assert ok is True
        assert out == "feature-x"

        ok, out = _run_git(["branch", "--no-color"], cwd=git_repo)
        assert ok is True
        assert "feature-x" in out

    def test_branch_checkout(self, git_repo):
        _run_git(["checkout", "-b", "dev"], cwd=git_repo)
        # Get default branch name
        ok, default_branch = _run_git(
            ["rev-parse", "--abbrev-ref", "HEAD"], cwd=git_repo
        )
        # We are on dev, switch back
        ok, out = _run_git(["log", "--oneline", "-1"], cwd=git_repo)
        assert ok is True

    def test_remote_add_and_list(self, git_repo):
        ok, _ = _run_git(
            ["remote", "add", "origin", "https://example.com/repo.git"],
            cwd=git_repo,
        )
        assert ok is True
        ok, out = _run_git(["remote", "-v"], cwd=git_repo)
        assert ok is True
        assert "origin" in out
        assert "example.com" in out

    def test_remote_remove(self, git_repo):
        _run_git(
            ["remote", "add", "upstream", "https://example.com/upstream.git"],
            cwd=git_repo,
        )
        ok, _ = _run_git(["remote", "remove", "upstream"], cwd=git_repo)
        assert ok is True
        ok, out = _run_git(["remote", "-v"], cwd=git_repo)
        assert ok is True
        assert "upstream" not in out

    def test_log_format(self, git_repo):
        """Verify the custom log format used by the commit history view."""
        fmt = "%H%x00%h%x00%s%x00%an%x00%aI%x00%D"
        ok, out = _run_git(
            ["log", "--all", "--graph", "--format=" + fmt, "-10"],
            cwd=git_repo,
        )
        assert ok is True
        assert "\x00" in out
        parts = out.split("\x00")
        assert len(parts) >= 6

    def test_tag_create_and_list(self, git_repo):
        ok, _ = _run_git(["tag", "v0.1.0"], cwd=git_repo)
        assert ok is True
        ok, out = _run_git(["tag", "-l"], cwd=git_repo)
        assert ok is True
        assert "v0.1.0" in out

    def test_branch_delete(self, git_repo):
        _run_git(["checkout", "-b", "temp-branch"], cwd=git_repo)
        _run_git(["checkout", "-"], cwd=git_repo)
        ok, _ = _run_git(["branch", "-d", "temp-branch"], cwd=git_repo)
        assert ok is True

    def test_init_already_initialized(self, git_repo):
        """Re-initializing should succeed (git init is idempotent)."""
        ok, out = _run_git(["init"], cwd=git_repo)
        assert ok is True

    def test_commit_empty_fails(self, git_repo):
        """Committing with nothing staged should fail."""
        ok, out = _run_git(
            ["commit", "-m", "empty commit"], cwd=git_repo
        )
        assert ok is False

    def test_multiple_commits_in_log(self, git_repo):
        """Create several commits and verify they appear in log."""
        for i in range(5):
            path = os.path.join(git_repo, f"file_{i}.txt")
            with open(path, "w") as f:
                f.write(f"content {i}\n")
            _run_git(["add", "-A"], cwd=git_repo)
            _run_git(["commit", "-m", f"Commit {i}"], cwd=git_repo)

        ok, out = _run_git(["log", "--oneline"], cwd=git_repo)
        assert ok is True
        lines = out.strip().split("\n")
        assert len(lines) >= 6  # 5 + initial

    def test_no_repo_status_fails(self, tmp_path):
        """Running status outside a repo should fail."""
        ok, out = _run_git(["status"], cwd=str(tmp_path))
        assert ok is False
