# Copyright 2026 The Commons(TM)
# SPDX-License-Identifier: Apache-2.0
"""Tests for forge.update -- validated auto-update system.

Requirement R-UPD-01: The Forge IDE SHALL provide an in-application update
mechanism that checks for new versions, validates them with local tests before
applying, and persists user preferences across sessions.

Model-user argument: An engineer accustomed to MATLAB's built-in update checker
expects to be notified of new Forge versions without leaving the IDE or running
manual pip commands. When an update is available, the system must download it,
run the test suite locally, and only apply the update if tests pass. If tests
fail, the version is deferred so the user is not nagged repeatedly about a
broken release. This protects production workflows from regressions.

Decomposition:
    R-UPD-01  Version comparison logic (is_newer, _parse_version).
    R-UPD-02  Manifest parsing from remote JSON (Manifest.from_dict).
    R-UPD-03  Settings persistence (save/load/toggle/deferred version).
    R-UPD-04  Manifest fetching with network error handling (fetch_manifest).
    R-UPD-05  Update availability detection (UpdateChecker).
    R-UPD-06  Pytest output parsing for validated updates (ValidatedUpdateWorker).
    R-UPD-07  UpdateResult dataclass correctness.
    R-UPD-08  End-to-end check_and_update integration (mocked).

Consistency argument: R-UPD-01 and R-UPD-02 establish the data model (versions
and manifests). R-UPD-03 ensures preferences survive restarts. R-UPD-04 handles
the network layer. R-UPD-05 composes the previous pieces into the availability
check. R-UPD-06 validates the local test gate. R-UPD-07 confirms the result
reporting structure. R-UPD-08 verifies the full orchestration path. Together
these cover the complete update lifecycle from check through apply or defer.
"""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from forge.update import (
    Manifest,
    UpdateChecker,
    UpdateResult,
    ValidatedUpdateWorker,
    _parse_version,
    check_and_update,
    fetch_manifest,
    get_deferred_version,
    is_auto_update_enabled,
    is_newer,
    load_settings,
    save_settings,
    set_auto_update_enabled,
    set_deferred_version,
)


# ---------------------------------------------------------------------------
# Version comparison
# ---------------------------------------------------------------------------

class TestVersionComparison:
    """R-UPD-01: The update module SHALL correctly compare semantic version
    strings to determine whether a remote version is newer than the installed
    version.

    Model-user argument: The engineer needs the IDE to distinguish patch, minor,
    and major version bumps so it can accurately report whether an update is
    available. False positives waste the user's time; false negatives leave them
    on a stale version.

    Decomposition:
        R-UPD-01.1  Newer patch version detected.
        R-UPD-01.2  Newer minor version detected.
        R-UPD-01.3  Newer major version detected.
        R-UPD-01.4  Same version returns False.
        R-UPD-01.5  Older version returns False.
        R-UPD-01.6  _parse_version returns correct integer tuple.

    Consistency: These six cases cover all ordering relationships (greater,
    equal, less) across all three version components, plus the parser that
    underlies the comparisons.
    """

    def test_newer_patch(self):
        """R-UPD-01.1: Patch bump is detected as newer."""
        assert is_newer("0.3.6", "0.3.5") is True

    def test_newer_minor(self):
        """R-UPD-01.2: Minor bump is detected as newer."""
        assert is_newer("0.4.0", "0.3.5") is True

    def test_newer_major(self):
        """R-UPD-01.3: Major bump is detected as newer."""
        assert is_newer("1.0.0", "0.3.5") is True

    def test_same_version(self):
        """R-UPD-01.4: Same version is not newer."""
        assert is_newer("0.3.5", "0.3.5") is False

    def test_older_version(self):
        """R-UPD-01.5: Older version is not newer."""
        assert is_newer("0.3.4", "0.3.5") is False

    def test_parse_version(self):
        """R-UPD-01.6: Version string parses to integer tuple."""
        assert _parse_version("1.2.3") == (1, 2, 3)
        assert _parse_version("0.3.5") == (0, 3, 5)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

class TestManifest:
    """R-UPD-02: The Manifest dataclass SHALL parse a remote JSON dictionary
    into a structured object with sensible defaults for missing fields.

    Model-user argument: The update manifest hosted on thecommons.cc/pypi may
    evolve over time (adding fields, omitting optional ones). The parser must
    handle both full and minimal payloads without crashing, so the engineer's
    update check never fails due to manifest format changes.

    Decomposition:
        R-UPD-02.1  Full manifest dict parses all fields.
        R-UPD-02.2  Minimal manifest dict applies defaults.
        R-UPD-02.3  Empty manifest dict does not crash.

    Consistency: These three cases cover the full, partial, and degenerate
    input scenarios for from_dict.
    """

    def test_from_dict_full(self):
        """R-UPD-02.1: Full manifest dict parses all fields correctly."""
        d = {
            "latest_version": "0.4.0",
            "min_supported": "0.3.0",
            "release_date": "2026-04-01",
            "changelog": "New features",
            "pip_index": "https://example.com/pypi/",
            "require_validation": False,
        }
        m = Manifest.from_dict(d)
        assert m.latest_version == "0.4.0"
        assert m.min_supported == "0.3.0"
        assert m.require_validation is False
        assert m.pip_index == "https://example.com/pypi/"

    def test_from_dict_minimal(self):
        """R-UPD-02.2: Minimal manifest applies correct defaults."""
        m = Manifest.from_dict({"latest_version": "1.0.0"})
        assert m.latest_version == "1.0.0"
        assert m.require_validation is True  # default
        assert m.min_supported == "0.0.0"

    def test_from_dict_empty(self):
        """R-UPD-02.3: Empty manifest does not crash."""
        m = Manifest.from_dict({})
        assert m.latest_version == "0.0.0"


# ---------------------------------------------------------------------------
# Settings persistence
# ---------------------------------------------------------------------------

class TestSettings:
    """R-UPD-03: The update module SHALL persist user preferences (auto-update
    toggle, deferred versions) to a JSON file that survives IDE restarts.

    Model-user argument: The engineer who disables auto-update or defers a
    broken version expects that preference to persist after closing and
    reopening the IDE. Losing preferences would cause unwanted update prompts
    or repeated attempts to install a known-bad version.

    Decomposition:
        R-UPD-03.1  save_settings writes and load_settings reads back.
        R-UPD-03.2  load_settings returns empty dict for missing file.
        R-UPD-03.3  Auto-update toggle round-trips correctly.
        R-UPD-03.4  Deferred version set/get/clear round-trips correctly.

    Consistency: These four cases cover the CRUD cycle for settings: create,
    read, update (toggle), and clear (deferred version). The missing-file case
    ensures a fresh install does not crash.
    """

    def test_save_and_load(self, tmp_path, monkeypatch):
        """R-UPD-03.1: Settings round-trip through save and load."""
        settings_file = tmp_path / "update_settings.json"
        monkeypatch.setattr(
            "forge.update._settings_path", lambda: settings_file
        )
        save_settings({"auto_update_enabled": True, "custom": "value"})
        loaded = load_settings()
        assert loaded["auto_update_enabled"] is True
        assert loaded["custom"] == "value"

    def test_load_missing_file(self, tmp_path, monkeypatch):
        """R-UPD-03.2: Missing settings file returns empty dict."""
        settings_file = tmp_path / "nonexistent.json"
        monkeypatch.setattr(
            "forge.update._settings_path", lambda: settings_file
        )
        assert load_settings() == {}

    def test_auto_update_toggle(self, tmp_path, monkeypatch):
        """R-UPD-03.3: Auto-update enable/disable persists correctly."""
        settings_file = tmp_path / "update_settings.json"
        monkeypatch.setattr(
            "forge.update._settings_path", lambda: settings_file
        )
        assert is_auto_update_enabled() is False
        set_auto_update_enabled(True)
        assert is_auto_update_enabled() is True
        set_auto_update_enabled(False)
        assert is_auto_update_enabled() is False

    def test_deferred_version(self, tmp_path, monkeypatch):
        """R-UPD-03.4: Deferred version set, get, and clear round-trips."""
        settings_file = tmp_path / "update_settings.json"
        monkeypatch.setattr(
            "forge.update._settings_path", lambda: settings_file
        )
        assert get_deferred_version() is None
        set_deferred_version("0.4.0", "3 tests failed")
        assert get_deferred_version() == "0.4.0"
        s = load_settings()
        assert s["deferred_reason"] == "3 tests failed"
        assert "deferred_at" in s
        set_deferred_version(None)
        assert get_deferred_version() is None


# ---------------------------------------------------------------------------
# Fetch manifest (mocked)
# ---------------------------------------------------------------------------

class TestFetchManifest:
    """R-UPD-04: The fetch_manifest function SHALL retrieve and parse a remote
    JSON manifest, returning None on network errors instead of crashing.

    Model-user argument: The engineer may be working offline or behind a
    restrictive firewall. The update check must degrade gracefully (return None)
    rather than throwing an unhandled exception that disrupts the IDE session.

    Decomposition:
        R-UPD-04.1  Successful fetch returns parsed Manifest.
        R-UPD-04.2  Network error returns None.

    Consistency: These two cases cover the success and failure paths for the
    network request.
    """

    def test_fetch_success(self, monkeypatch):
        """R-UPD-04.1: Successful fetch returns a Manifest object."""
        manifest_data = json.dumps({
            "latest_version": "0.4.0",
            "require_validation": True,
        }).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = manifest_data
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        def mock_urlopen(req, timeout=None):
            return mock_resp

        import urllib.request
        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        m = fetch_manifest("https://example.com/manifest.json")
        assert m is not None
        assert m.latest_version == "0.4.0"

    def test_fetch_network_error(self, monkeypatch):
        """R-UPD-04.2: Network error returns None without crashing."""
        import urllib.request
        import urllib.error

        def mock_urlopen(req, timeout=None):
            raise urllib.error.URLError("no internet")

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
        m = fetch_manifest("https://example.com/manifest.json")
        assert m is None


# ---------------------------------------------------------------------------
# UpdateChecker
# ---------------------------------------------------------------------------

class TestUpdateChecker:
    """R-UPD-05: The UpdateChecker SHALL determine whether a newer version is
    available and invoke a callback when one is found, respecting check
    intervals to avoid excessive network traffic.

    Model-user argument: The engineer expects the IDE to check for updates at
    startup (or on demand) and notify them when a new version exists, without
    hammering the server on every keystroke. The callback integration lets the
    GUI display an update banner.

    Decomposition:
        R-UPD-05.1  Update available triggers callback and returns manifest.
        R-UPD-05.2  No update available skips callback and returns None.
        R-UPD-05.3  Fetch failure returns None.
        R-UPD-05.4  First check is always allowed (should_check returns True).
        R-UPD-05.5  Recent check suppresses immediate re-check.

    Consistency: These five cases cover the three outcomes of a check (newer,
    same, failure) plus the rate-limiting logic (first-time vs. recent).
    """

    def test_update_available(self, monkeypatch):
        """R-UPD-05.1: Newer version triggers callback and returns manifest."""
        manifest = Manifest(latest_version="99.0.0")
        monkeypatch.setattr(
            "forge.update.fetch_manifest", lambda url: manifest
        )
        monkeypatch.setattr(
            "forge.update._installed_version", lambda: "0.3.5"
        )

        callback = MagicMock()
        checker = UpdateChecker(on_update_available=callback)
        result = checker.check_now()

        assert result is not None
        assert result.latest_version == "99.0.0"
        callback.assert_called_once_with(manifest)

    def test_no_update(self, monkeypatch):
        """R-UPD-05.2: Same version skips callback and returns None."""
        manifest = Manifest(latest_version="0.3.5")
        monkeypatch.setattr(
            "forge.update.fetch_manifest", lambda url: manifest
        )
        monkeypatch.setattr(
            "forge.update._installed_version", lambda: "0.3.5"
        )

        callback = MagicMock()
        checker = UpdateChecker(on_update_available=callback)
        result = checker.check_now()

        assert result is None
        callback.assert_not_called()

    def test_fetch_failure(self, monkeypatch):
        """R-UPD-05.3: Fetch failure returns None gracefully."""
        monkeypatch.setattr(
            "forge.update.fetch_manifest", lambda url: None
        )
        checker = UpdateChecker()
        assert checker.check_now() is None

    def test_should_check_initial(self):
        """R-UPD-05.4: First check is always allowed."""
        checker = UpdateChecker()
        assert checker.should_check() is True

    def test_should_check_after_recent(self, monkeypatch):
        """R-UPD-05.5: Recent check suppresses immediate re-check."""
        monkeypatch.setattr(
            "forge.update.fetch_manifest", lambda url: None
        )
        checker = UpdateChecker()
        checker.check_now()
        assert checker.should_check() is False


# ---------------------------------------------------------------------------
# ValidatedUpdateWorker -- parse pytest output
# ---------------------------------------------------------------------------

class TestParsePytest:
    """R-UPD-06: The ValidatedUpdateWorker SHALL correctly parse pytest summary
    lines to extract pass/fail counts for the local validation gate.

    Model-user argument: After downloading a new version, the update system runs
    the local test suite and parses the output to decide whether to apply or
    defer the update. Incorrect parsing could either block a good update (false
    negative) or apply a broken one (false positive).

    Decomposition:
        R-UPD-06.1  All-passed output parses correctly.
        R-UPD-06.2  Mixed pass/fail output parses correctly.
        R-UPD-06.3  Empty output returns None for both counts.
        R-UPD-06.4  Only-failed output parses correctly.

    Consistency: These four cases cover the three possible pytest summary
    formats (all pass, mixed, all fail) plus the degenerate empty case.
    """

    def test_all_passed(self):
        """R-UPD-06.1: All-passed pytest output parsed correctly."""
        output = "42 passed in 12.34s"
        p, f = ValidatedUpdateWorker._parse_pytest_output(output)
        assert p == 42
        assert f == 0

    def test_some_failed(self):
        """R-UPD-06.2: Mixed pass/fail pytest output parsed correctly."""
        output = "3 failed, 39 passed in 15.00s"
        p, f = ValidatedUpdateWorker._parse_pytest_output(output)
        assert p == 39
        assert f == 3

    def test_no_output(self):
        """R-UPD-06.3: Empty output returns None for both counts."""
        p, f = ValidatedUpdateWorker._parse_pytest_output("")
        assert p is None
        assert f is None

    def test_only_failed(self):
        """R-UPD-06.4: Only-failed pytest output parsed correctly."""
        output = "5 failed in 2.00s"
        p, f = ValidatedUpdateWorker._parse_pytest_output(output)
        assert p is None
        assert f == 5


# ---------------------------------------------------------------------------
# UpdateResult dataclass
# ---------------------------------------------------------------------------

class TestUpdateResult:
    """R-UPD-07: The UpdateResult dataclass SHALL accurately represent both
    successful and failed update outcomes including test counts and log paths.

    Model-user argument: The GUI displays update results to the engineer (e.g.
    "Updated to 0.4.0, 42 tests passed" or "Update deferred: 3 tests failed,
    see log at ..."). The dataclass must carry all the information needed for
    these messages.

    Decomposition:
        R-UPD-07.1  Success result carries correct fields.
        R-UPD-07.2  Failure result carries log path and fail count.

    Consistency: These two cases cover the two terminal states of an update
    attempt (success and failure).
    """

    def test_success_result(self):
        """R-UPD-07.1: Successful update result has correct fields."""
        r = UpdateResult(
            success=True, new_version="0.4.0",
            message="Updated", tests_passed=42, tests_failed=0,
        )
        assert r.success is True
        assert r.tests_passed == 42

    def test_failure_result(self):
        """R-UPD-07.2: Failed update result carries log path and fail count."""
        r = UpdateResult(
            success=False, new_version="0.4.0",
            message="3 tests failed", tests_passed=39, tests_failed=3,
            log_path="/tmp/test.log",
        )
        assert r.success is False
        assert r.log_path == "/tmp/test.log"


# ---------------------------------------------------------------------------
# check_and_update (integration, mocked)
# ---------------------------------------------------------------------------

class TestCheckAndUpdate:
    """R-UPD-08: The check_and_update orchestrator SHALL respect auto-update
    settings, handle missing manifests, and skip deferred versions.

    Model-user argument: The full update flow (check, download, validate, apply
    or defer) must honor the engineer's preferences. If auto-update is disabled,
    no check occurs. If a version was previously deferred, it is not re-offered.
    If no update is available, the function returns None without side effects.

    Decomposition:
        R-UPD-08.1  Auto-update disabled skips check entirely.
        R-UPD-08.2  No manifest available returns None.
        R-UPD-08.3  Deferred version is skipped with progress notification.

    Consistency: These three cases cover the three early-exit paths in the
    orchestrator. The full download-validate-apply path is not tested here
    because it requires real pip operations (covered by manual integration
    testing).
    """

    def test_auto_disabled_skips(self, tmp_path, monkeypatch):
        """R-UPD-08.1: Auto-update disabled returns None immediately."""
        settings_file = tmp_path / "update_settings.json"
        monkeypatch.setattr(
            "forge.update._settings_path", lambda: settings_file
        )
        result = check_and_update(auto=True)
        assert result is None

    def test_no_update_available(self, monkeypatch):
        """R-UPD-08.2: No manifest available returns None."""
        monkeypatch.setattr(
            "forge.update.fetch_manifest", lambda url: None
        )
        progress = MagicMock()
        result = check_and_update(auto=False, on_progress=progress)
        assert result is None

    def test_deferred_version_skipped(self, tmp_path, monkeypatch):
        """R-UPD-08.3: Previously deferred version is skipped."""
        settings_file = tmp_path / "update_settings.json"
        monkeypatch.setattr(
            "forge.update._settings_path", lambda: settings_file
        )
        set_deferred_version("0.4.0", "tests failed")

        manifest = Manifest(latest_version="0.4.0")
        monkeypatch.setattr(
            "forge.update.fetch_manifest", lambda url: manifest
        )
        monkeypatch.setattr(
            "forge.update._installed_version", lambda: "0.3.5"
        )

        progress = MagicMock()
        result = check_and_update(auto=False, on_progress=progress)
        assert result is None
        # Should have informed user about deferral
        progress.assert_called()
