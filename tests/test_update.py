# Copyright 2026 The Commons(TM)
# SPDX-License-Identifier: Apache-2.0
"""Tests for forge.update -- validated auto-update system."""
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
    def test_newer_patch(self):
        assert is_newer("0.3.6", "0.3.5") is True

    def test_newer_minor(self):
        assert is_newer("0.4.0", "0.3.5") is True

    def test_newer_major(self):
        assert is_newer("1.0.0", "0.3.5") is True

    def test_same_version(self):
        assert is_newer("0.3.5", "0.3.5") is False

    def test_older_version(self):
        assert is_newer("0.3.4", "0.3.5") is False

    def test_parse_version(self):
        assert _parse_version("1.2.3") == (1, 2, 3)
        assert _parse_version("0.3.5") == (0, 3, 5)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

class TestManifest:
    def test_from_dict_full(self):
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
        m = Manifest.from_dict({"latest_version": "1.0.0"})
        assert m.latest_version == "1.0.0"
        assert m.require_validation is True  # default
        assert m.min_supported == "0.0.0"

    def test_from_dict_empty(self):
        m = Manifest.from_dict({})
        assert m.latest_version == "0.0.0"


# ---------------------------------------------------------------------------
# Settings persistence
# ---------------------------------------------------------------------------

class TestSettings:
    def test_save_and_load(self, tmp_path, monkeypatch):
        settings_file = tmp_path / "update_settings.json"
        monkeypatch.setattr(
            "forge.update._settings_path", lambda: settings_file
        )
        save_settings({"auto_update_enabled": True, "custom": "value"})
        loaded = load_settings()
        assert loaded["auto_update_enabled"] is True
        assert loaded["custom"] == "value"

    def test_load_missing_file(self, tmp_path, monkeypatch):
        settings_file = tmp_path / "nonexistent.json"
        monkeypatch.setattr(
            "forge.update._settings_path", lambda: settings_file
        )
        assert load_settings() == {}

    def test_auto_update_toggle(self, tmp_path, monkeypatch):
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
    def test_fetch_success(self, monkeypatch):
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
    def test_update_available(self, monkeypatch):
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
        monkeypatch.setattr(
            "forge.update.fetch_manifest", lambda url: None
        )
        checker = UpdateChecker()
        assert checker.check_now() is None

    def test_should_check_initial(self):
        checker = UpdateChecker()
        assert checker.should_check() is True

    def test_should_check_after_recent(self, monkeypatch):
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
    def test_all_passed(self):
        output = "42 passed in 12.34s"
        p, f = ValidatedUpdateWorker._parse_pytest_output(output)
        assert p == 42
        assert f == 0

    def test_some_failed(self):
        output = "3 failed, 39 passed in 15.00s"
        p, f = ValidatedUpdateWorker._parse_pytest_output(output)
        assert p == 39
        assert f == 3

    def test_no_output(self):
        p, f = ValidatedUpdateWorker._parse_pytest_output("")
        assert p is None
        assert f is None

    def test_only_failed(self):
        output = "5 failed in 2.00s"
        p, f = ValidatedUpdateWorker._parse_pytest_output(output)
        assert p is None
        assert f == 5


# ---------------------------------------------------------------------------
# UpdateResult dataclass
# ---------------------------------------------------------------------------

class TestUpdateResult:
    def test_success_result(self):
        r = UpdateResult(
            success=True, new_version="0.4.0",
            message="Updated", tests_passed=42, tests_failed=0,
        )
        assert r.success is True
        assert r.tests_passed == 42

    def test_failure_result(self):
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
    def test_auto_disabled_skips(self, tmp_path, monkeypatch):
        settings_file = tmp_path / "update_settings.json"
        monkeypatch.setattr(
            "forge.update._settings_path", lambda: settings_file
        )
        result = check_and_update(auto=True)
        assert result is None

    def test_no_update_available(self, monkeypatch):
        monkeypatch.setattr(
            "forge.update.fetch_manifest", lambda url: None
        )
        progress = MagicMock()
        result = check_and_update(auto=False, on_progress=progress)
        assert result is None

    def test_deferred_version_skipped(self, tmp_path, monkeypatch):
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
