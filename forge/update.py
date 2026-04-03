# Copyright 2026 The Commons(TM)
# SPDX-License-Identifier: Apache-2.0
"""Locally-validated auto-update system for Forge IDE.

Provides:
- UpdateChecker: periodically polls the version manifest for new releases
- ValidatedUpdateWorker: downloads, validates via pytest, then applies updates
- Settings helpers for persisting auto-update preferences

The manifest is fetched from:
    https://thecommons.cc/pypi/forge-ide/manifest.json

Workflow:
    1. UpdateChecker fetches manifest and compares to installed version
    2. If a newer version exists and auto-update is enabled, spawn
       ValidatedUpdateWorker
    3. Worker installs new version into a temp venv, runs pytest
    4. If tests pass  -> apply upgrade to real environment
       If tests fail  -> defer, log failure, notify user
"""
from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MANIFEST_URL = "https://thecommons.cc/pypi/forge-ide/manifest.json"
PIP_INDEX = "https://thecommons.cc/pypi/"
CHECK_INTERVAL_SECONDS = 6 * 60 * 60  # 6 hours
PYTEST_TIMEOUT_SECONDS = 300
SETTINGS_FILE_NAME = "update_settings.json"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log_dir() -> Path:
    """Return ~/.forge/ directory, creating if needed."""
    d = Path.home() / ".forge"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("forge.update")
    if not logger.handlers:
        log_path = _log_dir() / "update.log"
        handler = logging.FileHandler(str(log_path), encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger


log = _setup_logger()

# ---------------------------------------------------------------------------
# Settings persistence (no Qt dependency -- plain JSON)
# ---------------------------------------------------------------------------

def _settings_path() -> Path:
    return _log_dir() / SETTINGS_FILE_NAME


def load_settings() -> dict:
    """Load update settings from ~/.forge/update_settings.json."""
    p = _settings_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_settings(settings: dict) -> None:
    """Persist update settings to disk."""
    p = _settings_path()
    p.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def is_auto_update_enabled() -> bool:
    """Return True if the user has opted in to validated auto-updates."""
    return load_settings().get("auto_update_enabled", False)


def set_auto_update_enabled(enabled: bool) -> None:
    """Toggle auto-update preference."""
    s = load_settings()
    s["auto_update_enabled"] = enabled
    save_settings(s)
    log.info("Auto-update %s by user", "enabled" if enabled else "disabled")


def get_deferred_version() -> Optional[str]:
    """Return the version string that was deferred, or None."""
    return load_settings().get("deferred_version")


def set_deferred_version(version: Optional[str], reason: str = "") -> None:
    s = load_settings()
    s["deferred_version"] = version
    s["deferred_reason"] = reason
    s["deferred_at"] = datetime.now(timezone.utc).isoformat()
    save_settings(s)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Manifest:
    latest_version: str
    min_supported: str = "0.0.0"
    release_date: str = ""
    changelog: str = ""
    pip_index: str = PIP_INDEX
    require_validation: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "Manifest":
        return cls(
            latest_version=d.get("latest_version", "0.0.0"),
            min_supported=d.get("min_supported", "0.0.0"),
            release_date=d.get("release_date", ""),
            changelog=d.get("changelog", ""),
            pip_index=d.get("pip_index", PIP_INDEX),
            require_validation=d.get("require_validation", True),
        )


@dataclass
class UpdateResult:
    success: bool
    new_version: str = ""
    message: str = ""
    tests_passed: Optional[int] = None
    tests_failed: Optional[int] = None
    log_path: str = ""

# ---------------------------------------------------------------------------
# Version comparison
# ---------------------------------------------------------------------------

def _parse_version(v: str) -> tuple:
    """Parse a PEP-440-ish version string to a comparable tuple."""
    parts = []
    for p in v.strip().split("."):
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(p)
    return tuple(parts)


def is_newer(remote: str, local: str) -> bool:
    return _parse_version(remote) > _parse_version(local)


def _installed_version() -> str:
    """Return the currently installed forge-ide version."""
    try:
        from forge import __version__
        return __version__
    except ImportError:
        return "0.0.0"

# ---------------------------------------------------------------------------
# Manifest fetcher
# ---------------------------------------------------------------------------

def fetch_manifest(url: str = MANIFEST_URL, timeout: int = 15) -> Optional[Manifest]:
    """Fetch the version manifest from the server.

    Uses urllib so we have no extra dependency beyond stdlib.
    """
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Forge-Update/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        log.debug("Manifest fetched: %s", data)
        return Manifest.from_dict(data)
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
        log.warning("Failed to fetch manifest from %s: %s", url, exc)
        return None

# ---------------------------------------------------------------------------
# Update checker (lightweight -- just compares versions)
# ---------------------------------------------------------------------------

class UpdateChecker:
    """Periodically checks for a newer version of forge-ide.

    This class is designed to be usable both from a Qt GUI (via QTimer
    calling check_now) and from a plain background thread.

    Attributes:
        on_update_available: callback(manifest) when a new version is found.
    """

    def __init__(
        self,
        on_update_available: Optional[Callable[[Manifest], None]] = None,
        manifest_url: str = MANIFEST_URL,
    ):
        self.on_update_available = on_update_available
        self.manifest_url = manifest_url
        self._last_check: Optional[float] = None

    def check_now(self) -> Optional[Manifest]:
        """Fetch manifest and compare.  Returns Manifest if update available."""
        manifest = fetch_manifest(self.manifest_url)
        self._last_check = time.time()
        if manifest is None:
            return None

        local = _installed_version()
        if is_newer(manifest.latest_version, local):
            log.info(
                "Update available: %s -> %s", local, manifest.latest_version
            )
            if self.on_update_available:
                self.on_update_available(manifest)
            return manifest
        else:
            log.debug(
                "Up to date: %s (remote: %s)", local, manifest.latest_version
            )
            return None

    def should_check(self) -> bool:
        """Return True if enough time has passed since last check."""
        if self._last_check is None:
            return True
        return (time.time() - self._last_check) >= CHECK_INTERVAL_SECONDS

# ---------------------------------------------------------------------------
# Validated update worker
# ---------------------------------------------------------------------------

class ValidatedUpdateWorker:
    """Download, validate, and apply a forge-ide update.

    The validation step installs the new version into a temporary virtualenv,
    runs ``python -m pytest tests/ -x -q``, and only applies the upgrade to
    the real environment if all tests pass.

    All heavy work runs in-process (call ``run()``).  For GUI usage, wrap
    in a QThread or threading.Thread.

    Parameters:
        manifest: the Manifest describing the target version.
        on_progress: callback(message: str) for status updates.
        on_finished: callback(UpdateResult) when done.
        force: skip validation (not recommended).
    """

    def __init__(
        self,
        manifest: Manifest,
        on_progress: Optional[Callable[[str], None]] = None,
        on_finished: Optional[Callable[[UpdateResult], None]] = None,
        force: bool = False,
    ):
        self.manifest = manifest
        self.on_progress = on_progress or (lambda m: None)
        self.on_finished = on_finished or (lambda r: None)
        self.force = force
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    # -- public entry point ------------------------------------------------

    def run(self) -> UpdateResult:
        """Execute the full validated-update pipeline.  Returns UpdateResult."""
        target = self.manifest.latest_version
        log.info("=== Starting validated update to %s ===", target)
        self.on_progress(f"Starting update to v{target}...")

        # If validation is required (and not forced), run in temp venv
        if self.manifest.require_validation and not self.force:
            result = self._validated_update(target)
        else:
            result = self._direct_update(target)

        if result.success:
            log.info("Update to %s succeeded", target)
            set_deferred_version(None)
        else:
            log.warning("Update to %s failed: %s", target, result.message)
            set_deferred_version(target, result.message)

        self.on_finished(result)
        return result

    # -- internal ----------------------------------------------------------

    def _validated_update(self, target: str) -> UpdateResult:
        """Install into temp venv, run tests, apply if passing."""
        tmpdir = None
        try:
            # 1. Create temporary virtualenv
            self.on_progress("Creating temporary validation environment...")
            tmpdir = tempfile.mkdtemp(prefix="forge_update_")
            venv_dir = os.path.join(tmpdir, "venv")
            log.debug("Temp venv at %s", venv_dir)

            rc = subprocess.call(
                [sys.executable, "-m", "venv", venv_dir],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            if rc != 0:
                return UpdateResult(
                    success=False,
                    new_version=target,
                    message="Failed to create temporary virtualenv",
                )

            if self._cancelled:
                return UpdateResult(False, target, "Cancelled by user")

            # Determine python in temp venv
            if platform.system() == "Windows":
                venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
            else:
                venv_python = os.path.join(venv_dir, "bin", "python")

            # 2. Install the new version + pytest into temp venv
            self.on_progress(
                f"Installing forge-ide=={target} in validation env..."
            )
            install_cmd = [
                venv_python, "-m", "pip", "install",
                "--index-url", self.manifest.pip_index,
                "--trusted-host", "thecommons.cc",
                f"forge-ide=={target}",
                "pytest",
            ]
            log.debug("Install cmd: %s", install_cmd)
            proc = subprocess.run(
                install_cmd,
                capture_output=True,
                text=True,
                timeout=180,
            )
            if proc.returncode != 0:
                log.error("pip install failed:\n%s", proc.stderr)
                return UpdateResult(
                    success=False,
                    new_version=target,
                    message=f"pip install failed: {proc.stderr[:500]}",
                )

            if self._cancelled:
                return UpdateResult(False, target, "Cancelled by user")

            # 3. Find the test directory
            find_tests_code = (
                "import forge, os; "
                "pkg = os.path.dirname(os.path.dirname(forge.__file__)); "
                "tp = os.path.join(pkg, 'tests'); "
                "print(tp if os.path.isdir(tp) else '')"
            )
            proc_find = subprocess.run(
                [venv_python, "-c", find_tests_code],
                capture_output=True, text=True, timeout=30,
            )
            test_dir = proc_find.stdout.strip()

            # Also check the source tree if available
            source_tests = Path(__file__).resolve().parent.parent / "tests"
            if not test_dir and source_tests.is_dir():
                test_dir = str(source_tests)

            if not test_dir:
                log.warning(
                    "No test directory found -- skipping validation"
                )
                self.on_progress(
                    "No tests found -- applying update directly..."
                )
                return self._direct_update(target)

            # 4. Run pytest
            self.on_progress("Running test suite against new version...")
            log.info("Running pytest on %s", test_dir)
            pytest_cmd = [
                venv_python, "-m", "pytest",
                test_dir,
                "-x", "-q",
                f"--timeout={PYTEST_TIMEOUT_SECONDS}",
                "--tb=short",
            ]
            try:
                proc_test = subprocess.run(
                    pytest_cmd,
                    capture_output=True,
                    text=True,
                    timeout=PYTEST_TIMEOUT_SECONDS + 60,
                    cwd=tmpdir,
                )
            except subprocess.TimeoutExpired:
                return UpdateResult(
                    success=False,
                    new_version=target,
                    message=(
                        f"Test suite timed out after "
                        f"{PYTEST_TIMEOUT_SECONDS}s"
                    ),
                )

            # Parse test results
            passed, failed = self._parse_pytest_output(proc_test.stdout)
            log.info(
                "Test results: passed=%s failed=%s rc=%d",
                passed, failed, proc_test.returncode,
            )

            # Write test output to log
            test_log_path = _log_dir() / f"update_test_{target}.log"
            test_log_path.write_text(
                proc_test.stdout + "\n---STDERR---\n" + proc_test.stderr,
                encoding="utf-8",
            )

            if proc_test.returncode != 0:
                msg = (
                    f"Update v{target} deferred -- "
                    f"{failed} test(s) failed. "
                    f"See {test_log_path} for details. "
                    f"Retry from Help > Check for Updates."
                )
                self.on_progress(msg)
                return UpdateResult(
                    success=False,
                    new_version=target,
                    message=msg,
                    tests_passed=passed,
                    tests_failed=failed,
                    log_path=str(test_log_path),
                )

            if self._cancelled:
                return UpdateResult(False, target, "Cancelled by user")

            # 5. Tests passed -- apply to real environment
            self.on_progress(
                f"All {passed} tests passed! "
                f"Applying update to v{target}..."
            )
            return self._direct_update(
                target, tests_passed=passed, tests_failed=0
            )

        except Exception as exc:
            log.exception("Unexpected error during validated update")
            return UpdateResult(
                success=False,
                new_version=target,
                message=f"Unexpected error: {exc}",
            )
        finally:
            if tmpdir and os.path.exists(tmpdir):
                try:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                except OSError:
                    pass

    def _direct_update(
        self,
        target: str,
        tests_passed: Optional[int] = None,
        tests_failed: Optional[int] = None,
    ) -> UpdateResult:
        """Apply the upgrade via pip install --upgrade."""
        self.on_progress(f"Applying upgrade to v{target}...")
        cmd = [
            sys.executable, "-m", "pip", "install",
            "--upgrade", "--no-cache-dir",
            "--index-url", self.manifest.pip_index,
            "--trusted-host", "thecommons.cc",
            f"forge-ide=={target}",
        ]
        log.debug("Upgrade cmd: %s", cmd)
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120,
            )
            if proc.returncode != 0:
                return UpdateResult(
                    success=False,
                    new_version=target,
                    message=f"pip upgrade failed: {proc.stderr[:500]}",
                    tests_passed=tests_passed,
                    tests_failed=tests_failed,
                )
            self.on_progress(f"Successfully updated to v{target}!")
            return UpdateResult(
                success=True,
                new_version=target,
                message=f"Updated to v{target}",
                tests_passed=tests_passed,
                tests_failed=tests_failed,
            )
        except subprocess.TimeoutExpired:
            return UpdateResult(
                success=False,
                new_version=target,
                message="pip upgrade timed out",
            )

    @staticmethod
    def _parse_pytest_output(output: str) -> tuple:
        """Extract passed/failed counts from pytest short summary."""
        passed = failed = None
        m_passed = re.search(r"(\d+)\s+passed", output)
        m_failed = re.search(r"(\d+)\s+failed", output)
        if m_passed:
            passed = int(m_passed.group(1))
        if m_failed:
            failed = int(m_failed.group(1))
        else:
            failed = 0 if passed is not None else None
        return passed, failed


# ---------------------------------------------------------------------------
# Convenience: one-shot check-and-update (for CLI / headless use)
# ---------------------------------------------------------------------------

def check_and_update(
    auto: bool = False,
    force: bool = False,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Optional[UpdateResult]:
    """Check for updates and optionally apply with validation.

    Args:
        auto: if True, only proceed if auto_update is enabled in settings.
        force: skip test validation (not recommended).
        on_progress: callback for status messages.

    Returns:
        UpdateResult if an update was attempted, None otherwise.
    """
    if auto and not is_auto_update_enabled():
        log.debug("Auto-update is disabled, skipping")
        return None

    progress = on_progress or (lambda m: print(f"  {m}"))
    checker = UpdateChecker()
    manifest = checker.check_now()

    if manifest is None:
        progress("No updates available (or fetch failed).")
        return None

    # Do not re-attempt a deferred version unless forced
    deferred = get_deferred_version()
    if deferred == manifest.latest_version and not force:
        progress(
            f"v{deferred} was previously deferred due to test failures. "
            "Use --force to retry."
        )
        return None

    worker = ValidatedUpdateWorker(
        manifest=manifest,
        on_progress=progress,
        force=force,
    )
    return worker.run()


# ---------------------------------------------------------------------------
# Qt integration helpers (only imported if PySide6 is available)
# ---------------------------------------------------------------------------

def get_qt_update_checker():
    """Return a QThread-based update checker class, or None if no Qt."""
    try:
        from PySide6.QtCore import QThread, Signal
    except ImportError:
        return None

    class QtUpdateChecker(QThread):
        """Background thread that periodically checks for updates.

        Signals:
            update_available(dict): emitted with manifest data
            check_finished(str): emitted with status message
        """
        update_available = Signal(dict)
        check_finished = Signal(str)

        def __init__(self, parent=None, interval_ms=None):
            super().__init__(parent)
            self._interval_ms = (
                interval_ms or (CHECK_INTERVAL_SECONDS * 1000)
            )
            self._checker = UpdateChecker()
            self._running = True

        def run(self):
            while self._running:
                manifest = self._checker.check_now()
                if manifest:
                    self.update_available.emit({
                        "latest_version": manifest.latest_version,
                        "changelog": manifest.changelog,
                        "release_date": manifest.release_date,
                        "require_validation": manifest.require_validation,
                    })
                    self.check_finished.emit(
                        f"Update available: v{manifest.latest_version}"
                    )
                else:
                    self.check_finished.emit("Up to date")
                # Sleep in small increments so we can stop promptly
                waited = 0
                while waited < self._interval_ms and self._running:
                    self.msleep(min(1000, self._interval_ms - waited))
                    waited += 1000

        def stop(self):
            self._running = False
            self.wait(5000)

    return QtUpdateChecker


def get_qt_validated_worker():
    """Return a QThread-based validated update worker, or None if no Qt."""
    try:
        from PySide6.QtCore import QThread, Signal
    except ImportError:
        return None

    class QtValidatedUpdateWorker(QThread):
        """QThread wrapper around ValidatedUpdateWorker.

        Signals:
            progress(str): status messages
            finished_result(dict): UpdateResult as dict when done
        """
        progress = Signal(str)
        finished_result = Signal(dict)

        def __init__(
            self, manifest_dict: dict, force: bool = False, parent=None
        ):
            super().__init__(parent)
            self._manifest = Manifest.from_dict(manifest_dict)
            self._force = force
            self._worker = None

        def run(self):
            self._worker = ValidatedUpdateWorker(
                manifest=self._manifest,
                on_progress=lambda m: self.progress.emit(m),
                force=self._force,
            )
            result = self._worker.run()
            self.finished_result.emit({
                "success": result.success,
                "new_version": result.new_version,
                "message": result.message,
                "tests_passed": result.tests_passed,
                "tests_failed": result.tests_failed,
                "log_path": result.log_path,
            })

        def cancel(self):
            if self._worker:
                self._worker.cancel()

    return QtValidatedUpdateWorker


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def cli_update(args=None) -> int:
    """CLI interface for the update system.

    Usage:
        forge --update              Check and update with validation
        forge --update --force      Skip validation
        forge --update --enable     Enable auto-updates
        forge --update --disable    Disable auto-updates
        forge --update --status     Show update settings and status
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="forge --update",
        description="Forge IDE validated auto-update system",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Skip test validation (not recommended)",
    )
    parser.add_argument(
        "--enable", action="store_true",
        help="Enable automatic validated updates",
    )
    parser.add_argument(
        "--disable", action="store_true",
        help="Disable automatic validated updates",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show current update settings and status",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Only check for updates, do not apply",
    )

    parsed = parser.parse_args(args or [])

    if parsed.enable:
        set_auto_update_enabled(True)
        print("Auto-validated updates: ENABLED")
        return 0

    if parsed.disable:
        set_auto_update_enabled(False)
        print("Auto-validated updates: DISABLED")
        return 0

    if parsed.status:
        settings = load_settings()
        enabled = settings.get("auto_update_enabled", False)
        deferred = settings.get("deferred_version")
        deferred_reason = settings.get("deferred_reason", "")
        installed = _installed_version()
        print(f"Installed version:  {installed}")
        print(f"Auto-update:        {'ENABLED' if enabled else 'DISABLED'}")
        if deferred:
            print(f"Deferred version:   {deferred}")
            print(f"Deferred reason:    {deferred_reason}")
        manifest = fetch_manifest()
        if manifest:
            print(f"Latest available:   {manifest.latest_version}")
            print(f"Changelog:          {manifest.changelog}")
            if is_newer(manifest.latest_version, installed):
                print("Status:             UPDATE AVAILABLE")
            else:
                print("Status:             Up to date")
        else:
            print("Manifest:           Could not fetch")
        return 0

    if parsed.check_only:
        checker = UpdateChecker()
        m = checker.check_now()
        if m:
            print(
                f"Update available: v{m.latest_version} "
                f"({m.changelog})"
            )
            return 0
        else:
            print(f"Up to date (v{_installed_version()})")
            return 0

    # Full check-and-update
    result = check_and_update(auto=False, force=parsed.force)
    if result is None:
        return 0
    return 0 if result.success else 1
