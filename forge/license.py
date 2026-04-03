"""
Forge IDE -- Client-side license management.

Handles activation, validation, and tier gating for Forge license keys.
Communicates with the license server at https://thecommons.cc/api/license/.

Tiers:
    free        -- Full engine, all functions, community edition title bar.
    pro         -- Priority updates, PyInstaller dist, cloud burst (future).
    academic    -- Same as pro, .edu verified.
    enterprise  -- Same as pro, volume licensing.

Grace period: 30 days offline use after last successful validation.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Networking -- prefer requests, fall back to urllib
# ---------------------------------------------------------------------------
try:
    import requests as _requests  # type: ignore[import-untyped]

    def _post(url: str, payload: dict, timeout: int = 10) -> dict:
        """Send a JSON POST request using the requests library."""
        resp = _requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def _get(url: str, timeout: int = 10) -> dict:
        """Send a GET request using the requests library."""
        resp = _requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

except ImportError:
    import urllib.request
    import urllib.error

    def _post(url: str, payload: dict, timeout: int = 10) -> dict:  # type: ignore[misc]
        """Send a JSON POST request using urllib (fallback)."""
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get(url: str, timeout: int = 10) -> dict:  # type: ignore[misc]
        """Send a GET request using urllib (fallback)."""
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BASE_URL = "https://thecommons.cc"
_ACTIVATE_URL = f"{_BASE_URL}/api/license/activate"
_VALIDATE_URL = f"{_BASE_URL}/api/license/validate"
_STATUS_URL = f"{_BASE_URL}/api/license/status"
_GRACE_PERIOD_DAYS = 30
_VALID_TIERS = frozenset({"free", "pro", "academic", "enterprise"})


class LicenseManager:
    """Manages Forge IDE license state on the client side.

    Usage::

        lm = LicenseManager()
        lm.startup_check()          # non-blocking background validation
        print(lm.tier)              # "free" | "pro" | "academic" | "enterprise"
        print(lm.is_pro)            # True if tier grants pro-level features
    """

    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._load()

    # ------------------------------------------------------------------
    # Config path (platform-specific)
    # ------------------------------------------------------------------
    @property
    def config_path(self) -> Path:
        """Return the platform-appropriate path for ``license.json``.

        - Windows: ``%APPDATA%/Forge/license.json``
        - macOS:   ``~/Library/Application Support/Forge/license.json``
        - Linux:   ``~/.config/forge/license.json``
        """
        if sys.platform == "win32":
            base = Path(
                os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")
            )
            return base / "Forge" / "license.json"
        elif sys.platform == "darwin":
            return (
                Path.home()
                / "Library"
                / "Application Support"
                / "Forge"
                / "license.json"
            )
        else:
            return Path.home() / ".config" / "forge" / "license.json"

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------
    @property
    def tier(self) -> str:
        """Current tier string.  Defaults to ``'free'``."""
        with self._lock:
            t = self._data.get("tier", "free")
            return t if t in _VALID_TIERS else "free"

    @property
    def is_pro(self) -> bool:
        """``True`` if the current tier grants pro-level features."""
        return self.tier in ("pro", "academic", "enterprise")

    @property
    def is_activated(self) -> bool:
        """``True`` if a license token is stored locally."""
        with self._lock:
            return bool(self._data.get("token"))

    @property
    def license_key(self) -> Optional[str]:
        """Return the stored key, masked for display (first 8 chars + ``'...'``)."""
        with self._lock:
            key = self._data.get("license_key", "")
        if not key:
            return None
        return key[:8] + "..." if len(key) > 8 else key

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------
    def activate(self, license_key: str) -> dict:
        """Activate a license key against the server.

        Parameters
        ----------
        license_key : str
            The license key provided to the user.

        Returns
        -------
        dict
            Server response with ``token``, ``tier``, ``status``, ``expires``.
            On network/server error the dict contains an ``"error"`` key.
        """
        machine_id = self._machine_id()
        try:
            resp = _post(
                _ACTIVATE_URL, {"key": license_key, "machine_id": machine_id}
            )
        except Exception as exc:
            return {"error": str(exc)}

        # Persist on success
        if resp.get("token"):
            with self._lock:
                self._data["license_key"] = license_key
                self._data["token"] = resp["token"]
                self._data["tier"] = resp.get("tier", "free")
                self._data["status"] = resp.get("status", "active")
                self._data["expires"] = resp.get("expires", "")
                self._data["last_validated"] = time.time()
            self._save()
        return resp

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> dict:
        """Validate the stored token with the license server.

        Returns
        -------
        dict
            ``{"valid": bool, "tier": str, ...}``.
            On network error, grants grace-period access (up to 30 days).
        """
        with self._lock:
            token = self._data.get("token")
            saved_tier = self._data.get("tier", "free")
            last_ok = self._data.get("last_validated", 0.0)

        if not token:
            return {"valid": False, "tier": "free"}

        try:
            resp = _post(_VALIDATE_URL, {"token": token})
        except Exception:
            # Grace period -- allow offline use for up to _GRACE_PERIOD_DAYS
            elapsed_days = (time.time() - last_ok) / 86400.0
            if elapsed_days <= _GRACE_PERIOD_DAYS:
                return {"valid": True, "tier": saved_tier, "offline_grace": True}
            # Grace expired -- downgrade
            with self._lock:
                self._data["tier"] = "free"
            self._save()
            return {"valid": False, "tier": "free", "grace_expired": True}

        if resp.get("valid"):
            new_tier = resp.get("tier", saved_tier)
            with self._lock:
                self._data["tier"] = new_tier
                self._data["last_validated"] = time.time()
                if "days_remaining" in resp:
                    self._data["days_remaining"] = resp["days_remaining"]
            self._save()
            return resp

        # Token invalid / expired
        with self._lock:
            self._data["tier"] = "free"
            self._data["status"] = resp.get("status", "expired")
        self._save()
        return resp

    # ------------------------------------------------------------------
    # Deactivation
    # ------------------------------------------------------------------
    def deactivate(self) -> None:
        """Remove the local license file and reset to free tier."""
        with self._lock:
            self._data = {}
        try:
            self.config_path.unlink(missing_ok=True)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Startup check (non-blocking)
    # ------------------------------------------------------------------
    def startup_check(self) -> None:
        """Run a background validation on app launch.

        Never blocks the UI thread.  Prints status to stdout.
        """
        if not self.is_activated:
            print("[forge] License: Community Edition (free tier)")
            return

        def _bg_validate() -> None:
            result = self.validate()
            tier = result.get("tier", "free")
            if result.get("valid"):
                grace = " (offline grace)" if result.get("offline_grace") else ""
                print(f"[forge] License validated -- tier: {tier}{grace}")
            elif result.get("grace_expired"):
                print(
                    "[forge] License grace period expired -- reverted to free tier"
                )
            else:
                print(f"[forge] License validation failed -- tier: {tier}")

        t = threading.Thread(
            target=_bg_validate, daemon=True, name="forge-license-check"
        )
        t.start()

    # ------------------------------------------------------------------
    # Machine fingerprint
    # ------------------------------------------------------------------
    @staticmethod
    def _machine_id() -> str:
        """Generate a deterministic machine fingerprint.

        Uses SHA-256 of hostname + MAC address integer + platform string.
        """
        raw = f"{platform.node()}{uuid.getnode()}{sys.platform}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _save(self) -> None:
        """Write ``license.json`` to disk."""
        path = self.config_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                payload = dict(self._data)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            print(f"[forge] Warning: could not save license file: {exc}")

    def _load(self) -> None:
        """Read ``license.json`` from disk into ``self._data``."""
        path = self.config_path
        try:
            if path.is_file():
                text = path.read_text(encoding="utf-8")
                with self._lock:
                    self._data = json.loads(text)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[forge] Warning: could not load license file: {exc}")
            with self._lock:
                self._data = {}


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------
_default_manager: Optional[LicenseManager] = None


def get_license_manager() -> LicenseManager:
    """Return the module-level singleton :class:`LicenseManager`."""
    global _default_manager
    if _default_manager is None:
        _default_manager = LicenseManager()
    return _default_manager
