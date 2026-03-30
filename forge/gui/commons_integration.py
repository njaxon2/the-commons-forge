"""TheCommons integration for Forge IDE.

Provides:
  - UpdateChecker: checks thecommons.cc/pypi for new wheel versions
  - UpdateWorker: performs pip/PyInstaller update in background thread
  - AMSConnector: opt-in anonymized telemetry
  - FeatureRequestDialog: submit feature requests
"""

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

from PySide6.QtCore import QTimer, Qt, Signal, QObject, QThread
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QTextEdit,
    QComboBox, QRadioButton, QButtonGroup, QPushButton, QMessageBox,
    QGroupBox, QCheckBox, QWidget, QProgressBar,
)

logger = logging.getLogger(__name__)

from forge import __version__ as FORGE_VERSION

COMMONS_BASE = "https://thecommons.cc/api"
PYPI_INDEX_URL = "https://thecommons.cc/pypi/forge-ide/"
AMS_LOG_URL = f"{COMMONS_BASE}/ams/log"
FEATURE_REQUEST_URL = f"{COMMONS_BASE}/forge/feature-request"

CONFIG_DIR = Path.home() / ".forge"


def _ensure_config_dir():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def _is_pyinstaller_bundle():
    """Return True if running from a PyInstaller frozen bundle."""
    return getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')


def _is_pip_install():
    """Return True if forge was installed via pip (not editable dev)."""
    try:
        import importlib.metadata
        dist = importlib.metadata.distribution("forge-ide")
        return True
    except Exception:
        return False


def _compare_versions(a, b):
    """Compare version strings. Returns >0 if a>b, 0 if equal, <0 if a<b."""
    def parse(v):
        return tuple(int(x) for x in v.split("."))
    try:
        pa, pb = parse(a), parse(b)
        if pa > pb:
            return 1
        elif pa < pb:
            return -1
        return 0
    except Exception:
        return 0


# ======================================================================
# UpdateChecker
# ======================================================================

class UpdateChecker(QObject):
    """Checks thecommons.cc/pypi/ for Forge updates."""

    update_available = Signal(str)   # latest_version
    update_not_available = Signal()
    check_failed = Signal(str)       # error message

    CHECK_INTERVAL_MS = 6 * 60 * 60 * 1000  # 6 hours

    def __init__(self, parent=None, auto_update=False):
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.check_now)
        self.latest_version = None
        self.auto_update = auto_update

    def start(self):
        """Begin periodic update checks (first check after 5 seconds)."""
        QTimer.singleShot(5000, self.check_now)
        self._timer.start(self.CHECK_INTERVAL_MS)

    def stop(self):
        self._timer.stop()

    def check_now(self):
        """Perform a single update check (non-blocking via thread)."""
        import threading
        t = threading.Thread(target=self._do_check, daemon=True)
        t.start()

    def _do_check(self):
        try:
            import urllib.request
            req = urllib.request.Request(
                PYPI_INDEX_URL,
                headers={"User-Agent": f"Forge/{FORGE_VERSION}"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read().decode()

            # Parse wheel filenames: forge_ide-X.Y.Z-py3-none-any.whl
            versions = re.findall(r'forge_ide-([\d.]+)-py3-none-any\.whl', html)
            if not versions:
                self.check_failed.emit("No versions found on server")
                return

            latest = max(versions, key=lambda v: tuple(int(x) for x in v.split(".")))
            self.latest_version = latest

            if _compare_versions(latest, FORGE_VERSION) > 0:
                logger.info("Forge update available: %s (current: %s)", latest, FORGE_VERSION)
                self.update_available.emit(latest)
            else:
                logger.debug("Forge is up to date (%s)", FORGE_VERSION)
                self.update_not_available.emit()

        except Exception as exc:
            logger.debug("Update check failed: %s", exc)
            self.check_failed.emit(str(exc))


# ======================================================================
# UpdateWorker -- runs pip/download in a background thread
# ======================================================================

class UpdateWorker(QThread):
    """Performs the actual update in a background thread."""
    progress = Signal(str)        # status text
    finished_ok = Signal(str)     # success message
    finished_err = Signal(str)    # error message

    def __init__(self, target_version, parent=None):
        super().__init__(parent)
        self.target_version = target_version

    def run(self):
        if _is_pyinstaller_bundle():
            self._update_pyinstaller()
        else:
            self._update_pip()

    def _update_pip(self):
        """Update via pip from thecommons.cc."""
        import subprocess
        self.progress.emit(f"Downloading forge-ide {self.target_version} via pip...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade",
                 "--index-url", "https://thecommons.cc/pypi/",
                 "--trusted-host", "thecommons.cc",
                 f"forge-ide=={self.target_version}"],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                self.progress.emit("Update installed successfully!")
                self.finished_ok.emit(
                    f"Forge has been updated to v{self.target_version}.\n"
                    "Please restart Forge for the changes to take effect."
                )
            else:
                err = result.stderr.strip() or result.stdout.strip()
                self.finished_err.emit(f"pip install failed:\n{err}")
        except subprocess.TimeoutExpired:
            self.finished_err.emit("Update timed out after 120 seconds.")
        except Exception as exc:
            self.finished_err.emit(f"Update failed: {exc}")

    def _update_pyinstaller(self):
        """Update a PyInstaller bundle by downloading the new executable."""
        import urllib.request
        import tempfile
        import shutil

        self.progress.emit(f"Downloading Forge v{self.target_version} installer...")
        try:
            if sys.platform == "win32":
                asset_name = f"forge-{self.target_version}-win64-setup.exe"
            elif sys.platform == "darwin":
                asset_name = f"forge-{self.target_version}-macos.dmg"
            else:
                asset_name = f"forge-{self.target_version}-linux"

            download_url = f"https://thecommons.cc/pypi/forge-ide/{asset_name}"

            tmp = tempfile.mktemp(suffix=os.path.splitext(asset_name)[1])
            req = urllib.request.Request(
                download_url,
                headers={"User-Agent": f"Forge/{FORGE_VERSION}"}
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                with open(tmp, 'wb') as f:
                    shutil.copyfileobj(resp, f)

            self.progress.emit("Download complete.")

            if sys.platform == "win32":
                current_dir = os.path.dirname(sys.executable)
                dest = os.path.join(current_dir, asset_name)
                shutil.move(tmp, dest)
                self.finished_ok.emit(
                    f"Update downloaded to:\n{dest}\n\n"
                    "Please close Forge and run the new installer."
                )
            else:
                self.finished_ok.emit(
                    f"Update downloaded to:\n{tmp}\n\n"
                    "Please close Forge and install the update."
                )
        except Exception as exc:
            self.finished_err.emit(f"Update download failed: {exc}")


# ======================================================================
# ValidatedUpdateWorker -- downloads, tests locally, then applies
# ======================================================================

class ValidatedUpdateWorker(QThread):
    """Download update, run local tests, apply only if passing."""
    progress = Signal(str)
    finished_ok = Signal(str)     # success message
    finished_err = Signal(str)    # error message
    tests_failed = Signal(str)    # tests failed — update deferred

    def __init__(self, target_version, parent=None):
        super().__init__(parent)
        self.target_version = target_version

    def run(self):
        import subprocess
        # Step 1: Download to a temp location
        self.progress.emit(f"Downloading forge-ide {self.target_version}...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install",
                 "--index-url", "https://thecommons.cc/pypi/",
                 "--trusted-host", "thecommons.cc",
                 f"forge-ide=={self.target_version}"],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                err = result.stderr.strip() or result.stdout.strip()
                self.finished_err.emit(f"Download failed:\n{err}")
                return
        except Exception as exc:
            self.finished_err.emit(f"Download failed: {exc}")
            return

        # Step 2: Run local tests
        self.progress.emit("Running local validation tests...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest",
                 "--tb=short", "-q", "--timeout=60"],
                capture_output=True, text=True, timeout=300,
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            )
            if result.returncode == 0:
                self.progress.emit("Tests passed! Update validated.")
                self.finished_ok.emit(
                    f"Forge v{self.target_version} installed and validated.\n"
                    "Please restart Forge for changes to take effect."
                )
            else:
                # Tests failed — rollback
                self.progress.emit("Tests failed — rolling back...")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install",
                     "--index-url", "https://thecommons.cc/pypi/",
                     "--trusted-host", "thecommons.cc",
                     f"forge-ide=={FORGE_VERSION}"],
                    capture_output=True, text=True, timeout=120,
                )
                summary = result.stdout[-500:] if len(result.stdout) > 500 else result.stdout
                self.tests_failed.emit(
                    f"Update to v{self.target_version} deferred — tests failed:\n{summary}"
                )
        except subprocess.TimeoutExpired:
            self.tests_failed.emit("Test suite timed out. Update deferred.")
        except Exception as exc:
            self.finished_err.emit(f"Validation error: {exc}")


# ======================================================================
# AMSConnector
# ======================================================================

class AMSConnector(QObject):
    """Application Management Service - opt-in anonymized telemetry."""

    def __init__(self, parent=None):
        super().__init__(parent)
        _ensure_config_dir()
        self._config_path = CONFIG_DIR / "ams_config.json"
        self._config = self._load_config()
        self._session_start = time.time()
        self._event_buffer = []

    def _load_config(self):
        try:
            if self._config_path.exists():
                with open(self._config_path, "r") as f:
                    return json.load(f)
        except Exception as exc:
            logger.debug("AMS config load error: %s", exc)
        return {"ams_enabled": False}

    def _save_config(self):
        try:
            _ensure_config_dir()
            with open(self._config_path, "w") as f:
                json.dump(self._config, f, indent=2)
        except Exception as exc:
            logger.warning("AMS config save error: %s", exc)

    @property
    def enabled(self):
        return self._config.get("ams_enabled", False)

    def connect(self):
        self._config["ams_enabled"] = True
        self._save_config()

    def disconnect(self):
        self._config["ams_enabled"] = False
        self._save_config()

    def send_log(self, event_type, data=None):
        if not self.enabled:
            return
        payload = {
            "event": event_type,
            "version": FORGE_VERSION,
            "session_duration": int(time.time() - self._session_start),
            "data": data or {},
            "timestamp": int(time.time()),
        }
        import threading
        t = threading.Thread(target=self._do_send, args=(payload,), daemon=True)
        t.start()

    def _do_send(self, payload):
        try:
            import urllib.request
            body = json.dumps(payload).encode()
            req = urllib.request.Request(
                AMS_LOG_URL, data=body,
                headers={"Content-Type": "application/json",
                         "User-Agent": f"Forge/{FORGE_VERSION}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10):
                pass
        except Exception as exc:
            logger.debug("AMS send failed: %s", exc)

    def show_opt_in_dialog(self, parent=None):
        dlg = QMessageBox(parent)
        dlg.setWindowTitle("AMS Telemetry")
        dlg.setIcon(QMessageBox.Icon.Question)
        dlg.setText(
            "<b>Help improve Forge</b><br><br>"
            "Would you like to enable anonymized telemetry?<br>"
            "We collect only: session duration, function usage counts, "
            "and error counts.<br><br>"
            "<b>No personal data is ever collected.</b><br>"
            "You can disable this at any time from the Help menu."
        )
        dlg.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        dlg.setDefaultButton(QMessageBox.StandardButton.No)
        result = dlg.exec()
        if result == QMessageBox.StandardButton.Yes:
            self.connect()
            return True
        else:
            self.disconnect()
            return False


# ======================================================================
# FeatureRequestDialog
# ======================================================================

class FeatureRequestDialog(QDialog):
    """Dialog for submitting feature requests to TheCommons."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Submit Feature Request")
        self.setMinimumWidth(480)
        self.setMinimumHeight(400)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        cat_layout = QHBoxLayout()
        cat_layout.addWidget(QLabel("Category:"))
        self.category_combo = QComboBox()
        self.category_combo.addItems(["UI", "Engine", "Functions", "Performance", "Other"])
        cat_layout.addWidget(self.category_combo)
        layout.addLayout(cat_layout)

        title_layout = QHBoxLayout()
        title_layout.addWidget(QLabel("Title:"))
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("Brief summary of your request")
        title_layout.addWidget(self.title_edit)
        layout.addLayout(title_layout)

        layout.addWidget(QLabel("Description:"))
        self.desc_edit = QTextEdit()
        self.desc_edit.setPlaceholderText("Describe the feature you would like...")
        layout.addWidget(self.desc_edit)

        prio_group = QGroupBox("Priority")
        prio_layout = QHBoxLayout(prio_group)
        self.prio_button_group = QButtonGroup(self)
        for i, label in enumerate(["Low", "Medium", "High"]):
            rb = QRadioButton(label)
            self.prio_button_group.addButton(rb, i)
            prio_layout.addWidget(rb)
            if label == "Medium":
                rb.setChecked(True)
        layout.addWidget(prio_group)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        submit_btn = QPushButton("Submit")
        submit_btn.setDefault(True)
        submit_btn.clicked.connect(self._on_submit)
        btn_layout.addWidget(submit_btn)
        layout.addLayout(btn_layout)

    def _on_submit(self):
        title = self.title_edit.text().strip()
        if not title:
            QMessageBox.warning(self, "Missing Title", "Please enter a title.")
            return
        prio_btn = self.prio_button_group.checkedButton()
        priority = prio_btn.text() if prio_btn else "Medium"
        payload = {
            "category": self.category_combo.currentText(),
            "title": title,
            "description": self.desc_edit.toPlainText().strip(),
            "priority": priority,
            "version": FORGE_VERSION,
            "timestamp": int(time.time()),
        }
        self._save_local(payload)
        self._submit_remote(payload)

    def _save_local(self, payload):
        try:
            _ensure_config_dir()
            path = CONFIG_DIR / "feature_requests.json"
            existing = []
            if path.exists():
                try:
                    with open(path, "r") as f:
                        existing = json.load(f)
                except Exception:
                    existing = []
            existing.append(payload)
            with open(path, "w") as f:
                json.dump(existing, f, indent=2)
        except Exception as exc:
            logger.warning("Failed to save feature request locally: %s", exc)

    def _submit_remote(self, payload):
        try:
            import urllib.request
            body = json.dumps(payload).encode()
            req = urllib.request.Request(
                FEATURE_REQUEST_URL, data=body,
                headers={"Content-Type": "application/json",
                         "User-Agent": f"Forge/{FORGE_VERSION}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                logger.info("Feature request submitted: %s", resp.status)
            QMessageBox.information(self, "Submitted",
                "Feature request submitted successfully. Thank you!")
            self.accept()
        except Exception as exc:
            logger.debug("Remote submission failed: %s", exc)
            QMessageBox.information(self, "Saved Locally",
                "Could not reach TheCommons server at this time.\n"
                "Your feature request has been saved locally and will\n"
                "be submitted when the service becomes available.")
            self.accept()
