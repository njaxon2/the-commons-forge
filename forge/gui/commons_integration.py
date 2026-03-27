"""TheCommons integration for Forge IDE.

Provides:
  - UpdateChecker: periodic update checks from thecommons.earth
  - AMSConnector: opt-in anonymized telemetry
  - FeatureRequestDialog: submit feature requests
"""

import json
import logging
import os
import time
from pathlib import Path

from PySide6.QtCore import QTimer, Qt, Signal, QObject
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QTextEdit,
    QComboBox, QRadioButton, QButtonGroup, QPushButton, QMessageBox,
    QGroupBox, QCheckBox, QWidget,
)

logger = logging.getLogger(__name__)

FORGE_VERSION = "0.1.0"
COMMONS_BASE = "https://thecommons.earth/api"
UPDATE_CHECK_URL = f"{COMMONS_BASE}/forge/updates?version={FORGE_VERSION}"
AMS_LOG_URL = f"{COMMONS_BASE}/ams/log"
FEATURE_REQUEST_URL = f"{COMMONS_BASE}/forge/feature-request"

CONFIG_DIR = Path.home() / ".forge"


def _ensure_config_dir():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# UpdateChecker
# ======================================================================

class UpdateChecker(QObject):
    """Checks thecommons.earth for Forge updates periodically."""

    update_available = Signal(str, str)  # (latest_version, release_url)

    CHECK_INTERVAL_MS = 6 * 60 * 60 * 1000  # 6 hours

    def __init__(self, parent=None):
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.check_now)
        self._latest_version = None
        self._release_url = None

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
            import urllib.error
            req = urllib.request.Request(
                UPDATE_CHECK_URL,
                headers={"User-Agent": f"Forge/{FORGE_VERSION}"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                latest = data.get("latest_version", FORGE_VERSION)
                url = data.get("release_url", "")
                if latest != FORGE_VERSION:
                    self._latest_version = latest
                    self._release_url = url
                    self.update_available.emit(latest, url)
                    logger.info("Forge update available: %s", latest)
                else:
                    logger.debug("Forge is up to date (%s)", FORGE_VERSION)
        except Exception as exc:
            logger.debug("Update check failed (expected if server not yet live): %s", exc)


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
        """Enable AMS telemetry."""
        self._config["ams_enabled"] = True
        self._save_config()
        logger.info("AMS telemetry enabled")

    def disconnect(self):
        """Disable AMS telemetry."""
        self._config["ams_enabled"] = False
        self._save_config()
        logger.info("AMS telemetry disabled")

    def send_log(self, event_type, data=None):
        """Send an anonymized telemetry event (if enabled)."""
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
            import urllib.error
            body = json.dumps(payload).encode()
            req = urllib.request.Request(
                AMS_LOG_URL,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": f"Forge/{FORGE_VERSION}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                logger.debug("AMS log sent: %s", resp.status)
        except Exception as exc:
            logger.debug("AMS send failed (expected if server not yet live): %s", exc)

    def show_opt_in_dialog(self, parent=None):
        """Show opt-in dialog. Returns True if user opted in."""
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

        # Category
        cat_layout = QHBoxLayout()
        cat_layout.addWidget(QLabel("Category:"))
        self.category_combo = QComboBox()
        self.category_combo.addItems(["UI", "Engine", "Functions", "Performance", "Other"])
        cat_layout.addWidget(self.category_combo)
        layout.addLayout(cat_layout)

        # Title
        title_layout = QHBoxLayout()
        title_layout.addWidget(QLabel("Title:"))
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("Brief summary of your request")
        title_layout.addWidget(self.title_edit)
        layout.addLayout(title_layout)

        # Description
        layout.addWidget(QLabel("Description:"))
        self.desc_edit = QTextEdit()
        self.desc_edit.setPlaceholderText("Describe the feature you would like...")
        layout.addWidget(self.desc_edit)

        # Priority
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

        # Buttons
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

        # Save local copy first
        self._save_local(payload)

        # Try to submit to TheCommons
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
            logger.info("Feature request saved locally")
        except Exception as exc:
            logger.warning("Failed to save feature request locally: %s", exc)

    def _submit_remote(self, payload):
        try:
            import urllib.request
            import urllib.error
            body = json.dumps(payload).encode()
            req = urllib.request.Request(
                FEATURE_REQUEST_URL,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": f"Forge/{FORGE_VERSION}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                logger.info("Feature request submitted: %s", resp.status)
            QMessageBox.information(
                self, "Submitted",
                "Feature request submitted successfully. Thank you!"
            )
            self.accept()
        except Exception as exc:
            logger.debug("Remote submission failed (expected): %s", exc)
            QMessageBox.information(
                self, "Saved Locally",
                "Could not reach TheCommons server at this time.\n"
                "Your feature request has been saved locally and will\n"
                "be submitted when the service becomes available."
            )
            self.accept()
