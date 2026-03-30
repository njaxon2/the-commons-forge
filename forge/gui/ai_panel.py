"""Forge AI Assistant Panel -- Premium feature with V-model context.

Requires active subscription at thecommons.cc. AI interactions are
charged by usage (15% above LLM costs) with deferred monthly billing.
Code context transits to the AI proxy but is NEVER stored.
"""

from PySide6.QtCore import Qt, Signal, QThread, QUrl
from PySide6.QtGui import QFont, QTextCursor, QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit,
    QPushButton, QLabel, QStackedWidget, QFrame, QScrollArea,
    QSizePolicy, QApplication,
)
import json
import urllib.request
import ssl

AI_PROXY_URL = "https://thecommons.cc/api/ai"


class _StreamWorker(QThread):
    """Background thread for streaming AI responses."""
    text_chunk = Signal(str)
    usage_info = Signal(dict)
    error = Signal(str)
    finished_signal = Signal()

    def __init__(self, url, token, payload):
        super().__init__()
        self._url = url
        self._token = token
        self._payload = payload

    def run(self):
        try:
            data = json.dumps(self._payload).encode()
            req = urllib.request.Request(
                f"{self._url}/chat",
                data=data,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
                method="POST",
            )
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
                for line in resp:
                    line = line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    try:
                        msg = json.loads(payload)
                        if msg.get("type") == "text":
                            self.text_chunk.emit(msg["text"])
                        elif msg.get("type") == "usage":
                            self.usage_info.emit(msg)
                        elif msg.get("type") == "error":
                            self.error.emit(msg.get("error", "Unknown error"))
                    except json.JSONDecodeError:
                        pass
        except urllib.error.HTTPError as e:
            body = e.read().decode() if e.readable() else ""
            try:
                detail = json.loads(body).get("detail", str(e))
            except Exception:
                detail = f"HTTP {e.code}: {body[:200]}"
            self.error.emit(detail)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished_signal.emit()


class AIPanel(QWidget):
    """AI Assistant panel -- premium feature requiring active subscription."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._token = None
        self._session = None  # Forge engine session
        self._editor = None   # Editor widget reference
        self._worker = None
        self._conversation = []  # message history
        self._total_cost = 0.0
        self._current_response = ""
        self.setObjectName("AIPanel")
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Stacked widget: login view vs chat view
        self._stack = QStackedWidget()
        layout.addWidget(self._stack)

        # --- Page 0: Login / Subscribe gate ---
        self._login_page = QWidget()
        login_layout = QVBoxLayout(self._login_page)
        login_layout.setAlignment(Qt.AlignCenter)

        badge = QLabel("Forge AI Assistant")
        badge.setAlignment(Qt.AlignCenter)
        badge.setFont(QFont("", 14, QFont.Bold))
        login_layout.addWidget(badge)

        desc = QLabel(
            "AI-powered pair programming with V-model methodology.\n"
            "Premium feature -- requires active subscription.\n\n"
            "* Engineering-focused AI assistant\n"
            "* V-model development guidance\n"
            "* Context-aware code help\n"
            "* Usage-based billing (pay only for what you use)\n\n"
            "Your code transits encrypted but is never stored."
        )
        desc.setAlignment(Qt.AlignCenter)
        desc.setWordWrap(True)
        login_layout.addWidget(desc)

        login_layout.addSpacing(20)

        self._email_input = QLineEdit()
        self._email_input.setPlaceholderText("Email address")
        self._email_input.setMaximumWidth(300)
        login_layout.addWidget(self._email_input, alignment=Qt.AlignCenter)

        self._pass_input = QLineEdit()
        self._pass_input.setPlaceholderText("Password")
        self._pass_input.setEchoMode(QLineEdit.Password)
        self._pass_input.setMaximumWidth(300)
        self._pass_input.returnPressed.connect(self._do_login)
        login_layout.addWidget(self._pass_input, alignment=Qt.AlignCenter)

        self._login_btn = QPushButton("Sign In")
        self._login_btn.setMaximumWidth(300)
        self._login_btn.clicked.connect(self._do_login)
        login_layout.addWidget(self._login_btn, alignment=Qt.AlignCenter)

        self._login_status = QLabel("")
        self._login_status.setAlignment(Qt.AlignCenter)
        self._login_status.setWordWrap(True)
        self._login_status.setStyleSheet("color: #f38ba8;")
        login_layout.addWidget(self._login_status)

        login_layout.addSpacing(10)

        sub_link = QLabel(
            '<a href="https://thecommons.cc/pricing">'
            "Subscribe at thecommons.cc/pricing</a>"
        )
        sub_link.setAlignment(Qt.AlignCenter)
        sub_link.setOpenExternalLinks(True)
        login_layout.addWidget(sub_link)

        self._stack.addWidget(self._login_page)

        # --- Page 1: Chat interface ---
        self._chat_page = QWidget()
        chat_layout = QVBoxLayout(self._chat_page)
        chat_layout.setContentsMargins(4, 4, 4, 4)
        chat_layout.setSpacing(4)

        header = QHBoxLayout()
        header_label = QLabel("Forge AI")
        header_label.setFont(QFont("", 10, QFont.Bold))
        header.addWidget(header_label)
        header.addStretch()

        self._cost_label = QLabel("Session: $0.00")
        self._cost_label.setStyleSheet("color: #a6adc8; font-size: 10px;")
        header.addWidget(self._cost_label)

        self._btn_clear = QPushButton("Clear")
        self._btn_clear.setFixedHeight(24)
        self._btn_clear.clicked.connect(self._clear_conversation)
        header.addWidget(self._btn_clear)

        self._btn_logout = QPushButton("Sign Out")
        self._btn_logout.setFixedHeight(24)
        self._btn_logout.clicked.connect(self._do_logout)
        header.addWidget(self._btn_logout)

        chat_layout.addLayout(header)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        chat_layout.addWidget(sep)

        self._chat_display = QTextEdit()
        self._chat_display.setReadOnly(True)
        self._chat_display.setFont(QFont("Consolas, Monaco, monospace", 10))
        chat_layout.addWidget(self._chat_display, 1)

        self._context_label = QLabel("")
        self._context_label.setStyleSheet("color: #6c7086; font-size: 9px;")
        chat_layout.addWidget(self._context_label)

        input_row = QHBoxLayout()

        self._input = QLineEdit()
        self._input.setPlaceholderText(
            "Ask the AI assistant... (Enter to send)"
        )
        self._input.returnPressed.connect(self._send_message)
        input_row.addWidget(self._input, 1)

        self._btn_send = QPushButton("Send")
        self._btn_send.clicked.connect(self._send_message)
        input_row.addWidget(self._btn_send)

        self._btn_context = QPushButton("+ Context")
        self._btn_context.setToolTip("Include current file as context")
        self._btn_context.setCheckable(True)
        self._btn_context.setChecked(True)
        input_row.addWidget(self._btn_context)

        chat_layout.addLayout(input_row)

        self._stack.addWidget(self._chat_page)
        self._stack.setCurrentIndex(0)

    def set_session(self, session):
        """Set the Forge engine session for context."""
        self._session = session

    def set_editor(self, editor_widget):
        """Set editor widget reference for file context."""
        self._editor = editor_widget

    def _do_login(self):
        """Authenticate with the AI proxy."""
        email = self._email_input.text().strip()
        password = self._pass_input.text().strip()
        if not email or not password:
            self._login_status.setText("Enter email and password")
            return

        self._login_btn.setEnabled(False)
        self._login_status.setText("Signing in...")
        self._login_status.setStyleSheet("color: #a6adc8;")
        QApplication.processEvents()

        try:
            data = json.dumps({"email": email, "password": password}).encode()
            req = urllib.request.Request(
                f"{AI_PROXY_URL}/auth/login",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
                result = json.loads(resp.read())
                self._token = result["token"]
                self._stack.setCurrentIndex(1)
                self._login_status.setText("")
                self._append_system(
                    "AI Assistant connected. V-model methodology active."
                )
                self._append_system(
                    "Your code context is encrypted in transit "
                    "and never stored."
                )
        except urllib.error.HTTPError as e:
            body = e.read().decode() if e.readable() else ""
            try:
                detail = json.loads(body).get("detail", str(e))
            except Exception:
                detail = f"Authentication failed (HTTP {e.code})"
            self._login_status.setText(detail)
            self._login_status.setStyleSheet("color: #f38ba8;")
        except Exception as e:
            self._login_status.setText(
                f"Connection error: {str(e)[:100]}"
            )
            self._login_status.setStyleSheet("color: #f38ba8;")
        finally:
            self._login_btn.setEnabled(True)

    def _do_logout(self):
        """Sign out and return to login page."""
        self._token = None
        self._conversation.clear()
        self._chat_display.clear()
        self._total_cost = 0.0
        self._cost_label.setText("Session: $0.00")
        self._stack.setCurrentIndex(0)

    def _get_file_context(self):
        """Get current file content from the editor (if context enabled)."""
        if not self._btn_context.isChecked():
            return None
        if not self._editor:
            return None
        try:
            editor = self._editor.current_editor()
            if editor is None:
                return None
            text = editor.toPlainText()
            path = getattr(editor, "file_path", "untitled")
            if text and len(text) > 50:
                if len(text) > 20000:
                    text = text[:20000] + "\n... (truncated)"
                self._context_label.setText(
                    f"Context: {path} ({len(text)} chars)"
                )
                return f"File: {path}\n```\n{text}\n```"
            return None
        except Exception:
            return None

    def _send_message(self):
        """Send user message to AI proxy."""
        text = self._input.text().strip()
        if not text or not self._token:
            return
        if self._worker and self._worker.isRunning():
            return

        self._input.clear()
        self._append_user(text)

        self._conversation.append({"role": "user", "content": text})

        context = self._get_file_context()

        payload = {
            "messages": self._conversation[-20:],
            "context": context,
        }

        self._btn_send.setEnabled(False)
        self._append_assistant_start()

        self._worker = _StreamWorker(AI_PROXY_URL, self._token, payload)
        self._worker.text_chunk.connect(self._on_text_chunk)
        self._worker.usage_info.connect(self._on_usage)
        self._worker.error.connect(self._on_error)
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.start()

    def _append_system(self, text):
        """Add system message to chat."""
        self._chat_display.append(
            '<div style="color: #6c7086; font-style: italic; '
            f'margin: 4px 0;">{text}</div>'
        )

    def _append_user(self, text):
        """Add user message to chat."""
        escaped = (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        self._chat_display.append(
            '<div style="color: #89b4fa; margin: 8px 0;">'
            f"<b>You:</b> {escaped}</div>"
        )

    def _append_assistant_start(self):
        """Start assistant message block."""
        self._chat_display.append(
            '<div style="color: #a6e3a1; margin: 8px 0;"><b>AI:</b> '
        )
        self._current_response = ""

    def _on_text_chunk(self, text):
        """Append streaming text chunk."""
        self._current_response += text
        cursor = self._chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self._chat_display.setTextCursor(cursor)
        self._chat_display.ensureCursorVisible()

    def _on_usage(self, info):
        """Handle usage/billing info."""
        cost = info.get("cost_usd", 0)
        self._total_cost += cost
        self._cost_label.setText(f"Session: ${self._total_cost:.3f}")
        tokens_in = info.get("input_tokens", 0)
        tokens_out = info.get("output_tokens", 0)
        self._append_system(
            f"({tokens_in} in / {tokens_out} out tokens"
            f" -- ${cost:.4f})"
        )

    def _on_error(self, error):
        """Handle error from proxy."""
        self._chat_display.append(
            f'<div style="color: #f38ba8;">Error: {error}</div>'
        )
        if (
            "subscription" in error.lower()
            or "token" in error.lower()
            or "401" in error
        ):
            self._do_logout()

    def _on_finished(self):
        """Streaming complete."""
        self._btn_send.setEnabled(True)
        if self._current_response:
            self._conversation.append(
                {"role": "assistant", "content": self._current_response}
            )
        self._chat_display.append("</div>")
        self._input.setFocus()

    def _clear_conversation(self):
        """Clear chat history."""
        self._conversation.clear()
        self._chat_display.clear()
        self._append_system(
            "Conversation cleared. V-model context retained."
        )

    def apply_theme(self, palette=None):
        """Update colors from theme."""
        if palette is None:
            from forge.gui.theme_utils import detect_palette
            palette = detect_palette()
        bg = palette.get("bg0", "#1e1e2e")
        fg = palette.get("fg0", "#cdd6f4")
        bg1 = palette.get("bg1", "#252536")
        accent = palette.get("accent", "#00BCD4")
        border0 = palette.get("border0", "#313145")
        border1 = palette.get("border1", "#44445a")
        bg3 = palette.get("bg3", "#313145")
        self.setStyleSheet(f"""
            #AIPanel {{
                background: {bg};
            }}
            QTextEdit {{
                background: {bg1};
                color: {fg};
                border: 1px solid {border0};
                border-radius: 4px;
            }}
            QLineEdit {{
                background: {bg1};
                color: {fg};
                border: 1px solid {border0};
                border-radius: 4px;
                padding: 6px;
            }}
            QPushButton {{
                background: {bg3};
                color: {fg};
                border: 1px solid {border1};
                border-radius: 4px;
                padding: 4px 12px;
            }}
            QPushButton:hover {{
                background: {accent};
                color: white;
            }}
        """)
