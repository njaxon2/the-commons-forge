"""Forge documentation viewer (forge/gui/help_viewer.py).

Provides a simple help browser for viewing function documentation.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QTextBrowser,
    QPushButton, QLabel, QCompleter,
)
from PySide6.QtGui import QFont


class HelpViewerWidget(QWidget):
    """Searchable documentation browser for Forge built-in functions."""

    def __init__(self, session=None, parent=None):
        super().__init__(parent)
        self.session = session
        self._build_ui()

    def set_session(self, session):
        self.session = session
        self._update_completer()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Search bar
        search_layout = QHBoxLayout()
        search_layout.setSpacing(4)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search functions...")
        self.search_edit.returnPressed.connect(self._on_search)
        search_layout.addWidget(self.search_edit)

        btn_go = QPushButton("Help")
        btn_go.clicked.connect(self._on_search)
        btn_go.setFixedWidth(60)
        search_layout.addWidget(btn_go)
        layout.addLayout(search_layout)

        # Content browser
        self.browser = QTextBrowser()
        self.browser.setOpenLinks(False)
        self.browser.anchorClicked.connect(self._on_link_clicked)
        font = QFont("Consolas", 10)
        self.browser.setFont(font)
        layout.addWidget(self.browser)

        # Show welcome
        self._show_welcome()

    def _show_welcome(self):
        self.browser.setHtml(
            "<h2 style='color:#89b4fa;'>Forge Help Browser</h2>"
            "<p>Type a function name above and press Enter to see its documentation.</p>"
            "<p>Examples: <a href='help:sin'>sin</a>, <a href='help:plot'>plot</a>, "
            "<a href='help:linspace'>linspace</a>, <a href='help:fft'>fft</a></p>"
            "<hr>"
            "<p><b>Keyboard:</b> Type <code>help funcname</code> in the command window</p>"
        )

    def _update_completer(self):
        if self.session and hasattr(self.session, '_engine'):
            names = sorted(self.session._engine.functions.keys())
            completer = QCompleter(names)
            completer.setCaseSensitivity(Qt.CaseInsensitive)
            self.search_edit.setCompleter(completer)

    def _on_search(self):
        name = self.search_edit.text().strip()
        if name:
            self.show_help(name)

    def _on_link_clicked(self, url):
        text = url.toString()
        if text.startswith("help:"):
            name = text[5:]
            self.search_edit.setText(name)
            self.show_help(name)

    def show_help(self, func_name: str):
        """Display help for the given function."""
        if not self.session or not hasattr(self.session, '_engine'):
            self.browser.setHtml(f"<p>No session available.</p>")
            return

        funcs = self.session._engine.functions
        if func_name not in funcs:
            # Try case-insensitive
            matches = [k for k in funcs if k.lower() == func_name.lower()]
            if matches:
                func_name = matches[0]
            else:
                # Show similar names
                similar = [k for k in sorted(funcs.keys())
                           if func_name.lower() in k.lower()][:20]
                html = f"<h3>Function \'{func_name}\' not found</h3>"
                if similar:
                    html += "<p>Did you mean:</p><ul>"
                    for s in similar:
                        html += f"<li><a href='help:{s}'>{s}</a></li>"
                    html += "</ul>"
                self.browser.setHtml(html)
                return

        func = funcs[func_name]
        doc = getattr(func, '__doc__', None) or "No documentation available."

        # Format the help text
        html = f"""
        <h2 style='color:#89b4fa;'>{func_name}</h2>
        <pre style='background:#313244; padding:8px; border-radius:4px;
                    color:#cdd6f4; font-family:Consolas;'>{doc}</pre>
        <hr>
        <p style='color:#6c7086;'>Type: {type(func).__name__}</p>
        """
        self.browser.setHtml(html)
