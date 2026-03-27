"""Enhanced Forge documentation viewer with navigation and cross-references."""

import re
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QTextBrowser,
    QPushButton, QLabel, QCompleter, QToolButton,
)
from PySide6.QtGui import QFont, QIcon


# Cross-reference categories for "See also" links
FUNC_CATEGORIES = {
    "trig": ["sin", "cos", "tan", "asin", "acos", "atan", "atan2",
             "sind", "cosd", "tand", "sinh", "cosh", "tanh"],
    "matrix_create": ["zeros", "ones", "eye", "rand", "randn", "randi",
                      "linspace", "logspace", "diag", "repmat"],
    "matrix_info": ["size", "length", "numel", "ndims", "rows", "columns",
                    "isempty", "isscalar", "isvector", "ismatrix"],
    "matrix_ops": ["inv", "det", "trace", "rank", "norm", "eig", "svd",
                   "lu", "qr", "chol", "pinv", "null", "orth"],
    "string": ["strcmp", "strcmpi", "strcat", "strsplit", "strrep",
               "num2str", "str2num", "sprintf", "upper", "lower"],
    "io": ["fprintf", "sprintf", "fopen", "fclose", "fread", "fwrite",
           "disp", "display", "csvread", "csvwrite"],
    "plot": ["plot", "figure", "xlabel", "ylabel", "title", "legend",
             "grid", "hold", "subplot", "bar", "histogram", "scatter"],
    "stats": ["mean", "median", "std", "var", "min", "max", "sum",
              "prod", "sort", "cumsum", "cumprod", "diff"],
    "signal": ["fft", "ifft", "filter", "conv", "xcorr", "spectrogram",
               "butter", "cheby1", "freqz", "periodogram"],
    "poly": ["poly", "roots", "polyval", "polyfit", "polyder", "polyint",
             "conv", "deconv", "residue"],
    "control": ["for", "while", "if", "switch", "try", "break",
                "continue", "return", "end"],
    "type_check": ["isnumeric", "ischar", "iscell", "isstruct", "islogical",
                   "isfloat", "isinteger", "isa", "class"],
}


def get_see_also(func_name):
    """Get related functions for See Also section."""
    related = set()
    for category, funcs in FUNC_CATEGORIES.items():
        if func_name in funcs:
            related.update(funcs)
    related.discard(func_name)
    return sorted(related)[:10]


class HelpViewerWidget(QWidget):
    """Searchable documentation browser with navigation history."""

    def __init__(self, session=None, parent=None):
        super().__init__(parent)
        self.session = session
        self._history = []
        self._history_pos = -1
        self._build_ui()

    def set_session(self, session):
        self.session = session
        self._update_completer()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Navigation bar
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(2)

        self._btn_back = QToolButton()
        self._btn_back.setText("\u25c0")
        self._btn_back.setToolTip("Back")
        self._btn_back.setFixedSize(28, 28)
        self._btn_back.clicked.connect(self._go_back)
        self._btn_back.setEnabled(False)
        nav_layout.addWidget(self._btn_back)

        self._btn_forward = QToolButton()
        self._btn_forward.setText("\u25b6")
        self._btn_forward.setToolTip("Forward")
        self._btn_forward.setFixedSize(28, 28)
        self._btn_forward.clicked.connect(self._go_forward)
        self._btn_forward.setEnabled(False)
        nav_layout.addWidget(self._btn_forward)

        self._btn_home = QToolButton()
        self._btn_home.setText("\u2302")
        self._btn_home.setToolTip("Home")
        self._btn_home.setFixedSize(28, 28)
        self._btn_home.clicked.connect(self._show_welcome)
        nav_layout.addWidget(self._btn_home)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search functions...")
        self.search_edit.returnPressed.connect(self._on_search)
        nav_layout.addWidget(self.search_edit)

        btn_go = QPushButton("Help")
        btn_go.clicked.connect(self._on_search)
        btn_go.setFixedWidth(50)
        nav_layout.addWidget(btn_go)

        layout.addLayout(nav_layout)

        # Content browser
        self.browser = QTextBrowser()
        self.browser.setOpenLinks(False)
        self.browser.anchorClicked.connect(self._on_link_clicked)
        self.browser.setFont(QFont("Consolas", 10))
        layout.addWidget(self.browser)

        self._show_welcome()

    def _show_welcome(self):
        html = """
        <h2 style='color:#89b4fa;'>Forge Documentation</h2>
        <p>Type a function name above and press Enter, or browse by category:</p>

        <h3 style='color:#cba6f7;'>Categories</h3>
        <table cellpadding='4' cellspacing='0' width='100%'>
        <tr>
            <td><b>Math:</b></td>
            <td><a href='help:sin'>Trigonometry</a> · <a href='help:abs'>Elementary</a> · <a href='help:beta'>Special</a></td>
        </tr>
        <tr>
            <td><b>Matrix:</b></td>
            <td><a href='help:zeros'>Construction</a> · <a href='help:size'>Information</a> · <a href='help:eig'>Linear Algebra</a></td>
        </tr>
        <tr>
            <td><b>Strings:</b></td>
            <td><a href='help:strcmp'>Comparison</a> · <a href='help:sprintf'>Formatting</a></td>
        </tr>
        <tr>
            <td><b>Plotting:</b></td>
            <td><a href='help:plot'>2D Plots</a> · <a href='help:surf'>3D Plots</a> · <a href='help:figure'>Figures</a></td>
        </tr>
        <tr>
            <td><b>Signal:</b></td>
            <td><a href='help:fft'>Transforms</a> · <a href='help:filter'>Filtering</a></td>
        </tr>
        <tr>
            <td><b>Stats:</b></td>
            <td><a href='help:mean'>Descriptive</a> · <a href='help:sort'>Sorting</a></td>
        </tr>
        <tr>
            <td><b>I/O:</b></td>
            <td><a href='help:fprintf'>File I/O</a> · <a href='help:disp'>Display</a></td>
        </tr>
        </table>

        <hr>
        <p style='color:#6c7086; font-size:11px;'>
            Tip: Use <code>help funcname</code> in the command window,
            or right-click a function name in the editor and select "Help".
        </p>
        """
        self.browser.setHtml(html)

    def _update_completer(self):
        if self.session and hasattr(self.session, '_engine'):
            names = sorted(self.session._engine.functions.keys())
            completer = QCompleter(names)
            completer.setCaseSensitivity(Qt.CaseInsensitive)
            completer.setFilterMode(Qt.MatchContains)
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

    def _go_back(self):
        if self._history_pos > 0:
            self._history_pos -= 1
            name = self._history[self._history_pos]
            self.search_edit.setText(name)
            self._show_help_no_history(name)
            self._update_nav_buttons()

    def _go_forward(self):
        if self._history_pos < len(self._history) - 1:
            self._history_pos += 1
            name = self._history[self._history_pos]
            self.search_edit.setText(name)
            self._show_help_no_history(name)
            self._update_nav_buttons()

    def _update_nav_buttons(self):
        self._btn_back.setEnabled(self._history_pos > 0)
        self._btn_forward.setEnabled(self._history_pos < len(self._history) - 1)

    def show_help(self, func_name: str):
        """Display help and add to history."""
        # Trim history after current position
        self._history = self._history[:self._history_pos + 1]
        self._history.append(func_name)
        self._history_pos = len(self._history) - 1
        self._update_nav_buttons()
        self._show_help_no_history(func_name)

    def _show_help_no_history(self, func_name: str):
        """Display help without modifying history."""
        if not self.session or not hasattr(self.session, '_engine'):
            self.browser.setHtml("<p style='color:#f38ba8;'>No session available.</p>")
            return

        funcs = self.session._engine.functions
        if func_name not in funcs:
            matches = [k for k in funcs if k.lower() == func_name.lower()]
            if matches:
                func_name = matches[0]
            else:
                similar = [k for k in sorted(funcs.keys())
                           if func_name.lower() in k.lower()][:20]
                html = f"<h3 style='color:#f38ba8;'>Function \'{func_name}\' not found</h3>"
                if similar:
                    html += "<p>Did you mean:</p><ul>"
                    for s in similar:
                        html += f"<li><a href='help:{s}'>{s}</a></li>"
                    html += "</ul>"
                else:
                    html += "<p>No similar functions found.</p>"
                self.browser.setHtml(html)
                return

        func = funcs[func_name]
        doc = getattr(func, '__doc__', None) or "No documentation available."

        # Auto-link function references in docstring
        doc_html = self._linkify_doc(doc, funcs)

        # See also section
        see_also = get_see_also(func_name)
        see_also_html = ""
        if see_also:
            links = ", ".join(f"<a href='help:{s}'>{s}</a>" for s in see_also)
            see_also_html = f"""
            <div style='margin-top:12px; padding:8px; background:#1e1e2e; border-radius:4px;'>
                <b style='color:#cba6f7;'>See also:</b> {links}
            </div>
            """

        html = f"""
        <h2 style='color:#89b4fa;'>{func_name}</h2>
        <div style='margin:4px 0;'>
            <span style='background:#313244; color:#a6e3a1; padding:2px 8px;
                         border-radius:4px; font-size:11px;'>
                {type(func).__name__}
            </span>
        </div>
        <pre style='background:#313244; padding:12px; border-radius:6px;
                    color:#cdd6f4; font-family:Consolas; margin-top:8px;
                    white-space:pre-wrap; line-height:1.4;'>{doc_html}</pre>
        {see_also_html}
        <hr style='border-color:#45475a;'>
        <p style='color:#6c7086; font-size:10px;'>
            Module: forge.engine.builtins
        </p>
        """
        self.browser.setHtml(html)

    def _linkify_doc(self, doc_text, funcs):
        """Convert function names in docstring to clickable links."""
        import html
        safe = html.escape(doc_text)
        # Find function-like words and link them
        words = set(re.findall(r'\b([a-zA-Z_]\w*)\b', safe))
        for word in words:
            if word in funcs and len(word) > 2:
                safe = re.sub(
                    rf'\b({re.escape(word)})\b',
                    f"<a href='help:{word}' style='color:#89b4fa;'>{word}</a>",
                    safe,
                    count=3  # Limit replacements to avoid noise
                )
        return safe
