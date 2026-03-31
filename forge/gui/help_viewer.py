# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Enhanced Forge documentation viewer with navigation, function list, and cross-references."""

import re
import html as html_mod
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QTextBrowser,
    QPushButton, QLabel, QCompleter, QToolButton, QSplitter,
    QListWidget, QListWidgetItem, QAbstractItemView,
)
from PySide6.QtGui import QFont


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

# Map categories to user-friendly group names
CATEGORY_LABELS = {
    "trig": "Trigonometry",
    "matrix_create": "Matrix Construction",
    "matrix_info": "Matrix Information",
    "matrix_ops": "Linear Algebra",
    "string": "String Operations",
    "io": "Input/Output",
    "plot": "Plotting",
    "stats": "Statistics",
    "signal": "Signal Processing",
    "poly": "Polynomials",
    "control": "Control Flow",
    "type_check": "Type Checking",
}


def get_see_also(func_name):
    """Get related functions for See Also section."""
    related = set()
    for category, funcs in FUNC_CATEGORIES.items():
        if func_name in funcs:
            related.update(funcs)
    related.discard(func_name)
    return sorted(related)[:10]


def get_category(func_name):
    """Get the category label for a function."""
    for cat, funcs in FUNC_CATEGORIES.items():
        if func_name in funcs:
            return CATEGORY_LABELS.get(cat, cat)
    return None


class FunctionListWidget(QListWidget):
    """Scrollable, filterable list of all available functions."""

    function_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._all_names = []
        self.setAlternatingRowColors(True)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.itemClicked.connect(self._on_click)
        self.setFont(QFont("Consolas", 9))

    def set_functions(self, names):
        """Populate with sorted function names."""
        self._all_names = sorted(names)
        self._populate(self._all_names)

    def filter_list(self, text):
        """Filter the displayed functions by substring match."""
        text = text.strip().lower()
        if not text:
            self._populate(self._all_names)
        else:
            filtered = [n for n in self._all_names if text in n.lower()]
            self._populate(filtered)

    def _populate(self, names):
        self.clear()
        for name in names:
            item = QListWidgetItem(name)
            self.addItem(item)

    def _on_click(self, item):
        if item:
            self.function_selected.emit(item.text())


class HelpViewerWidget(QWidget):
    """Searchable documentation browser with navigation history and function sidebar."""

    def __init__(self, session=None, parent=None):
        super().__init__(parent)
        self.session = session
        self._history = []
        self._history_pos = -1
        self._filter_timer = QTimer()
        self._filter_timer.setSingleShot(True)
        self._filter_timer.setInterval(200)
        self._filter_timer.timeout.connect(self._apply_filter)
        self._build_ui()

    def set_session(self, session):
        self.session = session
        self._update_completer()
        self._update_function_list()

    def _get_html_colors(self):
        """Get colors for HTML content based on current theme."""
        try:
            from forge.gui.theme_utils import detect_palette, is_light_theme
            p = detect_palette()
            light = is_light_theme()
            return {
                "bg0": p.get("bg0", "#1e1e2e"),
                "bg1": p.get("bg1", "#24243a"),
                "bg3": p.get("bg3", "#313145"),
                "fg0": p.get("fg0", "#cdd6f4"),
                "fg2": p.get("fg2", "#a6adc8"),
                "fg3": p.get("fg3", "#6c7086"),
                "border1": p.get("border1", "#44445a"),
                "accent": p.get("accent", "#00BCD4"),
                "info": p.get("info", "#89b4fa"),
                "success": p.get("success", "#a6e3a1"),
                "error": p.get("error", "#f38ba8"),
                "warning": "#cba6f7" if not light else "#7c3aed",
                "light": light,
            }
        except Exception:
            return {
                "bg0": "#1e1e2e", "bg1": "#24243a", "bg3": "#313145",
                "fg0": "#cdd6f4", "fg2": "#a6adc8", "fg3": "#6c7086",
                "border1": "#44445a", "accent": "#00BCD4", "info": "#89b4fa",
                "success": "#a6e3a1", "error": "#f38ba8", "warning": "#cba6f7",
                "light": False,
            }

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
        btn_go.setMinimumWidth(60)
        nav_layout.addWidget(btn_go)

        layout.addLayout(nav_layout)

        # Splitter: function list (left) | content browser (right)
        self._splitter = QSplitter(Qt.Horizontal)

        # Left panel: filter + function list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(2)

        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText("Filter functions...")
        self._filter_edit.textChanged.connect(self._on_filter_changed)
        self._filter_edit.setClearButtonEnabled(True)
        left_layout.addWidget(self._filter_edit)

        self._func_list = FunctionListWidget()
        self._func_list.function_selected.connect(self._on_func_selected)
        left_layout.addWidget(self._func_list)

        self._func_count_label = QLabel("0 functions")
        self._func_count_label.setAlignment(Qt.AlignCenter)
        font = self._func_count_label.font()
        font.setPointSize(8)
        self._func_count_label.setFont(font)
        left_layout.addWidget(self._func_count_label)

        self._splitter.addWidget(left_panel)

        # Right panel: content browser
        self.browser = QTextBrowser()
        self.browser.setOpenLinks(False)
        self.browser.anchorClicked.connect(self._on_link_clicked)
        self.browser.setFont(QFont("Consolas", 10))
        self._splitter.addWidget(self.browser)

        # Set splitter proportions: ~25% list, ~75% content
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 3)
        self._splitter.setSizes([180, 500])

        layout.addWidget(self._splitter)

        self._show_welcome()

    # ------------------------------------------------------------------
    # Function list management
    # ------------------------------------------------------------------

    def _update_function_list(self):
        """Refresh the function list sidebar from the session engine."""
        if self.session and hasattr(self.session, '_engine'):
            names = sorted(self.session._engine.functions.keys())
            self._func_list.set_functions(names)
            self._func_count_label.setText(f"{len(names)} functions")
        else:
            self._func_list.set_functions([])
            self._func_count_label.setText("0 functions")

    def _on_filter_changed(self, text):
        """Debounced filter for the function list."""
        self._filter_timer.start()

    def _apply_filter(self):
        text = self._filter_edit.text()
        self._func_list.filter_list(text)
        count = self._func_list.count()
        self._func_count_label.setText(f"{count} functions")

    def _on_func_selected(self, name):
        """When a function is clicked in the sidebar list."""
        self.search_edit.setText(name)
        self.show_help(name)

    # ------------------------------------------------------------------
    # Welcome / home page
    # ------------------------------------------------------------------

    def _show_welcome(self):
        c = self._get_html_colors()
        title_c = c["info"]
        cat_c = c["warning"]
        fg_c = c["fg0"]
        dim_c = c["fg3"]
        accent_c = c["accent"]
        bg3_c = c["bg3"]

        # Build category sections with function links
        cat_sections = ""
        for cat_key, label in CATEGORY_LABELS.items():
            funcs = FUNC_CATEGORIES.get(cat_key, [])
            links = " &middot; ".join(
                f"<a href='help:{f}' style='color:{accent_c}; text-decoration:none;'>{f}</a>"
                for f in funcs[:8]
            )
            if len(funcs) > 8:
                links += " ..."
            cat_sections += (
                f"<tr>"
                f"<td style='padding:4px 8px; vertical-align:top;'>"
                f"<b style='color:{cat_c};'>{label}</b></td>"
                f"<td style='padding:4px 8px;'>{links}</td>"
                f"</tr>"
            )

        html = (
            f"<h2 style='color:{title_c}; margin-bottom:4px;'>Forge Documentation</h2>"
            f"<p style='color:{fg_c};'>Type a function name in the search bar above, "
            f"click a function in the sidebar, or browse by category below.</p>"
            f"<table cellpadding='0' cellspacing='0' width='100%' "
            f"style='margin-top:8px;'>"
            f"{cat_sections}"
            f"</table>"
            f"<hr style='margin-top:12px;'>"
            f"<div style='background:{bg3_c}; padding:10px; border-radius:6px; "
            f"margin-top:8px;'>"
            f"<p style='color:{dim_c}; font-size:11px; margin:0;'>"
            f"<b>Tips:</b><br>"
            f"&bull; Use <code>help funcname</code> in the command window<br>"
            f"&bull; Right-click a function name in the editor and select Help<br>"
            f"&bull; Press F1 to open this panel<br>"
            f"&bull; Use the filter box above the function list to narrow results</p>"
            f"</div>"
        )
        self.browser.setHtml(html)

    # ------------------------------------------------------------------
    # Search / autocomplete
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Navigation history
    # ------------------------------------------------------------------

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

    def apply_theme(self):
        """Re-apply palette-derived styles after a theme switch."""
        from forge.gui.theme_utils import detect_palette, is_light_theme
        p = detect_palette()
        bg0 = p.get('bg0', '#1e1e2e')
        fg0 = p.get('fg0', '#cdd6f4')
        bg1 = p.get('bg1', '#252536')
        accent = p.get('accent', '#00BCD4')
        if hasattr(self, '_content'):
            css = (
                f"body {{ background: {bg0}; color: {fg0}; "
                f"font-family: sans-serif; padding: 12px; }}"
                f" a {{ color: {accent}; }}"
                f" pre, code {{ background: {bg1}; padding: 4px 8px; "
                f"border-radius: 4px; }}"
            )
            self._content.document().setDefaultStyleSheet(css)


    def show_help(self, func_name: str):
        """Display help and add to history."""
        self._history = self._history[:self._history_pos + 1]
        self._history.append(func_name)
        self._history_pos = len(self._history) - 1
        self._update_nav_buttons()
        self._show_help_no_history(func_name)

    # ------------------------------------------------------------------
    # Core help display
    # ------------------------------------------------------------------

    def _show_help_no_history(self, func_name: str):
        """Display help without modifying history."""
        c = self._get_html_colors()
        err_c = c["error"]
        info_c = c["info"]
        success_c = c["success"]
        bg3_c = c["bg3"]
        bg0_c = c["bg0"]
        fg0_c = c["fg0"]
        fg3_c = c["fg3"]
        border_c = c["border1"]
        warn_c = c["warning"]
        accent_c = c["accent"]

        if not self.session or not hasattr(self.session, '_engine'):
            self.browser.setHtml(
                f"<p style='color:{err_c};'>No session available.</p>"
            )
            return

        funcs = self.session._engine.functions
        if func_name not in funcs:
            # Case-insensitive match
            matches = [k for k in funcs if k.lower() == func_name.lower()]
            if matches:
                func_name = matches[0]
            else:
                # Fuzzy search: substring match
                similar = [k for k in sorted(funcs.keys())
                           if func_name.lower() in k.lower()][:30]
                html = (
                    f"<h3 style='color:{err_c};'>"
                    f"Function '{html_mod.escape(func_name)}' not found</h3>"
                )
                if similar:
                    html += (
                        f"<p style='color:{fg0_c};'>"
                        f"Did you mean one of these ({len(similar)} matches):</p>"
                    )
                    html += (
                        "<div style='column-count:2; column-gap:16px;'>"
                        "<ul style='margin:0; padding-left:20px;'>"
                    )
                    for s in similar:
                        doc = getattr(funcs[s], '__doc__', '') or ''
                        brief = doc.split('\n')[0][:60] if doc else ''
                        html += (
                            f"<li><a href='help:{s}' "
                            f"style='color:{accent_c};'>{s}</a>"
                            f"<span style='color:{fg3_c}; font-size:10px;'>"
                            f" - {html_mod.escape(brief)}</span></li>"
                        )
                    html += "</ul></div>"
                else:
                    html += (
                        f"<p style='color:{fg0_c};'>"
                        f"No similar functions found. "
                        f"Try a different search term.</p>"
                    )
                self.browser.setHtml(html)
                return

        func = funcs[func_name]
        doc = getattr(func, '__doc__', None) or "No documentation available."

        # Format the docstring as rich HTML
        doc_html = self._format_docstring(doc, funcs, c)

        # Category badge
        category = get_category(func_name)
        cat_badge = ""
        if category:
            cat_badge = (
                f" <span style='background:{warn_c}; color:{bg0_c}; "
                f"padding:1px 6px; border-radius:4px; font-size:10px; "
                f"margin-left:6px;'>{category}</span>"
            )

        # See also section
        see_also = get_see_also(func_name)
        see_also_html = ""
        if see_also:
            links = " &middot; ".join(
                f"<a href='help:{s}' style='color:{accent_c}; "
                f"text-decoration:none;'>{s}</a>"
                for s in see_also
            )
            see_also_html = (
                f"<div style='margin-top:12px; padding:10px; "
                f"background:{bg0_c}; "
                f"border-left:3px solid {warn_c}; border-radius:4px;'>"
                f"<b style='color:{warn_c};'>See also:</b> {links}"
                f"</div>"
            )

        # Module info
        module = getattr(func, '__module__', 'forge.engine.builtins')
        if not module:
            module = 'forge.engine.builtins'

        html = (
            f"<h2 style='color:{info_c}; margin-bottom:2px;'>"
            f"{html_mod.escape(func_name)}</h2>"
            f"<div style='margin:4px 0 8px 0;'>"
            f"<span style='background:{bg3_c}; color:{success_c}; "
            f"padding:2px 8px; border-radius:4px; font-size:11px;'>"
            f"{type(func).__name__}</span>"
            f"{cat_badge}</div>"
            f"{doc_html}"
            f"{see_also_html}"
            f"<hr style='border-color:{border_c}; margin-top:12px;'>"
            f"<p style='color:{fg3_c}; font-size:10px;'>"
            f"Module: {html_mod.escape(module)}</p>"
        )
        self.browser.setHtml(html)

    # ------------------------------------------------------------------
    # Docstring formatting
    # ------------------------------------------------------------------

    def _format_docstring(self, doc_text, funcs, colors):
        """Parse and format a docstring into styled HTML.

        Handles:
        - NumPy-style sections (Parameters, Returns, Examples, etc.)
        - Usage/synopsis lines
        - Cross-reference links to other functions
        """
        fg0 = colors["fg0"]
        fg3 = colors["fg3"]
        bg3 = colors["bg3"]
        info = colors["info"]
        accent = colors["accent"]
        warn = colors["warning"]
        border = colors["border1"]

        lines = doc_text.split('\n')
        safe_lines = [html_mod.escape(line) for line in lines]

        # Detect sections (NumPy-style: header followed by --- line)
        sections = []
        current_title = None
        current_lines = []
        i = 0
        while i < len(safe_lines):
            line = safe_lines[i]
            raw = line.strip()
            # Check if next line is a section underline
            if (i + 1 < len(safe_lines)
                    and re.match(r'^-{3,}$', safe_lines[i + 1].strip())
                    and raw and not raw.startswith('-')):
                # Save previous section
                if current_title is not None or current_lines:
                    sections.append((current_title, current_lines))
                current_title = raw
                current_lines = []
                i += 2  # skip underline
                continue
            current_lines.append(line)
            i += 1

        # Save last section
        if current_title is not None or current_lines:
            sections.append((current_title, current_lines))

        # Build HTML from sections
        html_parts = []

        for title, sec_lines in sections:
            if title:
                html_parts.append(
                    f"<h3 style='color:{warn}; margin:12px 0 4px 0; "
                    f"border-bottom:1px solid {border}; "
                    f"padding-bottom:2px;'>{title}</h3>"
                )

            for line in sec_lines:
                stripped = line.strip()

                # Detect parameter lines (name : type)
                param_match = re.match(
                    r'^(\s{2,})(\w+)\s*:\s*(.+)$', line
                )
                if (param_match and title in
                        ('Parameters', 'Returns', 'Raises', 'Attributes')):
                    _indent, pname, ptype = param_match.groups()
                    html_parts.append(
                        f"<div style='margin:4px 0 0 12px;'>"
                        f"<code style='color:{accent}; "
                        f"font-weight:bold;'>{pname}</code>"
                        f" <span style='color:{fg3};'>: {ptype}</span>"
                        f"</div>"
                    )
                    continue

                # Usage/synopsis lines (function calls)
                if (not title
                        and re.match(r'^[A-Za-z_]\w*\s*[=(]', stripped)):
                    linkified = self._linkify_line(stripped, funcs, info)
                    html_parts.append(
                        f"<div style='background:{bg3}; padding:4px 8px; "
                        f"margin:2px 0; border-radius:4px; "
                        f"font-family:Consolas; color:{fg0};'>"
                        f"{linkified}</div>"
                    )
                    continue

                # Indented description continuation
                if re.match(r'^\s{4,}', line) and stripped:
                    html_parts.append(
                        f"<div style='margin-left:24px; color:{fg0}; "
                        f"font-size:12px;'>{stripped}</div>"
                    )
                    continue

                # Regular text line
                if stripped:
                    linkified = self._linkify_line(stripped, funcs, info)
                    html_parts.append(
                        f"<p style='color:{fg0}; margin:3px 0; "
                        f"line-height:1.4;'>{linkified}</p>"
                    )
                elif html_parts:
                    html_parts.append("<div style='height:6px;'></div>")

        return "\n".join(html_parts)

    def _linkify_line(self, text, funcs, link_color):
        """Convert known function names in a line to clickable links.

        Only linkifies words that are actual function names in the engine,
        are longer than 2 characters, and limits replacements per line
        to avoid over-linking common words.
        """
        skip = {
            'all', 'any', 'and', 'end', 'not', 'for', 'the', 'are',
            'see', 'set', 'get', 'may', 'use', 'can', 'has', 'was',
            'did', 'its', 'one', 'two', 'via', 'etc', 'nan', 'inf',
            'true', 'false', 'none', 'type', 'name', 'list', 'input',
            'print', 'error', 'class', 'return', 'break', 'continue',
            'while', 'switch', 'case',
        }
        count = 0
        max_links = 5

        def replacer(m):
            nonlocal count
            word = m.group(0)
            if count >= max_links:
                return word
            if word.lower() in skip:
                return word
            if word in funcs and len(word) > 2:
                count += 1
                return (
                    f"<a href='help:{word}' style='color:{link_color}; "
                    f"text-decoration:none; "
                    f"border-bottom:1px dotted {link_color};'>"
                    f"{word}</a>"
                )
            return word

        return re.sub(r'\b([a-zA-Z_]\w*)\b', replacer, text)
