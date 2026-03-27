"""Side-by-side diff viewer for comparing file versions."""

import difflib
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPlainTextEdit, QLabel,
    QDialogButtonBox, QPushButton, QSplitter,
)


class DiffViewer(QDialog):
    """Side-by-side diff viewer for comparing two texts."""

    def __init__(self, text_a, text_b, title_a="Original", title_b="Modified", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Diff Viewer")
        self.setMinimumSize(900, 600)
        self._build_ui(text_a, text_b, title_a, title_b)

    def _build_ui(self, text_a, text_b, title_a, title_b):
        layout = QVBoxLayout(self)

        # Title row
        title_row = QHBoxLayout()
        lbl_a = QLabel(title_a)
        lbl_a.setFont(QFont("Consolas", 11, QFont.Bold))
        lbl_a.setStyleSheet("color: #f38ba8;")
        title_row.addWidget(lbl_a)

        lbl_b = QLabel(title_b)
        lbl_b.setFont(QFont("Consolas", 11, QFont.Bold))
        lbl_b.setStyleSheet("color: #a6e3a1;")
        title_row.addWidget(lbl_b)
        layout.addLayout(title_row)

        # Side-by-side editors
        splitter = QSplitter(Qt.Horizontal)

        self._editor_a = QPlainTextEdit()
        self._editor_a.setReadOnly(True)
        self._editor_a.setFont(QFont("Consolas", 10))
        self._editor_a.setLineWrapMode(QPlainTextEdit.NoWrap)
        splitter.addWidget(self._editor_a)

        self._editor_b = QPlainTextEdit()
        self._editor_b.setReadOnly(True)
        self._editor_b.setFont(QFont("Consolas", 10))
        self._editor_b.setLineWrapMode(QPlainTextEdit.NoWrap)
        splitter.addWidget(self._editor_b)

        layout.addWidget(splitter)

        # Sync scrolling
        self._editor_a.verticalScrollBar().valueChanged.connect(
            self._editor_b.verticalScrollBar().setValue
        )
        self._editor_b.verticalScrollBar().valueChanged.connect(
            self._editor_a.verticalScrollBar().setValue
        )

        # Stats
        lines_a = text_a.split('\n')
        lines_b = text_b.split('\n')
        diff = list(difflib.unified_diff(lines_a, lines_b, lineterm=''))
        additions = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
        deletions = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))

        stats = QLabel(f"  +{additions} additions, -{deletions} deletions")
        stats.setStyleSheet("color: #a6adc8; font-size: 11px; padding: 4px;")
        layout.addWidget(stats)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

        # Populate with diff highlighting
        self._populate_diff(text_a, text_b)

    def _populate_diff(self, text_a, text_b):
        """Populate both editors with diff-highlighted content."""
        lines_a = text_a.split('\n')
        lines_b = text_b.split('\n')

        # Use SequenceMatcher for line-by-line diff
        matcher = difflib.SequenceMatcher(None, lines_a, lines_b)

        result_a = []
        result_b = []
        colors_a = []
        colors_b = []

        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op == 'equal':
                for line in lines_a[i1:i2]:
                    result_a.append(line)
                    colors_a.append(None)
                for line in lines_b[j1:j2]:
                    result_b.append(line)
                    colors_b.append(None)
            elif op == 'replace':
                max_lines = max(i2 - i1, j2 - j1)
                for k in range(max_lines):
                    if i1 + k < i2:
                        result_a.append(lines_a[i1 + k])
                        colors_a.append('#f38ba8')  # red for changed
                    else:
                        result_a.append('')
                        colors_a.append('#45475a')  # grey padding
                    if j1 + k < j2:
                        result_b.append(lines_b[j1 + k])
                        colors_b.append('#a6e3a1')  # green for changed
                    else:
                        result_b.append('')
                        colors_b.append('#45475a')
            elif op == 'delete':
                for line in lines_a[i1:i2]:
                    result_a.append(line)
                    colors_a.append('#f38ba8')
                    result_b.append('')
                    colors_b.append('#45475a')
            elif op == 'insert':
                for line in lines_b[j1:j2]:
                    result_a.append('')
                    colors_a.append('#45475a')
                    result_b.append(line)
                    colors_b.append('#a6e3a1')

        # Set text
        self._editor_a.setPlainText('\n'.join(result_a))
        self._editor_b.setPlainText('\n'.join(result_b))

        # Apply colors
        self._apply_colors(self._editor_a, colors_a)
        self._apply_colors(self._editor_b, colors_b)

    def _apply_colors(self, editor, colors):
        """Apply background colors to lines."""
        cursor = QTextCursor(editor.document())
        block = editor.document().begin()

        extra = []
        for i, color in enumerate(colors):
            if color and block.isValid():
                sel = QPlainTextEdit.ExtraSelection()
                fmt = QTextCharFormat()
                fmt.setBackground(QColor(color))
                fmt.setProperty(0x600, True)  # FullWidthSelection
                cursor = QTextCursor(block)
                cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
                sel.cursor = cursor
                sel.format = fmt
                extra.append(sel)
            block = block.next()

        editor.setExtraSelections(extra)
