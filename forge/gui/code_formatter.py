# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""M-code (Octave / MATLAB) formatter and beautifier
(forge/gui/code_formatter.py)."""

import re


class MCodeFormatter:
    """Format / beautify Octave and MATLAB source code.

    Features
    --------
    * Auto-indent based on block keywords
    * Normalize spacing around operators
    * Remove trailing whitespace
    * Collapse runs of blank lines (max 2 consecutive)
    * Preserve comment indentation relative to code
    """

    # Keywords that OPEN a block  (increase indent after the line)
    _OPEN_KW = {
        "function", "if", "for", "while", "switch",
        "try", "do", "classdef", "properties", "methods",
    }

    # Keywords that CLOSE a block  (decrease indent before the line)
    _CLOSE_KW = {
        "end", "endif", "endfor", "endwhile", "endswitch",
        "endfunction", "end_try_catch",
    }

    # Keywords that are both close-then-open (decrease, print, increase)
    _MID_KW = {
        "else", "elseif", "case", "otherwise", "catch",
    }

    # Operators to normalise spacing around
    _OP_RE = re.compile(
        r'(?<!=)(?<!~)(?<!<)(?<!>)(?<!!)='    # = but not ==, ~=, <=, >=, !=
        r'|==|~=|!=|<=|>=|<|>'
        r'|(?<!\+)\+(?!\+)'                 # + but not ++
        r'|(?<!-)-(?!-)'                        # - but not --
        r'|(?<!\.)\*(?!\.)'                  # * but not .*
        r'|(?<!\.)/'                            # / but not ./
    )

    def __init__(self, tab_size: int = 4):
        self.tab_size = tab_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def format(self, source: str) -> str:
        """Return a formatted copy of *source*."""
        lines = source.split("\n")
        lines = self._normalize_blank_lines(lines)
        lines = self._remove_trailing_whitespace(lines)
        lines = self._normalize_operators(lines)
        lines = self._auto_indent(lines)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_strings_and_comments(line: str) -> str:
        """Return *line* with string literals and comments blanked out
        so keyword / operator detection ignores them."""
        result = []
        i = 0
        n = len(line)
        while i < n:
            ch = line[i]
            if ch == '%':
                # Rest is a comment
                result.append(' ' * (n - i))
                break
            if ch in ("'", '"'):
                quote = ch
                result.append(' ')
                i += 1
                while i < n and line[i] != quote:
                    result.append(' ')
                    i += 1
                if i < n:
                    result.append(' ')
                    i += 1
                continue
            result.append(ch)
            i += 1
        return "".join(result)

    # --- blank lines ---------------------------------------------------

    @staticmethod
    def _normalize_blank_lines(lines: list[str]) -> list[str]:
        out: list[str] = []
        blank_count = 0
        for ln in lines:
            if ln.strip() == "":
                blank_count += 1
                if blank_count <= 2:
                    out.append("")
            else:
                blank_count = 0
                out.append(ln)
        return out

    # --- trailing whitespace -------------------------------------------

    @staticmethod
    def _remove_trailing_whitespace(lines: list[str]) -> list[str]:
        return [ln.rstrip() for ln in lines]

    # --- operator spacing ----------------------------------------------

    def _normalize_operators(self, lines: list[str]) -> list[str]:
        out: list[str] = []
        for line in lines:
            stripped = line.lstrip()
            # Skip pure comment lines
            if stripped.startswith('%'):
                out.append(line)
                continue
            # Skip blank lines
            if stripped == '':
                out.append(line)
                continue

            clean = self._strip_strings_and_comments(line)
            # Find comment start in original line
            comment_start = len(line)
            ci = 0
            cn = len(line)
            while ci < cn:
                ch = line[ci]
                if ch == '%':
                    comment_start = ci
                    break
                if ch in ("'", '"'):
                    quote = ch
                    ci += 1
                    while ci < cn and line[ci] != quote:
                        ci += 1
                    if ci < cn:
                        ci += 1
                    continue
                ci += 1

            code_part = line[:comment_start]
            comment_part = line[comment_start:]

            # Normalise operators only in the code portion
            new_code = self._space_operators(code_part, clean[:comment_start])
            result = new_code.rstrip() + ('  ' + comment_part.strip() if comment_part else '')
            out.append(result.rstrip())
        return out

    def _space_operators(self, code: str, clean: str) -> str:
        """Add spaces around operators in *code*, using *clean*
        (strings blanked) for detection."""
        # We rebuild char-by-char, using the clean version to detect ops.
        # Simple approach: use regex on clean to find op positions,
        # then ensure spaces around those positions in code.
        spans: list[tuple[int, int]] = []
        for m in re.finditer(
            r'(?<!=)(?<!~)(?<!<)(?<!>)(?<!!)=(?!=)'
            r'|==|~=|!=|<=|>='
            r'|(?<!\.)\*(?!\.)'
            r'|(?<!\.)/'
            r'|\+'
            r'|-',
            clean
        ):
            spans.append((m.start(), m.end()))

        if not spans:
            return code

        # Build result by copying code and ensuring spaces around each op
        parts: list[str] = []
        prev_end = 0
        for start, end in spans:
            # Segment before the operator
            before = code[prev_end:start]
            op = code[start:end]
            # Strip trailing space from before, add exactly one space
            before_stripped = before.rstrip()
            if before_stripped:
                parts.append(before_stripped)
                parts.append(' ')
            else:
                parts.append(before)
            parts.append(op)
            parts.append(' ')
            prev_end = end

        # Remaining portion after last op
        after = code[prev_end:]
        after_stripped = after.lstrip()
        if after_stripped:
            parts.append(after_stripped)
        else:
            parts.append(after)

        result = "".join(parts)
        # Clean up any double spaces (but preserve leading indent)
        leading = len(result) - len(result.lstrip())
        body = re.sub(r'  +', ' ', result[leading:])
        return result[:leading] + body

    # --- auto-indent ---------------------------------------------------

    def _first_keyword(self, line: str) -> str | None:
        """Return the first significant keyword on *line*, ignoring
        strings and comments."""
        clean = self._strip_strings_and_comments(line)
        m = re.match(r'\s*(\w+)', clean)
        return m.group(1) if m else None

    def _auto_indent(self, lines: list[str]) -> list[str]:
        indent = 0
        out: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped == '':
                out.append('')
                continue

            kw = self._first_keyword(stripped)

            # Decrease indent for close / mid keywords
            if kw in self._CLOSE_KW or kw in self._MID_KW:
                indent = max(0, indent - 1)

            # Check if line is a pure comment
            is_comment = stripped.startswith('%')

            prefix = ' ' * (self.tab_size * indent)
            out.append(prefix + stripped)

            # Increase indent for open / mid keywords
            if kw in self._OPEN_KW or kw in self._MID_KW:
                indent += 1

        return out
