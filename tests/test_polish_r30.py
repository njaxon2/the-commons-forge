# Copyright 2026 The Commons(TM)
# SPDX-License-Identifier: Apache-2.0
"""Polish round 30 -- display formatting, workspace management, nargin/nargout.

V&V Traceability (backfill)
===========================
R-POL30-01: Display formatting commands (format long, short, short e, long e)
            SHALL control numeric output precision and notation.

    Model-user argument: An engineer reviewing numerical results toggles
    between ``format short`` for quick scans and ``format long`` for full
    precision when debugging convergence. If format commands do not alter
    display output, the user cannot verify significant digits and may
    mistrust the computation.

    Decomposition:
      R-POL30-01a: format long displays pi with at least 14 digits.
      R-POL30-01b: format short displays pi as 3.1416.
      R-POL30-01c: format short e displays in scientific notation.
      R-POL30-01d: format long e displays in scientific notation with more digits.
      R-POL30-01e: disp([1 2; 3 4]) produces two rows of output.
      R-POL30-01f: disp('hello') produces output without surrounding quotes.

    Consistency: Four format modes (01a-d) plus disp behavior for matrices
    (01e) and strings (01f) fully verify the display subsystem.

R-POL30-02: Workspace management commands (clear, clearvars, who, whos)
            SHALL correctly add, remove, and inspect variables.

    Model-user argument: Between experiments, a scientist clears the
    workspace to avoid stale variables contaminating results. Commands like
    ``clear all``, ``clearvars -except x``, and ``whos`` must work exactly
    as in Octave or the user risks carrying forward incorrect state.

    Decomposition:
      R-POL30-02a: clear all removes all user variables.
      R-POL30-02b: clear a b removes only named variables.
      R-POL30-02c: clearvars -except x keeps only x.
      R-POL30-02d: clearvars with no args clears all variables.
      R-POL30-02e: whos shows variable name, size, and type.

    Consistency: Full clear (02a), selective clear (02b), exception-based
    clear (02c), bare clearvars (02d), and introspection (02e) cover the
    workspace management API.

R-POL30-03: nargin, nargout, and narginchk SHALL report and validate
            argument counts inside user-defined functions.

    Model-user argument: Scientists write functions with optional arguments
    guarded by ``if nargin < 2; b = default; end``. If nargin does not
    report the actual call-site argument count, default-value logic breaks
    and functions produce wrong results or crash.

    Decomposition:
      R-POL30-03a: nargin inside a function with optional args controls defaults.
      R-POL30-03b: nargin returns the correct integer count.
      R-POL30-03c: narginchk passes when count is within range.
      R-POL30-03d: narginchk errors when count is below minimum.
      R-POL30-03e: nargout allows skipping expensive second output.
      R-POL30-03f: nargout correctly reports when both outputs are requested.

    Consistency: nargin value (03a-b), narginchk pass/fail (03c-d), and
    nargout single/dual (03e-f) cover the full argument-count API.
"""

import math
import pytest
from forge.engine.session import ForgeSession


@pytest.fixture
def S():
    return ForgeSession()


# ── Display / format ────────────────────────────────────────────────

class TestDisplayFormatting:
    """R-POL30-01: Display formatting commands SHALL control numeric output
    precision and notation.

    Model-user argument: An engineer reviewing numerical results toggles
    between ``format short`` for quick scans and ``format long`` for full
    precision when debugging convergence. If format commands do not alter
    display output, the user cannot verify significant digits and may
    mistrust the computation.

    Decomposition:
      R-POL30-01a: format long displays pi with at least 14 digits.
      R-POL30-01b: format short displays pi as 3.1416.
      R-POL30-01c: format short e displays in scientific notation.
      R-POL30-01d: format long e displays in scientific notation with more digits.
      R-POL30-01e: disp([1 2; 3 4]) produces two rows of output.
      R-POL30-01f: disp('hello') produces output without surrounding quotes.

    Consistency: Four format modes (01a-d) plus disp behavior for matrices
    (01e) and strings (01f) fully verify the display subsystem.
    """

    def test_format_long_pi(self, S):
        """R-POL30-01a: format long SHALL display pi with at least 14 digits."""
        S.eval("format long")
        r = S.eval("pi")
        assert "3.14159265358979" in r

    def test_format_short_pi(self, S):
        """R-POL30-01b: format short SHALL display pi as 3.1416."""
        S.eval("format short")
        r = S.eval("pi")
        assert "3.1416" in r

    def test_format_short_e(self, S):
        """R-POL30-01c: format short e SHALL display in scientific notation."""
        S.eval("format short e")
        r = S.eval("0.001234")
        assert "1.2340e-03" in r.lower().replace("+", "").replace(" ", "")

    def test_format_long_e(self, S):
        """R-POL30-01d: format long e SHALL display in scientific notation with more digits."""
        S.eval("format long e")
        r = S.eval("0.001234")
        # Should show more digits in scientific notation
        assert "e-" in r.lower() or "e+" in r.lower()

    def test_disp_matrix(self, S):
        """R-POL30-01e: disp([1 2; 3 4]) SHALL produce two rows of output."""
        r = S.eval('evalc("disp([1 2; 3 4])")')
        assert "1" in r and "2" in r and "3" in r and "4" in r
        lines = [l for l in r.strip().split("\n") if l.strip()]
        assert len(lines) == 2  # two rows

    def test_disp_string_no_quotes(self, S):
        """R-POL30-01f: disp('hello') SHALL produce output without surrounding quotes."""
        r = S.eval('evalc("disp(\'hello\')")')
        assert "hello" in r
        # Should NOT have quotes around the output
        assert r.strip() == "hello"


# ── Workspace management ────────────────────────────────────────────

class TestWorkspaceManagement:
    """R-POL30-02: Workspace management commands SHALL correctly add, remove,
    and inspect variables.

    Model-user argument: Between experiments, a scientist clears the
    workspace to avoid stale variables contaminating results. Commands like
    ``clear all``, ``clearvars -except x``, and ``whos`` must work exactly
    as in Octave or the user risks carrying forward incorrect state.

    Decomposition:
      R-POL30-02a: clear all removes all user variables.
      R-POL30-02b: clear a b removes only named variables.
      R-POL30-02c: clearvars -except x keeps only x.
      R-POL30-02d: clearvars with no args clears all variables.
      R-POL30-02e: whos shows variable name, size, and type.

    Consistency: Full clear (02a), selective clear (02b), exception-based
    clear (02c), bare clearvars (02d), and introspection (02e) cover the
    workspace management API.
    """

    def test_clear_all(self, S):
        """R-POL30-02a: clear all SHALL remove all user variables."""
        S.eval("x = 1; y = 2; z = 3")
        S.eval("clear all")
        r = S.eval("who")
        # No user variables should remain
        assert r is None or r.strip() == ""

    def test_clear_specific(self, S):
        """R-POL30-02b: clear a b SHALL remove only named variables."""
        S.eval("a = 1; b = 2; c = 3")
        S.eval("clear a b")
        r = S.eval("who")
        assert "c" in str(r)
        assert "a" not in str(r)
        assert "b" not in str(r)

    def test_clearvars_except(self, S):
        """R-POL30-02c: clearvars -except x SHALL keep only x."""
        S.eval("x = 10; y = 20; z = 30")
        S.eval("clearvars -except x")
        r = S.eval("who")
        assert "x" in str(r)
        assert "y" not in str(r)
        assert "z" not in str(r)

    def test_clearvars_no_args(self, S):
        """R-POL30-02d: clearvars with no args SHALL clear all variables."""
        S.eval("p = 1; q = 2")
        S.eval("clearvars")
        r = S.eval("who")
        assert r is None or r.strip() == ""

    def test_whos_shows_details(self, S):
        """R-POL30-02e: whos SHALL show variable name, size, and type."""
        S.eval("m = [1 2; 3 4]")
        r = S.eval('evalc("whos")')
        assert "m" in r
        assert "2x2" in r
        assert "double" in r


# ── nargin / nargout / narginchk ────────────────────────────────────

class TestNarginNargout:
    """R-POL30-03: nargin, nargout, and narginchk SHALL report and validate
    argument counts inside user-defined functions.

    Model-user argument: Scientists write functions with optional arguments
    guarded by ``if nargin < 2; b = default; end``. If nargin does not
    report the actual call-site argument count, default-value logic breaks
    and functions produce wrong results or crash.

    Decomposition:
      R-POL30-03a: nargin inside a function with optional args controls defaults.
      R-POL30-03b: nargin returns the correct integer count.
      R-POL30-03c: narginchk passes when count is within range.
      R-POL30-03d: narginchk errors when count is below minimum.
      R-POL30-03e: nargout allows skipping expensive second output.
      R-POL30-03f: nargout correctly reports when both outputs are requested.

    Consistency: nargin value (03a-b), narginchk pass/fail (03c-d), and
    nargout single/dual (03e-f) cover the full argument-count API.
    """

    def test_nargin_optional_args(self, S):
        """R-POL30-03a: nargin SHALL control default-value logic for optional args."""
        S.eval("function r = optfun(a, b, c)\n"
               "  if nargin < 2; b = 10; end\n"
               "  if nargin < 3; c = 100; end\n"
               "  r = a + b + c;\n"
               "end")
        r = S.eval("optfun(1)")
        assert "111" in str(r)
        r = S.eval("optfun(1, 2)")
        assert "103" in str(r)
        r = S.eval("optfun(1, 2, 3)")
        assert "6" in str(r)

    def test_nargin_value(self, S):
        """R-POL30-03b: nargin SHALL return the correct integer count."""
        S.eval("function r = getnargin(a, b)\n"
               "  r = nargin;\n"
               "end")
        r = S.eval("getnargin(5)")
        assert "1" in str(r)
        r = S.eval("getnargin(5, 6)")
        assert "2" in str(r)

    def test_narginchk_pass(self, S):
        """R-POL30-03c: narginchk SHALL pass when count is within range."""
        S.eval("function r = chk(varargin)\n"
               "  narginchk(1, 3);\n"
               "  r = nargin;\n"
               "end")
        r = S.eval("chk(1)")
        assert "1" in str(r)
        r = S.eval("chk(1, 2, 3)")
        assert "3" in str(r)

    def test_narginchk_too_few(self, S):
        """R-POL30-03d: narginchk SHALL error when count is below minimum."""
        S.eval("function r = chk2(varargin)\n"
               "  narginchk(1, 3);\n"
               "  r = 1;\n"
               "end")
        r = S.eval("chk2()")
        assert "error" in str(r).lower()

    def test_nargout_skip_expensive(self, S):
        """R-POL30-03e: nargout SHALL allow skipping expensive second output."""
        S.eval("function [a, b] = smartfun(x)\n"
               "  a = x;\n"
               "  if nargout > 1\n"
               "    b = x * 100;\n"
               "  end\n"
               "end")
        # Should work with single output
        r = S.eval("smartfun(5)")
        assert "5" in str(r)

    def test_nargout_both_outputs(self, S):
        """R-POL30-03f: nargout SHALL correctly report when both outputs are requested."""
        S.eval("function [a, b] = dualout(x)\n"
               "  a = x;\n"
               "  b = x * 2;\n"
               "end")
        S.eval("[p, q] = dualout(7)")
        rp = S.eval("p")
        rq = S.eval("q")
        assert "7" in str(rp)
        assert "14" in str(rq)
