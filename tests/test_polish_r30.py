# Copyright 2026 The Commons(TM)
# SPDX-License-Identifier: Apache-2.0
"""Polish round 30 – display formatting, workspace management, nargin/nargout."""

import math
import pytest
from forge.engine.session import ForgeSession


@pytest.fixture
def S():
    return ForgeSession()


# ── Display / format ────────────────────────────────────────────────

class TestDisplayFormatting:
    def test_format_long_pi(self, S):
        S.eval("format long")
        r = S.eval("pi")
        assert "3.14159265358979" in r

    def test_format_short_pi(self, S):
        S.eval("format short")
        r = S.eval("pi")
        assert "3.1416" in r

    def test_format_short_e(self, S):
        S.eval("format short e")
        r = S.eval("0.001234")
        assert "1.2340e-03" in r.lower().replace("+", "").replace(" ", "")

    def test_format_long_e(self, S):
        S.eval("format long e")
        r = S.eval("0.001234")
        # Should show more digits in scientific notation
        assert "e-" in r.lower() or "e+" in r.lower()

    def test_disp_matrix(self, S):
        r = S.eval('evalc("disp([1 2; 3 4])")')
        assert "1" in r and "2" in r and "3" in r and "4" in r
        lines = [l for l in r.strip().split("\n") if l.strip()]
        assert len(lines) == 2  # two rows

    def test_disp_string_no_quotes(self, S):
        r = S.eval('evalc("disp(\'hello\')")')
        assert "hello" in r
        # Should NOT have quotes around the output
        assert r.strip() == "hello"


# ── Workspace management ────────────────────────────────────────────

class TestWorkspaceManagement:
    def test_clear_all(self, S):
        S.eval("x = 1; y = 2; z = 3")
        S.eval("clear all")
        r = S.eval("who")
        # No user variables should remain
        assert r is None or r.strip() == ""

    def test_clear_specific(self, S):
        S.eval("a = 1; b = 2; c = 3")
        S.eval("clear a b")
        r = S.eval("who")
        assert "c" in str(r)
        assert "a" not in str(r)
        assert "b" not in str(r)

    def test_clearvars_except(self, S):
        S.eval("x = 10; y = 20; z = 30")
        S.eval("clearvars -except x")
        r = S.eval("who")
        assert "x" in str(r)
        assert "y" not in str(r)
        assert "z" not in str(r)

    def test_clearvars_no_args(self, S):
        S.eval("p = 1; q = 2")
        S.eval("clearvars")
        r = S.eval("who")
        assert r is None or r.strip() == ""

    def test_whos_shows_details(self, S):
        S.eval("m = [1 2; 3 4]")
        r = S.eval('evalc("whos")')
        assert "m" in r
        assert "2x2" in r
        assert "double" in r


# ── nargin / nargout / narginchk ────────────────────────────────────

class TestNarginNargout:
    def test_nargin_optional_args(self, S):
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
        S.eval("function r = getnargin(a, b)\n"
               "  r = nargin;\n"
               "end")
        r = S.eval("getnargin(5)")
        assert "1" in str(r)
        r = S.eval("getnargin(5, 6)")
        assert "2" in str(r)

    def test_narginchk_pass(self, S):
        S.eval("function r = chk(varargin)\n"
               "  narginchk(1, 3);\n"
               "  r = nargin;\n"
               "end")
        r = S.eval("chk(1)")
        assert "1" in str(r)
        r = S.eval("chk(1, 2, 3)")
        assert "3" in str(r)

    def test_narginchk_too_few(self, S):
        S.eval("function r = chk2(varargin)\n"
               "  narginchk(1, 3);\n"
               "  r = 1;\n"
               "end")
        r = S.eval("chk2()")
        assert "error" in str(r).lower()

    def test_nargout_skip_expensive(self, S):
        """Function uses nargout to skip computing second output."""
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
        S.eval("function [a, b] = dualout(x)\n"
               "  a = x;\n"
               "  b = x * 2;\n"
               "end")
        S.eval("[p, q] = dualout(7)")
        rp = S.eval("p")
        rq = S.eval("q")
        assert "7" in str(rp)
        assert "14" in str(rq)
