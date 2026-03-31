# Copyright 2026 The Commons (TM)
# SPDX-License-Identifier: Apache-2.0
"""Tests for error handling, eval/evalc, and inputParser (R26 polish)."""
import pytest
import numpy as np
from forge.engine.session import ForgeSession
from forge.engine.types import ForgeArray, _unwrap
from forge.engine.containers import ForgeChar


@pytest.fixture
def s():
    sess = ForgeSession()
    return sess._engine


def _scalar(v):
    arr = _unwrap(v)
    return float(np.asarray(arr).flat[0])


# -- Error handling edge cases ------------------------------------------------

class TestErrorHandling:
    def test_division_by_zero_gives_inf(self, s):
        """1/0 should produce inf (numpy/Octave behaviour), not an exception."""
        s.eval("x = 1/0")
        assert np.isinf(_scalar(s.workspace.get("x")))

    def test_try_catch_error_with_identifier(self, s):
        """error('id:sub', 'msg %s %d', 'hi', 7) should be catchable with e.identifier."""
        s.eval('try; error("id:sub", "formatted %s %d", "msg", 42); catch e; r = e.identifier; end')
        r = s.workspace.get("r")
        assert isinstance(r, ForgeChar)
        assert r.to_str() == "id:sub"

    def test_try_catch_index_out_of_range(self, s):
        """Indexing beyond array bounds should be catchable."""
        s.eval('try; A = [1 2]; A(5); catch e; r = "caught"; end')
        r = s.workspace.get("r")
        assert isinstance(r, ForgeChar)
        assert r.to_str() == "caught"

    def test_try_catch_error_message(self, s):
        """error('oops') should produce a catchable exception with e.message."""
        s.eval('try; error("oops"); catch e; r = e.message; end')
        r = s.workspace.get("r")
        assert "oops" in r.to_str()


# -- eval / evalc -------------------------------------------------------------

class TestEvalEvalc:
    def test_eval_sets_workspace_variable(self, s):
        """eval('x = 42') should set x in the workspace."""
        s.eval('eval("x = 42")')
        assert _scalar(s.workspace.get("x")) == 42

    def test_eval_returns_value(self, s):
        """y = eval('3 + 4') should set y = 7."""
        s.eval('y = eval("3 + 4")')
        assert _scalar(s.workspace.get("y")) == 7

    def test_evalc_captures_disp_output(self, s):
        """evalc('disp(42)') should capture the displayed text."""
        s.eval('sv = evalc("disp(42)")')
        sv = s.workspace.get("sv")
        assert isinstance(sv, ForgeChar)
        assert "42" in sv.to_str()

    def test_evalc_captures_fprintf_output(self, s):
        """evalc should capture fprintf output too."""
        s.eval('sv = evalc("fprintf(\\\"hello %d\\\", 7)")')
        sv = s.workspace.get("sv")
        text = sv.to_str()
        assert "hello" in text and "7" in text


# -- inputParser ---------------------------------------------------------------

class TestInputParser:
    def test_basic_required_argument(self, s):
        """inputParser with one required arg should store it in Results."""
        s.eval('p = inputParser; p.addRequired("x"); p.parse(42)')
        s.eval("rx = p.Results.x")
        assert _scalar(s.workspace.get("rx")) == 42

    def test_optional_argument_provided(self, s):
        """addOptional with default, then parse with value should use provided."""
        s.eval('p = inputParser; p.addRequired("a"); p.addOptional("b", 99); p.parse(1, 2)')
        s.eval("ra = p.Results.a; rb = p.Results.b")
        assert _scalar(s.workspace.get("ra")) == 1
        assert _scalar(s.workspace.get("rb")) == 2

    def test_optional_argument_default(self, s):
        """addOptional with default, parse without extra arg should use default."""
        s.eval('p = inputParser; p.addRequired("a"); p.addOptional("b", 99); p.parse(5)')
        s.eval("ra = p.Results.a; rb = p.Results.b")
        assert _scalar(s.workspace.get("ra")) == 5
        assert _scalar(s.workspace.get("rb")) == 99

    def test_parameter_name_value_pair(self, s):
        """addParameter with name-value pair parsing."""
        s.eval('p = inputParser; p.addRequired("x"); p.addParameter("verbose", 0); p.parse(10, "verbose", 1)')
        s.eval("rx = p.Results.x; rv = p.Results.verbose")
        assert _scalar(s.workspace.get("rx")) == 10
        assert _scalar(s.workspace.get("rv")) == 1
