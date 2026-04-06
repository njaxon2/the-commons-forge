# Copyright 2026 The Commons (TM)
# SPDX-License-Identifier: Apache-2.0
"""Tests for error handling, eval/evalc, and inputParser (R26 polish).

V&V Traceability (backfill)
===========================
R-POL26-01: Error handling SHALL produce Octave-compatible results for
            division by zero, catchable exceptions with identifiers, and
            out-of-range indexing.

    Model-user argument: An engineer migrating from Octave expects 1/0 to
    yield Inf silently, and try/catch blocks to capture error identifiers
    and messages exactly as Octave does. If these semantics differ, scripts
    ported from Octave will silently produce wrong results or crash
    unexpectedly.

    Decomposition:
      R-POL26-01a: 1/0 returns Inf (IEEE 754 / Octave convention).
      R-POL26-01b: error('id:sub', ...) is catchable with e.identifier.
      R-POL26-01c: Out-of-range indexing is catchable via try/catch.
      R-POL26-01d: error('msg') populates e.message in catch block.

    Consistency: The four sub-requirements cover the error-producing
    expression (01a), structured error identity (01b), runtime bounds
    errors (01c), and plain-text error messages (01d). Together they
    validate the full try/catch/error contract.

R-POL26-02: eval and evalc SHALL execute string expressions in the
            workspace, with evalc capturing displayed output.

    Model-user argument: Octave users rely on eval() to run dynamically
    constructed expressions and evalc() to capture console output into a
    variable for logging or post-processing. Both must work identically
    to Octave for migration scripts to function.

    Decomposition:
      R-POL26-02a: eval('x = 42') sets x in the workspace.
      R-POL26-02b: y = eval('3 + 4') returns the computed value.
      R-POL26-02c: evalc('disp(42)') captures disp output as a string.
      R-POL26-02d: evalc captures fprintf output as well.

    Consistency: Sub-requirements cover workspace side-effects (02a),
    return-value semantics (02b), and output capture for both disp (02c)
    and fprintf (02d), fully exercising eval/evalc behavior.

R-POL26-03: inputParser SHALL support addRequired, addOptional, and
            addParameter with correct default and provided-value semantics.

    Model-user argument: Scientists writing reusable Octave functions use
    inputParser to validate arguments with defaults. Forge must match this
    API so that existing function libraries with inputParser calls work
    without modification.

    Decomposition:
      R-POL26-03a: addRequired stores the provided argument in Results.
      R-POL26-03b: addOptional uses provided value when given.
      R-POL26-03c: addOptional falls back to default when omitted.
      R-POL26-03d: addParameter accepts name-value pair arguments.

    Consistency: The four sub-requirements cover required arguments (03a),
    optional with explicit value (03b), optional with default (03c), and
    name-value parameters (03d), spanning the full inputParser API surface.
"""
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
    """R-POL26-01: Error handling SHALL produce Octave-compatible results for
    division by zero, catchable exceptions with identifiers, and out-of-range
    indexing.

    Model-user argument: An engineer migrating from Octave expects 1/0 to
    yield Inf silently, and try/catch blocks to capture error identifiers
    and messages exactly as Octave does. If these semantics differ, scripts
    ported from Octave will silently produce wrong results or crash
    unexpectedly.

    Decomposition:
      R-POL26-01a: 1/0 returns Inf.
      R-POL26-01b: error('id:sub', ...) is catchable with e.identifier.
      R-POL26-01c: Out-of-range indexing is catchable via try/catch.
      R-POL26-01d: error('msg') populates e.message in catch block.

    Consistency: The four sub-requirements cover the error-producing
    expression (01a), structured error identity (01b), runtime bounds
    errors (01c), and plain-text error messages (01d). Together they
    validate the full try/catch/error contract.
    """

    def test_division_by_zero_gives_inf(self, s):
        """R-POL26-01a: 1/0 SHALL produce Inf (IEEE 754 / Octave convention)."""
        s.eval("x = 1/0")
        assert np.isinf(_scalar(s.workspace.get("x")))

    def test_try_catch_error_with_identifier(self, s):
        """R-POL26-01b: error('id:sub', ...) SHALL be catchable with e.identifier."""
        s.eval('try; error("id:sub", "formatted %s %d", "msg", 42); catch e; r = e.identifier; end')
        r = s.workspace.get("r")
        assert isinstance(r, ForgeChar)
        assert r.to_str() == "id:sub"

    def test_try_catch_index_out_of_range(self, s):
        """R-POL26-01c: Out-of-range indexing SHALL be catchable via try/catch."""
        s.eval('try; A = [1 2]; A(5); catch e; r = "caught"; end')
        r = s.workspace.get("r")
        assert isinstance(r, ForgeChar)
        assert r.to_str() == "caught"

    def test_try_catch_error_message(self, s):
        """R-POL26-01d: error('oops') SHALL populate e.message in catch block."""
        s.eval('try; error("oops"); catch e; r = e.message; end')
        r = s.workspace.get("r")
        assert "oops" in r.to_str()


# -- eval / evalc -------------------------------------------------------------

class TestEvalEvalc:
    """R-POL26-02: eval and evalc SHALL execute string expressions in the
    workspace, with evalc capturing displayed output.

    Model-user argument: Octave users rely on eval() to run dynamically
    constructed expressions and evalc() to capture console output into a
    variable for logging or post-processing. Both must work identically
    to Octave for migration scripts to function.

    Decomposition:
      R-POL26-02a: eval('x = 42') sets x in the workspace.
      R-POL26-02b: y = eval('3 + 4') returns the computed value.
      R-POL26-02c: evalc('disp(42)') captures disp output as a string.
      R-POL26-02d: evalc captures fprintf output as well.

    Consistency: Sub-requirements cover workspace side-effects (02a),
    return-value semantics (02b), and output capture for both disp (02c)
    and fprintf (02d), fully exercising eval/evalc behavior.
    """

    def test_eval_sets_workspace_variable(self, s):
        """R-POL26-02a: eval('x = 42') SHALL set x in the workspace."""
        s.eval('eval("x = 42")')
        assert _scalar(s.workspace.get("x")) == 42

    def test_eval_returns_value(self, s):
        """R-POL26-02b: y = eval('3 + 4') SHALL set y = 7."""
        s.eval('y = eval("3 + 4")')
        assert _scalar(s.workspace.get("y")) == 7

    def test_evalc_captures_disp_output(self, s):
        """R-POL26-02c: evalc('disp(42)') SHALL capture displayed text."""
        s.eval('sv = evalc("disp(42)")')
        sv = s.workspace.get("sv")
        assert isinstance(sv, ForgeChar)
        assert "42" in sv.to_str()

    def test_evalc_captures_fprintf_output(self, s):
        """R-POL26-02d: evalc SHALL capture fprintf output."""
        s.eval('sv = evalc("fprintf(\\\"hello %d\\\", 7)")')
        sv = s.workspace.get("sv")
        text = sv.to_str()
        assert "hello" in text and "7" in text


# -- inputParser ---------------------------------------------------------------

class TestInputParser:
    """R-POL26-03: inputParser SHALL support addRequired, addOptional, and
    addParameter with correct default and provided-value semantics.

    Model-user argument: Scientists writing reusable Octave functions use
    inputParser to validate arguments with defaults. Forge must match this
    API so that existing function libraries with inputParser calls work
    without modification.

    Decomposition:
      R-POL26-03a: addRequired stores the provided argument in Results.
      R-POL26-03b: addOptional uses provided value when given.
      R-POL26-03c: addOptional falls back to default when omitted.
      R-POL26-03d: addParameter accepts name-value pair arguments.

    Consistency: The four sub-requirements cover required arguments (03a),
    optional with explicit value (03b), optional with default (03c), and
    name-value parameters (03d), spanning the full inputParser API surface.
    """

    def test_basic_required_argument(self, s):
        """R-POL26-03a: addRequired SHALL store the provided argument in Results."""
        s.eval('p = inputParser; p.addRequired("x"); p.parse(42)')
        s.eval("rx = p.Results.x")
        assert _scalar(s.workspace.get("rx")) == 42

    def test_optional_argument_provided(self, s):
        """R-POL26-03b: addOptional SHALL use provided value when given."""
        s.eval('p = inputParser; p.addRequired("a"); p.addOptional("b", 99); p.parse(1, 2)')
        s.eval("ra = p.Results.a; rb = p.Results.b")
        assert _scalar(s.workspace.get("ra")) == 1
        assert _scalar(s.workspace.get("rb")) == 2

    def test_optional_argument_default(self, s):
        """R-POL26-03c: addOptional SHALL fall back to default when omitted."""
        s.eval('p = inputParser; p.addRequired("a"); p.addOptional("b", 99); p.parse(5)')
        s.eval("ra = p.Results.a; rb = p.Results.b")
        assert _scalar(s.workspace.get("ra")) == 5
        assert _scalar(s.workspace.get("rb")) == 99

    def test_parameter_name_value_pair(self, s):
        """R-POL26-03d: addParameter SHALL accept name-value pair arguments."""
        s.eval('p = inputParser; p.addRequired("x"); p.addParameter("verbose", 0); p.parse(10, "verbose", 1)')
        s.eval("rx = p.Results.x; rv = p.Results.verbose")
        assert _scalar(s.workspace.get("rx")) == 10
        assert _scalar(s.workspace.get("rv")) == 1
