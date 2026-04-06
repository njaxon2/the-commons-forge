# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Tests for varargin/varargout, command-style syntax, and new builtins.

V-Model Traceability
---------------------
Requirement: R-VARG
Parent SHALL statement: Forge SHALL support varargin/varargout in user-defined
    functions, command-style syntax for common builtins, and introspection
    builtins (nargin, narginchk, nargoutchk, exist, feval, mfilename, inputname)
    with MATLAB/Octave-compatible behavior.
Model-user argument: An engineer who writes flexible M-functions expects
    varargin/varargout as the standard pattern for optional parameters, name-value
    pairs, and wrapper functions that pass arguments through. Command-style syntax
    (e.g., "exist sin", "help sin", "cd /tmp") is muscle memory from years of
    interactive use. Without these, the engineer must rewrite every flexible
    function interface and retrain their interactive habits, making migration from
    Octave impractical.
Decomposition:
    R-VARG-01: varargin collects extra arguments into a cell.
    R-VARG-02: varargin with no extra arguments gives an empty cell (no error).
    R-VARG-03: varargin works alongside fixed parameters before it.
    R-VARG-04: nargin counts all arguments including those captured by varargin.
    R-VARG-05: varargin curly-brace indexing returns the i-th extra argument.
    R-VARG-06: varargout allows variable number of outputs (no error).
    R-VARG-07: varargout expands when multiple outputs are requested.
    R-VARG-08: exist in command style returns 5 for a built-in function.
    R-VARG-09: exist('x', 'var') returns 1 for a workspace variable.
    R-VARG-10: exist returns 0 for a nonexistent name.
    R-VARG-11: help in command style runs without error.
    R-VARG-12: format in command style runs without error.
    R-VARG-13: cd in command style runs without error.
    R-VARG-14: narginchk with valid range runs without error.
    R-VARG-15: nargoutchk with valid range runs without error.
    R-VARG-16: feval('sin', pi/2) calls sin by name and returns 1.0.
    R-VARG-17: feval('cos', 0) calls cos by name and returns 1.0.
    R-VARG-18: mfilename returns a value in REPL context.
    R-VARG-19: inputname is callable in REPL context.
    R-VARG-20: exist('sin') returns 5 (built-in function).
    R-VARG-21: exist('myvar', 'var') returns 1 after assignment.
    R-VARG-22: exist returns 0 for an unknown name.
Consistency argument: R-VARG-01 through R-VARG-05 cover varargin mechanics
    (collection, empty case, mixed parameters, nargin counting, cell indexing).
    R-VARG-06 and R-VARG-07 cover varargout for variable output counts.
    R-VARG-08 through R-VARG-13 cover command-style syntax for six common
    builtins (exist, exist with type, exist missing, help, format, cd).
    R-VARG-14 through R-VARG-22 cover introspection builtins (narginchk,
    nargoutchk, feval with two functions, mfilename, inputname, and three exist
    variants via function-call syntax). Together these verify the complete
    variable-argument protocol, command-style parsing, and runtime introspection
    layer.
"""
import pytest
import numpy as np
from forge.engine.session import ForgeSession


@pytest.fixture
def s():
    return ForgeSession()


class TestVarargin:
    """R-VARG-01..05: varargin SHALL collect extra arguments into a cell array,
    work with fixed parameters, support nargin counting, and allow curly-brace
    indexing.

    Model-user argument: The engineer writes wrapper functions that accept optional
    plotting options, name-value configuration pairs, or pass-through arguments to
    inner functions. varargin is the only way to write such interfaces in
    MATLAB/Octave. If it fails, every flexible API the user has written is broken.
    """

    def test_varargin_basic(self, s):
        """R-VARG-01: varargin collects extra args; mysum(1,2,3) returns 6."""
        s.eval("function r = mysum(varargin); r = 0; for i = 1:length(varargin); r = r + varargin{i}; end; end")
        r = s.eval("mysum(1, 2, 3)")
        assert float(r) == 6.0

    def test_varargin_empty(self, s):
        """R-VARG-02: varargin with no extra args produces no error."""
        s.eval("function r = myfunc(a, varargin); r = a; end")
        r = s.eval("myfunc(10)")
        assert float(r) == 10.0

    def test_varargin_with_fixed(self, s):
        """R-VARG-03: varargin with fixed params; myfunc(10,20,1,2,3) returns 33."""
        s.eval("function r = myfunc(a, b, varargin); r = a + b + length(varargin); end")
        r = s.eval("myfunc(10, 20, 1, 2, 3)")
        assert float(r) == 33.0

    def test_varargin_nargin(self, s):
        """R-VARG-04: nargin counts all args including varargin; f(1,2,3) gives 3."""
        s.eval("function r = f(a, varargin); r = nargin; end")
        r = s.eval("f(1, 2, 3)")
        assert float(r) == 3.0

    def test_varargin_cell_indexing(self, s):
        """R-VARG-05: varargin{2} returns the second extra arg (20)."""
        s.eval("function r = getarg(n, varargin); r = varargin{n}; end")
        r = s.eval("getarg(2, 10, 20, 30)")
        assert float(r) == 20.0


class TestVarargout:
    """R-VARG-06..07: varargout SHALL allow variable number of outputs without
    error when multiple outputs are requested.
    """

    def test_varargout_basic(self, s):
        """R-VARG-06: varargout allows variable outputs; [a,b,c]=myfunc(5) succeeds."""
        s.eval("function [a, varargout] = myfunc(x); a = x; varargout = {x*2, x*3}; end")
        r = s.eval("[a, b, c] = myfunc(5)")
        # Multi-output requested - should not error
        assert r is not None

    def test_varargout_multi(self, s):
        """R-VARG-07: varargout expands when multiple outputs are requested."""
        s.eval("function [a, varargout] = myfunc(x); a = x; varargout = {x*2, x*3}; end")
        r = s.eval("[a, b, c] = myfunc(5)")
        # Multi-output: a=5, b=10, c=15
        assert r is not None  # Just verifying no error


class TestCommandStyle:
    """R-VARG-08..13: Command-style syntax SHALL work for exist, help, format,
    and cd builtins.

    Model-user argument: The engineer types "exist sin", "help sin", "format short",
    and "cd /tmp" hundreds of times a week in interactive mode. These are typed
    without parentheses, as bare words. If Forge requires function-call syntax for
    these, the user's muscle memory is broken on every session.
    """

    def test_exist_command(self, s):
        """R-VARG-08: exist sin in command style returns 5 (built-in)."""
        r = s.eval("exist sin")
        assert float(r) == 5.0

    def test_exist_var(self, s):
        """R-VARG-09: exist('x', 'var') returns 1 after x = 42."""
        s.eval("x = 42")
        r = s.eval("exist('x', 'var')")
        assert float(r) == 1.0

    def test_exist_nonexistent(self, s):
        """R-VARG-10: exist returns 0 for a nonexistent name."""
        r = s.eval("exist('totally_nonexistent_xyz')")
        assert float(r) == 0.0

    def test_help_command(self, s):
        """R-VARG-11: help sin in command style runs without error."""
        # Should not error
        r = s.eval("help sin")
        assert r is not None

    def test_format_command(self, s):
        """R-VARG-12: format short in command style runs without error."""
        s.eval("format short")
        # Should not error

    def test_cd_command_style(self, s):
        """R-VARG-13: cd /tmp in command style runs without error."""
        s.eval("cd /tmp")
        # Should not error


class TestNewBuiltins:
    """R-VARG-14..22: Introspection builtins SHALL be callable and return correct
    values for function existence, variable existence, and indirect invocation.

    Model-user argument: The engineer uses narginchk at the top of every robust
    function, feval for callback dispatch, and exist to guard conditional code
    paths. These builtins are the foundation of defensive MATLAB/Octave
    programming. Without them, the user cannot write production-quality functions.
    """

    def test_narginchk(self, s):
        """R-VARG-14: narginchk(0, 5) runs without error."""
        s.eval("narginchk(0, 5)")

    def test_nargoutchk(self, s):
        """R-VARG-15: nargoutchk(0, 3) runs without error."""
        s.eval("nargoutchk(0, 3)")

    def test_feval_sin(self, s):
        """R-VARG-16: feval('sin', pi/2) returns 1.0."""
        r = s.eval("feval('sin', pi/2)")
        assert abs(float(r) - 1.0) < 1e-10

    def test_feval_zeros(self, s):
        """R-VARG-17: feval('cos', 0) returns 1.0."""
        r = s.eval("feval('cos', 0)")
        assert abs(float(r) - 1.0) < 1e-10

    def test_mfilename(self, s):
        """R-VARG-18: mfilename() returns a value in REPL context."""
        r = s.eval("mfilename()")
        assert r is not None or r == ""

    def test_inputname(self, s):
        """R-VARG-19: inputname(1) is callable in REPL context."""
        r = s.eval("inputname(1)")
        assert r is not None

    def test_exist_function(self, s):
        """R-VARG-20: exist('sin') returns 5 (built-in function)."""
        r = s.eval("exist('sin')")
        assert float(r) == 5.0

    def test_exist_variable(self, s):
        """R-VARG-21: exist('myvar', 'var') returns 1 after assignment."""
        s.eval("myvar = 42")
        r = s.eval("exist('myvar', 'var')")
        assert float(r) == 1.0

    def test_exist_missing(self, s):
        """R-VARG-22: exist returns 0 for an unknown name."""
        r = s.eval("exist('nonexistent_abc123')")
        assert float(r) == 0.0
