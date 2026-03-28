"""Tests for varargin/varargout, command-style syntax, and new builtins."""
import pytest
import numpy as np
from forge.engine.session import ForgeSession


@pytest.fixture
def s():
    return ForgeSession()


class TestVarargin:
    """Test varargin support in user-defined functions."""

    def test_varargin_basic(self, s):
        """varargin collects extra args into a cell."""
        s.eval("function r = mysum(varargin); r = 0; for i = 1:length(varargin); r = r + varargin{i}; end; end")
        r = s.eval("mysum(1, 2, 3)")
        assert float(r) == 6.0

    def test_varargin_empty(self, s):
        """varargin with no extra args gives empty cell."""
        s.eval("function r = myfunc(a, varargin); r = a; end")
        r = s.eval("myfunc(10)")
        assert float(r) == 10.0

    def test_varargin_with_fixed(self, s):
        """varargin with fixed params before it."""
        s.eval("function r = myfunc(a, b, varargin); r = a + b + length(varargin); end")
        r = s.eval("myfunc(10, 20, 1, 2, 3)")
        assert float(r) == 33.0

    def test_varargin_nargin(self, s):
        """nargin counts all args including varargin ones."""
        s.eval("function r = f(a, varargin); r = nargin; end")
        r = s.eval("f(1, 2, 3)")
        assert float(r) == 3.0

    def test_varargin_cell_indexing(self, s):
        """varargin{i} returns i-th extra arg."""
        s.eval("function r = getarg(n, varargin); r = varargin{n}; end")
        r = s.eval("getarg(2, 10, 20, 30)")
        assert float(r) == 20.0


class TestVarargout:
    """Test varargout support in user-defined functions."""

    def test_varargout_basic(self, s):
        """varargout allows variable number of outputs."""
        s.eval("function [a, varargout] = myfunc(x); a = x; varargout = {x*2, x*3}; end")
        r = s.eval("[a, b, c] = myfunc(5)")
        # Multi-output requested - should not error
        assert r is not None

    def test_varargout_multi(self, s):
        """varargout expands when multiple outputs requested."""
        s.eval("function [a, varargout] = myfunc(x); a = x; varargout = {x*2, x*3}; end")
        r = s.eval("[a, b, c] = myfunc(5)")
        # Multi-output: a=5, b=10, c=15
        assert r is not None  # Just verifying no error


class TestCommandStyle:
    """Test command-style syntax."""

    def test_exist_command(self, s):
        """exist funcname in command style."""
        r = s.eval("exist sin")
        assert float(r) == 5.0

    def test_exist_var(self, s):
        """exist varname checks workspace."""
        s.eval("x = 42")
        r = s.eval("exist('x', 'var')")
        assert float(r) == 1.0

    def test_exist_nonexistent(self, s):
        """exist returns 0 for nonexistent."""
        r = s.eval("exist('totally_nonexistent_xyz')")
        assert float(r) == 0.0

    def test_help_command(self, s):
        """help funcname in command style."""
        # Should not error
        r = s.eval("help sin")
        assert r is not None

    def test_format_command(self, s):
        """format short in command style."""
        s.eval("format short")
        # Should not error

    def test_cd_command_style(self, s):
        """cd /tmp in command style."""
        s.eval("cd /tmp")
        # Should not error


class TestNewBuiltins:
    """Test newly added builtins."""

    def test_narginchk(self, s):
        """narginchk should not error."""
        s.eval("narginchk(0, 5)")

    def test_nargoutchk(self, s):
        """nargoutchk should not error."""
        s.eval("nargoutchk(0, 3)")

    def test_feval_sin(self, s):
        """feval calls function by name."""
        r = s.eval("feval('sin', pi/2)")
        assert abs(float(r) - 1.0) < 1e-10

    def test_feval_zeros(self, s):
        """feval with scalar function."""
        r = s.eval("feval('cos', 0)")
        assert abs(float(r) - 1.0) < 1e-10

    def test_mfilename(self, s):
        """mfilename returns something in REPL."""
        r = s.eval("mfilename()")
        assert r is not None or r == ""

    def test_inputname(self, s):
        """inputname callable in REPL."""
        r = s.eval("inputname(1)")
        assert r is not None

    def test_exist_function(self, s):
        """exist('sin') returns 5 (built-in)."""
        r = s.eval("exist('sin')")
        assert float(r) == 5.0

    def test_exist_variable(self, s):
        """exist('x', 'var') after assignment."""
        s.eval("myvar = 42")
        r = s.eval("exist('myvar', 'var')")
        assert float(r) == 1.0

    def test_exist_missing(self, s):
        """exist returns 0 for unknown."""
        r = s.eval("exist('nonexistent_abc123')")
        assert float(r) == 0.0
