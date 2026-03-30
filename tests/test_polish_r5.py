"""tests/test_polish_r5.py -- Polish round 5: core engine improvements.

Covers:
  - ForgeCell rows/cols properties
  - exist with type argument ('builtin', 'var', 'file')
  - validatestring basic usage
  - nfields on a struct
  - deal function (single and multiple args)
  - bsxfun with different operations
  - inputname callable
  - validateattributes expanded checks (square, binary, increasing, etc.)
  - arguments block parsing in the parser
"""

import pytest
import numpy as np
from forge.engine.session import ForgeSession
from forge.engine.types import ForgeArray, _unwrap


class TestPolishR5:
    def setup_method(self):
        self.s = ForgeSession()

    def _val(self, name):
        """Get workspace variable as numpy array."""
        v = self.s._engine.workspace.get(name)
        return np.asarray(_unwrap(v), dtype=float)

    # -- ForgeCell rows/cols ---------------------------------------------------

    def test_cell_rows(self):
        self.s.eval("c = {1, 2, 3};")
        c = self.s._engine.workspace.get("c")
        assert c.rows == 1

    def test_cell_cols(self):
        self.s.eval("c = {1, 2, 3};")
        c = self.s._engine.workspace.get("c")
        assert c.cols == 3

    def test_cell_2d_shape(self):
        self.s.eval("c = cell(3, 4);")
        c = self.s._engine.workspace.get("c")
        assert c.rows == 3
        assert c.cols == 4

    # -- exist with type -------------------------------------------------------

    def test_exist_var(self):
        self.s.eval("myvar = 42;")
        self.s.eval("r = exist('myvar', 'var');")
        assert float(self._val("r").flat[0]) == 1.0

    def test_exist_var_missing(self):
        self.s.eval("r = exist('nonexistent_xyz', 'var');")
        assert float(self._val("r").flat[0]) == 0.0

    def test_exist_builtin(self):
        self.s.eval("r = exist('sin', 'builtin');")
        assert float(self._val("r").flat[0]) > 0

    def test_exist_builtin_missing(self):
        self.s.eval("r = exist('not_a_builtin_xyz', 'builtin');")
        assert float(self._val("r").flat[0]) == 0.0

    def test_exist_no_type(self):
        self.s.eval("abc = 10;")
        self.s.eval("r = exist('abc');")
        assert float(self._val("r").flat[0]) > 0

    # -- validatestring --------------------------------------------------------

    def test_validatestring_exact(self):
        self.s.eval("r = validatestring('linear', {'linear', 'quadratic', 'cubic'});")
        r = self.s._engine.workspace.get("r")
        assert r.to_str() == "linear"

    def test_validatestring_partial(self):
        self.s.eval("r = validatestring('lin', {'linear', 'quadratic', 'cubic'});")
        r = self.s._engine.workspace.get("r")
        assert r.to_str() == "linear"

    # -- nfields ---------------------------------------------------------------

    def test_nfields_basic(self):
        self.s.eval("s.a = 1; s.b = 2; s.c = 3;")
        self.s.eval("r = nfields(s);")
        assert float(self._val("r").flat[0]) == 3.0

    # -- deal ------------------------------------------------------------------

    def test_deal_single(self):
        self.s.eval("[a, b] = deal(42);")
        assert float(self._val("a").flat[0]) == 42.0
        assert float(self._val("b").flat[0]) == 42.0

    def test_deal_multiple(self):
        self.s.eval("[a, b, c] = deal(1, 2, 3);")
        assert float(self._val("a").flat[0]) == 1.0
        assert float(self._val("b").flat[0]) == 2.0
        assert float(self._val("c").flat[0]) == 3.0

    # -- bsxfun ----------------------------------------------------------------

    def test_bsxfun_plus(self):
        self.s.eval("a = [1; 2; 3]; b = [10, 20]; r = bsxfun(@plus, a, b);")
        expected = np.array([[11, 21], [12, 22], [13, 23]])
        np.testing.assert_array_equal(self._val("r"), expected)

    def test_bsxfun_times(self):
        self.s.eval("a = [1; 2]; b = [3, 4, 5]; r = bsxfun(@times, a, b);")
        expected = np.array([[3, 4, 5], [6, 8, 10]])
        np.testing.assert_array_equal(self._val("r"), expected)

    def test_bsxfun_minus(self):
        self.s.eval("r = bsxfun(@minus, [10; 20], [1, 2, 3]);")
        expected = np.array([[9, 8, 7], [19, 18, 17]])
        np.testing.assert_array_equal(self._val("r"), expected)

    # -- inputname callable ----------------------------------------------------

    def test_inputname_exists(self):
        f = self.s._engine.functions.get("inputname")
        assert f is not None

    # -- validateattributes expanded -------------------------------------------

    def test_va_square_pass(self):
        r = self.s.eval("validateattributes(eye(3), {'numeric'}, {'square'});")
        assert "error" not in str(r).lower()

    def test_va_square_fail(self):
        r = self.s.eval("validateattributes([1 2; 3 4; 5 6], {'numeric'}, {'square'});")
        assert "error" in str(r).lower()

    def test_va_binary_pass(self):
        r = self.s.eval("validateattributes([0 1 0 1], {'numeric'}, {'binary'});")
        assert "error" not in str(r).lower()

    def test_va_binary_fail(self):
        r = self.s.eval("validateattributes([0 1 2], {'numeric'}, {'binary'});")
        assert "error" in str(r).lower()

    def test_va_increasing_pass(self):
        r = self.s.eval("validateattributes([1 2 3 4], {'numeric'}, {'increasing'});")
        assert "error" not in str(r).lower()

    def test_va_increasing_fail(self):
        r = self.s.eval("validateattributes([1 3 2 4], {'numeric'}, {'increasing'});")
        assert "error" in str(r).lower()

    def test_va_nondecreasing_pass(self):
        r = self.s.eval("validateattributes([1 1 2 3], {'numeric'}, {'nondecreasing'});")
        assert "error" not in str(r).lower()

    def test_va_real_pass(self):
        r = self.s.eval("validateattributes([1 2 3], {'numeric'}, {'real'});")
        assert "error" not in str(r).lower()

    # -- arguments block parsing -----------------------------------------------

    def test_arguments_block_skipped(self):
        code = """
function y = myfunc(x)
    arguments
        x (1,1) double
    end
    y = x * 2;
end
"""
        self.s.eval(code)
        self.s.eval("r = myfunc(5);")
        assert float(self._val("r").flat[0]) == 10.0

    def test_arguments_block_multi_param(self):
        code = """
function z = adder(a, b)
    arguments
        a double
        b double
    end
    z = a + b;
end
"""
        self.s.eval(code)
        self.s.eval("r = adder(3, 7);")
        assert float(self._val("r").flat[0]) == 10.0
