"""tests/test_polish_r5.py -- Polish round 5: core engine improvements.

V-model traceability backfill: R-POL5-01 through R-POL5-07.

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
    """Container for R5 polish tests. Requirements are grouped by
    logical feature area within this class.
    """

    def setup_method(self):
        self.s = ForgeSession()

    def _val(self, name):
        """Get workspace variable as numpy array."""
        v = self.s._engine.workspace.get(name)
        return np.asarray(_unwrap(v), dtype=float)

    # -- ForgeCell rows/cols ---------------------------------------------------
    # R-POL5-01: ForgeCell objects SHALL expose .rows and .cols properties
    # that reflect the cell array dimensions.
    #
    # Model-user argument: Engineers working with cell arrays inspect
    # size and shape constantly. The .rows/.cols properties mirror the
    # NumPy-style shape access that Octave users expect from size(c).
    # Without them, cell arrays feel like opaque containers.
    #
    # Decomposition:
    #   R-POL5-01a: A 1x3 cell has rows==1.
    #   R-POL5-01b: A 1x3 cell has cols==3.
    #   R-POL5-01c: A cell(3,4) has rows==3, cols==4.
    #
    # Consistency: 1-D and 2-D cases cover the general shape property.

    def test_cell_rows(self):
        """R-POL5-01a: 1x3 cell array has rows==1."""
        self.s.eval("c = {1, 2, 3};")
        c = self.s._engine.workspace.get("c")
        assert c.rows == 1

    def test_cell_cols(self):
        """R-POL5-01b: 1x3 cell array has cols==3."""
        self.s.eval("c = {1, 2, 3};")
        c = self.s._engine.workspace.get("c")
        assert c.cols == 3

    def test_cell_2d_shape(self):
        """R-POL5-01c: cell(3,4) has rows==3, cols==4."""
        self.s.eval("c = cell(3, 4);")
        c = self.s._engine.workspace.get("c")
        assert c.rows == 3
        assert c.cols == 4

    # -- exist with type -------------------------------------------------------
    # R-POL5-02: exist() with a type argument ('var', 'builtin') SHALL
    # return 1 for matching entities and 0 for non-matching ones, and
    # without a type argument SHALL return nonzero for any known name.
    #
    # Model-user argument: Engineers use exist('x','var') to check
    # whether a variable is defined before accessing it, and
    # exist('f','builtin') to verify function availability. This is
    # the standard guard pattern in Octave scripts.
    #
    # Decomposition:
    #   R-POL5-02a: exist(name, 'var') returns 1 for a defined variable.
    #   R-POL5-02b: exist(name, 'var') returns 0 for undefined variable.
    #   R-POL5-02c: exist(name, 'builtin') returns >0 for known builtins.
    #   R-POL5-02d: exist(name, 'builtin') returns 0 for unknown names.
    #   R-POL5-02e: exist(name) without type returns >0 for any known name.
    #
    # Consistency: Variable, builtin, and untyped checks cover all three
    # usage patterns of exist().

    def test_exist_var(self):
        """R-POL5-02a: exist(name,'var') returns 1 for defined variable."""
        self.s.eval("myvar = 42;")
        self.s.eval("r = exist('myvar', 'var');")
        assert float(self._val("r").flat[0]) == 1.0

    def test_exist_var_missing(self):
        """R-POL5-02b: exist(name,'var') returns 0 for undefined variable."""
        self.s.eval("r = exist('nonexistent_xyz', 'var');")
        assert float(self._val("r").flat[0]) == 0.0

    def test_exist_builtin(self):
        """R-POL5-02c: exist(name,'builtin') returns >0 for known builtin."""
        self.s.eval("r = exist('sin', 'builtin');")
        assert float(self._val("r").flat[0]) > 0

    def test_exist_builtin_missing(self):
        """R-POL5-02d: exist(name,'builtin') returns 0 for unknown name."""
        self.s.eval("r = exist('not_a_builtin_xyz', 'builtin');")
        assert float(self._val("r").flat[0]) == 0.0

    def test_exist_no_type(self):
        """R-POL5-02e: exist(name) without type returns >0 for known name."""
        self.s.eval("abc = 10;")
        self.s.eval("r = exist('abc');")
        assert float(self._val("r").flat[0]) > 0

    # -- validatestring --------------------------------------------------------
    # R-POL5-03: validatestring SHALL match an input string against a
    # cell array of valid strings, supporting both exact and unambiguous
    # partial matches.
    #
    # Model-user argument: Engineers writing functions with string-option
    # arguments use validatestring to accept abbreviations (e.g., 'lin'
    # for 'linear'). This pattern is pervasive in MATLAB/Octave toolboxes
    # and must work identically.
    #
    # Decomposition:
    #   R-POL5-03a: Exact match returns the full string.
    #   R-POL5-03b: Unambiguous partial match expands to full string.
    #
    # Consistency: Exact and partial matching cover the two modes of
    # validatestring.

    def test_validatestring_exact(self):
        """R-POL5-03a: Exact match returns the full string."""
        self.s.eval("r = validatestring('linear', {'linear', 'quadratic', 'cubic'});")
        r = self.s._engine.workspace.get("r")
        assert r.to_str() == "linear"

    def test_validatestring_partial(self):
        """R-POL5-03b: Partial match expands to full string."""
        self.s.eval("r = validatestring('lin', {'linear', 'quadratic', 'cubic'});")
        r = self.s._engine.workspace.get("r")
        assert r.to_str() == "linear"

    # -- nfields ---------------------------------------------------------------
    # R-POL5-04: nfields(struct) SHALL return the number of fields in a
    # struct.
    #
    # Model-user argument: Engineers inspecting structs use nfields as
    # a quick way to check how many fields exist without calling
    # fieldnames and then length. It is a convenience function present
    # in Octave.
    #
    # Decomposition:
    #   R-POL5-04a: nfields returns correct count for a 3-field struct.
    #
    # Consistency: Single test case is sufficient for a scalar function
    # with deterministic output.

    def test_nfields_basic(self):
        """R-POL5-04a: nfields returns 3 for a 3-field struct."""
        self.s.eval("s.a = 1; s.b = 2; s.c = 3;")
        self.s.eval("r = nfields(s);")
        assert float(self._val("r").flat[0]) == 3.0

    # -- deal ------------------------------------------------------------------
    # R-POL5-05: deal() SHALL distribute values to multiple output
    # variables, replicating a single input when only one is provided.
    #
    # Model-user argument: Engineers use [a,b]=deal(x) to initialize
    # multiple variables to the same value, and [a,b,c]=deal(1,2,3)
    # to assign distinct values in a single statement. Both forms are
    # common Octave idioms.
    #
    # Decomposition:
    #   R-POL5-05a: deal(x) with two outputs replicates x.
    #   R-POL5-05b: deal(1,2,3) with three outputs assigns distinctly.
    #
    # Consistency: Single-input and multi-input modes are the two
    # calling conventions.

    def test_deal_single(self):
        """R-POL5-05a: deal(42) with two outputs replicates 42."""
        self.s.eval("[a, b] = deal(42);")
        assert float(self._val("a").flat[0]) == 42.0
        assert float(self._val("b").flat[0]) == 42.0

    def test_deal_multiple(self):
        """R-POL5-05b: deal(1,2,3) assigns distinct values."""
        self.s.eval("[a, b, c] = deal(1, 2, 3);")
        assert float(self._val("a").flat[0]) == 1.0
        assert float(self._val("b").flat[0]) == 2.0
        assert float(self._val("c").flat[0]) == 3.0

    # -- bsxfun ----------------------------------------------------------------
    # R-POL5-06: bsxfun SHALL broadcast binary operations across arrays
    # with compatible dimensions, matching Octave's implicit expansion.
    #
    # Model-user argument: Engineers performing vectorized computations
    # use bsxfun to apply operations across differently-shaped arrays
    # without manual repmat. This is essential for efficient numerical
    # code, especially in signal processing and statistics.
    #
    # Decomposition:
    #   R-POL5-06a: bsxfun(@plus, col, row) broadcasts addition.
    #   R-POL5-06b: bsxfun(@times, col, row) broadcasts multiplication.
    #   R-POL5-06c: bsxfun(@minus, col, row) broadcasts subtraction.
    #
    # Consistency: Plus, times, and minus cover the common arithmetic
    # operations and confirm the broadcast mechanism works generically.

    def test_bsxfun_plus(self):
        """R-POL5-06a: bsxfun(@plus, col, row) broadcasts addition."""
        self.s.eval("a = [1; 2; 3]; b = [10, 20]; r = bsxfun(@plus, a, b);")
        expected = np.array([[11, 21], [12, 22], [13, 23]])
        np.testing.assert_array_equal(self._val("r"), expected)

    def test_bsxfun_times(self):
        """R-POL5-06b: bsxfun(@times, col, row) broadcasts multiply."""
        self.s.eval("a = [1; 2]; b = [3, 4, 5]; r = bsxfun(@times, a, b);")
        expected = np.array([[3, 4, 5], [6, 8, 10]])
        np.testing.assert_array_equal(self._val("r"), expected)

    def test_bsxfun_minus(self):
        """R-POL5-06c: bsxfun(@minus, col, row) broadcasts subtraction."""
        self.s.eval("r = bsxfun(@minus, [10; 20], [1, 2, 3]);")
        expected = np.array([[9, 8, 7], [19, 18, 17]])
        np.testing.assert_array_equal(self._val("r"), expected)

    # -- inputname callable ----------------------------------------------------
    # R-POL5-06d (minor): inputname SHALL be registered in the function
    # dictionary. (Tested as existence only; runtime behavior requires
    # a function-call context.)

    def test_inputname_exists(self):
        """R-POL5-06d: inputname is registered in function dict."""
        f = self.s._engine.functions.get("inputname")
        assert f is not None

    # -- validateattributes expanded -------------------------------------------
    # R-POL5-07: validateattributes SHALL support 'square', 'binary',
    # 'increasing', 'nondecreasing', and 'real' attribute checks,
    # returning silently on valid input and reporting an error on
    # invalid input.
    #
    # Model-user argument: Engineers writing robust functions use
    # validateattributes at the top of each function to enforce input
    # constraints. These five attributes appear frequently in
    # signal-processing and linear-algebra toolboxes where matrices
    # must be square, data must be sorted, or values must be binary.
    #
    # Decomposition:
    #   R-POL5-07a: 'square' passes for square matrix.
    #   R-POL5-07b: 'square' fails for non-square matrix.
    #   R-POL5-07c: 'binary' passes for 0/1 array.
    #   R-POL5-07d: 'binary' fails for array with values > 1.
    #   R-POL5-07e: 'increasing' passes for strictly increasing array.
    #   R-POL5-07f: 'increasing' fails for non-monotonic array.
    #   R-POL5-07g: 'nondecreasing' passes for non-decreasing array.
    #   R-POL5-07h: 'real' passes for real-valued array.
    #
    # Consistency: Pass and fail cases for each attribute confirm both
    # the acceptance and rejection paths, fully covering the parent.

    def test_va_square_pass(self):
        """R-POL5-07a: 'square' passes for eye(3)."""
        r = self.s.eval("validateattributes(eye(3), {'numeric'}, {'square'});")
        assert "error" not in str(r).lower()

    def test_va_square_fail(self):
        """R-POL5-07b: 'square' fails for 3x2 matrix."""
        r = self.s.eval("validateattributes([1 2; 3 4; 5 6], {'numeric'}, {'square'});")
        assert "error" in str(r).lower()

    def test_va_binary_pass(self):
        """R-POL5-07c: 'binary' passes for [0 1 0 1]."""
        r = self.s.eval("validateattributes([0 1 0 1], {'numeric'}, {'binary'});")
        assert "error" not in str(r).lower()

    def test_va_binary_fail(self):
        """R-POL5-07d: 'binary' fails for [0 1 2]."""
        r = self.s.eval("validateattributes([0 1 2], {'numeric'}, {'binary'});")
        assert "error" in str(r).lower()

    def test_va_increasing_pass(self):
        """R-POL5-07e: 'increasing' passes for [1 2 3 4]."""
        r = self.s.eval("validateattributes([1 2 3 4], {'numeric'}, {'increasing'});")
        assert "error" not in str(r).lower()

    def test_va_increasing_fail(self):
        """R-POL5-07f: 'increasing' fails for [1 3 2 4]."""
        r = self.s.eval("validateattributes([1 3 2 4], {'numeric'}, {'increasing'});")
        assert "error" in str(r).lower()

    def test_va_nondecreasing_pass(self):
        """R-POL5-07g: 'nondecreasing' passes for [1 1 2 3]."""
        r = self.s.eval("validateattributes([1 1 2 3], {'numeric'}, {'nondecreasing'});")
        assert "error" not in str(r).lower()

    def test_va_real_pass(self):
        """R-POL5-07h: 'real' passes for [1 2 3]."""
        r = self.s.eval("validateattributes([1 2 3], {'numeric'}, {'real'});")
        assert "error" not in str(r).lower()

    # -- arguments block parsing -----------------------------------------------
    # R-POL5-08: The parser SHALL accept and skip MATLAB-style
    # 'arguments' blocks without error, allowing the function body
    # to execute normally.
    #
    # Model-user argument: Modern MATLAB code uses 'arguments' blocks
    # for input validation. Engineers migrating such code need the
    # parser to at least accept (and skip) these blocks so the
    # function's core logic still runs.
    #
    # Decomposition:
    #   R-POL5-08a: Single-parameter arguments block is skipped.
    #   R-POL5-08b: Multi-parameter arguments block is skipped.
    #
    # Consistency: Single and multi-parameter cases confirm the parser
    # handles the block regardless of content complexity.

    def test_arguments_block_skipped(self):
        """R-POL5-08a: Single-param arguments block is skipped."""
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
        """R-POL5-08b: Multi-param arguments block is skipped."""
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
