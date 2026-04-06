# Copyright 2026 The Commons(TM)
# SPDX-License-Identifier: Apache-2.0
"""Polish round 7: numeric-limit builtins, tab completion, continuation.

V-model traceability backfill: R-POL7-01 through R-POL7-04.
"""
import sys
import pytest
import numpy as np


def _val(session, varname):
    """Helper: extract scalar Python value from workspace variable."""
    from forge.engine.types import _unwrap
    ws = session.get_workspace_dict()
    v = _unwrap(ws[varname])
    if isinstance(v, np.ndarray):
        return v.item()
    return v


class TestNumericLimitBuiltins:
    """R-POL7-01: Numeric-limit builtins (flintmax, bitmax, intmax,
    intmin, realmax, realmin, nthroot, nargout, computer) SHALL return
    correct values matching Octave semantics.

    Model-user argument: Engineers writing portable numeric code depend
    on flintmax/realmax/intmax to guard against overflow and precision
    loss. These constants appear in range-checking code, scaling
    algorithms, and unit tests. nthroot is essential for real-valued
    root extraction of negative numbers (e.g., cube root of -8 must
    return -2, not a complex number).

    Decomposition:
      R-POL7-01a: flintmax returns 2^53.
      R-POL7-01b: bitmax is an alias for flintmax.
      R-POL7-01c: intmax('int32') returns 2147483647.
      R-POL7-01d: intmin('int32') returns -2147483648.
      R-POL7-01e: intmax with no argument defaults to int32.
      R-POL7-01f: realmax returns sys.float_info.max.
      R-POL7-01g: realmin returns sys.float_info.min.
      R-POL7-01h: nthroot(-8, 3) returns -2 (real, not complex).
      R-POL7-01i: nthroot(27, 3) returns 3.
      R-POL7-01j: nargout('size') returns a positive integer.
      R-POL7-01k: computer returns a non-empty string.

    Consistency: These eleven sub-requirements cover every numeric-limit
    builtin plus the related introspection functions in this round.
    """

    def setup_method(self):
        from forge.engine.session import ForgeSession
        self.s = ForgeSession()

    # -- flintmax / bitmax ------------------------------------------------
    def test_flintmax_value(self):
        """R-POL7-01a: flintmax returns 2^53."""
        self.s.eval("x = flintmax")
        assert _val(self.s, "x") == 2.0 ** 53

    def test_bitmax_alias(self):
        """R-POL7-01b: bitmax is a legacy alias for flintmax."""
        self.s.eval("a = bitmax; b = flintmax")
        assert _val(self.s, "a") == _val(self.s, "b")

    # -- intmax / intmin --------------------------------------------------
    def test_intmax_int32(self):
        """R-POL7-01c: intmax('int32') returns 2147483647."""
        self.s.eval("x = intmax('int32')")
        assert int(_val(self.s, "x")) == 2147483647

    def test_intmin_int32(self):
        """R-POL7-01d: intmin('int32') returns -2147483648."""
        self.s.eval("x = intmin('int32')")
        assert int(_val(self.s, "x")) == -2147483648

    def test_intmax_default(self):
        """R-POL7-01e: intmax with no argument defaults to int32."""
        self.s.eval("x = intmax")
        assert int(_val(self.s, "x")) == 2147483647

    # -- realmax / realmin ------------------------------------------------
    def test_realmax_value(self):
        """R-POL7-01f: realmax returns sys.float_info.max."""
        self.s.eval("x = realmax")
        assert _val(self.s, "x") == sys.float_info.max

    def test_realmin_value(self):
        """R-POL7-01g: realmin returns sys.float_info.min."""
        self.s.eval("x = realmin")
        assert _val(self.s, "x") == sys.float_info.min

    # -- nthroot ----------------------------------------------------------
    def test_nthroot_negative_cube(self):
        """R-POL7-01h: nthroot(-8, 3) returns -2, not complex."""
        self.s.eval("x = nthroot(-8, 3)")
        assert abs(_val(self.s, "x") - (-2.0)) < 1e-10

    def test_nthroot_positive(self):
        """R-POL7-01i: nthroot(27, 3) returns 3."""
        self.s.eval("x = nthroot(27, 3)")
        assert abs(_val(self.s, "x") - 3.0) < 1e-10

    # -- nargout as function ----------------------------------------------
    def test_nargout_builtin(self):
        """R-POL7-01j: nargout('size') returns a positive integer."""
        self.s.eval("n = nargout('size')")
        assert int(_val(self.s, "n")) >= 1

    # -- computer ---------------------------------------------------------
    def test_computer_returns_string(self):
        """R-POL7-01k: computer returns a non-empty string."""
        self.s.eval("c = computer")
        ws = self.s.get_workspace_dict()
        from forge.engine.types import _unwrap
        val = _unwrap(ws["c"])
        # Could be ndarray wrapping a string or a plain string
        if isinstance(val, np.ndarray):
            val = str(val.item()) if val.ndim == 0 else str(val.flat[0])
        assert isinstance(val, str) and len(val) > 0


class TestTabCompletion:
    """R-POL7-02: The session's function dictionary and workspace SHALL
    be discoverable for tab-completion, including all registered builtins
    and user-defined variables.

    Model-user argument: Engineers working interactively in the command
    window rely on tab completion to discover available functions and
    variables. If newly added builtins (bitmax, flintmax, nthroot) do
    not appear in the completion pool, the engineer will assume they
    are not implemented and write workarounds.

    Decomposition:
      R-POL7-02a: Function dict contains 'sin' and 'size' under 'si' prefix.
      R-POL7-02b: Workspace lists user-defined variables after eval.
      R-POL7-02c: New builtins (bitmax, flintmax, nthroot, intmax, intmin)
                   are in the function dict.

    Consistency: Function discovery, variable discovery, and new-builtin
    registration cover the three aspects of tab-completion readiness.
    """

    def test_function_names_start_with_prefix(self):
        """R-POL7-02a: Function dict contains sin, size under 'si' prefix."""
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        func_names = list(s._engine.functions.keys())
        si_funcs = [n for n in func_names if n.startswith("si")]
        assert "sin" in si_funcs
        assert "size" in si_funcs

    def test_workspace_vars_discoverable(self):
        """R-POL7-02b: Workspace lists variables after eval."""
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        s.eval("sigma = 42")
        ws_names = s._engine.workspace.names()
        assert "sigma" in ws_names

    def test_completion_pool_includes_builtins(self):
        """R-POL7-02c: New builtins are in the function dict."""
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        fns = set(s._engine.functions.keys())
        for name in ("bitmax", "flintmax", "nthroot", "intmax", "intmin"):
            assert name in fns, f"{name} missing from function dict"


class TestMultiLineContinuation:
    """R-POL7-03: The parser SHALL support '...' line-continuation,
    collecting continuation lines into a single logical line before
    evaluation.

    Model-user argument: Engineers writing long expressions or matrix
    literals routinely use '...' to break lines for readability. This
    is a fundamental Octave/MATLAB syntax feature; if it does not work,
    pasted code from existing scripts will fail.

    Decomposition:
      R-POL7-03a: Arithmetic expression with '...' continuation evaluates.
      R-POL7-03b: Matrix literal with '...' continuation evaluates.

    Consistency: Expression and matrix cases cover the two common uses
    of line continuation.
    """

    def setup_method(self):
        from forge.engine.session import ForgeSession
        self.s = ForgeSession()

    def test_dot_continuation(self):
        """R-POL7-03a: x = 1 + ... \\n 2 gives x = 3."""
        self.s.eval("x = 1 + ...\n2")
        assert _val(self.s, "x") == 3.0

    def test_multiline_matrix(self):
        """R-POL7-03b: Multi-line matrix with '...' evaluates correctly."""
        self.s.eval("A = [1 2; ...\n3 4]")
        ws = self.s.get_workspace_dict()
        from forge.engine.types import _unwrap
        val = _unwrap(ws["A"])
        expected = np.array([[1, 2], [3, 4]])
        np.testing.assert_array_equal(val, expected)
