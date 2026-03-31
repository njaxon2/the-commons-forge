# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Polish round 7 — numeric-limit builtins, tab completion, continuation."""
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
    """Tests for flintmax, bitmax, intmax, intmin, realmax, realmin."""

    def setup_method(self):
        from forge.engine.session import ForgeSession
        self.s = ForgeSession()

    # -- flintmax / bitmax ------------------------------------------------
    def test_flintmax_value(self):
        self.s.eval("x = flintmax")
        assert _val(self.s, "x") == 2.0 ** 53

    def test_bitmax_alias(self):
        """bitmax is a legacy alias for flintmax."""
        self.s.eval("a = bitmax; b = flintmax")
        assert _val(self.s, "a") == _val(self.s, "b")

    # -- intmax / intmin --------------------------------------------------
    def test_intmax_int32(self):
        self.s.eval("x = intmax('int32')")
        assert int(_val(self.s, "x")) == 2147483647

    def test_intmin_int32(self):
        self.s.eval("x = intmin('int32')")
        assert int(_val(self.s, "x")) == -2147483648

    def test_intmax_default(self):
        """Default (no arg) should return int32 max."""
        self.s.eval("x = intmax")
        assert int(_val(self.s, "x")) == 2147483647

    # -- realmax / realmin ------------------------------------------------
    def test_realmax_value(self):
        self.s.eval("x = realmax")
        assert _val(self.s, "x") == sys.float_info.max

    def test_realmin_value(self):
        self.s.eval("x = realmin")
        assert _val(self.s, "x") == sys.float_info.min

    # -- nthroot ----------------------------------------------------------
    def test_nthroot_negative_cube(self):
        """nthroot(-8, 3) should give -2, not complex."""
        self.s.eval("x = nthroot(-8, 3)")
        assert abs(_val(self.s, "x") - (-2.0)) < 1e-10

    def test_nthroot_positive(self):
        self.s.eval("x = nthroot(27, 3)")
        assert abs(_val(self.s, "x") - 3.0) < 1e-10

    # -- nargout as function ----------------------------------------------
    def test_nargout_builtin(self):
        """nargout('size') should return a positive integer."""
        self.s.eval("n = nargout('size')")
        assert int(_val(self.s, "n")) >= 1

    # -- computer ---------------------------------------------------------
    def test_computer_returns_string(self):
        self.s.eval("c = computer")
        ws = self.s.get_workspace_dict()
        from forge.engine.types import _unwrap
        val = _unwrap(ws["c"])
        # Could be ndarray wrapping a string or a plain string
        if isinstance(val, np.ndarray):
            val = str(val.item()) if val.ndim == 0 else str(val.flat[0])
        assert isinstance(val, str) and len(val) > 0


class TestTabCompletion:
    """Tab completion logic — test that function/var names are discoverable."""

    def test_function_names_start_with_prefix(self):
        """Session function dict should contain sin, size, etc."""
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        func_names = list(s._engine.functions.keys())
        si_funcs = [n for n in func_names if n.startswith("si")]
        assert "sin" in si_funcs
        assert "size" in si_funcs

    def test_workspace_vars_discoverable(self):
        """After eval, workspace should list the variable."""
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        s.eval("sigma = 42")
        ws_names = s._engine.workspace.names()
        assert "sigma" in ws_names

    def test_completion_pool_includes_builtins(self):
        """New builtins (bitmax, flintmax, nthroot) are in the function dict."""
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        fns = set(s._engine.functions.keys())
        for name in ("bitmax", "flintmax", "nthroot", "intmax", "intmin"):
            assert name in fns, f"{name} missing from function dict"


class TestMultiLineContinuation:
    """Legacy '...' line-continuation collects lines properly."""

    def setup_method(self):
        from forge.engine.session import ForgeSession
        self.s = ForgeSession()

    def test_dot_continuation(self):
        """x = 1 + ... \n 2 should give x = 3."""
        self.s.eval("x = 1 + ...\n2")
        assert _val(self.s, "x") == 3.0

    def test_multiline_matrix(self):
        """Multi-line matrix definition."""
        self.s.eval("A = [1 2; ...\n3 4]")
        ws = self.s.get_workspace_dict()
        from forge.engine.types import _unwrap
        val = _unwrap(ws["A"])
        expected = np.array([[1, 2], [3, 4]])
        np.testing.assert_array_equal(val, expected)
