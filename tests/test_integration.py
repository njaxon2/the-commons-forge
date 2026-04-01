# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Final integration tests -- end-to-end session workflows."""
import pytest
import numpy as np


class TestEndToEndScripts:
    def setup_method(self):
        from forge.engine.session import ForgeSession
        self.s = ForgeSession()

    def test_fibonacci_script(self):
        self.s.eval("a = 1; b = 1")
        for _ in range(8):
            self.s.eval("c = a + b; a = b; b = c")
        ws = self.s.get_workspace_dict()
        from forge.engine.types import _unwrap
        val = _unwrap(ws["b"])
        assert val.item() == 55.0

    def test_matrix_operations(self):
        self.s.eval("A = [1 2; 3 4]")
        self.s.eval("B = A * A")
        ws = self.s.get_workspace_dict()
        assert "A" in ws
        assert "B" in ws

    def test_function_composition(self):
        self.s.eval("x = sin(pi/6)")
        self.s.eval("y = asin(x)")
        ws = self.s.get_workspace_dict()
        from forge.engine.types import _unwrap
        val = _unwrap(ws["y"])
        np.testing.assert_allclose(val.item(), np.pi / 6, atol=1e-10)

    def test_statistics_workflow(self):
        self.s.eval("data = [2, 4, 4, 4, 5, 5, 7, 9]")
        self.s.eval("m = mean(data)")
        self.s.eval("s = std(data)")
        ws = self.s.get_workspace_dict()
        assert "m" in ws
        assert "s" in ws

    def test_polynomial_workflow(self):
        self.s.eval("x = [1, 2, 3, 4, 5]")
        self.s.eval("y = [1, 4, 9, 16, 25]")
        self.s.eval("p = polyfit(x, y, 2)")
        self.s.eval("yhat = polyval(p, x)")
        ws = self.s.get_workspace_dict()
        assert "p" in ws

    def test_who_whos_clear(self):
        self.s.eval("alpha = 1")
        self.s.eval("beta = 2")
        self.s.eval("who()")
        self.s.eval("clear()")
        ws = self.s.get_workspace_dict()
        assert "alpha" not in ws

    def test_error_recovery(self):
        result = self.s.eval("undefined_var_xyz123")
        self.s.eval("x = 42")
        ws = self.s.get_workspace_dict()
        assert "x" in ws

    def test_format_output(self):
        from forge.engine.types import ForgeArray
        result = self.s._format_result(ForgeArray(np.array(3.14159)))
        assert "3.14" in result


class TestToolboxIntegration:
    def setup_method(self):
        from forge.engine.session import ForgeSession
        self.s = ForgeSession()

    def test_elfun_accessible(self):
        self.s.eval("y = sind(30)")
        ws = self.s.get_workspace_dict()
        assert "y" in ws

    def test_linalg_accessible(self):
        fns = self.s._engine.functions
        assert "cond" in fns
        assert "rank" in fns
        assert "trace" in fns

    def test_specfun_accessible(self):
        self.s.eval("n = factorial(5)")
        ws = self.s.get_workspace_dict()
        assert "n" in ws

    def test_statistics_accessible(self):
        fns = self.s._engine.functions
        assert "mean" in fns
        assert "std" in fns
        assert "median" in fns

    def test_signal_accessible(self):
        fns = self.s._engine.functions
        has_signal = "butter" in fns or "hamming" in fns
        assert has_signal

    def test_optimization_accessible(self):
        fns = self.s._engine.functions
        has_opt = "fzero" in fns or "fzero" in fns
        assert has_opt

    def test_control_accessible(self):
        fns = self.s._engine.functions
        has_ctrl = "tf" in fns or "tf" in fns
        assert has_ctrl

    def test_financial_accessible(self):
        from forge.engine.builtins.financial import FINANCIAL_REGISTRY
        assert len(FINANCIAL_REGISTRY) >= 20


class TestRegistryCounts:
    def test_builtin_count(self):
        from forge.engine.builtins import BUILTIN_REGISTRY
        assert len(BUILTIN_REGISTRY) >= 400

    def test_all_registries_are_dicts(self):
        from forge.engine.builtins import BUILTIN_REGISTRY
        assert isinstance(BUILTIN_REGISTRY, dict)

    def test_session_has_builtins(self):
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        assert len(s._engine.functions) >= 400


class TestVersionAndMetadata:
    def test_version_string(self):
        import forge
        assert forge.__version__ == "0.2.7"

    def test_importable(self):
        import forge.engine.session
        import forge.engine.evaluator
        import forge.engine.parser
        import forge.engine.lexer
        import forge.engine.types
        import forge.engine.containers
        import forge.engine.builtins
        import forge.validation.framework
        import forge.validation.oqe
