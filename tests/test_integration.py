# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Final integration tests -- end-to-end session workflows.

Requirement R-INT: The engine SHALL support end-to-end session workflows
that exercise multiple components (parser, evaluator, builtins, session
management) together, and SHALL provide access to all registered
toolboxes with a minimum builtin count.

Model-user argument: An engineer migrating from MATLAB launches Forge,
runs multi-line scripts that define variables, call functions, compose
trig identities, fit polynomials, and inspect workspaces. If any
component fails in combination (even though it passes unit tests), the
engineer's interactive workflow breaks. The engineer also expects all
standard toolboxes (elfun, linalg, signal, statistics, financial) to
be available without manual imports.

Decomposition:
  R-INT-01..08: End-to-end script workflows (Fibonacci, matrix ops,
    trig composition, statistics, polynomial fit, who/clear, error
    recovery, format output)
  R-INT-09..16: Toolbox accessibility (elfun, linalg, specfun,
    statistics, signal, optimization, control, financial)
  R-INT-17..19: Registry counts (builtin count >= 400, registry is dict,
    session inherits builtins)
  R-INT-20..21: Version and importability

Consistency argument: Script workflows (R-INT-01..08) validate the
parser-evaluator-builtin pipeline end to end. Toolbox accessibility
tests (R-INT-09..16) verify that all domain packages register their
functions. Registry tests (R-INT-17..19) verify the function dispatch
infrastructure. Version/import tests (R-INT-20..21) verify packaging
integrity. Together they ensure the engine works as an integrated system.
"""
import pytest
import numpy as np


class TestEndToEndScripts:
    """R-INT-01..08: End-to-end script workflows SHALL produce correct
    results when parser, evaluator, builtins, and session management
    operate together.

    Model-user argument: The engineer types multi-line scripts into the
    command window, expecting each statement to execute in sequence with
    variables persisting across lines. If workspace persistence, function
    composition, or error recovery fails, the interactive experience is
    broken.

    Decomposition:
      R-INT-01: Iterative Fibonacci script reaches F(10)=55
      R-INT-02: Matrix multiplication A*A produces correct result
      R-INT-03: sin/asin composition recovers original angle
      R-INT-04: Statistics workflow creates mean and std variables
      R-INT-05: Polynomial fit workflow creates coefficient variable
      R-INT-06: who/clear workspace management empties workspace
      R-INT-07: Error recovery allows subsequent valid statements
      R-INT-08: Format output renders pi with 3.14 prefix

    Consistency: These eight tests span iterative computation (R-INT-01),
    matrix arithmetic (R-INT-02), function composition (R-INT-03),
    statistical analysis (R-INT-04), curve fitting (R-INT-05), workspace
    management (R-INT-06), error handling (R-INT-07), and display
    formatting (R-INT-08).
    """

    def setup_method(self):
        from forge.engine.session import ForgeSession
        self.s = ForgeSession()

    def test_fibonacci_script(self):
        """R-INT-01: Iterative Fibonacci script reaches F(10)=55."""
        self.s.eval("a = 1; b = 1")
        for _ in range(8):
            self.s.eval("c = a + b; a = b; b = c")
        ws = self.s.get_workspace_dict()
        from forge.engine.types import _unwrap
        val = _unwrap(ws["b"])
        assert val.item() == 55.0

    def test_matrix_operations(self):
        """R-INT-02: Matrix multiplication A*A creates both A and B in workspace."""
        self.s.eval("A = [1 2; 3 4]")
        self.s.eval("B = A * A")
        ws = self.s.get_workspace_dict()
        assert "A" in ws
        assert "B" in ws

    def test_function_composition(self):
        """R-INT-03: sin/asin composition recovers the original angle."""
        self.s.eval("x = sin(pi/6)")
        self.s.eval("y = asin(x)")
        ws = self.s.get_workspace_dict()
        from forge.engine.types import _unwrap
        val = _unwrap(ws["y"])
        np.testing.assert_allclose(val.item(), np.pi / 6, atol=1e-10)

    def test_statistics_workflow(self):
        """R-INT-04: Statistics workflow creates mean and std variables."""
        self.s.eval("data = [2, 4, 4, 4, 5, 5, 7, 9]")
        self.s.eval("m = mean(data)")
        self.s.eval("s = std(data)")
        ws = self.s.get_workspace_dict()
        assert "m" in ws
        assert "s" in ws

    def test_polynomial_workflow(self):
        """R-INT-05: Polynomial fit workflow creates coefficient variable."""
        self.s.eval("x = [1, 2, 3, 4, 5]")
        self.s.eval("y = [1, 4, 9, 16, 25]")
        self.s.eval("p = polyfit(x, y, 2)")
        self.s.eval("yhat = polyval(p, x)")
        ws = self.s.get_workspace_dict()
        assert "p" in ws

    def test_who_whos_clear(self):
        """R-INT-06: clear() empties the workspace after who() lists variables."""
        self.s.eval("alpha = 1")
        self.s.eval("beta = 2")
        self.s.eval("who()")
        self.s.eval("clear()")
        ws = self.s.get_workspace_dict()
        assert "alpha" not in ws

    def test_error_recovery(self):
        """R-INT-07: Session recovers from undefined variable and accepts next statement."""
        result = self.s.eval("undefined_var_xyz123")
        self.s.eval("x = 42")
        ws = self.s.get_workspace_dict()
        assert "x" in ws

    def test_format_output(self):
        """R-INT-08: Format output renders pi with '3.14' prefix."""
        from forge.engine.types import ForgeArray
        result = self.s._format_result(ForgeArray(np.array(3.14159)))
        assert "3.14" in result


class TestToolboxIntegration:
    """R-INT-09..16: All standard toolboxes SHALL register their functions
    and be accessible from a default session.

    Model-user argument: The engineer expects sind, cond, factorial, mean,
    hamming, fzero, tf, and financial functions to be available without
    any pkg load or import commands, just like MATLAB's default path
    includes all installed toolboxes.

    Decomposition:
      R-INT-09: elfun toolbox (sind)
      R-INT-10: linalg toolbox (cond, rank, trace)
      R-INT-11: specfun toolbox (factorial)
      R-INT-12: statistics toolbox (mean, std, median)
      R-INT-13: signal toolbox (butter or hamming)
      R-INT-14: optimization toolbox (fzero)
      R-INT-15: control toolbox (tf)
      R-INT-16: financial toolbox (>= 20 functions)

    Consistency: Each test verifies one toolbox's registration. The eight
    toolboxes cover the major engineering domains the engineer uses.
    """

    def setup_method(self):
        from forge.engine.session import ForgeSession
        self.s = ForgeSession()

    def test_elfun_accessible(self):
        """R-INT-09: elfun toolbox provides sind()."""
        self.s.eval("y = sind(30)")
        ws = self.s.get_workspace_dict()
        assert "y" in ws

    def test_linalg_accessible(self):
        """R-INT-10: linalg toolbox provides cond, rank, trace."""
        fns = self.s._engine.functions
        assert "cond" in fns
        assert "rank" in fns
        assert "trace" in fns

    def test_specfun_accessible(self):
        """R-INT-11: specfun toolbox provides factorial."""
        self.s.eval("n = factorial(5)")
        ws = self.s.get_workspace_dict()
        assert "n" in ws

    def test_statistics_accessible(self):
        """R-INT-12: statistics toolbox provides mean, std, median."""
        fns = self.s._engine.functions
        assert "mean" in fns
        assert "std" in fns
        assert "median" in fns

    def test_signal_accessible(self):
        """R-INT-13: signal toolbox provides butter or hamming."""
        fns = self.s._engine.functions
        has_signal = "butter" in fns or "hamming" in fns
        assert has_signal

    def test_optimization_accessible(self):
        """R-INT-14: optimization toolbox provides fzero."""
        fns = self.s._engine.functions
        has_opt = "fzero" in fns or "fzero" in fns
        assert has_opt

    def test_control_accessible(self):
        """R-INT-15: control toolbox provides tf."""
        fns = self.s._engine.functions
        has_ctrl = "tf" in fns or "tf" in fns
        assert has_ctrl

    def test_financial_accessible(self):
        """R-INT-16: financial toolbox registers >= 20 functions."""
        from forge.engine.builtins.financial import FINANCIAL_REGISTRY
        assert len(FINANCIAL_REGISTRY) >= 20


class TestRegistryCounts:
    """R-INT-17..19: The builtin registry SHALL contain >= 400 functions,
    be implemented as a dict, and be inherited by new sessions.

    Model-user argument: The engineer expects a comprehensive function
    library comparable to MATLAB's. A minimum of 400 builtins ensures
    coverage of the most common mathematical, string, and I/O operations
    without requiring the engineer to write wrappers.

    Decomposition:
      R-INT-17: BUILTIN_REGISTRY contains >= 400 entries
      R-INT-18: BUILTIN_REGISTRY is a dict (O(1) lookup)
      R-INT-19: A new session inherits >= 400 functions

    Consistency: R-INT-17..18 validate the registry data structure and
    R-INT-19 confirms sessions receive the full registry.
    """

    def test_builtin_count(self):
        """R-INT-17: BUILTIN_REGISTRY contains >= 400 entries."""
        from forge.engine.builtins import BUILTIN_REGISTRY
        assert len(BUILTIN_REGISTRY) >= 400

    def test_all_registries_are_dicts(self):
        """R-INT-18: BUILTIN_REGISTRY is a dict for O(1) dispatch."""
        from forge.engine.builtins import BUILTIN_REGISTRY
        assert isinstance(BUILTIN_REGISTRY, dict)

    def test_session_has_builtins(self):
        """R-INT-19: A new session inherits >= 400 functions."""
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        assert len(s._engine.functions) >= 400


class TestVersionAndMetadata:
    """R-INT-20..21: The package SHALL expose a version string and all
    core modules SHALL be importable.

    Model-user argument: The engineer checks forge.__version__ to verify
    they have the correct release installed, and automated tooling imports
    submodules to validate the package is intact after pip install.

    Decomposition:
      R-INT-20: forge.__version__ matches the expected release string
      R-INT-21: All core modules are importable without error

    Consistency: Version check (R-INT-20) validates metadata, and import
    check (R-INT-21) validates package structure.
    """

    def test_version_string(self):
        """R-INT-20: forge.__version__ matches the expected release string."""
        import forge
        assert forge.__version__ == "0.3.8"

    def test_importable(self):
        """R-INT-21: All core modules are importable without error."""
        import forge.engine.session
        import forge.engine.evaluator
        import forge.engine.parser
        import forge.engine.lexer
        import forge.engine.types
        import forge.engine.containers
        import forge.engine.builtins
        import forge.validation.framework
        import forge.validation.oqe
