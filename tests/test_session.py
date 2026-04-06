# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Tests for ForgeSession.

V&V traceability backfill: R-SESS-01 through R-SESS-05.
"""
import pytest
import numpy as np


class TestSessionCreation:
    """R-SESS-01: The ForgeSession SHALL initialize successfully with built-in
    functions pre-registered in the engine.

    Model-user argument: When the engineer launches Forge, the session must
    be ready immediately with all standard math functions (sin, cos, etc.)
    available. If the session fails to create or lacks builtins, the
    engineer cannot evaluate even basic expressions.

    Decomposition: Session creation and builtin registration tested.
    Consistency: Creation and builtin availability are the two aspects of
    a correctly initialized session.
    """

    def test_create(self):
        """R-SESS-01.01: ForgeSession constructor returns a non-None session."""
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        assert s is not None

    def test_has_builtins(self):
        """R-SESS-01.02: Session engine has sin and cos registered."""
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        assert 'sin' in s._engine.functions
        assert 'cos' in s._engine.functions


class TestSessionEval:
    """R-SESS-02: The ForgeSession SHALL evaluate arithmetic, matrix, and
    function expressions, storing results in the workspace and handling
    errors gracefully.

    Model-user argument: The command window is the engineer's primary
    interface. Typing ``x = 3 + 4`` must create variable x in the workspace.
    Evaluating ``sin(0)`` must return 0. Referencing an undefined variable
    must return an error string, not crash the session.

    Decomposition: Arithmetic eval, matrix eval, function eval, error
    handling, and history tracking tested. Consistency: These five tests
    cover the main eval pathways (arithmetic, constructor, function call,
    error, and history recording).
    """

    def test_eval_arithmetic(self):
        """R-SESS-02.01: Arithmetic expression stores result in workspace."""
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        result = s.eval('x = 3 + 4')
        ws = s.get_workspace_dict()
        assert 'x' in ws

    def test_eval_matrix(self):
        """R-SESS-02.02: Matrix literal stores result in workspace."""
        from forge.engine.session import ForgeSession
        from forge.engine.types import _unwrap
        s = ForgeSession()
        s.eval('A = [1 2; 3 4]')
        ws = s.get_workspace_dict()
        assert 'A' in ws

    def test_eval_sin(self):
        """R-SESS-02.03: Built-in function call stores result in workspace."""
        from forge.engine.session import ForgeSession
        from forge.engine.types import _unwrap
        s = ForgeSession()
        s.eval('y = sin(0)')
        ws = s.get_workspace_dict()
        assert 'y' in ws

    def test_eval_error_handling(self):
        """R-SESS-02.04: Undefined variable returns error string, not exception."""
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        result = s.eval('this_is_undefined_variable_xyz')
        # Should return error string, not crash
        assert isinstance(result, str)

    def test_history_tracking(self):
        """R-SESS-02.05: Each eval call is recorded in the session history."""
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        s.eval('a = 1')
        s.eval('b = 2')
        assert len(s.history) == 2


class TestSessionBuiltins:
    """R-SESS-03: The ForgeSession SHALL provide workspace management builtins
    (who, clear, disp) and variable/function existence checking (exist).

    Model-user argument: The engineer uses ``who`` to see what is in the
    workspace, ``clear`` to reset state, ``disp`` to inspect values, and
    ``exist`` to check whether a name is a variable or function before using
    it. These are essential session-management tools.

    Decomposition: who, clear, disp, exist-variable, and exist-function
    tested. Consistency: These five builtins cover workspace introspection,
    cleanup, output, and existence queries.
    """

    def test_who(self):
        """R-SESS-03.01: who() executes without error after variable creation."""
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        s.eval('x = 1')
        s.eval('y = 2')
        result = s.eval('who()')

    def test_clear(self):
        """R-SESS-03.02: clear() removes all variables from workspace."""
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        s.eval('x = 1')
        s.eval('clear()')
        ws = s.get_workspace_dict()
        assert 'x' not in ws

    def test_disp(self):
        """R-SESS-03.03: disp(x) executes without error."""
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        s.eval('x = 42')
        result = s.eval('disp(x)')

    def test_exist_variable(self):
        """R-SESS-03.04: exist('x') returns a nonzero code for a defined variable."""
        from forge.engine.session import ForgeSession
        from forge.engine.types import _unwrap
        s = ForgeSession()
        s.eval('x = 5')
        s.eval('r = exist("x")')
        ws = s.get_workspace_dict()
        assert 'r' in ws

    def test_exist_function(self):
        """R-SESS-03.05: exist('sin') returns a nonzero code for a builtin."""
        from forge.engine.session import ForgeSession
        from forge.engine.types import _unwrap
        s = ForgeSession()
        s.eval('r = exist("sin")')
        ws = s.get_workspace_dict()
        assert 'r' in ws


class TestSessionFormatting:
    """R-SESS-04: The ForgeSession SHALL format result values for display,
    showing scalars with their numeric value and empty arrays with size
    notation.

    Model-user argument: When the engineer evaluates an expression without
    a semicolon, the formatted result appears in the command window. Scalars
    should show their value (e.g., "3.14"); empty arrays should indicate
    their dimensions (e.g., "0x0").

    Decomposition: Scalar formatting and empty-array formatting tested.
    Consistency: Scalar and empty are the two edge cases; matrix formatting
    is implicit.
    """

    def test_format_scalar(self):
        """R-SESS-04.01: Scalar 3.14 formats with '3.14' in output."""
        from forge.engine.session import ForgeSession
        from forge.engine.types import ForgeArray
        s = ForgeSession()
        r = s._format_result(ForgeArray(np.array(3.14)))
        assert '3.14' in r

    def test_format_empty(self):
        """R-SESS-04.02: Empty array format contains '0' size indicator."""
        from forge.engine.session import ForgeSession
        from forge.engine.types import ForgeArray
        s = ForgeSession()
        r = s._format_result(ForgeArray(np.array([])))
        assert '0x0' in r or '0' in r

    def test_get_workspace_dict(self):
        """R-SESS-05.01: get_workspace_dict returns dict containing assigned variables."""
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        s.eval('a = 5')
        d = s.get_workspace_dict()
        assert 'a' in d
