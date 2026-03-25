"""Tests for ForgeSession."""
import pytest
import numpy as np


class TestSessionCreation:
    def test_create(self):
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        assert s is not None

    def test_has_builtins(self):
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        assert 'sin' in s._engine.functions
        assert 'cos' in s._engine.functions


class TestSessionEval:
    def test_eval_arithmetic(self):
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        result = s.eval('x = 3 + 4')
        ws = s.get_workspace_dict()
        assert 'x' in ws

    def test_eval_matrix(self):
        from forge.engine.session import ForgeSession
        from forge.engine.types import _unwrap
        s = ForgeSession()
        s.eval('A = [1 2; 3 4]')
        ws = s.get_workspace_dict()
        assert 'A' in ws

    def test_eval_sin(self):
        from forge.engine.session import ForgeSession
        from forge.engine.types import _unwrap
        s = ForgeSession()
        s.eval('y = sin(0)')
        ws = s.get_workspace_dict()
        assert 'y' in ws

    def test_eval_error_handling(self):
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        result = s.eval('this_is_undefined_variable_xyz')
        # Should return error string, not crash
        assert isinstance(result, str)

    def test_history_tracking(self):
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        s.eval('a = 1')
        s.eval('b = 2')
        assert len(s.history) == 2


class TestSessionBuiltins:
    def test_who(self):
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        s.eval('x = 1')
        s.eval('y = 2')
        result = s.eval('who()')

    def test_clear(self):
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        s.eval('x = 1')
        s.eval('clear()')
        ws = s.get_workspace_dict()
        assert 'x' not in ws

    def test_disp(self):
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        s.eval('x = 42')
        result = s.eval('disp(x)')

    def test_exist_variable(self):
        from forge.engine.session import ForgeSession
        from forge.engine.types import _unwrap
        s = ForgeSession()
        s.eval('x = 5')
        s.eval('r = exist("x")')
        ws = s.get_workspace_dict()
        assert 'r' in ws

    def test_exist_function(self):
        from forge.engine.session import ForgeSession
        from forge.engine.types import _unwrap
        s = ForgeSession()
        s.eval('r = exist("sin")')
        ws = s.get_workspace_dict()
        assert 'r' in ws


class TestSessionFormatting:
    def test_format_scalar(self):
        from forge.engine.session import ForgeSession
        from forge.engine.types import ForgeArray
        s = ForgeSession()
        r = s._format_result(ForgeArray(np.array(3.14)))
        assert '3.14' in r

    def test_format_empty(self):
        from forge.engine.session import ForgeSession
        from forge.engine.types import ForgeArray
        s = ForgeSession()
        r = s._format_result(ForgeArray(np.array([])))
        assert '0x0' in r or '0' in r

    def test_get_workspace_dict(self):
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        s.eval('a = 5')
        d = s.get_workspace_dict()
        assert 'a' in d
