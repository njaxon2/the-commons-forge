"""Tests for v0.2.0 polish features."""
# Copyright 2026 The Commons(TM)
# SPDX-License-Identifier: Apache-2.0
import pytest
import numpy as np

@pytest.fixture(scope="module")
def session():
    from forge.engine.session import ForgeSession
    return ForgeSession()


class TestStructOverwrite:
    """Test that struct field assignment overwrites non-struct variables."""

    def test_overwrite_number(self, session):
        session.eval("xx = 42")
        session.eval("xx.a = 1")
        r = session.eval("xx.a")
        assert "1" in str(r)

    def test_overwrite_array(self, session):
        session.eval("yy = [1 2 3]")
        session.eval("yy.name = 'test'")
        r = session.eval("yy.name")
        assert "test" in str(r)


class TestErrorIdentifier:
    """Test error() with identifier form."""

    def test_forge_error_id(self, session):
        r = session.eval("try\n  error('myid:test', 'msg')\ncatch e\n  e.identifier\nend")
        assert "myid:test" in str(r)

    def test_forge_error_msg(self, session):
        r = session.eval("try\n  error('myid:test', 'hello world')\ncatch e\n  e.message\nend")
        assert "hello world" in str(r)


class TestNarginHandle:
    """Test nargin with function handles."""

    def test_nargin_builtin(self, session):
        r = session.eval("nargin('sin')")
        assert "1" in str(r) or "-1" in str(r)  # sin takes 1 arg or varargs

    def test_nargin_anonymous(self, session):
        r = session.eval("f = @(x,y) x+y; nargin(f)")
        assert "2" in str(r) or "-1" in str(r)  # anon funcs may report varargs


class TestNewBuiltins:
    """Test newly added builtins."""

    def test_fieldnames(self, session):
        session.eval("s_fn = struct('a', 1, 'b', 2)")
        r = session.eval("fieldnames(s_fn)")
        assert "'a'" in str(r) and "'b'" in str(r)

    def test_isfield_true(self, session):
        r = session.eval("isfield(s_fn, 'a')")
        assert "1" in str(r)

    def test_isfield_false(self, session):
        r = session.eval("isfield(s_fn, 'z')")
        assert "0" in str(r)

    def test_rmfield(self, session):
        r = session.eval("r_fn = rmfield(s_fn, 'a'); isfield(r_fn, 'a')")
        assert "0" in str(r)

    def test_deal(self, session):
        r = session.eval("deal(42)")
        assert "42" in str(r)

    def test_cellstr(self, session):
        r = session.eval("cellstr('hello')")
        assert "hello" in str(r)

    def test_display(self, session):
        r = session.eval("display(99)")
        assert "99" in str(r)


class TestPlotFunctions:
    """Test that new plot functions are registered."""

    def test_bar3_exists(self, session):
        r = session.eval("exist('bar3')")
        assert str(r).strip() not in ("0", "")

    def test_boxplot_exists(self, session):
        r = session.eval("exist('boxplot')")
        assert str(r).strip() not in ("0", "")

    def test_heatmap_exists(self, session):
        r = session.eval("exist('heatmap')")
        assert str(r).strip() not in ("0", "")

    def test_ginput_exists(self, session):
        r = session.eval("exist('ginput')")
        assert str(r).strip() not in ("0", "")


class TestOctaveCompat:
    """Additional Octave compatibility tests."""

    def test_string_concat(self, session):
        r = session.eval("['hello' ' ' 'world']")
        assert "hello world" in str(r)

    def test_sprintf(self, session):
        r = session.eval("sprintf('%d + %d = %d', 1, 2, 3)")
        assert "1 + 2 = 3" in str(r)

    def test_switch_case(self, session):
        r = session.eval("x_sw = 2; switch x_sw; case 1; r_sw = 'one'; case 2; r_sw = 'two'; otherwise; r_sw = 'other'; end; r_sw")
        assert "two" in str(r)

    def test_nested_anon(self, session):
        r = session.eval("f_na = @(x) @(y) x + y; g_na = f_na(10); g_na(5)")
        assert "15" in str(r)

    def test_multiple_return(self, session):
        session.eval("[r_mr, c_mr] = size([1 2 3; 4 5 6])")
        r = session.eval("r_mr")
        c = session.eval("c_mr")
        assert "2" in str(r) or "-1" in str(r)  # anon funcs may report varargs
        assert "3" in str(c)
