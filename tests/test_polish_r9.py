# Copyright 2026 The Commons
# SPDX-License-Identifier: Apache-2.0
"""Polish R9 -- evaluator edge-case tests."""
import pytest
from forge.engine.session import ForgeSession


@pytest.fixture
def s():
    return ForgeSession()


# ---------- 1. switch with string cases ----------
class TestSwitchStringCases:
    def test_switch_matches_first_string(self, s):
        s.eval('switch "hello"; case "hello"; x=1; case "world"; x=2; end')
        x = s._engine.workspace.get("x")
        assert float(x.data.flat[0]) == 1.0

    def test_switch_matches_second_string(self, s):
        s.eval('switch "world"; case "hello"; x=1; case "world"; x=2; end')
        x = s._engine.workspace.get("x")
        assert float(x.data.flat[0]) == 2.0

    def test_switch_no_match(self, s):
        s.eval('x=0; switch "other"; case "hello"; x=1; case "world"; x=2; end')
        x = s._engine.workspace.get("x")
        assert float(x.data.flat[0]) == 0.0

    def test_switch_otherwise(self, s):
        s.eval('switch "other"; case "hello"; x=1; otherwise; x=99; end')
        x = s._engine.workspace.get("x")
        assert float(x.data.flat[0]) == 99.0


# ---------- 2. for loop over cell array ----------
class TestForCellArray:
    def test_for_cell_sum(self, s):
        r = s.eval("x=0; for i={1,2,3}; x=x+i; end; x")
        assert "6" in r

    def test_for_cell_count(self, s):
        r = s.eval("n=0; for i={10,20,30,40}; n=n+1; end; n")
        assert "4" in r

    def test_for_cell_mixed_types(self, s):
        # Should iterate without error even with mixed types
        s.eval('for i={"a", 1, "b"}; end')


# ---------- 3. string concatenation via horzcat ----------
class TestStringConcat:
    def test_horzcat_strings(self, s):
        s.eval('s = ["hello" " " "world"]')
        val = s._engine.workspace.get("s")
        assert val.to_str() == "hello world"

    def test_horzcat_two_strings(self, s):
        s.eval('s = ["foo" "bar"]')
        val = s._engine.workspace.get("s")
        assert val.to_str() == "foobar"


# ---------- 4. end keyword in indexing ----------
class TestEndKeyword:
    def test_end_last_element(self, s):
        r = s.eval("A=[1 2 3 4 5]; A(end)")
        assert "5" in r

    def test_end_minus_one(self, s):
        r = s.eval("A=[1 2 3 4 5]; A(end-1)")
        assert "4" in r

    def test_end_minus_two(self, s):
        r = s.eval("A=[10 20 30 40 50]; A(end-2)")
        assert "30" in r


# ---------- 5. isfield ----------
class TestIsfield:
    def test_isfield_true(self, s):
        r = s.eval('st.x=1; st.y=2; isfield(st, "x")')
        assert "1" in r

    def test_isfield_false(self, s):
        r = s.eval('st.x=1; isfield(st, "z")')
        assert "0" in r


# ---------- 6. error ID handling ----------
class TestErrorId:
    def test_error_id_and_message(self, s):
        r = s.eval('try; error("myid:myerr", "msg %d", 42); catch e; disp(e.identifier); disp(e.message); end')
        assert "myid:myerr" in r
        assert "msg 42" in r

    def test_error_simple(self, s):
        r = s.eval('try; error("something broke"); catch e; disp(e.message); end')
        assert "something broke" in r


# ---------- 7. warning function ----------
class TestWarning:
    def test_warning_simple(self, s):
        r = s.eval('warning("hello")')
        assert "warning: hello" in r

    def test_warning_format(self, s):
        r = s.eval('warning("hello %s", "world")')
        assert "warning: hello world" in r

    def test_warning_id_format(self, s):
        r = s.eval('warning("test:warn", "hello %s", "world")')
        assert "warning: hello world" in r

    def test_warning_id_numeric_format(self, s):
        r = s.eval('warning("test:warn", "value is %d", 42)')
        assert "warning: value is 42" in r
