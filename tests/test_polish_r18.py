# Copyright 2026 The Commons(TM)
# SPDX-License-Identifier: Apache-2.0
"""
Polish round 18: containers.Map, table, typecast/cast, cell2struct/struct2cell.
"""
import pytest
import numpy as np
from forge.engine.session import ForgeSession


@pytest.fixture
def s():
    return ForgeSession()


def _ws(s, name):
    return s._engine.workspace.get(name)


# ===================================================================
# containers.Map
# ===================================================================

class TestContainersMap:

    def test_map_create_and_read(self, s):
        """containers.Map then m('b') == 2."""
        s.eval('m = containers.Map({"a","b","c"}, {1,2,3})')
        r = s.eval('m("b")')
        assert float(r) == 2.0

    def test_map_add_key(self, s):
        """m('d') = 4 adds a new key."""
        s.eval('m = containers.Map({"a","b"}, {1,2})')
        s.eval('m("d") = 4')
        r = s.eval('m("d")')
        assert float(r) == 4.0

    def test_map_keys(self, s):
        """keys(m) returns a cell array of string keys."""
        s.eval('m = containers.Map({"x","y"}, {10,20})')
        s.eval("k = keys(m)")
        k = _ws(s, "k")
        from forge.engine.containers import ForgeCell
        assert isinstance(k, ForgeCell)
        names = sorted([str(c) for c in k._data])
        assert names == ["x", "y"]

    def test_map_values(self, s):
        """values(m) returns a cell array of values."""
        s.eval('m = containers.Map({"a","b"}, {10,20})')
        s.eval("v = values(m)")
        v = _ws(s, "v")
        from forge.engine.containers import ForgeCell
        assert isinstance(v, ForgeCell)
        vals = sorted([float(x) for x in v._data])
        assert vals == [10.0, 20.0]

    def test_map_count(self, s):
        """m.Count returns number of entries."""
        s.eval('m = containers.Map({"a","b","c"}, {1,2,3})')
        r = s.eval("m.Count")
        assert float(r) == 3.0

    def test_map_iskey(self, s):
        """isKey(m, 'a') returns true."""
        s.eval('m = containers.Map({"a","b"}, {1,2})')
        s.eval('r1 = isKey(m, "a")')
        s.eval('r2 = isKey(m, "z")')
        assert float(_ws(s, "r1")) == 1.0
        assert float(_ws(s, "r2")) == 0.0

    def test_map_remove(self, s):
        """remove(m, 'a') removes the key."""
        s.eval('m = containers.Map({"a","b","c"}, {1,2,3})')
        s.eval('m = remove(m, "a")')
        r = s.eval("m.Count")
        assert float(r) == 2.0

    def test_map_overwrite_value(self, s):
        """Overwriting an existing key."""
        s.eval('m = containers.Map({"x"}, {10})')
        s.eval('m("x") = 99')
        r = s.eval('m("x")')
        assert float(r) == 99.0

    def test_map_class(self, s):
        """class(m) returns containers.Map."""
        s.eval('m = containers.Map({"a"}, {1})')
        r = s.eval("class(m)")
        assert str(r) == "containers.Map"


# ===================================================================
# table
# ===================================================================

class TestTable:

    def test_table_create_with_varnames(self, s):
        """table with VariableNames creates proper ForgeTable."""
        s.eval('t = table([1;2;3], {"a";"b";"c"}, "VariableNames", {"ID","Name"})')
        t = _ws(s, "t")
        from forge.engine.containers import ForgeTable
        assert isinstance(t, ForgeTable)

    def test_table_dot_column(self, s):
        """t.Val returns the column data."""
        s.eval('t = table([10;20;30], "VariableNames", {"Val"})')
        s.eval("v = t.Val")
        v = _ws(s, "v")
        assert float(v.data.flat[0]) == 10.0
        assert float(v.data.flat[2]) == 30.0

    def test_table_height(self, s):
        """height(t) returns row count."""
        s.eval('t = table([1;2;3;4], "VariableNames", {"X"})')
        r = s.eval("height(t)")
        assert float(r) == 4.0

    def test_table_width(self, s):
        """width(t) returns column count."""
        s.eval('t = table([1;2], [3;4], "VariableNames", {"A","B"})')
        r = s.eval("width(t)")
        assert float(r) == 2.0

    def test_table_properties_varnames(self, s):
        """t.Properties.VariableNames returns cell of names."""
        s.eval('t = table([1;2], {"x";"y"}, "VariableNames", {"ID","Label"})')
        s.eval("vn = t.Properties.VariableNames")
        vn = _ws(s, "vn")
        from forge.engine.containers import ForgeCell
        assert isinstance(vn, ForgeCell)
        names = [str(c) for c in vn._data]
        assert names == ["ID", "Label"]

    def test_table_set_column(self, s):
        """t.NewCol = data adds a column."""
        s.eval('t = table([1;2;3], "VariableNames", {"A"})')
        s.eval("t.B = [4;5;6]")
        s.eval("bv = t.B")
        bv = _ws(s, "bv")
        assert float(bv.data.flat[0]) == 4.0

    def test_table_class(self, s):
        """class(t) returns table."""
        s.eval('t = table([1], "VariableNames", {"X"})')
        r = s.eval("class(t)")
        assert str(r) == "table"

    def test_istable(self, s):
        """istable(t) returns 1, istable(42) returns 0."""
        s.eval('t = table([1], "VariableNames", {"X"})')
        assert float(s.eval("istable(t)")) == 1.0
        assert float(s.eval("istable(42)")) == 0.0


# ===================================================================
# typecast (byte reinterpretation) and cast (value conversion)
# ===================================================================

class TestTypecastAndCast:

    def test_typecast_single_to_uint32(self, s):
        """typecast(single(1), uint32) reinterprets bytes."""
        r = s.eval('typecast(single(1.0), "uint32")')
        # IEEE 754: single 1.0 = 0x3F800000 = 1065353216
        assert int(float(r)) == 1065353216

    def test_typecast_double_to_uint8(self, s):
        """typecast(1.0, uint8) returns 8 bytes."""
        s.eval('r = typecast(1.0, "uint8")')
        r = _ws(s, "r")
        assert r.data.size == 8  # double = 8 bytes

    def test_typecast_uint8_to_single(self, s):
        """typecast(uint8([0 0 128 63]), single) gives 1.0."""
        s.eval('r = typecast(uint8([0 0 128 63]), "single")')
        r = _ws(s, "r")
        assert abs(float(r.data.flat[0]) - 1.0) < 1e-6

    def test_cast_double_to_int32(self, s):
        """cast(3.7, int32) converts value."""
        r = s.eval('cast(3.7, "int32")')
        assert int(float(r)) in (3, 4)

    def test_cast_preserves_value(self, s):
        """cast(42, single) keeps value 42."""
        r = s.eval('cast(42, "single")')
        assert abs(float(r) - 42.0) < 1e-6


# ===================================================================
# cell2struct / struct2cell round-trip
# ===================================================================

class TestCellStructRoundTrip:

    def test_cell2struct_basic(self, s):
        """cell2struct creates struct from cell + fieldnames."""
        s.eval('c = {1; "hello"; true}')
        s.eval('f = {"a"; "b"; "c"}')
        s.eval("st = cell2struct(c, f)")
        r = s.eval("st.b")
        assert "hello" in str(r)

    def test_struct2cell_basic(self, s):
        """struct2cell returns cell of values."""
        s.eval('st = struct("x", 10, "y", 20)')
        s.eval("c = struct2cell(st)")
        c = _ws(s, "c")
        from forge.engine.containers import ForgeCell
        assert isinstance(c, ForgeCell)
        vals = [float(v) for v in c._data]
        assert 10.0 in vals
        assert 20.0 in vals

    def test_round_trip(self, s):
        """struct -> struct2cell -> cell2struct recovers fields."""
        s.eval('orig = struct("alpha", 10, "beta", 20)')
        s.eval("c = struct2cell(orig)")
        s.eval("f = fieldnames(orig)")
        s.eval("rebuilt = cell2struct(c, f)")
        assert float(s.eval("rebuilt.alpha")) == 10.0
        assert float(s.eval("rebuilt.beta")) == 20.0

    def test_cell2struct_three_fields(self, s):
        """cell2struct with 3 fields."""
        s.eval('st = cell2struct({100; "test"; 3.14}, {"id"; "name"; "val"})')
        assert float(s.eval("st.id")) == 100.0
        assert "test" in str(s.eval("st.name"))
        assert abs(float(s.eval("st.val")) - 3.14) < 0.01
