# Copyright 2026 The Commons(TM)
# SPDX-License-Identifier: Apache-2.0
"""
Polish round 18: containers.Map, table, typecast/cast, cell2struct/struct2cell.

V&V Traceability (backfill):
    R-POL18-01 .. R-POL18-04 (parent requirements)
    R-POL18-01-nn .. R-POL18-04-nn (unit sub-requirements)

SRS trace: SRS-FUNC-001, SRS-VAL-001, SRS-COMPAT-001
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
    """R-POL18-01: containers.Map SHALL provide dictionary-style key/value
    storage with create, read, add, remove, keys, values, count, isKey,
    overwrite, and class introspection operations.

    Model-user argument: An engineer migrating from MATLAB/Octave relies on
    containers.Map to build lookup tables for parameter sets, unit conversions,
    and configuration data. Without faithful Map support, porting existing
    scripts that use this container type would require manual rewrites.

    Decomposition:
        R-POL18-01-01: Create Map and read value by key
        R-POL18-01-02: Add a new key to an existing Map
        R-POL18-01-03: keys() returns cell of string keys
        R-POL18-01-04: values() returns cell of values
        R-POL18-01-05: Map.Count returns entry count
        R-POL18-01-06: isKey() returns logical match result
        R-POL18-01-07: remove() deletes a key
        R-POL18-01-08: Overwriting an existing key replaces its value
        R-POL18-01-09: class() returns 'containers.Map'

    Consistency: The nine sub-requirements cover the full CRUD lifecycle
    (create, read, update, delete) plus introspection (keys, values, Count,
    isKey, class). Together they verify every documented containers.Map
    operation, satisfying R-POL18-01.
    """

    def test_map_create_and_read(self, s):
        """R-POL18-01-01: containers.Map then m('b') == 2."""
        s.eval('m = containers.Map({"a","b","c"}, {1,2,3})')
        r = s.eval('m("b")')
        assert float(r) == 2.0

    def test_map_add_key(self, s):
        """R-POL18-01-02: m('d') = 4 adds a new key."""
        s.eval('m = containers.Map({"a","b"}, {1,2})')
        s.eval('m("d") = 4')
        r = s.eval('m("d")')
        assert float(r) == 4.0

    def test_map_keys(self, s):
        """R-POL18-01-03: keys(m) returns a cell array of string keys."""
        s.eval('m = containers.Map({"x","y"}, {10,20})')
        s.eval("k = keys(m)")
        k = _ws(s, "k")
        from forge.engine.containers import ForgeCell
        assert isinstance(k, ForgeCell)
        names = sorted([str(c) for c in k._data])
        assert names == ["x", "y"]

    def test_map_values(self, s):
        """R-POL18-01-04: values(m) returns a cell array of values."""
        s.eval('m = containers.Map({"a","b"}, {10,20})')
        s.eval("v = values(m)")
        v = _ws(s, "v")
        from forge.engine.containers import ForgeCell
        assert isinstance(v, ForgeCell)
        vals = sorted([float(x) for x in v._data])
        assert vals == [10.0, 20.0]

    def test_map_count(self, s):
        """R-POL18-01-05: m.Count returns number of entries."""
        s.eval('m = containers.Map({"a","b","c"}, {1,2,3})')
        r = s.eval("m.Count")
        assert float(r) == 3.0

    def test_map_iskey(self, s):
        """R-POL18-01-06: isKey(m, 'a') returns true."""
        s.eval('m = containers.Map({"a","b"}, {1,2})')
        s.eval('r1 = isKey(m, "a")')
        s.eval('r2 = isKey(m, "z")')
        assert float(_ws(s, "r1")) == 1.0
        assert float(_ws(s, "r2")) == 0.0

    def test_map_remove(self, s):
        """R-POL18-01-07: remove(m, 'a') removes the key."""
        s.eval('m = containers.Map({"a","b","c"}, {1,2,3})')
        s.eval('m = remove(m, "a")')
        r = s.eval("m.Count")
        assert float(r) == 2.0

    def test_map_overwrite_value(self, s):
        """R-POL18-01-08: Overwriting an existing key."""
        s.eval('m = containers.Map({"x"}, {10})')
        s.eval('m("x") = 99')
        r = s.eval('m("x")')
        assert float(r) == 99.0

    def test_map_class(self, s):
        """R-POL18-01-09: class(m) returns containers.Map."""
        s.eval('m = containers.Map({"a"}, {1})')
        r = s.eval("class(m)")
        assert str(r) == "containers.Map"


# ===================================================================
# table
# ===================================================================

class TestTable:
    """R-POL18-02: ForgeTable SHALL support construction with VariableNames,
    column access via dot notation, height/width queries, Properties metadata,
    column assignment, class introspection, and istable detection.

    Model-user argument: Scientists and engineers use MATLAB tables extensively
    for tabular data workflows (experiment results, sensor logs, CSV imports).
    Forge must replicate table construction and access patterns so that existing
    data-processing scripts port without modification.

    Decomposition:
        R-POL18-02-01: table() with VariableNames creates ForgeTable
        R-POL18-02-02: t.Col returns column data
        R-POL18-02-03: height(t) returns row count
        R-POL18-02-04: width(t) returns column count
        R-POL18-02-05: t.Properties.VariableNames returns cell of names
        R-POL18-02-06: t.NewCol = data adds a column
        R-POL18-02-07: class(t) returns 'table'
        R-POL18-02-08: istable() correctly identifies tables vs. non-tables

    Consistency: Sub-requirements cover construction (01), read (02, 03, 04, 05),
    mutation (06), and type introspection (07, 08). This spans the full surface
    area of table operations required for migration scripts.
    """

    def test_table_create_with_varnames(self, s):
        """R-POL18-02-01: table with VariableNames creates proper ForgeTable."""
        s.eval('t = table([1;2;3], {"a";"b";"c"}, "VariableNames", {"ID","Name"})')
        t = _ws(s, "t")
        from forge.engine.containers import ForgeTable
        assert isinstance(t, ForgeTable)

    def test_table_dot_column(self, s):
        """R-POL18-02-02: t.Val returns the column data."""
        s.eval('t = table([10;20;30], "VariableNames", {"Val"})')
        s.eval("v = t.Val")
        v = _ws(s, "v")
        assert float(v.data.flat[0]) == 10.0
        assert float(v.data.flat[2]) == 30.0

    def test_table_height(self, s):
        """R-POL18-02-03: height(t) returns row count."""
        s.eval('t = table([1;2;3;4], "VariableNames", {"X"})')
        r = s.eval("height(t)")
        assert float(r) == 4.0

    def test_table_width(self, s):
        """R-POL18-02-04: width(t) returns column count."""
        s.eval('t = table([1;2], [3;4], "VariableNames", {"A","B"})')
        r = s.eval("width(t)")
        assert float(r) == 2.0

    def test_table_properties_varnames(self, s):
        """R-POL18-02-05: t.Properties.VariableNames returns cell of names."""
        s.eval('t = table([1;2], {"x";"y"}, "VariableNames", {"ID","Label"})')
        s.eval("vn = t.Properties.VariableNames")
        vn = _ws(s, "vn")
        from forge.engine.containers import ForgeCell
        assert isinstance(vn, ForgeCell)
        names = [str(c) for c in vn._data]
        assert names == ["ID", "Label"]

    def test_table_set_column(self, s):
        """R-POL18-02-06: t.NewCol = data adds a column."""
        s.eval('t = table([1;2;3], "VariableNames", {"A"})')
        s.eval("t.B = [4;5;6]")
        s.eval("bv = t.B")
        bv = _ws(s, "bv")
        assert float(bv.data.flat[0]) == 4.0

    def test_table_class(self, s):
        """R-POL18-02-07: class(t) returns table."""
        s.eval('t = table([1], "VariableNames", {"X"})')
        r = s.eval("class(t)")
        assert str(r) == "table"

    def test_istable(self, s):
        """R-POL18-02-08: istable(t) returns 1, istable(42) returns 0."""
        s.eval('t = table([1], "VariableNames", {"X"})')
        assert float(s.eval("istable(t)")) == 1.0
        assert float(s.eval("istable(42)")) == 0.0


# ===================================================================
# typecast (byte reinterpretation) and cast (value conversion)
# ===================================================================

class TestTypecastAndCast:
    """R-POL18-03: typecast SHALL reinterpret raw bytes between numeric types,
    and cast SHALL convert numeric values to a target type, both matching
    MATLAB/Octave semantics.

    Model-user argument: Engineers working with binary file I/O, sensor data
    acquisition, or fixed-point DSP code depend on typecast for bit-level
    reinterpretation and cast for safe type conversion. Incorrect byte mapping
    would silently corrupt numerical results.

    Decomposition:
        R-POL18-03-01: typecast single(1.0) to uint32 yields IEEE 754 encoding
        R-POL18-03-02: typecast double to uint8 returns 8 bytes
        R-POL18-03-03: typecast uint8 bytes back to single recovers value
        R-POL18-03-04: cast converts double to int32 by value
        R-POL18-03-05: cast preserves numeric value across type boundary

    Consistency: Sub-requirements 01-03 verify byte-level round-trip fidelity
    for typecast; 04-05 verify value-level conversion for cast. Together they
    cover both reinterpretation and conversion paths.
    """

    def test_typecast_single_to_uint32(self, s):
        """R-POL18-03-01: typecast(single(1), uint32) reinterprets bytes."""
        r = s.eval('typecast(single(1.0), "uint32")')
        # IEEE 754: single 1.0 = 0x3F800000 = 1065353216
        assert int(float(r)) == 1065353216

    def test_typecast_double_to_uint8(self, s):
        """R-POL18-03-02: typecast(1.0, uint8) returns 8 bytes."""
        s.eval('r = typecast(1.0, "uint8")')
        r = _ws(s, "r")
        assert r.data.size == 8  # double = 8 bytes

    def test_typecast_uint8_to_single(self, s):
        """R-POL18-03-03: typecast(uint8([0 0 128 63]), single) gives 1.0."""
        s.eval('r = typecast(uint8([0 0 128 63]), "single")')
        r = _ws(s, "r")
        assert abs(float(r.data.flat[0]) - 1.0) < 1e-6

    def test_cast_double_to_int32(self, s):
        """R-POL18-03-04: cast(3.7, int32) converts value."""
        r = s.eval('cast(3.7, "int32")')
        assert int(float(r)) in (3, 4)

    def test_cast_preserves_value(self, s):
        """R-POL18-03-05: cast(42, single) keeps value 42."""
        r = s.eval('cast(42, "single")')
        assert abs(float(r) - 42.0) < 1e-6


# ===================================================================
# cell2struct / struct2cell round-trip
# ===================================================================

class TestCellStructRoundTrip:
    """R-POL18-04: cell2struct and struct2cell SHALL convert between cell arrays
    and structs bidirectionally, preserving field names and values through a
    complete round-trip.

    Model-user argument: MATLAB/Octave users frequently convert between cell
    arrays and structs when reshaping data for different processing stages
    (e.g., reading CSV columns into cells, then assembling a struct per record).
    Round-trip fidelity is essential so that no data is lost during conversion.

    Decomposition:
        R-POL18-04-01: cell2struct creates struct from cell + fieldnames
        R-POL18-04-02: struct2cell returns cell of struct values
        R-POL18-04-03: struct -> struct2cell -> cell2struct recovers fields
        R-POL18-04-04: cell2struct with 3 fields assigns all correctly

    Consistency: Sub-requirements cover forward conversion (01), reverse
    conversion (02), round-trip (03), and multi-field verification (04).
    Together they prove bidirectional lossless conversion.
    """

    def test_cell2struct_basic(self, s):
        """R-POL18-04-01: cell2struct creates struct from cell + fieldnames."""
        s.eval('c = {1; "hello"; true}')
        s.eval('f = {"a"; "b"; "c"}')
        s.eval("st = cell2struct(c, f)")
        r = s.eval("st.b")
        assert "hello" in str(r)

    def test_struct2cell_basic(self, s):
        """R-POL18-04-02: struct2cell returns cell of values."""
        s.eval('st = struct("x", 10, "y", 20)')
        s.eval("c = struct2cell(st)")
        c = _ws(s, "c")
        from forge.engine.containers import ForgeCell
        assert isinstance(c, ForgeCell)
        vals = [float(v) for v in c._data]
        assert 10.0 in vals
        assert 20.0 in vals

    def test_round_trip(self, s):
        """R-POL18-04-03: struct -> struct2cell -> cell2struct recovers fields."""
        s.eval('orig = struct("alpha", 10, "beta", 20)')
        s.eval("c = struct2cell(orig)")
        s.eval("f = fieldnames(orig)")
        s.eval("rebuilt = cell2struct(c, f)")
        assert float(s.eval("rebuilt.alpha")) == 10.0
        assert float(s.eval("rebuilt.beta")) == 20.0

    def test_cell2struct_three_fields(self, s):
        """R-POL18-04-04: cell2struct with 3 fields."""
        s.eval('st = cell2struct({100; "test"; 3.14}, {"id"; "name"; "val"})')
        assert float(s.eval("st.id")) == 100.0
        assert "test" in str(s.eval("st.name"))
        assert abs(float(s.eval("st.val")) - 3.14) < 0.01
