# Copyright 2026 The Commons
# SPDX-License-Identifier: Apache-2.0
"""Polish R13 -- struct and cell array operations."""
import numpy as np
import pytest
from forge.engine.evaluator import Session
from forge.engine.types import ForgeArray
from forge.engine.containers import ForgeCell, ForgeStruct, ForgeChar


def _eval(code):
    """Helper: create session, eval code, return session."""
    s = Session()
    s.eval(code)
    return s


def _val(s, name):
    """Get scalar float from workspace variable."""
    v = s.workspace.get(name)
    if isinstance(v, ForgeArray):
        return float(v.data.flat[0])
    return float(v)


# ---------- Struct basics ----------

class TestStructOperations:

    def test_struct_array_indexing(self):
        """s(1).x=1; s(2).x=2; s(2).x -> 2"""
        s = _eval("s(1).x=1; s(2).x=2; r=s(2).x;")
        assert _val(s, "r") == 2.0

    def test_dynamic_field_access(self):
        """s.x=1; f='x'; s.(f) -> 1"""
        s = _eval("s.x=1; f='x'; r=s.(f);")
        assert _val(s, "r") == 1.0

    def test_nested_struct(self):
        """s.a.b.c = 42"""
        s = _eval("s.a.b.c = 42; r = s.a.b.c;")
        assert _val(s, "r") == 42.0

    def test_struct_constructor(self):
        """struct('x',1,'y',2)"""
        s = _eval("s=struct('x',1,'y',2); rx=s.x; ry=s.y;")
        assert _val(s, "rx") == 1.0
        assert _val(s, "ry") == 2.0

    def test_fieldnames(self):
        """fieldnames returns cell of field names."""
        s = _eval("s.alpha=1; s.beta=2; n=fieldnames(s);")
        names = s.workspace.get("n")
        assert isinstance(names, ForgeCell)
        name_strs = [x.to_str() if isinstance(x, ForgeChar) else str(x)
                      for x in names._data]
        assert "alpha" in name_strs
        assert "beta" in name_strs

    def test_rmfield(self):
        """rmfield removes a field."""
        s = _eval("s.x=1; s.y=2; s=rmfield(s,'y'); n=fieldnames(s);")
        names = s.workspace.get("n")
        name_strs = [x.to_str() if isinstance(x, ForgeChar) else str(x)
                      for x in names._data]
        assert name_strs == ["x"]

    def test_isstruct(self):
        """isstruct returns true for structs."""
        s = _eval("s.x=1; r=isstruct(s);")
        assert _val(s, "r") == 1.0

    def test_isfield(self):
        """isfield checks field existence."""
        s = _eval("s.x=1; r1=isfield(s,'x'); r2=isfield(s,'z');")
        assert _val(s, "r1") == 1.0
        assert _val(s, "r2") == 0.0

    def test_isequal_structs(self):
        """isequal works on structs."""
        s = _eval("s1.x=1; s2.x=1; r=isequal(s1,s2);")
        assert bool(s.workspace.get("r").data.flat[0]) is True

    def test_isequal_structs_false(self):
        """isequal detects different structs."""
        s = _eval("s1.x=1; s2.x=2; r=isequal(s1,s2);")
        assert bool(s.workspace.get("r").data.flat[0]) is False

    def test_getfield(self):
        """getfield(s, 'x')"""
        s = _eval("s=struct('x',42); r=getfield(s,'x');")
        assert _val(s, "r") == 42.0

    def test_setfield(self):
        """setfield(s, 'y', 99)"""
        s = _eval("s=struct('x',1); s=setfield(s,'y',99); r=s.y;")
        assert _val(s, "r") == 99.0


# ---------- Cell operations ----------

class TestCellOperations:

    def test_cell_creation(self):
        """Basic cell array creation."""
        s = _eval("c = {1, 'hello', [1 2 3]};")
        c = s.workspace.get("c")
        assert isinstance(c, ForgeCell)
        assert c.numel() == 3

    def test_cellfun_isempty(self):
        """cellfun(@isempty, {[], 1, '', [1 2]}) -> [1 0 1 0]"""
        s = _eval("r=cellfun(@isempty, {[], 1, '', [1 2]});")
        r = s.workspace.get("r")
        assert list(r.data.flatten()) == [1.0, 0.0, 1.0, 0.0]

    def test_cell2mat_2d(self):
        """cell2mat({1 2; 3 4}) -> [1 2; 3 4]"""
        s = _eval("r=cell2mat({1 2; 3 4});")
        r = s.workspace.get("r")
        assert r.data.shape == (2, 2)
        assert np.array_equal(r.data, np.array([[1, 2], [3, 4]]))

    def test_cell2mat_1d(self):
        """cell2mat({1 2 3}) -> [1 2 3]"""
        s = _eval("r=cell2mat({1 2 3});")
        r = s.workspace.get("r")
        vals = r.data.flatten()
        assert list(vals) == [1.0, 2.0, 3.0]

    def test_iscell(self):
        """iscell returns true for cells."""
        s = _eval("c={1,2}; r=iscell(c);")
        assert _val(s, "r") == 1.0

    def test_cell_content_access(self):
        """c{2} content indexing."""
        s = _eval("c = {10, 20, 30}; r = c{2};")
        assert _val(s, "r") == 20.0

    def test_cell_content_assign(self):
        """c{1} = 99"""
        s = _eval("c = {0, 0}; c{1} = 99; r = c{1};")
        assert _val(s, "r") == 99.0

    def test_cellfun_custom_function(self):
        """cellfun with anonymous function."""
        s = _eval("r = cellfun(@(x) x*2, {1, 2, 3});")
        r = s.workspace.get("r")
        assert list(r.data.flatten()) == [2.0, 4.0, 6.0]

    def test_cell_2d_shape(self):
        """Cell {a b; c d} has shape (2,2)."""
        s = _eval("c = {1 2; 3 4};")
        c = s.workspace.get("c")
        assert c.shape == (2, 2)
        assert c.numel() == 4


# ---------- Mixed / edge cases ----------

class TestStructCellEdgeCases:

    def test_struct_with_cell_field(self):
        """Struct field containing a cell."""
        s = _eval("s.data = {1, 2, 3}; r = s.data;")
        r = s.workspace.get("r")
        assert isinstance(r, ForgeCell)

    def test_nested_struct_three_levels(self):
        """Three levels of nesting."""
        s = _eval("s.a.b.c = 7; s.a.b.d = 8; r1=s.a.b.c; r2=s.a.b.d;")
        assert _val(s, "r1") == 7.0
        assert _val(s, "r2") == 8.0

    def test_rmfield_preserves_other_fields(self):
        """rmfield keeps remaining fields intact."""
        s = _eval("s.a=1; s.b=2; s.c=3; s=rmfield(s,'b'); ra=s.a; rc=s.c;")
        assert _val(s, "ra") == 1.0
        assert _val(s, "rc") == 3.0

    def test_struct_overwrite_field(self):
        """Overwriting a struct field."""
        s = _eval("s.x=1; s.x=42; r=s.x;")
        assert _val(s, "r") == 42.0
