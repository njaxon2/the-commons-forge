# Copyright 2026 The Commons
# SPDX-License-Identifier: Apache-2.0
"""Polish R13 -- struct and cell array operations.

Requirement R-POL13-01:
    The struct subsystem SHALL support array indexing, dynamic field access,
    nested structs, the struct() constructor, fieldnames, rmfield, isfield,
    isstruct, isequal, getfield, and setfield with correct values and types.

    Model-user argument:
    An engineer migrating from MATLAB/Octave stores simulation parameters,
    mesh metadata, and result bundles in structs. Nested structs (e.g.,
    config.solver.tol) are ubiquitous. If dynamic field access s.(f) fails,
    or rmfield silently corrupts remaining fields, parameter-sweep scripts
    that programmatically iterate over field names will break. Correct
    struct behavior is foundational to porting real codebases.

    Decomposition:
    R-POL13-01a: Struct array indexing (s(2).x) returns the correct element.
    R-POL13-01b: Dynamic field access s.(f) resolves the field by variable.
    R-POL13-01c: Three-level nested struct assignment and retrieval works.
    R-POL13-01d: struct('x',1,'y',2) constructor creates correct fields.
    R-POL13-01e: fieldnames returns a cell of all field names.
    R-POL13-01f: rmfield removes the target field only.
    R-POL13-01g: isstruct returns 1 for structs.
    R-POL13-01h: isfield correctly detects field presence and absence.
    R-POL13-01i: isequal returns true for identical structs, false otherwise.
    R-POL13-01j: getfield retrieves the named field value.
    R-POL13-01k: setfield adds or updates a field.

    Consistency argument:
    Sub-requirements 01a-01k each test one struct operation in isolation.
    Together they cover every struct function and access pattern listed in
    the parent requirement.

Requirement R-POL13-02:
    The cell array subsystem SHALL support creation, content indexing,
    content assignment, cellfun (built-in and anonymous), cell2mat, iscell,
    and 2D cell shapes.

    Model-user argument:
    An engineer uses cell arrays to store heterogeneous data: file paths,
    mixed-type sensor readings, and variable-length string lists. cellfun
    is the standard way to apply transforms across such collections. If
    cell content indexing c{2} returns the wrong element, or cellfun skips
    entries, batch-processing loops over measurement files silently produce
    incomplete results.

    Decomposition:
    R-POL13-02a: Cell literal {1, 'hello', [1 2 3]} creates 3-element cell.
    R-POL13-02b: cellfun(@isempty, ...) returns correct logical vector.
    R-POL13-02c: cell2mat on 2D cell produces correct numeric matrix.
    R-POL13-02d: cell2mat on 1D cell produces correct row vector.
    R-POL13-02e: iscell returns 1 for cell arrays.
    R-POL13-02f: Content indexing c{2} returns the correct element.
    R-POL13-02g: Content assignment c{1}=99 updates the cell in place.
    R-POL13-02h: cellfun with anonymous function applies element-wise.
    R-POL13-02i: Cell literal {a b; c d} has shape (2,2).

    Consistency argument:
    Sub-requirements 02a-02i each test one cell operation. Together they
    cover creation, indexing, assignment, iteration, conversion, type
    checking, and shape, which are all operations listed in the parent.

Requirement R-POL13-03:
    Mixed struct/cell edge cases SHALL work correctly: struct fields
    containing cells, deep nesting, rmfield preserving siblings, and
    field overwrite.

    Model-user argument:
    Real MATLAB/Octave code frequently nests cells inside structs (e.g.,
    s.files = {'a.csv','b.csv'}) and overwrites fields during iterative
    refinement. If a struct field containing a cell silently converts to
    a numeric array, or if rmfield corrupts sibling fields, data
    structures built up over long simulation runs become unreliable.

    Decomposition:
    R-POL13-03a: Struct field containing a cell preserves cell type.
    R-POL13-03b: Three-level nested struct with two leaf fields works.
    R-POL13-03c: rmfield preserves all non-removed fields.
    R-POL13-03d: Overwriting a struct field replaces the value.

    Consistency argument:
    Sub-requirements 03a-03d each test one mixed/edge-case scenario.
    Together they cover cell-in-struct, deep nesting, selective removal,
    and overwrite, completing the edge-case coverage.
"""
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


# ---------- Struct basics (R-POL13-01) ----------

class TestStructOperations:
    """R-POL13-01: Struct creation, access, mutation, and introspection."""

    def test_struct_array_indexing(self):
        """R-POL13-01a: s(2).x returns correct element from struct array."""
        s = _eval("s(1).x=1; s(2).x=2; r=s(2).x;")
        assert _val(s, "r") == 2.0

    def test_dynamic_field_access(self):
        """R-POL13-01b: s.(f) resolves field name from variable."""
        s = _eval("s.x=1; f='x'; r=s.(f);")
        assert _val(s, "r") == 1.0

    def test_nested_struct(self):
        """R-POL13-01c: Three-level nested struct s.a.b.c works."""
        s = _eval("s.a.b.c = 42; r = s.a.b.c;")
        assert _val(s, "r") == 42.0

    def test_struct_constructor(self):
        """R-POL13-01d: struct('x',1,'y',2) creates correct fields."""
        s = _eval("s=struct('x',1,'y',2); rx=s.x; ry=s.y;")
        assert _val(s, "rx") == 1.0
        assert _val(s, "ry") == 2.0

    def test_fieldnames(self):
        """R-POL13-01e: fieldnames returns cell of field names."""
        s = _eval("s.alpha=1; s.beta=2; n=fieldnames(s);")
        names = s.workspace.get("n")
        assert isinstance(names, ForgeCell)
        name_strs = [x.to_str() if isinstance(x, ForgeChar) else str(x)
                      for x in names._data]
        assert "alpha" in name_strs
        assert "beta" in name_strs

    def test_rmfield(self):
        """R-POL13-01f: rmfield removes the target field only."""
        s = _eval("s.x=1; s.y=2; s=rmfield(s,'y'); n=fieldnames(s);")
        names = s.workspace.get("n")
        name_strs = [x.to_str() if isinstance(x, ForgeChar) else str(x)
                      for x in names._data]
        assert name_strs == ["x"]

    def test_isstruct(self):
        """R-POL13-01g: isstruct returns 1 for structs."""
        s = _eval("s.x=1; r=isstruct(s);")
        assert _val(s, "r") == 1.0

    def test_isfield(self):
        """R-POL13-01h: isfield detects presence and absence of fields."""
        s = _eval("s.x=1; r1=isfield(s,'x'); r2=isfield(s,'z');")
        assert _val(s, "r1") == 1.0
        assert _val(s, "r2") == 0.0

    def test_isequal_structs(self):
        """R-POL13-01i: isequal returns true for identical structs."""
        s = _eval("s1.x=1; s2.x=1; r=isequal(s1,s2);")
        assert bool(s.workspace.get("r").data.flat[0]) is True

    def test_isequal_structs_false(self):
        """R-POL13-01i: isequal returns false for different structs."""
        s = _eval("s1.x=1; s2.x=2; r=isequal(s1,s2);")
        assert bool(s.workspace.get("r").data.flat[0]) is False

    def test_getfield(self):
        """R-POL13-01j: getfield(s, 'x') retrieves field value."""
        s = _eval("s=struct('x',42); r=getfield(s,'x');")
        assert _val(s, "r") == 42.0

    def test_setfield(self):
        """R-POL13-01k: setfield adds a new field to the struct."""
        s = _eval("s=struct('x',1); s=setfield(s,'y',99); r=s.y;")
        assert _val(s, "r") == 99.0


# ---------- Cell operations (R-POL13-02) ----------

class TestCellOperations:
    """R-POL13-02: Cell creation, indexing, mutation, and iteration."""

    def test_cell_creation(self):
        """R-POL13-02a: Cell literal creates a 3-element cell."""
        s = _eval("c = {1, 'hello', [1 2 3]};")
        c = s.workspace.get("c")
        assert isinstance(c, ForgeCell)
        assert c.numel() == 3

    def test_cellfun_isempty(self):
        """R-POL13-02b: cellfun(@isempty, ...) returns correct logical vector."""
        s = _eval("r=cellfun(@isempty, {[], 1, '', [1 2]});")
        r = s.workspace.get("r")
        assert list(r.data.flatten()) == [1.0, 0.0, 1.0, 0.0]

    def test_cell2mat_2d(self):
        """R-POL13-02c: cell2mat on 2x2 cell produces 2x2 numeric matrix."""
        s = _eval("r=cell2mat({1 2; 3 4});")
        r = s.workspace.get("r")
        assert r.data.shape == (2, 2)
        assert np.array_equal(r.data, np.array([[1, 2], [3, 4]]))

    def test_cell2mat_1d(self):
        """R-POL13-02d: cell2mat on 1D cell produces row vector."""
        s = _eval("r=cell2mat({1 2 3});")
        r = s.workspace.get("r")
        vals = r.data.flatten()
        assert list(vals) == [1.0, 2.0, 3.0]

    def test_iscell(self):
        """R-POL13-02e: iscell returns 1 for cell arrays."""
        s = _eval("c={1,2}; r=iscell(c);")
        assert _val(s, "r") == 1.0

    def test_cell_content_access(self):
        """R-POL13-02f: c{2} returns the correct element."""
        s = _eval("c = {10, 20, 30}; r = c{2};")
        assert _val(s, "r") == 20.0

    def test_cell_content_assign(self):
        """R-POL13-02g: c{1}=99 updates the cell in place."""
        s = _eval("c = {0, 0}; c{1} = 99; r = c{1};")
        assert _val(s, "r") == 99.0

    def test_cellfun_custom_function(self):
        """R-POL13-02h: cellfun with anonymous function applies element-wise."""
        s = _eval("r = cellfun(@(x) x*2, {1, 2, 3});")
        r = s.workspace.get("r")
        assert list(r.data.flatten()) == [2.0, 4.0, 6.0]

    def test_cell_2d_shape(self):
        """R-POL13-02i: Cell {a b; c d} has shape (2,2)."""
        s = _eval("c = {1 2; 3 4};")
        c = s.workspace.get("c")
        assert c.shape == (2, 2)
        assert c.numel() == 4


# ---------- Mixed / edge cases (R-POL13-03) ----------

class TestStructCellEdgeCases:
    """R-POL13-03: Mixed struct/cell edge cases."""

    def test_struct_with_cell_field(self):
        """R-POL13-03a: Struct field containing a cell preserves cell type."""
        s = _eval("s.data = {1, 2, 3}; r = s.data;")
        r = s.workspace.get("r")
        assert isinstance(r, ForgeCell)

    def test_nested_struct_three_levels(self):
        """R-POL13-03b: Three-level nested struct with two leaf fields."""
        s = _eval("s.a.b.c = 7; s.a.b.d = 8; r1=s.a.b.c; r2=s.a.b.d;")
        assert _val(s, "r1") == 7.0
        assert _val(s, "r2") == 8.0

    def test_rmfield_preserves_other_fields(self):
        """R-POL13-03c: rmfield preserves all non-removed fields."""
        s = _eval("s.a=1; s.b=2; s.c=3; s=rmfield(s,'b'); ra=s.a; rc=s.c;")
        assert _val(s, "ra") == 1.0
        assert _val(s, "rc") == 3.0

    def test_struct_overwrite_field(self):
        """R-POL13-03d: Overwriting a struct field replaces the value."""
        s = _eval("s.x=1; s.x=42; r=s.x;")
        assert _val(s, "r") == 42.0
