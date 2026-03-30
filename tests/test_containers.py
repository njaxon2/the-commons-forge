# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Tests for char arrays, cell arrays, structs, Map, and sparse matrices.
Covers Stages 1.5-1.9 from V&V plan."""
import numpy as np
import pytest
from forge.engine.types import ForgeArray
from forge.engine.containers import (
    ForgeChar, forge_char, forge_strcmp, forge_strcmpi, forge_strncmp, forge_strncmpi,
    ForgeCell, forge_cell, forge_iscell, forge_cell2mat, forge_num2cell, forge_cellfun,
    ForgeStruct, forge_struct, forge_isstruct, forge_fieldnames, forge_rmfield,
    forge_setfield, forge_getfield, forge_orderfields, forge_structfun,
    ForgeMap,
    ForgeSparse, forge_sparse, forge_issparse, forge_full,
)


# ============================================================
# Stage 1.5: Char Arrays
# ============================================================

class TestCharArrays:
    def test_from_string(self):
        c = ForgeChar("hello")
        assert c.to_str() == "hello"
        assert c.numel() == 5
        assert c.type_name() == "char"

    def test_from_string_shape(self):
        c = ForgeChar("abc")
        assert c.shape == (1, 3)

    def test_indexing(self):
        c = ForgeChar("hello")
        assert c[1] == ord("h")
        assert c[5] == ord("o")

    def test_concatenation(self):
        a = ForgeChar("hello")
        b = ForgeChar(" world")
        c = a + b
        assert c.to_str() == "hello world"

    def test_string_concat(self):
        a = ForgeChar("hello")
        c = a + " world"
        assert c.to_str() == "hello world"

    def test_multi_row(self):
        c = ForgeChar(["abc", "de"])
        result = c.to_str()
        assert result == ["abc", "de "]  # Padded with space

    def test_equality(self):
        a = ForgeChar("abc")
        b = ForgeChar("abc")
        r = (a == b)
        assert np.all(r.data)

    def test_forge_char_function(self):
        c = forge_char("test")
        assert isinstance(c, ForgeChar)
        assert c.to_str() == "test"

    def test_repr(self):
        c = ForgeChar("hi")
        assert "'hi'" in repr(c)


class TestStringComparison:
    def test_strcmp_equal(self):
        assert forge_strcmp("hello", "hello")

    def test_strcmp_not_equal(self):
        assert not forge_strcmp("hello", "world")

    def test_strcmp_case_sensitive(self):
        assert not forge_strcmp("Hello", "hello")

    def test_strcmpi_case_insensitive(self):
        assert forge_strcmpi("Hello", "hello")

    def test_strncmp(self):
        assert forge_strncmp("hello", "help", 3)
        assert not forge_strncmp("hello", "help", 4)

    def test_strncmpi(self):
        assert forge_strncmpi("Hello", "HELP", 3)

    def test_strcmp_with_char(self):
        a = ForgeChar("test")
        assert forge_strcmp(a, "test")


# ============================================================
# Stage 1.6: Cell Arrays
# ============================================================

class TestCellArrays:
    def test_creation_from_list(self):
        c = ForgeCell([1, "hello", ForgeArray([1, 2, 3])])
        assert c.numel() == 3
        assert c.shape == (1, 3)

    def test_content_access(self):
        c = ForgeCell([10, 20, 30])
        assert c.content_get(1) == 10
        assert c.content_get(2) == 20
        assert c.content_get(3) == 30

    def test_content_set(self):
        c = ForgeCell([10, 20, 30])
        c.content_set(99, 2)
        assert c.content_get(2) == 99

    def test_cell_mn(self):
        c = forge_cell(2, 3)
        assert c.shape == (2, 3)
        assert c.numel() == 6

    def test_2d_access(self):
        c = ForgeCell(2, 3)
        c.content_set("hello", 1, 2)
        assert c.content_get(1, 2) == "hello"

    def test_iscell(self):
        assert forge_iscell(ForgeCell([1, 2]))
        assert not forge_iscell(ForgeArray([1, 2]))

    def test_nested_cell(self):
        inner = ForgeCell([1, 2])
        outer = ForgeCell([inner, 3])
        assert forge_iscell(outer.content_get(1))

    def test_empty_cell(self):
        c = ForgeCell()
        assert c.isempty()
        assert c.numel() == 0

    def test_cell2mat(self):
        c = ForgeCell([ForgeArray([1, 2]), ForgeArray([3, 4])])
        m = forge_cell2mat(c)
        np.testing.assert_array_equal(m.data.ravel(), [1, 2, 3, 4])

    def test_num2cell(self):
        a = ForgeArray([[1, 2], [3, 4]])
        c = forge_num2cell(a)
        assert c.numel() == 4

    def test_cellfun(self):
        c = ForgeCell([1.0, 4.0, 9.0])
        r = forge_cellfun(lambda x: x ** 0.5, c)
        np.testing.assert_array_almost_equal(r.data.ravel(), [1, 2, 3])

    def test_auto_grow(self):
        c = ForgeCell([10])
        c.content_set(99, 5)  # Auto-grow to index 5
        assert c.content_get(5) == 99
        assert c.numel() == 5


# ============================================================
# Stage 1.7: Structs
# ============================================================

class TestStructs:
    def test_creation_kwargs(self):
        s = forge_struct(x=1, y=2)
        assert s.x == 1
        assert s.y == 2

    def test_creation_alternating(self):
        s = forge_struct("a", 10, "b", 20)
        assert s.a == 10
        assert s.b == 20

    def test_field_set(self):
        s = ForgeStruct()
        s.name = "test"
        s.value = 42
        assert s.name == "test"
        assert s.value == 42

    def test_dynamic_field(self):
        s = forge_struct(x=1)
        field = "x"
        assert s._fields[field] == 1

    def test_isstruct(self):
        assert forge_isstruct(forge_struct(a=1))
        assert not forge_isstruct(ForgeArray([1]))

    def test_fieldnames(self):
        s = forge_struct("x", 1, "y", 2, "z", 3)
        names = forge_fieldnames(s)
        assert forge_iscell(names)
        assert names.content_get(1) == "x"
        assert names.content_get(2) == "y"
        assert names.content_get(3) == "z"

    def test_rmfield(self):
        s = forge_struct("a", 1, "b", 2, "c", 3)
        s2 = forge_rmfield(s, "b")
        assert "a" in s2
        assert "b" not in s2
        assert "c" in s2

    def test_setfield(self):
        s = forge_struct(x=1)
        s2 = forge_setfield(s, "y", 2)
        assert s2._fields["y"] == 2

    def test_getfield(self):
        s = forge_struct(x=42)
        assert forge_getfield(s, "x") == 42

    def test_orderfields(self):
        s = forge_struct("c", 3, "a", 1, "b", 2)
        s2 = forge_orderfields(s)
        names = list(s2._fields.keys())
        assert names == ["a", "b", "c"]

    def test_structfun(self):
        s = forge_struct(x=ForgeArray([1, 2]), y=ForgeArray([3, 4]))
        r = forge_structfun(lambda v: ForgeArray(v.data * 2), s)
        np.testing.assert_array_equal(r.x.data.ravel(), [2, 4])
        np.testing.assert_array_equal(r.y.data.ravel(), [6, 8])

    def test_nested_struct(self):
        inner = forge_struct(val=42)
        outer = forge_struct(child=inner)
        assert outer.child.val == 42

    def test_contains(self):
        s = forge_struct(x=1, y=2)
        assert "x" in s
        assert "z" not in s

    def test_missing_field_error(self):
        s = forge_struct(x=1)
        with pytest.raises(AttributeError, match="no_such"):
            _ = s.no_such

    def test_equality(self):
        a = forge_struct(x=1, y=2)
        b = forge_struct(x=1, y=2)
        assert a == b

    def test_copy(self):
        s = forge_struct(x=1)
        s2 = ForgeStruct(s)
        s2.x = 99
        assert s.x == 1  # Original unchanged


# ============================================================
# Stage 1.8: Containers.Map
# ============================================================

class TestContainersMap:
    def test_creation(self):
        m = ForgeMap(["a", "b", "c"], [1, 2, 3])
        assert m["a"] == 1
        assert m["b"] == 2
        assert m["c"] == 3

    def test_set_get(self):
        m = ForgeMap()
        m["key1"] = 100
        assert m["key1"] == 100

    def test_isKey(self):
        m = ForgeMap(["x"], [1])
        assert m.isKey("x")
        assert not m.isKey("y")

    def test_remove(self):
        m = ForgeMap(["a", "b"], [1, 2])
        m.remove("a")
        assert not m.isKey("a")
        assert m.isKey("b")

    def test_keys_values(self):
        m = ForgeMap(["x", "y"], [10, 20])
        k = m.keys()
        assert forge_iscell(k)
        assert k.numel() == 2

    def test_length(self):
        m = ForgeMap(["a", "b", "c"], [1, 2, 3])
        assert m.length() == 3

    def test_contains(self):
        m = ForgeMap(["a"], [1])
        assert "a" in m
        assert "b" not in m

    def test_overwrite(self):
        m = ForgeMap(["a"], [1])
        m["a"] = 99
        assert m["a"] == 99

    def test_empty(self):
        m = ForgeMap()
        assert m.length() == 0

    def test_numeric_keys(self):
        m = ForgeMap([1, 2, 3], ["a", "b", "c"], key_type="double")
        assert m[1] == "a"
        assert m[3] == "c"


# ============================================================
# Stage 1.9: Sparse Matrices
# ============================================================

class TestSparseMatrices:
    def test_from_dense(self):
        a = ForgeArray([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        s = forge_sparse(a)
        assert forge_issparse(s)
        assert s.shape == (3, 3)
        assert s.nnz() == 3

    def test_from_triplets(self):
        i = ForgeArray([1, 2, 3])
        j = ForgeArray([1, 2, 3])
        v = ForgeArray([10, 20, 30])
        s = forge_sparse(i, j, v, 3, 3)
        assert s.nnz() == 3

    def test_to_full(self):
        a = ForgeArray([[1, 0], [0, 2]])
        s = forge_sparse(a)
        f = forge_full(s)
        assert isinstance(f, ForgeArray)
        np.testing.assert_array_equal(f.data, a.data)

    def test_round_trip(self):
        a = ForgeArray(np.eye(5))
        s = forge_sparse(a)
        f = s.full()
        np.testing.assert_array_equal(f.data, a.data)

    def test_addition(self):
        a = forge_sparse(ForgeArray([[1, 0], [0, 2]]))
        b = forge_sparse(ForgeArray([[0, 3], [4, 0]]))
        c = a + b
        f = c.full()
        np.testing.assert_array_equal(f.data, [[1, 3], [4, 2]])

    def test_scalar_multiply(self):
        a = forge_sparse(ForgeArray([[1, 0], [0, 2]]))
        b = a * 3
        f = b.full()
        np.testing.assert_array_equal(f.data, [[3, 0], [0, 6]])

    def test_matmul(self):
        a = forge_sparse(ForgeArray(np.eye(3)))
        x = ForgeArray(np.array([[1], [2], [3]]))  # Column vector
        result = a @ x
        np.testing.assert_array_equal(result.data.ravel(), [1, 2, 3])

    def test_issparse(self):
        assert forge_issparse(forge_sparse(ForgeArray(np.eye(2))))
        assert not forge_issparse(ForgeArray([1, 2]))

    def test_not_sparse_array(self):
        assert not forge_issparse(42)

    def test_nnz_empty(self):
        s = forge_sparse(ForgeArray(np.zeros((3, 3))))
        assert s.nnz() == 0

    def test_repr(self):
        s = forge_sparse(ForgeArray(np.eye(3)))
        assert "3" in repr(s)
        assert "nnz" in repr(s)

    def test_sparse_x_dense_matmul(self):
        A = forge_sparse(ForgeArray([[2, 0], [0, 3]]))
        x = ForgeArray([[1], [1]])
        r = A @ x
        np.testing.assert_array_equal(r.data.ravel(), [2, 3])
