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


class TestCharArrays:
    """R-CONT-01: Forge SHALL represent character data as 2-D numeric arrays
    of character codes, matching Octave char semantics including shape,
    indexing, concatenation, and multi-row padding.

    Model-user argument: The migrating engineer uses char arrays constantly
    for file paths, axis labels, and legend entries. Octave treats strings
    as row vectors of character codes with 1-based indexing and automatic
    padding for multi-row construction. If Forge deviates from these
    semantics, scripts that build file paths with string concatenation or
    index into labels will silently produce wrong results.

    Decomposition:
      R-CONT-01.1  Construction from a Python string
      R-CONT-01.2  Shape is (1, N) for a single string
      R-CONT-01.3  1-based element indexing returns character codes
      R-CONT-01.4  Concatenation with + operator
      R-CONT-01.5  Concatenation with a plain Python string
      R-CONT-01.6  Multi-row char with automatic space-padding
      R-CONT-01.7  Element-wise equality comparison
      R-CONT-01.8  forge_char() factory function
      R-CONT-01.9  Human-readable repr

    Consistency: R-CONT-01.1 through R-CONT-01.3 verify storage and access
    fundamentals. R-CONT-01.4 and R-CONT-01.5 cover both homogeneous and
    mixed-type concatenation. R-CONT-01.6 tests the multi-row padding rule.
    R-CONT-01.7 through R-CONT-01.9 cover comparison, construction helpers,
    and display. Together these span the full char array interface.
    """

    def test_from_string(self):
        """R-CONT-01.1: Verify char construction, content, length, and type name."""
        c = ForgeChar("hello")
        assert c.to_str() == "hello"
        assert c.numel() == 5
        assert c.type_name() == "char"

    def test_from_string_shape(self):
        """R-CONT-01.2: Verify single-string char has shape (1, N)."""
        c = ForgeChar("abc")
        assert c.shape == (1, 3)

    def test_indexing(self):
        """R-CONT-01.3: Verify 1-based indexing returns character codes."""
        c = ForgeChar("hello")
        assert c[1] == ord("h")
        assert c[5] == ord("o")

    def test_concatenation(self):
        """R-CONT-01.4: Verify char + char concatenation."""
        a = ForgeChar("hello")
        b = ForgeChar(" world")
        c = a + b
        assert c.to_str() == "hello world"

    def test_string_concat(self):
        """R-CONT-01.5: Verify char + Python string concatenation."""
        a = ForgeChar("hello")
        c = a + " world"
        assert c.to_str() == "hello world"

    def test_multi_row(self):
        """R-CONT-01.6: Verify multi-row char pads shorter rows with spaces."""
        c = ForgeChar(["abc", "de"])
        result = c.to_str()
        assert result == ["abc", "de "]

    def test_equality(self):
        """R-CONT-01.7: Verify element-wise equality of two char arrays."""
        a = ForgeChar("abc")
        b = ForgeChar("abc")
        r = (a == b)
        assert np.all(r.data)

    def test_forge_char_function(self):
        """R-CONT-01.8: Verify forge_char() factory returns ForgeChar."""
        c = forge_char("test")
        assert isinstance(c, ForgeChar)
        assert c.to_str() == "test"

    def test_repr(self):
        """R-CONT-01.9: Verify repr includes the quoted string content."""
        c = ForgeChar("hi")
        assert "'hi'" in repr(c)


class TestStringComparison:
    """R-CONT-02: Forge SHALL provide string comparison functions (strcmp,
    strcmpi, strncmp, strncmpi) that accept both char arrays and Python
    strings and return boolean results consistent with Octave semantics.

    Model-user argument: The engineer uses strcmp/strcmpi in switch-like
    logic to dispatch on solver names, file extensions, or configuration
    flags read from .m files. Case-insensitive and prefix comparisons are
    common when parsing mixed-case user input or checking file headers.
    Incorrect comparison semantics would cause silent logic errors in
    configuration-driven workflows.

    Decomposition:
      R-CONT-02.1  strcmp returns true for equal strings
      R-CONT-02.2  strcmp returns false for unequal strings
      R-CONT-02.3  strcmp is case-sensitive
      R-CONT-02.4  strcmpi is case-insensitive
      R-CONT-02.5  strncmp compares first N characters
      R-CONT-02.6  strncmpi compares first N characters case-insensitively
      R-CONT-02.7  strcmp accepts ForgeChar operands

    Consistency: R-CONT-02.1 through R-CONT-02.3 establish baseline
    exact-match behavior and case sensitivity. R-CONT-02.4 adds the
    case-insensitive variant. R-CONT-02.5 and R-CONT-02.6 add prefix
    comparison in both case modes. R-CONT-02.7 confirms interoperability
    with ForgeChar objects. All four Octave comparison functions are covered.
    """

    def test_strcmp_equal(self):
        """R-CONT-02.1: strcmp returns true for identical strings."""
        assert forge_strcmp("hello", "hello")

    def test_strcmp_not_equal(self):
        """R-CONT-02.2: strcmp returns false for different strings."""
        assert not forge_strcmp("hello", "world")

    def test_strcmp_case_sensitive(self):
        """R-CONT-02.3: strcmp distinguishes case."""
        assert not forge_strcmp("Hello", "hello")

    def test_strcmpi_case_insensitive(self):
        """R-CONT-02.4: strcmpi ignores case differences."""
        assert forge_strcmpi("Hello", "hello")

    def test_strncmp(self):
        """R-CONT-02.5: strncmp compares only the first N characters."""
        assert forge_strncmp("hello", "help", 3)
        assert not forge_strncmp("hello", "help", 4)

    def test_strncmpi(self):
        """R-CONT-02.6: strncmpi prefix comparison is case-insensitive."""
        assert forge_strncmpi("Hello", "HELP", 3)

    def test_strcmp_with_char(self):
        """R-CONT-02.7: strcmp accepts ForgeChar as an operand."""
        a = ForgeChar("test")
        assert forge_strcmp(a, "test")


class TestCellArrays:
    """R-CONT-03: Forge SHALL implement cell arrays as heterogeneous
    containers with 1-based content indexing, 2-D shaping, automatic
    growth, nesting, and standard utility functions (iscell, cell2mat,
    num2cell, cellfun).

    Model-user argument: The engineer stores heterogeneous collections in
    cell arrays: lists of matrices from different sensors (varying sizes),
    mixed numeric/string parameter sets, or nested result structures from
    iterative solvers. Cell arrays are the primary heterogeneous container
    in Octave. Incorrect content indexing, shape handling, or utility
    function behavior would break data aggregation pipelines that the
    engineer relies on daily.

    Decomposition:
      R-CONT-03.1   Construction from a Python list
      R-CONT-03.2   1-based content_get
      R-CONT-03.3   content_set mutation
      R-CONT-03.4   forge_cell(M, N) pre-allocation
      R-CONT-03.5   2-D content access with row, column indices
      R-CONT-03.6   forge_iscell type predicate
      R-CONT-03.7   Nested cell arrays
      R-CONT-03.8   Empty cell construction
      R-CONT-03.9   cell2mat flattens cells of arrays
      R-CONT-03.10  num2cell converts array elements to cell entries
      R-CONT-03.11  cellfun applies a function across cell contents
      R-CONT-03.12  Automatic growth on out-of-bounds content_set

    Consistency: R-CONT-03.1 and R-CONT-03.4 cover both list-based and
    pre-allocated construction. R-CONT-03.2, R-CONT-03.3, and R-CONT-03.5
    cover read/write access in 1-D and 2-D. R-CONT-03.6 through R-CONT-03.8
    test type detection, nesting, and empty edge case. R-CONT-03.9 through
    R-CONT-03.11 cover the three key utility functions. R-CONT-03.12 tests
    auto-grow, which Octave supports for cell assignment beyond bounds.
    """

    def test_creation_from_list(self):
        """R-CONT-03.1: Cell created from list has correct numel and shape."""
        c = ForgeCell([1, "hello", ForgeArray([1, 2, 3])])
        assert c.numel() == 3
        assert c.shape == (1, 3)

    def test_content_access(self):
        """R-CONT-03.2: 1-based content_get retrieves correct elements."""
        c = ForgeCell([10, 20, 30])
        assert c.content_get(1) == 10
        assert c.content_get(2) == 20
        assert c.content_get(3) == 30

    def test_content_set(self):
        """R-CONT-03.3: content_set mutates the specified cell entry."""
        c = ForgeCell([10, 20, 30])
        c.content_set(99, 2)
        assert c.content_get(2) == 99

    def test_cell_mn(self):
        """R-CONT-03.4: forge_cell(M, N) creates an M-by-N empty cell."""
        c = forge_cell(2, 3)
        assert c.shape == (2, 3)
        assert c.numel() == 6

    def test_2d_access(self):
        """R-CONT-03.5: 2-D content_set and content_get with row, col indices."""
        c = ForgeCell(2, 3)
        c.content_set("hello", 1, 2)
        assert c.content_get(1, 2) == "hello"

    def test_iscell(self):
        """R-CONT-03.6: forge_iscell returns true for cells, false otherwise."""
        assert forge_iscell(ForgeCell([1, 2]))
        assert not forge_iscell(ForgeArray([1, 2]))

    def test_nested_cell(self):
        """R-CONT-03.7: Cell arrays can nest other cell arrays."""
        inner = ForgeCell([1, 2])
        outer = ForgeCell([inner, 3])
        assert forge_iscell(outer.content_get(1))

    def test_empty_cell(self):
        """R-CONT-03.8: Default-constructed cell is empty with numel 0."""
        c = ForgeCell()
        assert c.isempty()
        assert c.numel() == 0

    def test_cell2mat(self):
        """R-CONT-03.9: cell2mat concatenates cell of arrays into one array."""
        c = ForgeCell([ForgeArray([1, 2]), ForgeArray([3, 4])])
        m = forge_cell2mat(c)
        np.testing.assert_array_equal(m.data.ravel(), [1, 2, 3, 4])

    def test_num2cell(self):
        """R-CONT-03.10: num2cell converts each array element to a cell entry."""
        a = ForgeArray([[1, 2], [3, 4]])
        c = forge_num2cell(a)
        assert c.numel() == 4

    def test_cellfun(self):
        """R-CONT-03.11: cellfun applies a function to each cell element."""
        c = ForgeCell([1.0, 4.0, 9.0])
        r = forge_cellfun(lambda x: x ** 0.5, c)
        np.testing.assert_array_almost_equal(r.data.ravel(), [1, 2, 3])

    def test_auto_grow(self):
        """R-CONT-03.12: content_set beyond bounds auto-grows the cell."""
        c = ForgeCell([10])
        c.content_set(99, 5)
        assert c.content_get(5) == 99
        assert c.numel() == 5


class TestStructs:
    """R-CONT-04: Forge SHALL implement structs as ordered field containers
    supporting attribute access, dynamic field names, introspection
    (fieldnames, isstruct), mutation (setfield, rmfield), ordering
    (orderfields), iteration (structfun), nesting, equality, and copy
    semantics consistent with Octave.

    Model-user argument: The engineer uses structs for all structured data:
    mesh.nodes, mesh.elements, config.solver_type, results.eigenvalues.
    Structs are the primary record type in Octave and appear in virtually
    every non-trivial script. Field introspection (fieldnames) drives
    generic processing loops, rmfield prunes optional parameters, and
    nested structs represent hierarchical configurations. Incorrect field
    ordering, missing copy-on-assign, or broken dynamic access would
    corrupt data pipelines the engineer depends on.

    Decomposition:
      R-CONT-04.1   Keyword construction via forge_struct(key=val)
      R-CONT-04.2   Alternating-argument construction forge_struct("a",1,"b",2)
      R-CONT-04.3   Attribute-style field set and get
      R-CONT-04.4   Dynamic field access via _fields dict
      R-CONT-04.5   forge_isstruct type predicate
      R-CONT-04.6   forge_fieldnames returns cell of field names
      R-CONT-04.7   forge_rmfield removes a field
      R-CONT-04.8   forge_setfield adds or updates a field
      R-CONT-04.9   forge_getfield retrieves a field value by name
      R-CONT-04.10  forge_orderfields sorts fields alphabetically
      R-CONT-04.11  forge_structfun applies a function to each field
      R-CONT-04.12  Nested struct access
      R-CONT-04.13  'in' operator for field membership
      R-CONT-04.14  AttributeError on missing field access
      R-CONT-04.15  Struct equality comparison
      R-CONT-04.16  Copy constructor produces independent copy

    Consistency: R-CONT-04.1 through R-CONT-04.4 cover all construction and
    access modes. R-CONT-04.5 through R-CONT-04.10 cover the six standard
    introspection/mutation functions. R-CONT-04.11 adds functional iteration.
    R-CONT-04.12 through R-CONT-04.16 cover nesting, membership, error
    handling, equality, and value semantics. Together these span the full
    Octave struct interface.
    """

    def test_creation_kwargs(self):
        """R-CONT-04.1: forge_struct with keyword args sets fields."""
        s = forge_struct(x=1, y=2)
        assert s.x == 1
        assert s.y == 2

    def test_creation_alternating(self):
        """R-CONT-04.2: forge_struct with alternating name-value pairs."""
        s = forge_struct("a", 10, "b", 20)
        assert s.a == 10
        assert s.b == 20

    def test_field_set(self):
        """R-CONT-04.3: Attribute-style field assignment and retrieval."""
        s = ForgeStruct()
        s.name = "test"
        s.value = 42
        assert s.name == "test"
        assert s.value == 42

    def test_dynamic_field(self):
        """R-CONT-04.4: Dynamic field access through _fields dict."""
        s = forge_struct(x=1)
        field = "x"
        assert s._fields[field] == 1

    def test_isstruct(self):
        """R-CONT-04.5: forge_isstruct returns true for structs only."""
        assert forge_isstruct(forge_struct(a=1))
        assert not forge_isstruct(ForgeArray([1]))

    def test_fieldnames(self):
        """R-CONT-04.6: forge_fieldnames returns a cell of field name strings."""
        s = forge_struct("x", 1, "y", 2, "z", 3)
        names = forge_fieldnames(s)
        assert forge_iscell(names)
        assert names.content_get(1) == "x"
        assert names.content_get(2) == "y"
        assert names.content_get(3) == "z"

    def test_rmfield(self):
        """R-CONT-04.7: forge_rmfield removes the named field."""
        s = forge_struct("a", 1, "b", 2, "c", 3)
        s2 = forge_rmfield(s, "b")
        assert "a" in s2
        assert "b" not in s2
        assert "c" in s2

    def test_setfield(self):
        """R-CONT-04.8: forge_setfield adds a new field to the struct."""
        s = forge_struct(x=1)
        s2 = forge_setfield(s, "y", 2)
        assert s2._fields["y"] == 2

    def test_getfield(self):
        """R-CONT-04.9: forge_getfield retrieves a field value by name."""
        s = forge_struct(x=42)
        assert forge_getfield(s, "x") == 42

    def test_orderfields(self):
        """R-CONT-04.10: forge_orderfields sorts fields alphabetically."""
        s = forge_struct("c", 3, "a", 1, "b", 2)
        s2 = forge_orderfields(s)
        names = list(s2._fields.keys())
        assert names == ["a", "b", "c"]

    def test_structfun(self):
        """R-CONT-04.11: forge_structfun applies function to each field value."""
        s = forge_struct(x=ForgeArray([1, 2]), y=ForgeArray([3, 4]))
        r = forge_structfun(lambda v: ForgeArray(v.data * 2), s)
        np.testing.assert_array_equal(r.x.data.ravel(), [2, 4])
        np.testing.assert_array_equal(r.y.data.ravel(), [6, 8])

    def test_nested_struct(self):
        """R-CONT-04.12: Nested struct fields are accessible via chained dot access."""
        inner = forge_struct(val=42)
        outer = forge_struct(child=inner)
        assert outer.child.val == 42

    def test_contains(self):
        """R-CONT-04.13: 'in' operator checks field membership."""
        s = forge_struct(x=1, y=2)
        assert "x" in s
        assert "z" not in s

    def test_missing_field_error(self):
        """R-CONT-04.14: Accessing a nonexistent field raises AttributeError."""
        s = forge_struct(x=1)
        with pytest.raises(AttributeError, match="no_such"):
            _ = s.no_such

    def test_equality(self):
        """R-CONT-04.15: Structs with identical fields and values are equal."""
        a = forge_struct(x=1, y=2)
        b = forge_struct(x=1, y=2)
        assert a == b

    def test_copy(self):
        """R-CONT-04.16: Copy constructor produces an independent copy."""
        s = forge_struct(x=1)
        s2 = ForgeStruct(s)
        s2.x = 99
        assert s.x == 1


class TestContainersMap:
    """R-CONT-05: Forge SHALL implement containers.Map as a key-value
    dictionary supporting string and numeric keys, membership testing,
    removal, key/value enumeration, and length queries consistent with
    Octave containers.Map semantics.

    Model-user argument: The engineer uses containers.Map for lookup tables:
    material property databases keyed by material name, unit conversion
    factors, or memoization caches in iterative solvers. Map is the only
    associative container in Octave and is the standard way to implement
    O(1) lookups. Incorrect key handling, missing membership checks, or
    broken removal would corrupt lookup-driven workflows.

    Decomposition:
      R-CONT-05.1   Construction with key and value lists
      R-CONT-05.2   Bracket-style set and get
      R-CONT-05.3   isKey membership test
      R-CONT-05.4   remove deletes a key
      R-CONT-05.5   keys() returns a cell of keys
      R-CONT-05.6   length() returns the number of entries
      R-CONT-05.7   'in' operator for membership
      R-CONT-05.8   Overwriting an existing key
      R-CONT-05.9   Empty map construction
      R-CONT-05.10  Numeric keys with key_type='double'

    Consistency: R-CONT-05.1 and R-CONT-05.9 cover non-empty and empty
    construction. R-CONT-05.2 and R-CONT-05.8 cover set/get including
    overwrite. R-CONT-05.3, R-CONT-05.4, and R-CONT-05.7 cover membership
    and removal. R-CONT-05.5 and R-CONT-05.6 cover enumeration and sizing.
    R-CONT-05.10 verifies numeric key support. Together these cover the
    full containers.Map interface.
    """

    def test_creation(self):
        """R-CONT-05.1: Map constructed from key and value lists."""
        m = ForgeMap(["a", "b", "c"], [1, 2, 3])
        assert m["a"] == 1
        assert m["b"] == 2
        assert m["c"] == 3

    def test_set_get(self):
        """R-CONT-05.2: Bracket-style assignment and retrieval."""
        m = ForgeMap()
        m["key1"] = 100
        assert m["key1"] == 100

    def test_isKey(self):
        """R-CONT-05.3: isKey returns true for present keys, false otherwise."""
        m = ForgeMap(["x"], [1])
        assert m.isKey("x")
        assert not m.isKey("y")

    def test_remove(self):
        """R-CONT-05.4: remove deletes a key while preserving others."""
        m = ForgeMap(["a", "b"], [1, 2])
        m.remove("a")
        assert not m.isKey("a")
        assert m.isKey("b")

    def test_keys_values(self):
        """R-CONT-05.5: keys() returns a cell array of all keys."""
        m = ForgeMap(["x", "y"], [10, 20])
        k = m.keys()
        assert forge_iscell(k)
        assert k.numel() == 2

    def test_length(self):
        """R-CONT-05.6: length() returns the number of key-value pairs."""
        m = ForgeMap(["a", "b", "c"], [1, 2, 3])
        assert m.length() == 3

    def test_contains(self):
        """R-CONT-05.7: 'in' operator tests key membership."""
        m = ForgeMap(["a"], [1])
        assert "a" in m
        assert "b" not in m

    def test_overwrite(self):
        """R-CONT-05.8: Assigning to an existing key overwrites the value."""
        m = ForgeMap(["a"], [1])
        m["a"] = 99
        assert m["a"] == 99

    def test_empty(self):
        """R-CONT-05.9: Default-constructed map has length 0."""
        m = ForgeMap()
        assert m.length() == 0

    def test_numeric_keys(self):
        """R-CONT-05.10: Map supports numeric keys with key_type='double'."""
        m = ForgeMap([1, 2, 3], ["a", "b", "c"], key_type="double")
        assert m[1] == "a"
        assert m[3] == "c"


class TestSparseMatrices:
    """R-CONT-06: Forge SHALL implement sparse matrices backed by SciPy
    sparse storage, supporting construction from dense arrays and COO
    triplets, round-trip dense conversion, arithmetic (addition, scalar
    multiply, matrix multiply with both sparse and dense operands), type
    detection, nnz counting, and human-readable repr.

    Model-user argument: The engineer uses sparse matrices extensively for
    finite element method (FEM) stiffness and mass matrices, which are
    large but mostly zeros. Octave's sparse() with triplet input is the
    standard FEM assembly pattern. The engineer also multiplies sparse
    system matrices by dense load vectors. If sparse construction, nnz
    reporting, or mixed sparse-dense matmul are incorrect, FEM solves
    produce wrong displacements or forces with no obvious error message.

    Decomposition:
      R-CONT-06.1   Construction from a dense ForgeArray
      R-CONT-06.2   Construction from COO triplets (i, j, v, m, n)
      R-CONT-06.3   Conversion to full dense array via forge_full
      R-CONT-06.4   Round-trip dense-to-sparse-to-dense preserves values
      R-CONT-06.5   Sparse + sparse element-wise addition
      R-CONT-06.6   Sparse * scalar multiplication
      R-CONT-06.7   Sparse @ sparse matrix multiplication
      R-CONT-06.8   forge_issparse type predicate for sparse
      R-CONT-06.9   forge_issparse returns false for non-sparse
      R-CONT-06.10  nnz returns 0 for all-zero sparse matrix
      R-CONT-06.11  repr includes dimensions and nnz
      R-CONT-06.12  Sparse @ dense matrix multiplication

    Consistency: R-CONT-06.1 and R-CONT-06.2 cover both construction paths.
    R-CONT-06.3 and R-CONT-06.4 verify dense recovery. R-CONT-06.5 through
    R-CONT-06.7 cover the three arithmetic operations. R-CONT-06.8 and
    R-CONT-06.9 cover type detection for both positive and negative cases.
    R-CONT-06.10 tests the zero-element edge case. R-CONT-06.11 covers
    display. R-CONT-06.12 tests the critical FEM use case of sparse times
    dense. Together these span the full sparse interface.
    """

    def test_from_dense(self):
        """R-CONT-06.1: Sparse from dense array preserves shape and nnz."""
        a = ForgeArray([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        s = forge_sparse(a)
        assert forge_issparse(s)
        assert s.shape == (3, 3)
        assert s.nnz() == 3

    def test_from_triplets(self):
        """R-CONT-06.2: Sparse from COO triplets (i, j, v, m, n)."""
        i = ForgeArray([1, 2, 3])
        j = ForgeArray([1, 2, 3])
        v = ForgeArray([10, 20, 30])
        s = forge_sparse(i, j, v, 3, 3)
        assert s.nnz() == 3

    def test_to_full(self):
        """R-CONT-06.3: forge_full converts sparse back to dense ForgeArray."""
        a = ForgeArray([[1, 0], [0, 2]])
        s = forge_sparse(a)
        f = forge_full(s)
        assert isinstance(f, ForgeArray)
        np.testing.assert_array_equal(f.data, a.data)

    def test_round_trip(self):
        """R-CONT-06.4: Dense to sparse to full round-trip preserves values."""
        a = ForgeArray(np.eye(5))
        s = forge_sparse(a)
        f = s.full()
        np.testing.assert_array_equal(f.data, a.data)

    def test_addition(self):
        """R-CONT-06.5: Sparse + sparse element-wise addition."""
        a = forge_sparse(ForgeArray([[1, 0], [0, 2]]))
        b = forge_sparse(ForgeArray([[0, 3], [4, 0]]))
        c = a + b
        f = c.full()
        np.testing.assert_array_equal(f.data, [[1, 3], [4, 2]])

    def test_scalar_multiply(self):
        """R-CONT-06.6: Sparse * scalar multiplication."""
        a = forge_sparse(ForgeArray([[1, 0], [0, 2]]))
        b = a * 3
        f = b.full()
        np.testing.assert_array_equal(f.data, [[3, 0], [0, 6]])

    def test_matmul(self):
        """R-CONT-06.7: Sparse @ sparse matrix multiplication (identity test)."""
        a = forge_sparse(ForgeArray(np.eye(3)))
        x = ForgeArray(np.array([[1], [2], [3]]))
        result = a @ x
        np.testing.assert_array_equal(result.data.ravel(), [1, 2, 3])

    def test_issparse(self):
        """R-CONT-06.8: forge_issparse returns true for sparse matrices."""
        assert forge_issparse(forge_sparse(ForgeArray(np.eye(2))))
        assert not forge_issparse(ForgeArray([1, 2]))

    def test_not_sparse_array(self):
        """R-CONT-06.9: forge_issparse returns false for plain scalars."""
        assert not forge_issparse(42)

    def test_nnz_empty(self):
        """R-CONT-06.10: nnz returns 0 for an all-zero sparse matrix."""
        s = forge_sparse(ForgeArray(np.zeros((3, 3))))
        assert s.nnz() == 0

    def test_repr(self):
        """R-CONT-06.11: repr includes dimension and nnz information."""
        s = forge_sparse(ForgeArray(np.eye(3)))
        assert "3" in repr(s)
        assert "nnz" in repr(s)

    def test_sparse_x_dense_matmul(self):
        """R-CONT-06.12: Sparse @ dense matmul (FEM load-vector pattern)."""
        A = forge_sparse(ForgeArray([[2, 0], [0, 3]]))
        x = ForgeArray([[1], [1]])
        r = A @ x
        np.testing.assert_array_equal(r.data.ravel(), [2, 3])
