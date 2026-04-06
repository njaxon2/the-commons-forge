# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Tests for ForgeArray data types, 1-based indexing, type conversion, and matrix construction.
Covers Stages 1.1, 1.2, 1.3 from the V&V plan.

V&V traceability backfill: R-TYPE-01 through R-TYPE-14.
"""
import numpy as np
import pytest
from forge.engine.types import (
    ForgeArray, _unwrap, DTYPE_MAP,
    forge_double, forge_single, forge_int8, forge_int16, forge_int32, forge_int64,
    forge_uint8, forge_uint16, forge_uint32, forge_uint64, forge_logical, forge_complex,
    forge_isa, forge_isnumeric, forge_islogical, forge_isfloat, forge_isinteger, forge_ischar,
    forge_isnan, forge_isinf, forge_isfinite,
    FORGE_PI, FORGE_E, FORGE_INF, FORGE_NAN, FORGE_EPS, FORGE_I, FORGE_J,
    FORGE_TRUE, FORGE_FALSE,
    forge_eye, forge_ones, forge_zeros, forge_rand, forge_randn, forge_randi,
    forge_true, forge_false, forge_diag, forge_linspace, forge_colon, forge_repmat,
)


# ============================================================
# Stage 1.1: Numeric Types & ForgeArray
# ============================================================

class TestForgeArrayCreation:
    """R-TYPE-01: ForgeArray SHALL be constructable from scalars, lists, 2-D
    lists, NumPy arrays, and other ForgeArrays, with MATLAB-compatible
    shapes (scalars as 1x1, lists as row vectors) and correct dtype
    propagation.

    Model-user argument: Everything in MATLAB is a matrix. A scalar is 1x1,
    a list is a row vector, and 2-D data is a matrix. The engineer expects
    these conventions to hold so that size(), numel(), and indexing behave
    identically to MATLAB. An empty array must report zero elements.

    Decomposition: Scalar, 1-D list, 2-D list, NumPy array, ForgeArray
    copy, dtype specification, and empty array tested. Consistency: These
    seven constructors cover every input pathway into ForgeArray.
    """

    def test_from_scalar(self):
        """R-TYPE-01.01: Scalar creates a 1x1 ForgeArray."""
        a = ForgeArray(5.0)
        assert a.shape == (1, 1)  # Scalar is 1x1 in Octave
        assert a.isscalar()
        assert float(a) == 5.0

    def test_from_list(self):
        """R-TYPE-01.02: 1-D list creates a row vector."""
        a = ForgeArray([1, 2, 3])
        assert a.shape == (1, 3)  # Row vector
        assert a.isvector()

    def test_from_2d_list(self):
        """R-TYPE-01.03: 2-D list creates a square matrix."""
        a = ForgeArray([[1, 2], [3, 4]])
        assert a.shape == (2, 2)
        assert a.issquare()

    def test_from_numpy(self):
        """R-TYPE-01.04: NumPy array creates a row vector with matching shape."""
        arr = np.array([1.0, 2.0, 3.0])
        a = ForgeArray(arr)
        assert a.shape == (1, 3)

    def test_from_forge_array(self):
        """R-TYPE-01.05: ForgeArray copy has same shape and data."""
        a = ForgeArray([1, 2, 3])
        b = ForgeArray(a)
        assert b.shape == a.shape
        np.testing.assert_array_equal(b.data, a.data)

    def test_dtype_specification(self):
        """R-TYPE-01.06: Explicit dtype creates array with matching type."""
        a = ForgeArray([1, 2, 3], dtype="int32")
        assert a.dtype == np.int32
        assert a.type_name() == "int32"

    def test_empty_array(self):
        """R-TYPE-01.07: Empty NumPy array creates empty ForgeArray with 0 elements."""
        a = ForgeArray(np.array([]))
        assert a.isempty()
        assert a.numel() == 0


class TestForgeArrayIndexing:
    """R-TYPE-02: ForgeArray SHALL support 1-based scalar indexing, 2-D
    indexing, slice indexing, indexed assignment, and logical indexing,
    raising IndexError for index 0.

    Model-user argument: MATLAB uses 1-based indexing throughout. The
    engineer accesses the first element with ``x(1)``, not ``x(0)``.
    Attempting index 0 must raise an error. Slicing with ``x(2:4)`` must
    return elements 2 through 4 inclusive. Logical indexing (boolean masks)
    is the primary way to filter data arrays.

    Decomposition: 1-based scalar, error-on-zero, 2-D, slice, scalar
    assignment, 2-D assignment, and logical indexing tested. Consistency:
    These seven tests cover all indexing access and mutation patterns.
    """

    def test_1based_scalar(self):
        """R-TYPE-02.01: Index 1 returns first element, index 3 returns third."""
        a = ForgeArray([10, 20, 30])
        assert a[1] == 10  # First element
        assert a[2] == 20
        assert a[3] == 30

    def test_1based_error_on_zero(self):
        """R-TYPE-02.02: Index 0 raises IndexError."""
        a = ForgeArray([10, 20, 30])
        with pytest.raises(IndexError):
            _ = a[0]

    def test_1based_2d(self):
        """R-TYPE-02.03: 2-D (row, col) indexing returns correct elements."""
        a = ForgeArray([[1, 2, 3], [4, 5, 6]])
        assert a[1, 1] == 1
        assert a[1, 3] == 3
        assert a[2, 1] == 4
        assert a[2, 3] == 6

    def test_slice_1based(self):
        """R-TYPE-02.04: Slice 2:4 returns elements 2, 3, 4 (1-based)."""
        a = ForgeArray([10, 20, 30, 40, 50])
        s = a[2:4]  # Elements 2,3,4 (1-based) → indices 1,2,3 (0-based)
        np.testing.assert_array_equal(s.data.ravel(), [20, 30, 40])

    def test_assignment_1based(self):
        """R-TYPE-02.05: Assignment to index 1 modifies the first element."""
        a = ForgeArray([10, 20, 30])
        a[1] = 99
        assert a[1] == 99
        assert a[2] == 20

    def test_2d_assignment(self):
        """R-TYPE-02.06: 2-D assignment modifies the correct cell."""
        a = ForgeArray([[1, 2], [3, 4]])
        a[1, 2] = 99
        assert a[1, 2] == 99
        assert a[1, 1] == 1

    def test_logical_indexing(self):
        """R-TYPE-02.07: Logical mask selects elements where mask is true."""
        a = ForgeArray([10, 20, 30, 40])
        mask = ForgeArray(np.array([True, False, True, False]))
        result = a[mask]
        np.testing.assert_array_equal(result.data.ravel(), [10, 30])


class TestForgeArraySizeInfo:
    """R-TYPE-03: ForgeArray SHALL report correct size information via numel(),
    length(), rows(), columns(), ismatrix(), and isvector().

    Model-user argument: The engineer queries array dimensions with numel(),
    length(), size() and predicates like isvector() to write shape-aware
    code. Wrong size info causes off-by-one errors, dimension mismatches, and
    incorrect loop bounds.

    Decomposition: numel, length, rows/columns, ismatrix, isvector (row and
    column), and not-vector tested. Consistency: These cover all
    size-reporting methods and shape predicates.
    """

    def test_numel(self):
        """R-TYPE-03.01: numel() returns total element count."""
        assert ForgeArray([[1, 2], [3, 4]]).numel() == 4

    def test_length(self):
        """R-TYPE-03.02: length() returns the largest dimension."""
        assert ForgeArray([[1, 2, 3], [4, 5, 6]]).length() == 3  # max(2,3)

    def test_rows_columns(self):
        """R-TYPE-03.03: rows() and columns() return row and column counts."""
        a = ForgeArray([[1, 2, 3], [4, 5, 6]])
        assert a.rows() == 2
        assert a.columns() == 3

    def test_ismatrix(self):
        """R-TYPE-03.04: 2-D array reports ismatrix() = true."""
        assert ForgeArray([[1, 2], [3, 4]]).ismatrix()

    def test_isvector_row(self):
        """R-TYPE-03.05: Row vector reports isvector() = true."""
        assert ForgeArray([1, 2, 3]).isvector()

    def test_isvector_column(self):
        """R-TYPE-03.06: Column vector reports isvector() = true."""
        assert ForgeArray(np.array([[1], [2], [3]])).isvector()

    def test_not_vector(self):
        """R-TYPE-03.07: 2x2 matrix reports isvector() = false."""
        assert not ForgeArray([[1, 2], [3, 4]]).isvector()


class TestForgeArrayArithmetic:
    """R-TYPE-04: ForgeArray SHALL support element-wise arithmetic (+, -, *,
    /, **), scalar broadcasting, matrix multiplication (@), and element-wise
    comparison (==, <).

    Model-user argument: Array arithmetic is the core of numerical computing.
    The engineer adds vectors, multiplies by scalars, computes matrix
    products, and compares arrays element-wise. Each operation must produce
    correct results with correct shapes.

    Decomposition: Element-wise add, scalar add, subtract, multiply, divide,
    power, negation, matmul, equality comparison, and less-than comparison
    tested. Consistency: These ten operations cover the complete set of
    ForgeArray arithmetic and comparison operators.
    """

    def test_add(self):
        """R-TYPE-04.01: Element-wise addition returns correct sums."""
        a = ForgeArray([1, 2, 3])
        b = ForgeArray([10, 20, 30])
        c = a + b
        np.testing.assert_array_equal(c.data.ravel(), [11, 22, 33])

    def test_scalar_add(self):
        """R-TYPE-04.02: Scalar addition broadcasts to all elements."""
        a = ForgeArray([1, 2, 3])
        c = a + 10
        np.testing.assert_array_equal(c.data.ravel(), [11, 12, 13])

    def test_sub(self):
        """R-TYPE-04.03: Element-wise subtraction returns correct differences."""
        a = ForgeArray([10, 20, 30])
        b = ForgeArray([1, 2, 3])
        np.testing.assert_array_equal((a - b).data.ravel(), [9, 18, 27])

    def test_mul(self):
        """R-TYPE-04.04: Scalar multiplication broadcasts correctly."""
        a = ForgeArray([2, 3, 4])
        np.testing.assert_array_equal((a * 2).data.ravel(), [4, 6, 8])

    def test_div(self):
        """R-TYPE-04.05: Scalar division returns correct quotients."""
        a = ForgeArray([10.0, 20.0, 30.0])
        np.testing.assert_array_equal((a / 10).data.ravel(), [1, 2, 3])

    def test_power(self):
        """R-TYPE-04.06: Element-wise power returns correct results."""
        a = ForgeArray([2, 3, 4])
        np.testing.assert_array_equal((a ** 2).data.ravel(), [4, 9, 16])

    def test_negation(self):
        """R-TYPE-04.07: Unary negation flips signs."""
        a = ForgeArray([1, -2, 3])
        np.testing.assert_array_equal((-a).data.ravel(), [-1, 2, -3])

    def test_matmul(self):
        """R-TYPE-04.08: Matrix multiplication returns correct product."""
        a = ForgeArray([[1, 2], [3, 4]])
        b = ForgeArray([[5, 6], [7, 8]])
        c = a @ b
        np.testing.assert_array_equal(c.data, [[19, 22], [43, 50]])

    def test_comparison_eq(self):
        """R-TYPE-04.09: Element-wise equality returns boolean array."""
        a = ForgeArray([1, 2, 3])
        b = ForgeArray([1, 5, 3])
        r = (a == b)
        np.testing.assert_array_equal(r.data.ravel(), [True, False, True])

    def test_comparison_lt(self):
        """R-TYPE-04.10: Element-wise less-than with scalar broadcast."""
        a = ForgeArray([1, 5, 3])
        r = (a < 3)
        np.testing.assert_array_equal(r.data.ravel(), [True, False, False])


class TestTypeConversion:
    """R-TYPE-05: ForgeArray SHALL support type conversion between double,
    single, int8/16/32/64, uint8/16/32/64, and logical types with correct
    overflow saturation and truncation semantics.

    Model-user argument: The engineer converts between types for memory
    efficiency (uint8 for images), hardware interfaces (int16 for DAC
    values), and logical operations (logical for masks). Overflow must
    saturate to type limits (Octave semantics), not wrap around.

    Decomposition: double, single, int8 overflow, uint8 range, logical,
    all integer type names, and astype truncation tested. Consistency:
    These tests cover the conversion functions, saturation behavior, and
    the astype generic converter.
    """

    def test_double(self):
        """R-TYPE-05.01: forge_double converts int32 array to float64."""
        a = forge_double(ForgeArray([1, 2, 3], dtype="int32"))
        assert a.dtype == np.float64
        assert a.type_name() == "double"

    def test_single(self):
        """R-TYPE-05.02: forge_single creates float32 array."""
        a = forge_single([1.0, 2.0])
        assert a.dtype == np.float32

    def test_int8_overflow(self):
        """R-TYPE-05.03: int8 saturates 200 to 127 (Octave semantics)."""
        a = forge_int8([200])
        # int8 range is -128..127, Octave saturates to 127
        assert a.dtype == np.int8
        assert a.data.ravel()[0] == 127  # Saturates (Octave semantics)

    def test_uint8_range(self):
        """R-TYPE-05.04: uint8 preserves values within 0..255 range."""
        a = forge_uint8([0, 128, 255])
        np.testing.assert_array_equal(a.data.ravel(), [0, 128, 255])

    def test_logical(self):
        """R-TYPE-05.05: forge_logical converts nonzero to true, zero to false."""
        a = forge_logical([0, 1, 5, 0])
        np.testing.assert_array_equal(a.data.ravel(), [False, True, True, False])

    def test_all_integer_types(self):
        """R-TYPE-05.06: All integer type constructors produce correct type names."""
        for name, func in [("int8", forge_int8), ("int16", forge_int16),
                           ("int32", forge_int32), ("int64", forge_int64),
                           ("uint8", forge_uint8), ("uint16", forge_uint16),
                           ("uint32", forge_uint32), ("uint64", forge_uint64)]:
            a = func([1, 2, 3])
            assert a.type_name() == name

    def test_astype(self):
        """R-TYPE-05.07: astype('int32') truncates floats to integers."""
        a = ForgeArray([1.5, 2.7, 3.9])
        b = a.astype("int32")
        assert b.dtype == np.int32
        np.testing.assert_array_equal(b.data.ravel(), [1, 2, 3])


# ============================================================
# Stage 1.2: Complex Numbers & Special Values
# ============================================================

class TestComplex:
    """R-TYPE-06: ForgeArray SHALL support complex number creation and
    arithmetic with correct real/imaginary components.

    Model-user argument: Complex numbers are essential for electrical
    engineering (impedance), signal processing (FFT), and control theory
    (poles/zeros). The engineer creates complex values with
    ``complex(3, 4)`` and expects standard complex arithmetic.

    Decomposition: Scalar complex, complex array, real-only complex,
    complex addition, and complex multiplication tested. Consistency:
    These cover creation (scalar, array, real-only) and arithmetic
    (addition, multiplication) for complex types.
    """

    def test_complex_creation(self):
        """R-TYPE-06.01: forge_complex(3, 4) creates 3+4j scalar."""
        a = forge_complex(3, 4)
        assert a.isscalar()
        assert complex(a.data.flat[0]) == 3 + 4j

    def test_complex_array(self):
        """R-TYPE-06.02: forge_complex with arrays creates element-wise complex."""
        a = forge_complex([1, 2, 3], [4, 5, 6])
        np.testing.assert_array_equal(a.data.ravel(), [1+4j, 2+5j, 3+6j])

    def test_complex_real_only(self):
        """R-TYPE-06.03: forge_complex with real only produces complex128 dtype."""
        a = forge_complex([1.0, 2.0])
        assert a.dtype == np.complex128

    def test_complex_arithmetic(self):
        """R-TYPE-06.04: Complex addition combines real and imaginary parts."""
        a = ForgeArray(np.array([1+2j, 3+4j]))
        b = ForgeArray(np.array([1-2j, 3-4j]))
        c = a + b
        np.testing.assert_array_equal(c.data.ravel(), [2+0j, 6+0j])

    def test_complex_multiply(self):
        """R-TYPE-06.05: (1+i)(1-i) = 2 verifies complex multiplication."""
        a = ForgeArray(np.array([1+1j]))
        b = ForgeArray(np.array([1-1j]))
        c = a * b  # (1+i)(1-i) = 2
        assert abs(c.data.flat[0] - 2.0) < 1e-15


class TestSpecialValues:
    """R-TYPE-07: The type system SHALL provide correct special constants (pi,
    e, Inf, NaN, eps, i, j, true, false) with proper arithmetic propagation.

    Model-user argument: The engineer uses pi for trigonometry, Inf for
    unbounded limits, NaN for missing data, eps for numerical tolerance
    checks, and i/j for complex numbers. These constants must have exact
    IEEE 754 semantics: NaN propagates through arithmetic, Inf absorbs
    finite additions, and Inf-Inf yields NaN.

    Decomposition: pi, e, Inf, NaN, eps, i/j, true/false, NaN propagation,
    and Inf arithmetic tested. Consistency: These cover all special
    constants and their critical arithmetic behaviors.
    """

    def test_pi(self):
        """R-TYPE-07.01: FORGE_PI equals IEEE 754 pi."""
        assert abs(FORGE_PI - 3.141592653589793) < 1e-15

    def test_e(self):
        """R-TYPE-07.02: FORGE_E equals IEEE 754 e."""
        assert abs(FORGE_E - 2.718281828459045) < 1e-15

    def test_inf(self):
        """R-TYPE-07.03: FORGE_INF is positive infinity."""
        assert np.isinf(FORGE_INF)
        assert FORGE_INF > 0

    def test_nan(self):
        """R-TYPE-07.04: FORGE_NAN is NaN."""
        assert np.isnan(FORGE_NAN)

    def test_eps(self):
        """R-TYPE-07.05: FORGE_EPS is positive, less than 1e-15, and distinguishes 1.0+eps from 1.0."""
        assert FORGE_EPS > 0
        assert FORGE_EPS < 1e-15
        assert 1.0 + FORGE_EPS != 1.0

    def test_i_j(self):
        """R-TYPE-07.06: FORGE_I and FORGE_J both equal 1j."""
        assert FORGE_I == 1j
        assert FORGE_J == 1j

    def test_true_false(self):
        """R-TYPE-07.07: FORGE_TRUE is truthy, FORGE_FALSE is falsy."""
        assert bool(FORGE_TRUE)
        assert not bool(FORGE_FALSE)

    def test_nan_propagation(self):
        """R-TYPE-07.08: NaN propagates through addition without infecting neighbors."""
        a = ForgeArray([1.0, FORGE_NAN, 3.0])
        b = a + 1
        assert not np.isnan(b.data.ravel()[0])
        assert np.isnan(b.data.ravel()[1])
        assert not np.isnan(b.data.ravel()[2])

    def test_inf_arithmetic(self):
        """R-TYPE-07.09: Inf+1=Inf, Inf*2=Inf, Inf-Inf=NaN."""
        a = ForgeArray([FORGE_INF])
        assert np.isinf((a + 1).data.flat[0])
        assert np.isinf((a * 2).data.flat[0])
        assert np.isnan((a - a).data.flat[0])  # Inf - Inf = NaN


class TestTypeChecking:
    """R-TYPE-08: The type system SHALL provide type-checking predicates
    (isnumeric, islogical, isfloat, isinteger, isnan, isinf, isfinite, isa)
    that return correct results for all ForgeArray types.

    Model-user argument: The engineer uses type checks to write generic code
    that adapts to input types (e.g., ``if isfloat(x)``). These predicates
    must be reliable or type-dispatch logic fails silently.

    Decomposition: isnumeric, islogical, isfloat, isinteger, isnan, isinf,
    isfinite, and isa tested with positive and negative cases. Consistency:
    These eight predicates cover all type-checking functions.
    """

    def test_isnumeric(self):
        """R-TYPE-08.01: isnumeric returns true for double and int32, false for strings."""
        assert forge_isnumeric(ForgeArray([1.0]))
        assert forge_isnumeric(ForgeArray([1], dtype="int32"))
        assert not forge_isnumeric("hello")

    def test_islogical(self):
        """R-TYPE-08.02: islogical returns true for logical arrays, false for doubles."""
        assert forge_islogical(forge_logical([1, 0]))
        assert not forge_islogical(ForgeArray([1.0]))

    def test_isfloat(self):
        """R-TYPE-08.03: isfloat returns true for double and single, false for int32."""
        assert forge_isfloat(ForgeArray([1.0]))
        assert forge_isfloat(forge_single([1.0]))
        assert not forge_isfloat(ForgeArray([1], dtype="int32"))

    def test_isinteger(self):
        """R-TYPE-08.04: isinteger returns true for int32, false for double."""
        assert forge_isinteger(ForgeArray([1], dtype="int32"))
        assert not forge_isinteger(ForgeArray([1.0]))

    def test_isnan_array(self):
        """R-TYPE-08.05: isnan returns element-wise NaN detection."""
        r = forge_isnan(ForgeArray([1.0, FORGE_NAN, 3.0]))
        np.testing.assert_array_equal(r.data.ravel(), [False, True, False])

    def test_isinf_array(self):
        """R-TYPE-08.06: isinf returns true for both +Inf and -Inf."""
        r = forge_isinf(ForgeArray([1.0, FORGE_INF, -FORGE_INF]))
        np.testing.assert_array_equal(r.data.ravel(), [False, True, True])

    def test_isfinite_array(self):
        """R-TYPE-08.07: isfinite returns false for Inf and NaN."""
        r = forge_isfinite(ForgeArray([1.0, FORGE_INF, FORGE_NAN]))
        np.testing.assert_array_equal(r.data.ravel(), [True, False, False])

    def test_isa(self):
        """R-TYPE-08.08: isa checks class name match for double and int32."""
        assert forge_isa(ForgeArray([1.0]), "double")
        assert forge_isa(forge_int32([1]), "int32")
        assert not forge_isa(ForgeArray([1.0]), "single")


# ============================================================
# Stage 1.3: Matrix Construction
# ============================================================

class TestMatrixConstruction:
    """R-TYPE-09: The type system SHALL provide matrix construction functions
    (eye, ones, zeros, rand, randn, randi, true, false, diag, linspace,
    colon, repmat) that produce arrays with correct shape, values, and dtype.

    Model-user argument: Matrix constructors are among the most-used
    functions. The engineer calls ``zeros(100,100)`` to preallocate,
    ``eye(3)`` for identity matrices, ``linspace(0,1,100)`` for sample
    grids, and ``colon(1,0.5,10)`` for stepped ranges. Each must produce
    the correct array or numerical results are wrong from the start.

    Decomposition: eye (square, rectangular), ones (square, rect), zeros
    (square, typed), rand, randn, randi, true, false, diag (create, extract,
    superdiag), linspace (explicit, default), colon (basic, step, fractional,
    negative, empty), and repmat (tiling, identity) tested. Consistency:
    These 24 sub-tests cover every matrix constructor and its parameter
    variations.
    """

    def test_eye_square(self):
        """R-TYPE-09.01: eye(3) creates 3x3 identity."""
        a = forge_eye(3)
        assert a.shape == (3, 3)
        np.testing.assert_array_equal(a.data, np.eye(3))

    def test_eye_rectangular(self):
        """R-TYPE-09.02: eye(2,3) creates 2x3 matrix with diagonal ones."""
        a = forge_eye(2, 3)
        assert a.shape == (2, 3)
        assert a[1, 1] == 1
        assert a[1, 3] == 0

    def test_ones_square(self):
        """R-TYPE-09.03: ones(3) creates 3x3 matrix of ones."""
        a = forge_ones(3)
        assert a.shape == (3, 3)
        assert np.all(a.data == 1)

    def test_ones_rect(self):
        """R-TYPE-09.04: ones(2,3) creates 2x3 matrix."""
        a = forge_ones(2, 3)
        assert a.shape == (2, 3)

    def test_zeros_square(self):
        """R-TYPE-09.05: zeros(3) creates 3x3 matrix of zeros."""
        a = forge_zeros(3)
        assert a.shape == (3, 3)
        assert np.all(a.data == 0)

    def test_zeros_typed(self):
        """R-TYPE-09.06: zeros with int32 dtype creates int32 array."""
        a = forge_zeros(2, 3, dtype="int32")
        assert a.dtype == np.int32

    def test_rand_shape(self):
        """R-TYPE-09.07: rand(3,4) creates 3x4 array in [0,1)."""
        a = forge_rand(3, 4)
        assert a.shape == (3, 4)
        assert np.all(a.data >= 0) and np.all(a.data < 1)

    def test_randn_shape(self):
        """R-TYPE-09.08: randn(100,1) creates 100x1 array with mean near 0."""
        a = forge_randn(100, 1)
        assert a.shape == (100, 1)
        # Mean should be near 0 for large sample
        assert abs(np.mean(a.data)) < 0.5

    def test_randi_range(self):
        """R-TYPE-09.09: randi(10,100,1) values are in [1,10]."""
        a = forge_randi(10, 100, 1)
        assert a.shape == (100, 1)
        assert np.all(a.data >= 1) and np.all(a.data <= 10)

    def test_true_matrix(self):
        """R-TYPE-09.10: true(2,3) creates 2x3 boolean matrix, all true."""
        a = forge_true(2, 3)
        assert a.shape == (2, 3)
        assert a.dtype == np.bool_
        assert np.all(a.data)

    def test_false_matrix(self):
        """R-TYPE-09.11: false(2,3) creates 2x3 boolean matrix, all false."""
        a = forge_false(2, 3)
        assert a.shape == (2, 3)
        assert not np.any(a.data)

    def test_diag_create(self):
        """R-TYPE-09.12: diag([1,2,3]) creates 3x3 diagonal matrix."""
        a = forge_diag(ForgeArray([1, 2, 3]))
        assert a.shape == (3, 3)
        assert a[1, 1] == 1
        assert a[2, 2] == 2
        assert a[3, 3] == 3
        assert a[1, 2] == 0

    def test_diag_extract(self):
        """R-TYPE-09.13: diag(matrix) extracts the main diagonal."""
        a = ForgeArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        d = forge_diag(a)
        np.testing.assert_array_equal(d.data.ravel(), [1, 5, 9])

    def test_diag_superdiag(self):
        """R-TYPE-09.14: diag(matrix, 1) extracts the first super-diagonal."""
        a = ForgeArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        d = forge_diag(a, 1)
        np.testing.assert_array_equal(d.data.ravel(), [2, 6])

    def test_linspace(self):
        """R-TYPE-09.15: linspace(0,1,5) produces [0, 0.25, 0.5, 0.75, 1.0]."""
        a = forge_linspace(0, 1, 5)
        np.testing.assert_array_almost_equal(a.data.ravel(), [0, 0.25, 0.5, 0.75, 1.0])

    def test_linspace_default_100(self):
        """R-TYPE-09.16: linspace(0,1) defaults to 100 points."""
        a = forge_linspace(0, 1)
        assert a.numel() == 100

    def test_colon_basic(self):
        """R-TYPE-09.17: colon(1,5) produces [1, 2, 3, 4, 5]."""
        a = forge_colon(1, 5)
        np.testing.assert_array_equal(a.data.ravel(), [1, 2, 3, 4, 5])

    def test_colon_step(self):
        """R-TYPE-09.18: colon(1,2,9) produces [1, 3, 5, 7, 9]."""
        a = forge_colon(1, 2, 9)
        np.testing.assert_array_equal(a.data.ravel(), [1, 3, 5, 7, 9])

    def test_colon_fractional(self):
        """R-TYPE-09.19: colon(0,0.5,2) produces [0, 0.5, 1.0, 1.5, 2.0]."""
        a = forge_colon(0, 0.5, 2)
        np.testing.assert_array_almost_equal(a.data.ravel(), [0, 0.5, 1.0, 1.5, 2.0])

    def test_colon_negative_step(self):
        """R-TYPE-09.20: colon(5,-1,1) produces [5, 4, 3, 2, 1]."""
        a = forge_colon(5, -1, 1)
        np.testing.assert_array_equal(a.data.ravel(), [5, 4, 3, 2, 1])

    def test_colon_empty(self):
        """R-TYPE-09.21: colon(5,1) with step 1 produces empty array."""
        a = forge_colon(5, 1)  # 5:1 with step 1 = empty
        assert a.numel() == 0

    def test_repmat(self):
        """R-TYPE-09.22: repmat([1 2; 3 4], 2, 3) produces 4x6 tiled matrix."""
        a = ForgeArray([[1, 2], [3, 4]])
        b = forge_repmat(a, 2, 3)
        assert b.shape == (4, 6)
        assert b[1, 1] == 1
        assert b[3, 3] == 1
        assert b[1, 3] == 1

    def test_repmat_identity(self):
        """R-TYPE-09.23: repmat(A, 1, 1) returns a copy with same data."""
        a = ForgeArray([[1, 2], [3, 4]])
        b = forge_repmat(a, 1, 1)
        np.testing.assert_array_equal(b.data, a.data)


class TestForgeArrayMisc:
    """R-TYPE-10: ForgeArray SHALL support copy semantics, transpose, repr,
    float/int/bool conversion, and unwrap to raw NumPy arrays.

    Model-user argument: The engineer expects value semantics (modifying a
    copy does not affect the original), transpose via .T, inspectable repr
    for debugging, and seamless conversion to Python scalars for interfacing
    with non-array code.

    Decomposition: copy independence, transpose shape, repr for scalar and
    matrix, float/int/bool conversion, and _unwrap tested. Consistency:
    These cover all utility methods on ForgeArray.
    """

    def test_copy(self):
        """R-TYPE-10.01: Copy is independent; modifying copy does not affect original."""
        a = ForgeArray([1, 2, 3])
        b = a.copy()
        b[1] = 99
        assert a[1] == 1  # Original unchanged

    def test_transpose(self):
        """R-TYPE-10.02: Transpose swaps rows and columns."""
        a = ForgeArray([[1, 2, 3], [4, 5, 6]])
        t = a.T
        assert t.shape == (3, 2)
        assert t[1, 1] == 1
        assert t[1, 2] == 4

    def test_repr_scalar(self):
        """R-TYPE-10.03: Scalar repr includes the numeric value."""
        a = ForgeArray(5.0)
        assert "5.0" in repr(a)

    def test_repr_matrix(self):
        """R-TYPE-10.04: Matrix repr includes shape dimensions."""
        a = ForgeArray([[1, 2], [3, 4]])
        assert "(2, 2)" in repr(a)

    def test_float_conversion(self):
        """R-TYPE-10.05: float() conversion returns correct Python float."""
        a = ForgeArray(3.14)
        assert abs(float(a) - 3.14) < 1e-15

    def test_int_conversion(self):
        """R-TYPE-10.06: int() conversion returns correct Python int."""
        a = ForgeArray(42)
        assert int(a) == 42

    def test_bool_conversion(self):
        """R-TYPE-10.07: bool() returns true for nonzero, false for zero."""
        assert bool(ForgeArray(1))
        assert not bool(ForgeArray(0))

    def test_unwrap(self):
        """R-TYPE-10.08: _unwrap extracts raw ndarray from ForgeArray and passes through scalars."""
        a = ForgeArray([1, 2, 3])
        assert isinstance(_unwrap(a), np.ndarray)
        assert isinstance(_unwrap(5), int)
