# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Tests for ForgeArray data types, 1-based indexing, type conversion, and matrix construction.
Covers Stages 1.1, 1.2, 1.3 from the V&V plan."""
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
    def test_from_scalar(self):
        a = ForgeArray(5.0)
        assert a.shape == (1, 1)  # Scalar is 1x1 in Octave
        assert a.isscalar()
        assert float(a) == 5.0

    def test_from_list(self):
        a = ForgeArray([1, 2, 3])
        assert a.shape == (1, 3)  # Row vector
        assert a.isvector()

    def test_from_2d_list(self):
        a = ForgeArray([[1, 2], [3, 4]])
        assert a.shape == (2, 2)
        assert a.issquare()

    def test_from_numpy(self):
        arr = np.array([1.0, 2.0, 3.0])
        a = ForgeArray(arr)
        assert a.shape == (1, 3)

    def test_from_forge_array(self):
        a = ForgeArray([1, 2, 3])
        b = ForgeArray(a)
        assert b.shape == a.shape
        np.testing.assert_array_equal(b.data, a.data)

    def test_dtype_specification(self):
        a = ForgeArray([1, 2, 3], dtype="int32")
        assert a.dtype == np.int32
        assert a.type_name() == "int32"

    def test_empty_array(self):
        a = ForgeArray(np.array([]))
        assert a.isempty()
        assert a.numel() == 0


class TestForgeArrayIndexing:
    def test_1based_scalar(self):
        a = ForgeArray([10, 20, 30])
        assert a[1] == 10  # First element
        assert a[2] == 20
        assert a[3] == 30

    def test_1based_error_on_zero(self):
        a = ForgeArray([10, 20, 30])
        with pytest.raises(IndexError):
            _ = a[0]

    def test_1based_2d(self):
        a = ForgeArray([[1, 2, 3], [4, 5, 6]])
        assert a[1, 1] == 1
        assert a[1, 3] == 3
        assert a[2, 1] == 4
        assert a[2, 3] == 6

    def test_slice_1based(self):
        a = ForgeArray([10, 20, 30, 40, 50])
        s = a[2:4]  # Elements 2,3,4 (1-based) → indices 1,2,3 (0-based)
        np.testing.assert_array_equal(s.data.ravel(), [20, 30, 40])

    def test_assignment_1based(self):
        a = ForgeArray([10, 20, 30])
        a[1] = 99
        assert a[1] == 99
        assert a[2] == 20

    def test_2d_assignment(self):
        a = ForgeArray([[1, 2], [3, 4]])
        a[1, 2] = 99
        assert a[1, 2] == 99
        assert a[1, 1] == 1

    def test_logical_indexing(self):
        a = ForgeArray([10, 20, 30, 40])
        mask = ForgeArray(np.array([True, False, True, False]))
        result = a[mask]
        np.testing.assert_array_equal(result.data.ravel(), [10, 30])


class TestForgeArraySizeInfo:
    def test_numel(self):
        assert ForgeArray([[1, 2], [3, 4]]).numel() == 4

    def test_length(self):
        assert ForgeArray([[1, 2, 3], [4, 5, 6]]).length() == 3  # max(2,3)

    def test_rows_columns(self):
        a = ForgeArray([[1, 2, 3], [4, 5, 6]])
        assert a.rows() == 2
        assert a.columns() == 3

    def test_ismatrix(self):
        assert ForgeArray([[1, 2], [3, 4]]).ismatrix()

    def test_isvector_row(self):
        assert ForgeArray([1, 2, 3]).isvector()

    def test_isvector_column(self):
        assert ForgeArray(np.array([[1], [2], [3]])).isvector()

    def test_not_vector(self):
        assert not ForgeArray([[1, 2], [3, 4]]).isvector()


class TestForgeArrayArithmetic:
    def test_add(self):
        a = ForgeArray([1, 2, 3])
        b = ForgeArray([10, 20, 30])
        c = a + b
        np.testing.assert_array_equal(c.data.ravel(), [11, 22, 33])

    def test_scalar_add(self):
        a = ForgeArray([1, 2, 3])
        c = a + 10
        np.testing.assert_array_equal(c.data.ravel(), [11, 12, 13])

    def test_sub(self):
        a = ForgeArray([10, 20, 30])
        b = ForgeArray([1, 2, 3])
        np.testing.assert_array_equal((a - b).data.ravel(), [9, 18, 27])

    def test_mul(self):
        a = ForgeArray([2, 3, 4])
        np.testing.assert_array_equal((a * 2).data.ravel(), [4, 6, 8])

    def test_div(self):
        a = ForgeArray([10.0, 20.0, 30.0])
        np.testing.assert_array_equal((a / 10).data.ravel(), [1, 2, 3])

    def test_power(self):
        a = ForgeArray([2, 3, 4])
        np.testing.assert_array_equal((a ** 2).data.ravel(), [4, 9, 16])

    def test_negation(self):
        a = ForgeArray([1, -2, 3])
        np.testing.assert_array_equal((-a).data.ravel(), [-1, 2, -3])

    def test_matmul(self):
        a = ForgeArray([[1, 2], [3, 4]])
        b = ForgeArray([[5, 6], [7, 8]])
        c = a @ b
        np.testing.assert_array_equal(c.data, [[19, 22], [43, 50]])

    def test_comparison_eq(self):
        a = ForgeArray([1, 2, 3])
        b = ForgeArray([1, 5, 3])
        r = (a == b)
        np.testing.assert_array_equal(r.data.ravel(), [True, False, True])

    def test_comparison_lt(self):
        a = ForgeArray([1, 5, 3])
        r = (a < 3)
        np.testing.assert_array_equal(r.data.ravel(), [True, False, False])


class TestTypeConversion:
    def test_double(self):
        a = forge_double(ForgeArray([1, 2, 3], dtype="int32"))
        assert a.dtype == np.float64
        assert a.type_name() == "double"

    def test_single(self):
        a = forge_single([1.0, 2.0])
        assert a.dtype == np.float32

    def test_int8_overflow(self):
        a = forge_int8([200])
        # int8 range is -128..127, 200 overflows/wraps
        assert a.dtype == np.int8
        assert a.data.ravel()[0] == np.array(200).astype(np.int8)  # Wraps

    def test_uint8_range(self):
        a = forge_uint8([0, 128, 255])
        np.testing.assert_array_equal(a.data.ravel(), [0, 128, 255])

    def test_logical(self):
        a = forge_logical([0, 1, 5, 0])
        np.testing.assert_array_equal(a.data.ravel(), [False, True, True, False])

    def test_all_integer_types(self):
        for name, func in [("int8", forge_int8), ("int16", forge_int16),
                           ("int32", forge_int32), ("int64", forge_int64),
                           ("uint8", forge_uint8), ("uint16", forge_uint16),
                           ("uint32", forge_uint32), ("uint64", forge_uint64)]:
            a = func([1, 2, 3])
            assert a.type_name() == name

    def test_astype(self):
        a = ForgeArray([1.5, 2.7, 3.9])
        b = a.astype("int32")
        assert b.dtype == np.int32
        np.testing.assert_array_equal(b.data.ravel(), [1, 2, 3])


# ============================================================
# Stage 1.2: Complex Numbers & Special Values
# ============================================================

class TestComplex:
    def test_complex_creation(self):
        a = forge_complex(3, 4)
        assert a.isscalar()
        assert complex(a.data.flat[0]) == 3 + 4j

    def test_complex_array(self):
        a = forge_complex([1, 2, 3], [4, 5, 6])
        np.testing.assert_array_equal(a.data.ravel(), [1+4j, 2+5j, 3+6j])

    def test_complex_real_only(self):
        a = forge_complex([1.0, 2.0])
        assert a.dtype == np.complex128

    def test_complex_arithmetic(self):
        a = ForgeArray(np.array([1+2j, 3+4j]))
        b = ForgeArray(np.array([1-2j, 3-4j]))
        c = a + b
        np.testing.assert_array_equal(c.data.ravel(), [2+0j, 6+0j])

    def test_complex_multiply(self):
        a = ForgeArray(np.array([1+1j]))
        b = ForgeArray(np.array([1-1j]))
        c = a * b  # (1+i)(1-i) = 2
        assert abs(c.data.flat[0] - 2.0) < 1e-15


class TestSpecialValues:
    def test_pi(self):
        assert abs(FORGE_PI - 3.141592653589793) < 1e-15

    def test_e(self):
        assert abs(FORGE_E - 2.718281828459045) < 1e-15

    def test_inf(self):
        assert np.isinf(FORGE_INF)
        assert FORGE_INF > 0

    def test_nan(self):
        assert np.isnan(FORGE_NAN)

    def test_eps(self):
        assert FORGE_EPS > 0
        assert FORGE_EPS < 1e-15
        assert 1.0 + FORGE_EPS != 1.0

    def test_i_j(self):
        assert FORGE_I == 1j
        assert FORGE_J == 1j

    def test_true_false(self):
        assert bool(FORGE_TRUE)
        assert not bool(FORGE_FALSE)

    def test_nan_propagation(self):
        a = ForgeArray([1.0, FORGE_NAN, 3.0])
        b = a + 1
        assert not np.isnan(b.data.ravel()[0])
        assert np.isnan(b.data.ravel()[1])
        assert not np.isnan(b.data.ravel()[2])

    def test_inf_arithmetic(self):
        a = ForgeArray([FORGE_INF])
        assert np.isinf((a + 1).data.flat[0])
        assert np.isinf((a * 2).data.flat[0])
        assert np.isnan((a - a).data.flat[0])  # Inf - Inf = NaN


class TestTypeChecking:
    def test_isnumeric(self):
        assert forge_isnumeric(ForgeArray([1.0]))
        assert forge_isnumeric(ForgeArray([1], dtype="int32"))
        assert not forge_isnumeric("hello")

    def test_islogical(self):
        assert forge_islogical(forge_logical([1, 0]))
        assert not forge_islogical(ForgeArray([1.0]))

    def test_isfloat(self):
        assert forge_isfloat(ForgeArray([1.0]))
        assert forge_isfloat(forge_single([1.0]))
        assert not forge_isfloat(ForgeArray([1], dtype="int32"))

    def test_isinteger(self):
        assert forge_isinteger(ForgeArray([1], dtype="int32"))
        assert not forge_isinteger(ForgeArray([1.0]))

    def test_isnan_array(self):
        r = forge_isnan(ForgeArray([1.0, FORGE_NAN, 3.0]))
        np.testing.assert_array_equal(r.data.ravel(), [False, True, False])

    def test_isinf_array(self):
        r = forge_isinf(ForgeArray([1.0, FORGE_INF, -FORGE_INF]))
        np.testing.assert_array_equal(r.data.ravel(), [False, True, True])

    def test_isfinite_array(self):
        r = forge_isfinite(ForgeArray([1.0, FORGE_INF, FORGE_NAN]))
        np.testing.assert_array_equal(r.data.ravel(), [True, False, False])

    def test_isa(self):
        assert forge_isa(ForgeArray([1.0]), "double")
        assert forge_isa(forge_int32([1]), "int32")
        assert not forge_isa(ForgeArray([1.0]), "single")


# ============================================================
# Stage 1.3: Matrix Construction
# ============================================================

class TestMatrixConstruction:
    def test_eye_square(self):
        a = forge_eye(3)
        assert a.shape == (3, 3)
        np.testing.assert_array_equal(a.data, np.eye(3))

    def test_eye_rectangular(self):
        a = forge_eye(2, 3)
        assert a.shape == (2, 3)
        assert a[1, 1] == 1
        assert a[1, 3] == 0

    def test_ones_square(self):
        a = forge_ones(3)
        assert a.shape == (3, 3)
        assert np.all(a.data == 1)

    def test_ones_rect(self):
        a = forge_ones(2, 3)
        assert a.shape == (2, 3)

    def test_zeros_square(self):
        a = forge_zeros(3)
        assert a.shape == (3, 3)
        assert np.all(a.data == 0)

    def test_zeros_typed(self):
        a = forge_zeros(2, 3, dtype="int32")
        assert a.dtype == np.int32

    def test_rand_shape(self):
        a = forge_rand(3, 4)
        assert a.shape == (3, 4)
        assert np.all(a.data >= 0) and np.all(a.data < 1)

    def test_randn_shape(self):
        a = forge_randn(100, 1)
        assert a.shape == (100, 1)
        # Mean should be near 0 for large sample
        assert abs(np.mean(a.data)) < 0.5

    def test_randi_range(self):
        a = forge_randi(10, 100, 1)
        assert a.shape == (100, 1)
        assert np.all(a.data >= 1) and np.all(a.data <= 10)

    def test_true_matrix(self):
        a = forge_true(2, 3)
        assert a.shape == (2, 3)
        assert a.dtype == np.bool_
        assert np.all(a.data)

    def test_false_matrix(self):
        a = forge_false(2, 3)
        assert a.shape == (2, 3)
        assert not np.any(a.data)

    def test_diag_create(self):
        a = forge_diag(ForgeArray([1, 2, 3]))
        assert a.shape == (3, 3)
        assert a[1, 1] == 1
        assert a[2, 2] == 2
        assert a[3, 3] == 3
        assert a[1, 2] == 0

    def test_diag_extract(self):
        a = ForgeArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        d = forge_diag(a)
        np.testing.assert_array_equal(d.data.ravel(), [1, 5, 9])

    def test_diag_superdiag(self):
        a = ForgeArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        d = forge_diag(a, 1)
        np.testing.assert_array_equal(d.data.ravel(), [2, 6])

    def test_linspace(self):
        a = forge_linspace(0, 1, 5)
        np.testing.assert_array_almost_equal(a.data.ravel(), [0, 0.25, 0.5, 0.75, 1.0])

    def test_linspace_default_100(self):
        a = forge_linspace(0, 1)
        assert a.numel() == 100

    def test_colon_basic(self):
        a = forge_colon(1, 5)
        np.testing.assert_array_equal(a.data.ravel(), [1, 2, 3, 4, 5])

    def test_colon_step(self):
        a = forge_colon(1, 2, 9)
        np.testing.assert_array_equal(a.data.ravel(), [1, 3, 5, 7, 9])

    def test_colon_fractional(self):
        a = forge_colon(0, 0.5, 2)
        np.testing.assert_array_almost_equal(a.data.ravel(), [0, 0.5, 1.0, 1.5, 2.0])

    def test_colon_negative_step(self):
        a = forge_colon(5, -1, 1)
        np.testing.assert_array_equal(a.data.ravel(), [5, 4, 3, 2, 1])

    def test_colon_empty(self):
        a = forge_colon(5, 1)  # 5:1 with step 1 = empty
        assert a.numel() == 0

    def test_repmat(self):
        a = ForgeArray([[1, 2], [3, 4]])
        b = forge_repmat(a, 2, 3)
        assert b.shape == (4, 6)
        assert b[1, 1] == 1
        assert b[3, 3] == 1
        assert b[1, 3] == 1

    def test_repmat_identity(self):
        a = ForgeArray([[1, 2], [3, 4]])
        b = forge_repmat(a, 1, 1)
        np.testing.assert_array_equal(b.data, a.data)


class TestForgeArrayMisc:
    def test_copy(self):
        a = ForgeArray([1, 2, 3])
        b = a.copy()
        b[1] = 99
        assert a[1] == 1  # Original unchanged

    def test_transpose(self):
        a = ForgeArray([[1, 2, 3], [4, 5, 6]])
        t = a.T
        assert t.shape == (3, 2)
        assert t[1, 1] == 1
        assert t[1, 2] == 4

    def test_repr_scalar(self):
        a = ForgeArray(5.0)
        assert "5.0" in repr(a)

    def test_repr_matrix(self):
        a = ForgeArray([[1, 2], [3, 4]])
        assert "(2, 2)" in repr(a)

    def test_float_conversion(self):
        a = ForgeArray(3.14)
        assert abs(float(a) - 3.14) < 1e-15

    def test_int_conversion(self):
        a = ForgeArray(42)
        assert int(a) == 42

    def test_bool_conversion(self):
        assert bool(ForgeArray(1))
        assert not bool(ForgeArray(0))

    def test_unwrap(self):
        a = ForgeArray([1, 2, 3])
        assert isinstance(_unwrap(a), np.ndarray)
        assert isinstance(_unwrap(5), int)
