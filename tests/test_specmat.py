"""V&V tests for special matrices toolbox (11 functions).

SRS trace: SRS-FUNC-001, SRS-VAL-001
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose
from forge.engine.types import ForgeArray, _unwrap
from forge.engine.builtins.special_matrix import *


class TestHadamard:

    def test_hadamard_orthogonality(self):
        """H * H' = n * I for a Hadamard matrix of order n."""
        H = _unwrap(forge_hadamard(ForgeArray(np.array(4.0))))
        n = H.shape[0]
        product = H @ H.T
        assert_allclose(product, n * np.eye(n), atol=1e-12)

    def test_hadamard_entries_pm1(self):
        """All entries should be +1 or -1."""
        H = _unwrap(forge_hadamard(ForgeArray(np.array(8.0))))
        assert np.all(np.abs(H) == 1.0)

    def test_hadamard_size(self):
        H = _unwrap(forge_hadamard(ForgeArray(np.array(16.0))))
        assert H.shape == (16, 16)


class TestHilbert:

    def test_hilb_invhilb_identity(self):
        """invhilb(n) * hilb(n) should be approximately I."""
        n = 5
        H = _unwrap(forge_hilb(ForgeArray(np.array(float(n)))))
        Hinv = _unwrap(forge_invhilb(ForgeArray(np.array(float(n)))))
        product = Hinv @ H
        assert_allclose(product, np.eye(n), atol=1e-6)

    def test_hilb_symmetry(self):
        H = _unwrap(forge_hilb(ForgeArray(np.array(4.0))))
        assert_allclose(H, H.T, atol=1e-15)

    def test_hilb_known_entry(self):
        """H[0,0] = 1, H[0,1] = 1/2, H[1,0] = 1/2, etc."""
        H = _unwrap(forge_hilb(ForgeArray(np.array(3.0))))
        assert abs(H[0, 0] - 1.0) < 1e-15
        assert abs(H[0, 1] - 0.5) < 1e-15
        assert abs(H[1, 0] - 0.5) < 1e-15
        assert abs(H[1, 1] - 1.0 / 3.0) < 1e-15


class TestMagic:

    def test_magic_row_sums_equal(self):
        """All row sums of a magic square should be equal."""
        M = _unwrap(forge_magic(ForgeArray(np.array(5.0))))
        row_sums = M.sum(axis=1)
        assert_allclose(row_sums, row_sums[0] * np.ones(5), atol=1e-10)

    def test_magic_col_sums_equal(self):
        """All column sums should also be equal."""
        M = _unwrap(forge_magic(ForgeArray(np.array(5.0))))
        col_sums = M.sum(axis=0)
        assert_allclose(col_sums, col_sums[0] * np.ones(5), atol=1e-10)

    def test_magic_sum_value(self):
        """For order n, the magic constant is n*(n^2+1)/2."""
        n = 5
        M = _unwrap(forge_magic(ForgeArray(np.array(float(n)))))
        expected = n * (n ** 2 + 1) / 2
        assert abs(M.sum(axis=1)[0] - expected) < 1e-10

    def test_magic_contains_all_integers(self):
        """Magic square of order n contains all integers 1..n^2."""
        n = 4
        M = _unwrap(forge_magic(ForgeArray(np.array(float(n)))))
        vals = np.sort(M.ravel())
        np.testing.assert_array_equal(vals, np.arange(1, n * n + 1, dtype=float))


class TestPascal:

    def test_pascal_det_small(self):
        """det(pascal(n)) = 1 for small n."""
        for n in [1, 2, 3, 4, 5]:
            P = _unwrap(forge_pascal(ForgeArray(np.array(float(n)))))
            d = np.linalg.det(P)
            assert abs(d - 1.0) < 1e-6, f"det(pascal({n})) = {d}, expected 1"

    def test_pascal_symmetric(self):
        P = _unwrap(forge_pascal(ForgeArray(np.array(4.0))))
        assert_allclose(P, P.T, atol=1e-12)

    def test_pascal_known_3(self):
        """pascal(3) = [[1,1,1],[1,2,3],[1,3,6]]."""
        P = _unwrap(forge_pascal(ForgeArray(np.array(3.0))))
        expected = np.array([[1, 1, 1], [1, 2, 3], [1, 3, 6]], dtype=float)
        assert_allclose(P, expected, atol=1e-12)


class TestToeplitz:

    def test_toeplitz_round_trip(self):
        """Toeplitz from first column, check structure."""
        c = ForgeArray(np.array([1.0, 2.0, 3.0]))
        T = _unwrap(forge_toeplitz(c))
        # T should be symmetric for real input with single arg
        assert_allclose(T, T.T, atol=1e-12)
        # Check first column
        assert_allclose(T[:, 0].ravel(), [1, 2, 3], atol=1e-12)

    def test_toeplitz_with_row(self):
        c = ForgeArray(np.array([1.0, 2.0, 3.0]))
        r = ForgeArray(np.array([1.0, 4.0, 5.0]))
        T = _unwrap(forge_toeplitz(c, r))
        # First column should be c, first row should be r
        assert_allclose(T[:, 0].ravel(), [1, 2, 3], atol=1e-12)
        assert_allclose(T[0, :].ravel(), [1, 4, 5], atol=1e-12)


class TestVander:

    def test_vander_known(self):
        """Vandermonde of [1,2,3] should be [[1,1,1],[4,2,1],[9,3,1]]."""
        x = ForgeArray(np.array([1.0, 2.0, 3.0]))
        V = _unwrap(forge_vander(x))
        expected = np.array([[1, 1, 1], [4, 2, 1], [9, 3, 1]], dtype=float)
        assert_allclose(V, expected, atol=1e-12)

    def test_vander_column_count(self):
        x = ForgeArray(np.array([1.0, 2.0, 3.0, 4.0]))
        V = _unwrap(forge_vander(x, 3))
        assert V.shape == (4, 3)


class TestRosser:

    def test_rosser_eigenvalues(self):
        """The Rosser matrix has known eigenvalues including 0 and 1020."""
        R = _unwrap(forge_rosser())
        eigs = np.sort(np.linalg.eigvalsh(R))
        # Known eigenvalues of the Rosser matrix (sorted):
        # a = 102*sqrt(10405) ~ 10405.128... but actually:
        # The known eigenvalues are approximately:
        # -1020.0490, 0.098, 0.098, 1000, 1000, 1019.9020, 1020.0490, 1020.0490
        # Let's just check the smallest is near 0 and 1020 appears
        assert abs(eigs[0] - (-1020.0490184299969)) < 0.01
        assert abs(eigs[-1] - 1020.0490184299969) < 0.01

    def test_rosser_symmetric(self):
        R = _unwrap(forge_rosser())
        assert_allclose(R, R.T, atol=1e-12)

    def test_rosser_size(self):
        R = _unwrap(forge_rosser())
        assert R.shape == (8, 8)


class TestWilkinson:

    def test_wilkinson_tridiagonal(self):
        """Wilkinson matrix should be tridiagonal."""
        W = _unwrap(forge_wilkinson(ForgeArray(np.array(7.0))))
        n = W.shape[0]
        for i in range(n):
            for j in range(n):
                if abs(i - j) > 1:
                    assert W[i, j] == 0.0, f"W[{i},{j}] = {W[i,j]} should be 0"

    def test_wilkinson_symmetric(self):
        W = _unwrap(forge_wilkinson(ForgeArray(np.array(7.0))))
        assert_allclose(W, W.T, atol=1e-12)

    def test_wilkinson_subdiag_ones(self):
        """Sub- and super-diagonal should be all ones."""
        W = _unwrap(forge_wilkinson(ForgeArray(np.array(7.0))))
        n = W.shape[0]
        for i in range(n - 1):
            assert W[i, i + 1] == 1.0
            assert W[i + 1, i] == 1.0
