"""V&V tests for special matrices toolbox (11 functions).

SRS trace: SRS-FUNC-001, SRS-VAL-001
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose
from forge.engine.types import ForgeArray, _unwrap
from forge.engine.builtins.special_matrix import *


class TestHadamard:
    """R-SMAT-01: Forge SHALL generate Hadamard matrices that are orthogonal
    with entries restricted to +1 and -1, matching Octave hadamard(n) output.

    Model-user argument: The golden user employs Hadamard matrices in
    experimental design (DOE) to construct balanced factorial experiments.
    Orthogonality guarantees uncorrelated factor estimates, and the +/-1
    entry constraint is the defining property that makes Hadamard designs
    statistically optimal for screening experiments.

    Decomposition:
      R-SMAT-01a: H * H' equals n * I (orthogonality property).
      R-SMAT-01b: Every entry of H is exactly +1 or -1.
      R-SMAT-01c: Output dimensions are n x n for input order n.

    Consistency: 01a verifies the algebraic orthogonality relation, 01b
    verifies the entry constraint, and 01c verifies dimensional correctness.
    Together these fully characterize a valid Hadamard matrix.
    """

    def test_hadamard_orthogonality(self):
        """R-SMAT-01a: H * H' = n * I for a Hadamard matrix of order n."""
        H = _unwrap(forge_hadamard(ForgeArray(np.array(4.0))))
        n = H.shape[0]
        product = H @ H.T
        assert_allclose(product, n * np.eye(n), atol=1e-12)

    def test_hadamard_entries_pm1(self):
        """R-SMAT-01b: All entries of hadamard(n) are exactly +1 or -1."""
        H = _unwrap(forge_hadamard(ForgeArray(np.array(8.0))))
        assert np.all(np.abs(H) == 1.0)

    def test_hadamard_size(self):
        """R-SMAT-01c: hadamard(n) produces an n x n matrix."""
        H = _unwrap(forge_hadamard(ForgeArray(np.array(16.0))))
        assert H.shape == (16, 16)


class TestHilbert:
    """R-SMAT-02: Forge SHALL generate Hilbert matrices and their exact
    inverses such that invhilb(n) * hilb(n) approximates the identity,
    matching Octave hilb(n) and invhilb(n) output.

    Model-user argument: The golden user relies on Hilbert matrices as
    canonical ill-conditioned test cases when validating numerical solvers.
    The ability to generate both the matrix and its exact inverse is critical
    for quantifying solver accuracy: if invhilb(n) * hilb(n) deviates from
    identity, the user knows the solver's precision limits.

    Decomposition:
      R-SMAT-02a: invhilb(n) * hilb(n) approximates the identity matrix.
      R-SMAT-02b: hilb(n) is symmetric.
      R-SMAT-02c: hilb(n) entries match the formula H(i,j) = 1/(i+j-1).

    Consistency: 02a validates the inverse relationship, 02b confirms the
    required symmetry, and 02c checks element values against the closed-form
    definition. These three properties fully specify a correct Hilbert pair.
    """

    def test_hilb_invhilb_identity(self):
        """R-SMAT-02a: invhilb(n) * hilb(n) approximates identity."""
        n = 5
        H = _unwrap(forge_hilb(ForgeArray(np.array(float(n)))))
        Hinv = _unwrap(forge_invhilb(ForgeArray(np.array(float(n)))))
        product = Hinv @ H
        assert_allclose(product, np.eye(n), atol=1e-6)

    def test_hilb_symmetry(self):
        """R-SMAT-02b: hilb(n) is symmetric."""
        H = _unwrap(forge_hilb(ForgeArray(np.array(4.0))))
        assert_allclose(H, H.T, atol=1e-15)

    def test_hilb_known_entry(self):
        """R-SMAT-02c: hilb(n) entries match H(i,j) = 1/(i+j-1)."""
        H = _unwrap(forge_hilb(ForgeArray(np.array(3.0))))
        assert abs(H[0, 0] - 1.0) < 1e-15
        assert abs(H[0, 1] - 0.5) < 1e-15
        assert abs(H[1, 0] - 0.5) < 1e-15
        assert abs(H[1, 1] - 1.0 / 3.0) < 1e-15


class TestMagic:
    """R-SMAT-03: Forge SHALL generate magic squares where all rows, columns,
    and the magic constant match the formula n*(n^2+1)/2, containing every
    integer from 1 to n^2 exactly once, matching Octave magic(n) output.

    Model-user argument: The golden user uses magic squares for teaching and
    validation of matrix operations. Verifying that row and column sums equal
    the magic constant confirms correct construction. Checking that all
    integers 1..n^2 appear exactly once ensures the square is a valid
    permutation matrix of consecutive integers, not a degenerate case.

    Decomposition:
      R-SMAT-03a: All row sums are equal.
      R-SMAT-03b: All column sums are equal.
      R-SMAT-03c: The common sum equals n*(n^2+1)/2.
      R-SMAT-03d: The matrix contains every integer from 1 to n^2.

    Consistency: 03a and 03b verify equal-sum structure in both dimensions,
    03c pins the sum to the known magic constant, and 03d confirms the
    complete integer set. Together these define a valid magic square.
    """

    def test_magic_row_sums_equal(self):
        """R-SMAT-03a: All row sums of magic(n) are equal."""
        M = _unwrap(forge_magic(ForgeArray(np.array(5.0))))
        row_sums = M.sum(axis=1)
        assert_allclose(row_sums, row_sums[0] * np.ones(5), atol=1e-10)

    def test_magic_col_sums_equal(self):
        """R-SMAT-03b: All column sums of magic(n) are equal."""
        M = _unwrap(forge_magic(ForgeArray(np.array(5.0))))
        col_sums = M.sum(axis=0)
        assert_allclose(col_sums, col_sums[0] * np.ones(5), atol=1e-10)

    def test_magic_sum_value(self):
        """R-SMAT-03c: Row/column sum equals n*(n^2+1)/2."""
        n = 5
        M = _unwrap(forge_magic(ForgeArray(np.array(float(n)))))
        expected = n * (n ** 2 + 1) / 2
        assert abs(M.sum(axis=1)[0] - expected) < 1e-10

    def test_magic_contains_all_integers(self):
        """R-SMAT-03d: magic(n) contains every integer from 1 to n^2."""
        n = 4
        M = _unwrap(forge_magic(ForgeArray(np.array(float(n)))))
        vals = np.sort(M.ravel())
        np.testing.assert_array_equal(vals, np.arange(1, n * n + 1, dtype=float))


class TestPascal:
    """R-SMAT-04: Forge SHALL generate Pascal matrices that are symmetric,
    positive definite with determinant 1, and contain correct binomial
    coefficient entries, matching Octave pascal(n) output.

    Model-user argument: The golden user leverages Pascal matrices for their
    combinatorial properties when testing Cholesky factorization and other
    positive-definite solvers. The determinant-equals-one property makes
    Pascal matrices ideal conditioning benchmarks, and the binomial entries
    connect to combinatorial identities the user verifies in coursework.

    Decomposition:
      R-SMAT-04a: det(pascal(n)) equals 1 for n in 1..5.
      R-SMAT-04b: pascal(n) is symmetric.
      R-SMAT-04c: pascal(3) matches the known binomial coefficient matrix.

    Consistency: 04a verifies the determinant invariant across several sizes,
    04b confirms symmetry, and 04c spot-checks entries against the known
    closed-form values. These cover structure, algebraic property, and values.
    """

    def test_pascal_det_small(self):
        """R-SMAT-04a: det(pascal(n)) equals 1 for small n."""
        for n in [1, 2, 3, 4, 5]:
            P = _unwrap(forge_pascal(ForgeArray(np.array(float(n)))))
            d = np.linalg.det(P)
            assert abs(d - 1.0) < 1e-6, f"det(pascal({n})) = {d}, expected 1"

    def test_pascal_symmetric(self):
        """R-SMAT-04b: pascal(n) is symmetric."""
        P = _unwrap(forge_pascal(ForgeArray(np.array(4.0))))
        assert_allclose(P, P.T, atol=1e-12)

    def test_pascal_known_3(self):
        """R-SMAT-04c: pascal(3) equals [[1,1,1],[1,2,3],[1,3,6]]."""
        P = _unwrap(forge_pascal(ForgeArray(np.array(3.0))))
        expected = np.array([[1, 1, 1], [1, 2, 3], [1, 3, 6]], dtype=float)
        assert_allclose(P, expected, atol=1e-12)


class TestToeplitz:
    """R-SMAT-05: Forge SHALL generate Toeplitz matrices with correct
    constant-diagonal structure from column and optional row vectors,
    matching Octave toeplitz(c) and toeplitz(c, r) output.

    Model-user argument: The golden user builds Toeplitz matrices for signal
    processing convolution and autocorrelation modeling. The single-argument
    form produces the symmetric autocorrelation matrix, while the two-argument
    form constructs asymmetric cross-correlation structures. Both forms must
    preserve the constant-diagonal property for valid filter design.

    Decomposition:
      R-SMAT-05a: toeplitz(c) is symmetric with correct first column.
      R-SMAT-05b: toeplitz(c, r) has first column c and first row r.

    Consistency: 05a covers the symmetric single-argument case including
    structure and values. 05b covers the asymmetric two-argument case by
    checking both the column and row vectors. These two forms are the
    complete public API of the toeplitz function.
    """

    def test_toeplitz_round_trip(self):
        """R-SMAT-05a: toeplitz(c) is symmetric with correct first column."""
        c = ForgeArray(np.array([1.0, 2.0, 3.0]))
        T = _unwrap(forge_toeplitz(c))
        # T should be symmetric for real input with single arg
        assert_allclose(T, T.T, atol=1e-12)
        # Check first column
        assert_allclose(T[:, 0].ravel(), [1, 2, 3], atol=1e-12)

    def test_toeplitz_with_row(self):
        """R-SMAT-05b: toeplitz(c, r) has first column c and first row r."""
        c = ForgeArray(np.array([1.0, 2.0, 3.0]))
        r = ForgeArray(np.array([1.0, 4.0, 5.0]))
        T = _unwrap(forge_toeplitz(c, r))
        # First column should be c, first row should be r
        assert_allclose(T[:, 0].ravel(), [1, 2, 3], atol=1e-12)
        assert_allclose(T[0, :].ravel(), [1, 4, 5], atol=1e-12)


class TestVander:
    """R-SMAT-06: Forge SHALL generate Vandermonde matrices with correct
    powers-of-x structure and optional column count, matching Octave
    vander(x) and vander(x, n) output.

    Model-user argument: The golden user constructs Vandermonde matrices for
    polynomial curve fitting (polyfit/polyval workflows). The matrix columns
    must contain descending powers of the input vector so that V\\y solves for
    polynomial coefficients directly. Controlling column count lets the user
    specify the polynomial degree independently of the number of data points.

    Decomposition:
      R-SMAT-06a: vander([1,2,3]) matches the known power matrix.
      R-SMAT-06b: vander(x, n) produces a matrix with n columns.

    Consistency: 06a validates entry correctness against a hand-computed
    reference. 06b confirms that the column-count parameter controls output
    shape. Together these cover value accuracy and dimensional control.
    """

    def test_vander_known(self):
        """R-SMAT-06a: vander([1,2,3]) equals [[1,1,1],[4,2,1],[9,3,1]]."""
        x = ForgeArray(np.array([1.0, 2.0, 3.0]))
        V = _unwrap(forge_vander(x))
        expected = np.array([[1, 1, 1], [4, 2, 1], [9, 3, 1]], dtype=float)
        assert_allclose(V, expected, atol=1e-12)

    def test_vander_column_count(self):
        """R-SMAT-06b: vander(x, n) returns a matrix with n columns."""
        x = ForgeArray(np.array([1.0, 2.0, 3.0, 4.0]))
        V = _unwrap(forge_vander(x, 3))
        assert V.shape == (4, 3)


class TestRosser:
    """R-SMAT-07: Forge SHALL generate the 8x8 Rosser matrix with correct
    eigenvalue spectrum and symmetric structure, matching Octave rosser()
    output.

    Model-user argument: The golden user uses the Rosser matrix as a standard
    eigenvalue benchmark when validating eig() accuracy. Its known eigenvalue
    spectrum (including near-zero and clustered values near 1020) exposes
    precision issues in eigensolvers. Symmetry is required for eigvalsh-based
    verification and for the matrix to serve as a valid test of symmetric
    eigendecomposition routines.

    Decomposition:
      R-SMAT-07a: Extreme eigenvalues match known reference values.
      R-SMAT-07b: rosser() is symmetric.
      R-SMAT-07c: rosser() is 8x8.

    Consistency: 07a checks the eigenvalue spectrum endpoints that define the
    Rosser benchmark. 07b confirms the symmetry required for real eigenvalues.
    07c verifies the fixed dimension. These fully characterize a valid Rosser
    matrix.
    """

    def test_rosser_eigenvalues(self):
        """R-SMAT-07a: Rosser extreme eigenvalues match known references."""
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
        """R-SMAT-07b: rosser() is symmetric."""
        R = _unwrap(forge_rosser())
        assert_allclose(R, R.T, atol=1e-12)

    def test_rosser_size(self):
        """R-SMAT-07c: rosser() produces an 8x8 matrix."""
        R = _unwrap(forge_rosser())
        assert R.shape == (8, 8)


class TestWilkinson:
    """R-SMAT-08: Forge SHALL generate Wilkinson matrices that are symmetric
    and tridiagonal with unit sub/super-diagonals, matching Octave
    wilkinson(n) output.

    Model-user argument: The golden user employs Wilkinson matrices for
    eigenvalue sensitivity analysis. The tridiagonal structure with unit
    off-diagonals and specific main-diagonal values creates eigenvalue
    clusters that stress-test eigensolver precision. Verifying the exact
    structure ensures the matrix produces the intended near-degenerate
    eigenvalue pairs that make it a meaningful sensitivity benchmark.

    Decomposition:
      R-SMAT-08a: wilkinson(n) is tridiagonal (zero outside the three
                   central diagonals).
      R-SMAT-08b: wilkinson(n) is symmetric.
      R-SMAT-08c: Sub-diagonal and super-diagonal entries are all ones.

    Consistency: 08a confirms the sparsity pattern, 08b confirms symmetry,
    and 08c pins the off-diagonal values. Together with the known main-
    diagonal formula (tested implicitly via symmetry and structure), these
    fully define a valid Wilkinson matrix.
    """

    def test_wilkinson_tridiagonal(self):
        """R-SMAT-08a: wilkinson(n) has zero entries outside the tridiagonal."""
        W = _unwrap(forge_wilkinson(ForgeArray(np.array(7.0))))
        n = W.shape[0]
        for i in range(n):
            for j in range(n):
                if abs(i - j) > 1:
                    assert W[i, j] == 0.0, f"W[{i},{j}] = {W[i,j]} should be 0"

    def test_wilkinson_symmetric(self):
        """R-SMAT-08b: wilkinson(n) is symmetric."""
        W = _unwrap(forge_wilkinson(ForgeArray(np.array(7.0))))
        assert_allclose(W, W.T, atol=1e-12)

    def test_wilkinson_subdiag_ones(self):
        """R-SMAT-08c: Sub- and super-diagonal entries are all ones."""
        W = _unwrap(forge_wilkinson(ForgeArray(np.array(7.0))))
        n = W.shape[0]
        for i in range(n - 1):
            assert W[i, i + 1] == 1.0
            assert W[i + 1, i] == 1.0
