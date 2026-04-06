# Copyright 2026 The Commons(TM)
# SPDX-License-Identifier: Apache-2.0
"""Round-2 sparse matrix tests -- session-level integration tests for sparse
matrix creation, arithmetic, solvers, display, and utility functions."""
import pytest
import numpy as np
import scipy.sparse as sp

from forge.engine.session import ForgeSession
from forge.engine.types import ForgeArray, _unwrap


@pytest.fixture
def sess():
    """Fresh ForgeSession for each test."""
    return ForgeSession()


class TestSparseCreation:
    """R-SPAR-07: Forge SHALL support all standard sparse matrix construction
    forms: triplet (sparse(i,j,v,m,n)), dense-to-sparse conversion (sparse(A)),
    zero-matrix shorthand (sparse(m,n)), sparse identity (speye), sparse random
    (sprand), and sparse diagonal (spdiags).

    Model-user argument: The migrating engineer builds FEM stiffness matrices
    with sparse(I,J,V,m,n) from element connectivity, converts prototype dense
    matrices with sparse(A), preallocates with sparse(m,n), and uses speye,
    sprand, and spdiags as building blocks for preconditioners and test problems.
    All six forms must be available or the workflow breaks at assembly time.

    Decomposition:
      R-SPAR-07.1: sparse(i,j,v,m,n) creates correct sparse matrix from triplets.
      R-SPAR-07.2: sparse(A) converts a dense matrix to sparse representation.
      R-SPAR-07.3: sparse(m,n) creates an all-zero sparse matrix.
      R-SPAR-07.4: speye(n) returns a sparse identity with n nonzeros.
      R-SPAR-07.5: sprand(m,n,d) returns a sparse random matrix.
      R-SPAR-07.6: spdiags returns a sparse diagonal matrix.

    Consistency: The six sub-requirements cover every documented sparse
    construction function. Together they ensure that any matrix the engineer
    would build (from triplets, from dense, from generators) enters the sparse
    pipeline correctly.
    """

    def test_sparse_triplet(self, sess):
        """R-SPAR-07.1: sparse(i,j,v,m,n) creates correct sparse matrix."""
        sess.eval("S = sparse([1 2 3], [1 2 3], [10 20 30], 3, 3)")
        result = sess.eval("full(S)")
        assert "10" in result and "20" in result and "30" in result

    def test_sparse_from_dense(self, sess):
        """R-SPAR-07.2: sparse(A) converts dense to sparse."""
        sess.eval("A = [1 0; 0 2]")
        sess.eval("S = sparse(A)")
        result = sess.eval("issparse(S)")
        assert "1" in result

    def test_sparse_zeros(self, sess):
        """R-SPAR-07.3: sparse(m,n) creates zero sparse matrix."""
        result = sess.eval("S = sparse(5, 5)")
        assert "All zero sparse" in result

    def test_speye_returns_sparse(self, sess):
        """R-SPAR-07.4: speye(n) returns a sparse identity matrix."""
        result = sess.eval("S = speye(4)")
        assert "Compressed Column Sparse" in result
        nnz_result = sess.eval("nnz(S)")
        assert "4" in nnz_result

    def test_sprand_returns_sparse(self, sess):
        """R-SPAR-07.5: sprand(m,n,d) returns sparse random matrix."""
        result = sess.eval("S = sprand(10, 10, 0.3)")
        assert "Compressed Column Sparse" in result
        iss = sess.eval("issparse(S)")
        assert "1" in iss

    def test_spdiags_returns_sparse(self, sess):
        """R-SPAR-07.6: spdiags returns a sparse diagonal matrix."""
        result = sess.eval("S = spdiags([1; 2; 3], 0, 3, 3)")
        assert "Compressed Column Sparse" in result


class TestSparseArithmetic:
    """R-SPAR-08: Forge SHALL preserve sparsity through addition, subtraction,
    matrix multiplication, and element-wise multiplication of sparse operands.

    Model-user argument: During FEM assembly the engineer accumulates element
    stiffness matrices into a global matrix (K = K + ke) and applies material
    scaling (K .* mask). If any arithmetic silently densifies the result, memory
    explodes on real-world meshes with millions of DOFs. The engineer expects
    every sparse-on-sparse operation to return a sparse matrix.

    Decomposition:
      R-SPAR-08.1: Sparse + sparse produces a sparse result with correct values.
      R-SPAR-08.2: Sparse - sparse produces a sparse result with correct values.
      R-SPAR-08.3: Sparse * sparse performs matrix multiplication correctly.
      R-SPAR-08.4: Sparse .* sparse performs element-wise multiplication correctly.
      R-SPAR-08.5: issparse confirms result of sparse arithmetic remains sparse.

    Consistency: The four arithmetic operators (+, -, *, .*) are the complete set
    used in FEM assembly and preconditioning. R-SPAR-08.5 cross-checks the
    sparsity invariant that the other four sub-requirements assume.
    """

    def test_sparse_addition(self, sess):
        """R-SPAR-08.1: Sparse + sparse produces sparse result."""
        sess.eval("A = sparse([1 0; 0 2]); B = sparse([3 0; 0 4])")
        result = sess.eval("C = A + B")
        assert "Compressed Column Sparse" in result
        full_result = sess.eval("full(C)")
        assert "4" in full_result and "6" in full_result

    def test_sparse_subtraction(self, sess):
        """R-SPAR-08.2: Sparse - sparse produces sparse result."""
        sess.eval("A = sparse([5 0; 0 8]); B = sparse([1 0; 0 3])")
        result = sess.eval("C = A - B")
        full_result = sess.eval("full(C)")
        assert "4" in full_result and "5" in full_result

    def test_sparse_matrix_multiply(self, sess):
        """R-SPAR-08.3: Sparse * sparse does matrix multiplication."""
        sess.eval("A = sparse([2 0; 0 3]); B = sparse([4 0; 0 5])")
        result = sess.eval("C = A * B")
        full_result = sess.eval("full(C)")
        assert "8" in full_result and "15" in full_result

    def test_sparse_elementwise_multiply(self, sess):
        """R-SPAR-08.4: Sparse .* sparse does element-wise multiplication."""
        sess.eval("A = sparse([1 2; 3 4]); B = sparse([5 6; 7 8])")
        result = sess.eval("C = A .* B")
        full_result = sess.eval("full(C)")
        assert "5" in full_result and "32" in full_result

    def test_sparse_issparse_after_arithmetic(self, sess):
        """R-SPAR-08.5: Result of sparse arithmetic is still sparse."""
        sess.eval("A = sparse([1 0; 0 2]); B = sparse([3 0; 0 4])")
        sess.eval("C = A + B")
        result = sess.eval("issparse(C)")
        assert "1" in result


class TestSparseFullInteraction:
    """R-SPAR-09: Forge SHALL support mixed sparse/dense arithmetic and provide
    issparse for type interrogation on both sparse and dense operands.

    Model-user argument: The engineer frequently adds a sparse stiffness matrix
    to a dense load or boundary-condition patch (K + F_dense). They also guard
    code paths with issparse() checks to choose between sparse and dense solvers.
    Both the mixed arithmetic and the type query must work or conditional logic
    and hybrid assembly fail silently.

    Decomposition:
      R-SPAR-09.1: Sparse + full dense matrix produces correct numeric result.
      R-SPAR-09.2: issparse returns 1 (true) for a sparse matrix.
      R-SPAR-09.3: issparse returns 0 (false) for a full (dense) matrix.

    Consistency: R-SPAR-09.1 covers the mixed-operand case. R-SPAR-09.2 and
    R-SPAR-09.3 together verify both branches of the issparse predicate, which
    is the only runtime type check for sparsity in MATLAB/Octave.
    """

    def test_sparse_plus_full(self, sess):
        """R-SPAR-09.1: Sparse + full dense matrix works."""
        sess.eval("S = sparse([1 2; 3 4]); F = [10 20; 30 40]")
        result = sess.eval("R = S + F")
        assert "11" in result and "44" in result

    def test_issparse_on_sparse(self, sess):
        """R-SPAR-09.2: issparse returns 1 for sparse matrix."""
        sess.eval("S = sparse([1 0; 0 1])")
        result = sess.eval("issparse(S)")
        assert "1" in result

    def test_issparse_on_full(self, sess):
        """R-SPAR-09.3: issparse returns 0 for full matrix."""
        result = sess.eval("issparse([1 2; 3 4])")
        assert "0" in result


class TestSparseSolvers:
    """R-SPAR-10: Forge SHALL solve sparse linear systems via the backslash
    operator, dispatching to a sparse-aware solver for both small and larger
    diagonal systems.

    Model-user argument: After assembling the global stiffness matrix, the
    engineer solves K\\F to get nodal displacements. This is the single most
    critical operation in any FEM workflow. If backslash does not dispatch to a
    sparse solver, the system either fails on large problems or silently
    densifies and runs out of memory.

    Decomposition:
      R-SPAR-10.1: S \\ b solves a 2x2 sparse diagonal system correctly.
      R-SPAR-10.2: S \\ b solves a 5x5 sparse diagonal system correctly.

    Consistency: The two sub-requirements test backslash at minimal size (2x2,
    verifying basic dispatch) and at moderate size (5x5, verifying scaling
    beyond trivial). Both use diagonal systems with known analytic solutions.
    """

    def test_backslash_sparse(self, sess):
        """R-SPAR-10.1: S \\ b uses sparse solver for sparse S."""
        sess.eval("S = sparse([2 0; 0 3]); b = [4; 9]")
        result = sess.eval("x = S \\ b")
        assert "2" in result and "3" in result

    def test_backslash_sparse_larger(self, sess):
        """R-SPAR-10.2: Sparse backslash on a larger diagonal system."""
        sess.eval("S = sparse([1 2 3 4 5], [1 2 3 4 5], [2 4 6 8 10], 5, 5)")
        sess.eval("b = [2; 8; 18; 32; 50]")
        result = sess.eval("x = S \\ b")
        assert "1" in result and "5" in result


class TestSparseUtilities:
    """R-SPAR-11: Forge SHALL provide sparse utility and query functions: nnz,
    nonzeros, find (1-output and 3-output forms), full round-trip, display in
    (row,col)->value format, and spconvert for triplet-matrix import.

    Model-user argument: The engineer inspects assembled matrices with nnz to
    confirm fill counts, extracts nonzero structure with find(S) to build
    adjacency graphs, round-trips through full() for dense post-processing, and
    reads the (row,col)->value display to spot-check entries during debugging.
    spconvert is used to import triplet data exported from other tools. Without
    these utilities the engineer cannot inspect, validate, or import sparse data.

    Decomposition:
      R-SPAR-11.1: nnz returns correct count of stored nonzeros.
      R-SPAR-11.2: nonzeros returns the nonzero values as a vector.
      R-SPAR-11.3: [i,j,v] = find(S) returns row indices, column indices, and values.
      R-SPAR-11.4: find(S) with one output returns column-major linear indices.
      R-SPAR-11.5: full(sparse(A)) recovers the original dense matrix.
      R-SPAR-11.6: Sparse display uses (row, col) -> value format.
      R-SPAR-11.7: spconvert builds a sparse matrix from a triplet matrix.

    Consistency: The seven sub-requirements cover every sparse query and
    conversion function available in Octave. R-SPAR-11.1 and R-SPAR-11.2 query
    scalar and vector nonzero info; R-SPAR-11.3 and R-SPAR-11.4 cover both
    find() output signatures; R-SPAR-11.5 verifies lossless dense recovery;
    R-SPAR-11.6 validates human-readable output; R-SPAR-11.7 covers external
    triplet import. Together they span the full inspection and import surface.
    """

    def test_nnz(self, sess):
        """R-SPAR-11.1: nnz returns correct count of nonzeros."""
        sess.eval("S = sparse([1 2 3], [1 2 3], [10 20 30], 3, 3)")
        result = sess.eval("nnz(S)")
        assert "3" in result

    def test_nonzeros(self, sess):
        """R-SPAR-11.2: nonzeros returns the nonzero values."""
        sess.eval("S = sparse([1 2], [1 2], [10 20], 3, 3)")
        result = sess.eval("nz = nonzeros(S)")
        assert "10" in result and "20" in result

    def test_find_sparse_3output(self, sess):
        """R-SPAR-11.3: [i,j,v] = find(S) returns row/col indices and values."""
        sess.eval("S = sparse([1 3], [2 1], [5 7], 3, 3)")
        sess.eval("[i,j,v] = find(S)")
        i_result = sess.eval("i")
        j_result = sess.eval("j")
        v_result = sess.eval("v")
        assert "3" in i_result and "1" in i_result
        assert "1" in j_result and "2" in j_result
        assert "5" in v_result and "7" in v_result

    def test_find_sparse_1output(self, sess):
        """R-SPAR-11.4: find(S) returns column-major linear indices."""
        sess.eval("S = sparse([1 2], [1 2], [10 20], 2, 2)")
        result = sess.eval("idx = find(S)")
        assert "1" in result and "4" in result

    def test_full_roundtrip(self, sess):
        """R-SPAR-11.5: full(sparse(A)) recovers original dense matrix."""
        sess.eval("A = [1 0 2; 0 3 0; 4 0 5]")
        sess.eval("S = sparse(A)")
        sess.eval("B = full(S)")
        result = sess.eval("B")
        assert "1" in result and "3" in result and "5" in result

    def test_sparse_display_format(self, sess):
        """R-SPAR-11.6: Sparse display shows (row, col) -> value format."""
        result = sess.eval("S = sparse([1 2], [1 3], [99 77], 2, 3)")
        assert "(1, 1) -> 99" in result
        assert "(2, 3) -> 77" in result

    def test_spconvert(self, sess):
        """R-SPAR-11.7: spconvert builds sparse from triplet matrix."""
        sess.eval("T = [1 1 10; 2 2 20; 3 3 30]")
        result = sess.eval("S = spconvert(T)")
        assert "Compressed Column Sparse" in result
        nnz_result = sess.eval("nnz(S)")
        assert "3" in nnz_result
