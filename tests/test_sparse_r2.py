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


# ── Construction ──────────────────────────────────────────────────────────

class TestSparseCreation:
    def test_sparse_triplet(self, sess):
        """sparse(i, j, v, m, n) creates correct sparse matrix."""
        sess.eval("S = sparse([1 2 3], [1 2 3], [10 20 30], 3, 3)")
        result = sess.eval("full(S)")
        assert "10" in result and "20" in result and "30" in result

    def test_sparse_from_dense(self, sess):
        """sparse(A) converts dense to sparse."""
        sess.eval("A = [1 0; 0 2]")
        sess.eval("S = sparse(A)")
        result = sess.eval("issparse(S)")
        assert "1" in result

    def test_sparse_zeros(self, sess):
        """sparse(m, n) creates zero sparse matrix."""
        result = sess.eval("S = sparse(5, 5)")
        assert "All zero sparse" in result

    def test_speye_returns_sparse(self, sess):
        """speye(n) returns a sparse identity matrix."""
        result = sess.eval("S = speye(4)")
        assert "Compressed Column Sparse" in result
        nnz_result = sess.eval("nnz(S)")
        assert "4" in nnz_result

    def test_sprand_returns_sparse(self, sess):
        """sprand(m, n, d) returns sparse random matrix."""
        result = sess.eval("S = sprand(10, 10, 0.3)")
        assert "Compressed Column Sparse" in result
        iss = sess.eval("issparse(S)")
        assert "1" in iss

    def test_spdiags_returns_sparse(self, sess):
        """spdiags returns a sparse diagonal matrix."""
        result = sess.eval("S = spdiags([1; 2; 3], 0, 3, 3)")
        assert "Compressed Column Sparse" in result


# ── Arithmetic ────────────────────────────────────────────────────────────

class TestSparseArithmetic:
    def test_sparse_addition(self, sess):
        """Sparse + sparse produces sparse result."""
        sess.eval("A = sparse([1 0; 0 2]); B = sparse([3 0; 0 4])")
        result = sess.eval("C = A + B")
        assert "Compressed Column Sparse" in result
        full_result = sess.eval("full(C)")
        assert "4" in full_result and "6" in full_result

    def test_sparse_subtraction(self, sess):
        """Sparse - sparse produces sparse result."""
        sess.eval("A = sparse([5 0; 0 8]); B = sparse([1 0; 0 3])")
        result = sess.eval("C = A - B")
        full_result = sess.eval("full(C)")
        assert "4" in full_result and "5" in full_result

    def test_sparse_matrix_multiply(self, sess):
        """Sparse * sparse does matrix multiplication."""
        sess.eval("A = sparse([2 0; 0 3]); B = sparse([4 0; 0 5])")
        result = sess.eval("C = A * B")
        full_result = sess.eval("full(C)")
        assert "8" in full_result and "15" in full_result

    def test_sparse_elementwise_multiply(self, sess):
        """Sparse .* sparse does element-wise multiplication."""
        sess.eval("A = sparse([1 2; 3 4]); B = sparse([5 6; 7 8])")
        result = sess.eval("C = A .* B")
        full_result = sess.eval("full(C)")
        assert "5" in full_result and "32" in full_result

    def test_sparse_issparse_after_arithmetic(self, sess):
        """Result of sparse arithmetic is still sparse."""
        sess.eval("A = sparse([1 0; 0 2]); B = sparse([3 0; 0 4])")
        sess.eval("C = A + B")
        result = sess.eval("issparse(C)")
        assert "1" in result


# ── Sparse-full interaction ───────────────────────────────────────────────

class TestSparseFullInteraction:
    def test_sparse_plus_full(self, sess):
        """Sparse + full dense matrix works."""
        sess.eval("S = sparse([1 2; 3 4]); F = [10 20; 30 40]")
        result = sess.eval("R = S + F")
        assert "11" in result and "44" in result

    def test_issparse_on_sparse(self, sess):
        """issparse returns 1 for sparse matrix."""
        sess.eval("S = sparse([1 0; 0 1])")
        result = sess.eval("issparse(S)")
        assert "1" in result

    def test_issparse_on_full(self, sess):
        """issparse returns 0 for full matrix."""
        result = sess.eval("issparse([1 2; 3 4])")
        assert "0" in result


# ── Solvers ───────────────────────────────────────────────────────────────

class TestSparseSolvers:
    def test_backslash_sparse(self, sess):
        """S \\ b uses sparse solver for sparse S."""
        sess.eval("S = sparse([2 0; 0 3]); b = [4; 9]")
        result = sess.eval("x = S \\ b")
        assert "2" in result and "3" in result

    def test_backslash_sparse_larger(self, sess):
        """Sparse backslash on a larger diagonal system."""
        sess.eval("S = sparse([1 2 3 4 5], [1 2 3 4 5], [2 4 6 8 10], 5, 5)")
        sess.eval("b = [2; 8; 18; 32; 50]")
        result = sess.eval("x = S \\ b")
        # x should be [1; 2; 3; 4; 5]
        assert "1" in result and "5" in result


# ── Utility functions ─────────────────────────────────────────────────────

class TestSparseUtilities:
    def test_nnz(self, sess):
        """nnz returns correct count of nonzeros."""
        sess.eval("S = sparse([1 2 3], [1 2 3], [10 20 30], 3, 3)")
        result = sess.eval("nnz(S)")
        assert "3" in result

    def test_nonzeros(self, sess):
        """nonzeros returns the nonzero values."""
        sess.eval("S = sparse([1 2], [1 2], [10 20], 3, 3)")
        result = sess.eval("nz = nonzeros(S)")
        assert "10" in result and "20" in result

    def test_find_sparse_3output(self, sess):
        """[i,j,v] = find(S) returns row/col indices and values."""
        sess.eval("S = sparse([1 3], [2 1], [5 7], 3, 3)")
        sess.eval("[i,j,v] = find(S)")
        i_result = sess.eval("i")
        j_result = sess.eval("j")
        v_result = sess.eval("v")
        # (3,1)->7 and (1,2)->5
        assert "3" in i_result and "1" in i_result
        assert "1" in j_result and "2" in j_result
        assert "5" in v_result and "7" in v_result

    def test_find_sparse_1output(self, sess):
        """find(S) returns column-major linear indices."""
        sess.eval("S = sparse([1 2], [1 2], [10 20], 2, 2)")
        result = sess.eval("idx = find(S)")
        # (1,1)->idx 1, (2,2)->idx 4 in 2x2 column-major
        assert "1" in result and "4" in result

    def test_full_roundtrip(self, sess):
        """full(sparse(A)) recovers original dense matrix."""
        sess.eval("A = [1 0 2; 0 3 0; 4 0 5]")
        sess.eval("S = sparse(A)")
        sess.eval("B = full(S)")
        result = sess.eval("B")
        assert "1" in result and "3" in result and "5" in result

    def test_sparse_display_format(self, sess):
        """Sparse display shows (row, col) -> value format."""
        result = sess.eval("S = sparse([1 2], [1 3], [99 77], 2, 3)")
        assert "(1, 1) -> 99" in result
        assert "(2, 3) -> 77" in result

    def test_spconvert(self, sess):
        """spconvert builds sparse from triplet matrix."""
        sess.eval("T = [1 1 10; 2 2 20; 3 3 30]")
        result = sess.eval("S = spconvert(T)")
        assert "Compressed Column Sparse" in result
        nnz_result = sess.eval("nnz(S)")
        assert "3" in nnz_result
