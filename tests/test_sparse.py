# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Tests for sparse matrix toolbox."""
import pytest
import numpy as np
from forge.engine.types import _unwrap
import scipy.sparse as sp


class TestSparseConstruction:
    def test_speye_identity(self):
        from forge.engine.builtins.sparse import forge_speye
        S = forge_speye(3)
        assert sp.issparse(S)
        np.testing.assert_array_equal(S.toarray(), np.eye(3))

    def test_speye_rectangular(self):
        from forge.engine.builtins.sparse import forge_speye
        S = forge_speye(3, 5)
        assert S.shape == (3, 5)

    def test_sprand_density(self):
        from forge.engine.builtins.sparse import forge_sprand
        S = forge_sprand(100, 100, 0.1)
        assert sp.issparse(S)
        assert S.nnz < 1500

    def test_sprandn(self):
        from forge.engine.builtins.sparse import forge_sprandn
        S = forge_sprandn(100, 100, 0.1)
        assert sp.issparse(S)

    def test_spones(self):
        from forge.engine.builtins.sparse import forge_spones, forge_sprand
        S = forge_sprand(10, 10, 0.5)
        O = forge_spones(S)
        assert np.all(O.data == 1.0)
        assert O.nnz == S.nnz


class TestSparseInfo:
    def test_issparse_true(self):
        from forge.engine.builtins.sparse import forge_issparse, forge_speye
        assert forge_issparse(forge_speye(3)) == True

    def test_issparse_false(self):
        from forge.engine.builtins.sparse import forge_issparse
        assert forge_issparse(np.eye(3)) == False

    def test_nnz(self):
        from forge.engine.builtins.sparse import forge_nnz, forge_speye
        assert forge_nnz(forge_speye(5)) == 5

    def test_nonzeros(self):
        from forge.engine.builtins.sparse import forge_nonzeros, forge_speye
        nz = forge_nonzeros(forge_speye(3))
        arr = _unwrap(nz).ravel() if hasattr(nz, '_data') else np.asarray(nz).ravel()
        assert len(arr) == 3

    def test_full_roundtrip(self):
        from forge.engine.builtins.sparse import forge_sparse, forge_full
        A = np.array([[1, 0], [0, 2]], dtype=float)
        S = forge_sparse(A)
        assert sp.issparse(S)
        F = forge_full(S)
        np.testing.assert_array_equal(_unwrap(F), A)


class TestSparseSolvers:
    def test_pcg_identity(self):
        from forge.engine.builtins.sparse import forge_pcg
        A = sp.eye(5, format='csc')
        b = np.array([1, 2, 3, 4, 5], dtype=float)
        result = forge_pcg(A, b)
        if isinstance(result, tuple):
            x = result[0]
        else:
            x = result
        x_arr = _unwrap(x).ravel()
        np.testing.assert_allclose(x_arr, b, atol=1e-8)

    def test_gmres_diag_dominant(self):
        from forge.engine.builtins.sparse import forge_gmres
        A = sp.random(10, 10, density=0.5, format='csc') + sp.eye(10) * 10
        b = np.ones(10)
        result = forge_gmres(A, b)
        # Should not crash

    def test_bicgstab(self):
        from forge.engine.builtins.sparse import forge_bicgstab
        A = sp.random(10, 10, density=0.5, format='csc') + sp.eye(10) * 10
        b = np.ones(10)
        result = forge_bicgstab(A, b)


class TestSparseEig:
    def test_eigs_identity(self):
        from forge.engine.builtins.sparse import forge_eigs
        A = sp.eye(10, format='csc') * 2.0
        result = forge_eigs(A, 3)
        # Eigenvalues of 2*I should be 2

    def test_svds(self):
        from forge.engine.builtins.sparse import forge_svds
        A = sp.eye(10, format='csc')
        result = forge_svds(A, 3)


class TestSparsePreconditioner:
    def test_ilu(self):
        from forge.engine.builtins.sparse import forge_ilu
        A = sp.diags([[-1] * 4, [2] * 5, [-1] * 4], [-1, 0, 1], shape=(5, 5), format='csc')
        result = forge_ilu(A)
        assert result is not None
