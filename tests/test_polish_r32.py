# Copyright 2026 The Commons
# SPDX-License-Identifier: Apache-2.0
"""Polish round 32 — bitwise, set ops, accumarray, reshape/permute/squeeze, repmat.

SRS trace: SRS-FUNC-001 (Octave-compatible function library)
"""
import pytest
import numpy as np
from forge.engine.session import ForgeSession
from forge.engine.types import ForgeArray, _unwrap


@pytest.fixture(scope="module")
def S():
    return ForgeSession()


def _val(session, expr):
    """Evaluate and return numpy array."""
    r = session.eval(expr)
    if isinstance(r, str):
        # Try to get from workspace
        return r
    return np.asarray(_unwrap(r)).ravel() if isinstance(r, ForgeArray) else r


def _scalar(session, expr):
    """Evaluate and return a scalar float."""
    r = session.eval(expr)
    ws = session.workspace
    # If result is display string, re-read from ans
    if isinstance(r, str):
        ans = ws.get("ans") if ws.has("ans") else None
        if ans is not None:
            return float(np.asarray(_unwrap(ans)).ravel()[0])
        return float(r.strip())
    return float(np.asarray(_unwrap(r)).ravel()[0])


# ── Bitwise operations ────────────────────────────────────────────
class TestBitwise:
    def test_bitand(self, S):
        assert _scalar(S, "bitand(12, 10)") == 8

    def test_bitor(self, S):
        assert _scalar(S, "bitor(12, 10)") == 14

    def test_bitxor(self, S):
        assert _scalar(S, "bitxor(12, 10)") == 6

    def test_bitshift_left(self, S):
        assert _scalar(S, "bitshift(1, 4)") == 16

    def test_bitshift_right(self, S):
        assert _scalar(S, "bitshift(16, -2)") == 4

    def test_bitget_lsb(self, S):
        assert _scalar(S, "bitget(13, 1)") == 1

    def test_bitset(self, S):
        assert _scalar(S, "bitset(0, 3)") == 4


# ── Set operations edge cases ─────────────────────────────────────
class TestSetOps:
    def test_ismember_string_in_cell(self, S):
        v = _scalar(S, 'ismember("hello", {"hello", "world", "test"})')
        assert v == 1

    def test_ismember_string_not_in_cell(self, S):
        v = _scalar(S, 'ismember("foo", {"hello", "world", "test"})')
        assert v == 0

    def test_ismember_multi_output_tf(self, S):
        S.eval("[tf, loc] = ismember([2 4 6], [1 2 3 4 5]);")
        tf = np.asarray(_unwrap(S.workspace.get("tf"))).ravel()
        np.testing.assert_array_equal(tf, [1, 1, 0])

    def test_ismember_multi_output_loc(self, S):
        S.eval("[tf, loc] = ismember([2 4 6], [1 2 3 4 5]);")
        loc = np.asarray(_unwrap(S.workspace.get("loc"))).ravel()
        np.testing.assert_array_equal(loc, [2, 4, 0])


# ── accumarray with function handles ──────────────────────────────
class TestAccumarray:
    def test_accumarray_default_sum(self, S):
        S.eval("r = accumarray([1;1;2;2;3], [10;20;30;40;50]);")
        r = np.asarray(_unwrap(S.workspace.get("r"))).ravel()
        np.testing.assert_array_equal(r, [30, 70, 50])

    def test_accumarray_mean(self, S):
        S.eval("r = accumarray([1;1;2;2;3], [10;20;30;40;50], [], @mean);")
        r = np.asarray(_unwrap(S.workspace.get("r"))).ravel()
        np.testing.assert_array_equal(r, [15, 35, 50])

    def test_accumarray_max(self, S):
        S.eval("r = accumarray([1;1;2;2;3], [10;20;30;40;50], [], @max);")
        r = np.asarray(_unwrap(S.workspace.get("r"))).ravel()
        np.testing.assert_array_equal(r, [20, 40, 50])


# ── reshape / permute / squeeze ───────────────────────────────────
class TestReshapePermuteSqueze:
    def test_reshape_2x3(self, S):
        S.eval("r = reshape([1 2 3 4 5 6], 2, 3);")
        r = np.asarray(_unwrap(S.workspace.get("r")))
        assert r.shape == (2, 3)

    def test_reshape_auto_rows(self, S):
        S.eval("r = reshape([1 2 3 4 5 6], [], 2);")
        r = np.asarray(_unwrap(S.workspace.get("r")))
        assert r.shape == (3, 2)

    def test_permute_reorder(self, S):
        S.eval("r = size(permute(rand(2,3,4), [3 1 2]));")
        r = np.asarray(_unwrap(S.workspace.get("r"))).ravel()
        np.testing.assert_array_equal(r, [4, 2, 3])

    def test_squeeze_removes_singletons(self, S):
        S.eval("r = size(squeeze(rand(1,3,1,4)));")
        r = np.asarray(_unwrap(S.workspace.get("r"))).ravel()
        np.testing.assert_array_equal(r, [3, 4])


# ── repmat ─────────────────────────────────────────────────────────
class TestRepmat:
    def test_repmat_2x3(self, S):
        S.eval("r = repmat([1 2; 3 4], 2, 3);")
        r = np.asarray(_unwrap(S.workspace.get("r")))
        assert r.shape == (4, 6)

    def test_repmat_values(self, S):
        S.eval("r = repmat([1 2; 3 4], 2, 3);")
        r = np.asarray(_unwrap(S.workspace.get("r")))
        # Top-left 2x2 block should be original
        np.testing.assert_array_equal(r[:2, :2], [[1, 2], [3, 4]])
        # Second block row should repeat
        np.testing.assert_array_equal(r[2:4, :2], [[1, 2], [3, 4]])
