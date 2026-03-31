# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Tests for image processing, data manipulation, and advanced array ops (R23)."""
import pytest
import numpy as np
from forge.engine.evaluator import Session, ForgeError
from forge.engine.types import ForgeArray, _unwrap


@pytest.fixture
def s():
    return Session()


# ── Image processing ──────────────────────────────────────────────


class TestImageProcessing:
    def test_cat3_creates_3d(self, s):
        """cat(3, ...) should create a 3-D array."""
        s.eval("img = cat(3, [1 0; 0 1], [0 1; 1 0], [0.5 0.5; 0.5 0.5])")
        img = _unwrap(s.workspace.get("img"))
        assert img.shape == (2, 2, 3)

    def test_rgb2gray(self, s):
        """rgb2gray should convert 3-D RGB to 2-D grayscale."""
        s.eval("img = cat(3, ones(2,2), zeros(2,2), zeros(2,2))")
        s.eval("g = rgb2gray(img)")
        g = _unwrap(s.workspace.get("g"))
        assert g.shape == (2, 2)
        # Pure red → 0.2989
        np.testing.assert_allclose(g, 0.2989, atol=1e-4)

    def test_im2double(self, s):
        """im2double should map uint8 [0,255] → double [0,1]."""
        s.eval("d = im2double(uint8([0 128 255]))")
        d = _unwrap(s.workspace.get("d"))
        np.testing.assert_allclose(d.ravel(), [0.0, 128 / 255, 1.0], atol=1e-4)

    def test_imresize_scalar(self, s):
        """imresize(A, 0.5) should halve dimensions."""
        s.eval("r = imresize(ones(4, 4), 0.5)")
        r = _unwrap(s.workspace.get("r"))
        assert r.shape == (2, 2)

    def test_imresize_target_size(self, s):
        """imresize(A, [rows cols]) should resize to exact dims."""
        s.eval("r = imresize(ones(4, 6), [2 3])")
        r = _unwrap(s.workspace.get("r"))
        assert r.shape == (2, 3)

    def test_fft2_ifft2_roundtrip(self, s):
        """fft2 → ifft2 should recover original matrix."""
        s.eval("M = [1 2; 3 4]")
        s.eval("R = real(ifft2(fft2(M)))")
        R = _unwrap(s.workspace.get("R"))
        np.testing.assert_allclose(R, [[1, 2], [3, 4]], atol=1e-10)

    def test_imread_imwrite_roundtrip(self, s, tmp_path):
        """imwrite then imread should preserve shape."""
        fpath = str(tmp_path / "test_img.png").replace("\\", "/")
        s.eval("test_img = uint8(ones(8, 8, 3) * 128)")
        s.eval(f'imwrite(test_img, "{fpath}")')
        s.eval(f'img_back = imread("{fpath}")')
        img_back = _unwrap(s.workspace.get("img_back"))
        assert img_back.shape[0] == 8
        assert img_back.shape[1] == 8


# ── Data manipulation ─────────────────────────────────────────────


class TestDataManipulation:
    def test_accumarray(self, s):
        """accumarray([1;1;2;2;3], [10;20;30;40;50]) → [30;70;50]."""
        s.eval("r = accumarray([1;1;2;2;3], [10;20;30;40;50])")
        r = _unwrap(s.workspace.get("r"))
        np.testing.assert_allclose(r.ravel(), [30, 70, 50])

    def test_sortrows(self, s):
        """sortrows should sort by first column."""
        s.eval("r = sortrows([3 1; 1 3; 2 2])")
        r = _unwrap(s.workspace.get("r"))
        expected = [[1, 3], [2, 2], [3, 1]]
        np.testing.assert_allclose(r, expected)

    def test_unique_vector(self, s):
        """unique([3 1 2 1 3]) → [1 2 3]."""
        s.eval("r = unique([3 1 2 1 3])")
        r = _unwrap(s.workspace.get("r"))
        np.testing.assert_allclose(r.ravel(), [1, 2, 3])

    def test_unique_rows(self, s):
        """unique(M, 'rows') should remove duplicate rows."""
        s.eval('r = unique([1 2; 3 4; 1 2], "rows")')
        r = _unwrap(s.workspace.get("r"))
        expected = [[1, 2], [3, 4]]
        np.testing.assert_allclose(r, expected)

    def test_intersect(self, s):
        """intersect([1 2 3], [2 3 4]) → [2 3]."""
        s.eval("r = intersect([1 2 3], [2 3 4])")
        r = _unwrap(s.workspace.get("r"))
        np.testing.assert_allclose(r.ravel(), [2, 3])

    def test_union(self, s):
        """union([1 2 3], [3 4 5]) → [1 2 3 4 5]."""
        s.eval("r = union([1 2 3], [3 4 5])")
        r = _unwrap(s.workspace.get("r"))
        np.testing.assert_allclose(r.ravel(), [1, 2, 3, 4, 5])

    def test_setdiff(self, s):
        """setdiff([1 2 3 4], [2 4]) → [1 3]."""
        s.eval("r = setdiff([1 2 3 4], [2 4])")
        r = _unwrap(s.workspace.get("r"))
        np.testing.assert_allclose(r.ravel(), [1, 3])

    def test_ismember(self, s):
        """ismember([1 2 3 4 5], [2 4]) → [0 1 0 1 0]."""
        s.eval("r = ismember([1 2 3 4 5], [2 4])")
        r = _unwrap(s.workspace.get("r"))
        np.testing.assert_allclose(r.ravel(), [0, 1, 0, 1, 0])


# ── Advanced array ops ────────────────────────────────────────────


class TestAdvancedArrayOps:
    def test_circshift(self, s):
        """circshift([1 2 3 4 5], 2) → [4 5 1 2 3]."""
        s.eval("r = circshift([1 2 3 4 5], 2)")
        r = _unwrap(s.workspace.get("r"))
        np.testing.assert_allclose(r.ravel(), [4, 5, 1, 2, 3])

    def test_repelem(self, s):
        """repelem([1 2 3], 2) → [1 1 2 2 3 3]."""
        s.eval("r = repelem([1 2 3], 2)")
        r = _unwrap(s.workspace.get("r"))
        np.testing.assert_allclose(r.ravel(), [1, 1, 2, 2, 3, 3])

    def test_kron(self, s):
        """kron(eye(2), [1 2; 3 4]) → 4×4 block matrix."""
        s.eval("r = kron(eye(2), [1 2; 3 4])")
        r = _unwrap(s.workspace.get("r"))
        expected = [[1, 2, 0, 0], [3, 4, 0, 0],
                    [0, 0, 1, 2], [0, 0, 3, 4]]
        np.testing.assert_allclose(r, expected)

    def test_blkdiag(self, s):
        """blkdiag(A, B) → block diagonal matrix."""
        s.eval("r = blkdiag([1 2; 3 4], [5 6; 7 8])")
        r = _unwrap(s.workspace.get("r"))
        expected = [[1, 2, 0, 0], [3, 4, 0, 0],
                    [0, 0, 5, 6], [0, 0, 7, 8]]
        np.testing.assert_allclose(r, expected)
