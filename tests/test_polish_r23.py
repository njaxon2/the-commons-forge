# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Tests for image processing, data manipulation, and advanced array ops (R23).

V&V Traceability (backfill):
    R-POL23-01 .. R-POL23-03 (parent requirements)
    R-POL23-01-nn .. R-POL23-03-nn (unit sub-requirements)

SRS trace: SRS-FUNC-001, SRS-VAL-001, SRS-COMPAT-001
"""
import pytest
import numpy as np
from forge.engine.evaluator import Session, ForgeError
from forge.engine.types import ForgeArray, _unwrap


@pytest.fixture
def s():
    return Session()


# ── Image processing ──────────────────────────────────────────────


class TestImageProcessing:
    """R-POL23-01: Forge SHALL provide image processing functions (cat along
    dim 3, rgb2gray, im2double, imresize, fft2/ifft2, imread/imwrite) that
    construct, convert, resize, transform, and persist image data matching
    MATLAB/Octave semantics.

    Model-user argument: A scientist porting image analysis pipelines from
    Octave uses cat(3,...) for channel stacking, rgb2gray for luminance
    extraction, im2double for normalization, imresize for downsampling, and
    fft2/ifft2 for frequency-domain filtering. Incorrect conversions would
    silently corrupt pixel values, producing wrong measurements in downstream
    analysis (e.g., particle counting, spectral classification).

    Decomposition:
        R-POL23-01-01: cat(3,...) creates a 3-D array
        R-POL23-01-02: rgb2gray converts 3-D RGB to 2-D grayscale
        R-POL23-01-03: im2double maps uint8 [0,255] to double [0,1]
        R-POL23-01-04: imresize with scalar factor halves dimensions
        R-POL23-01-05: imresize with [rows cols] target resizes exactly
        R-POL23-01-06: fft2 then ifft2 recovers original matrix
        R-POL23-01-07: imwrite then imread preserves image shape

    Consistency: Sub-requirements cover construction (01), color conversion (02),
    type conversion (03), resizing by factor and target (04-05), frequency
    transform round-trip (06), and file I/O round-trip (07). Together they
    validate the full image processing pipeline.
    """

    def test_cat3_creates_3d(self, s):
        """R-POL23-01-01: cat(3,...) creates a (2,2,3) array."""
        s.eval("img = cat(3, [1 0; 0 1], [0 1; 1 0], [0.5 0.5; 0.5 0.5])")
        img = _unwrap(s.workspace.get("img"))
        assert img.shape == (2, 2, 3)

    def test_rgb2gray(self, s):
        """R-POL23-01-02: rgb2gray of pure red yields 0.2989."""
        s.eval("img = cat(3, ones(2,2), zeros(2,2), zeros(2,2))")
        s.eval("g = rgb2gray(img)")
        g = _unwrap(s.workspace.get("g"))
        assert g.shape == (2, 2)
        # Pure red → 0.2989
        np.testing.assert_allclose(g, 0.2989, atol=1e-4)

    def test_im2double(self, s):
        """R-POL23-01-03: im2double maps [0,128,255] to [0, 128/255, 1]."""
        s.eval("d = im2double(uint8([0 128 255]))")
        d = _unwrap(s.workspace.get("d"))
        np.testing.assert_allclose(d.ravel(), [0.0, 128 / 255, 1.0], atol=1e-4)

    def test_imresize_scalar(self, s):
        """R-POL23-01-04: imresize(ones(4,4), 0.5) yields (2,2)."""
        s.eval("r = imresize(ones(4, 4), 0.5)")
        r = _unwrap(s.workspace.get("r"))
        assert r.shape == (2, 2)

    def test_imresize_target_size(self, s):
        """R-POL23-01-05: imresize(ones(4,6), [2 3]) yields (2,3)."""
        s.eval("r = imresize(ones(4, 6), [2 3])")
        r = _unwrap(s.workspace.get("r"))
        assert r.shape == (2, 3)

    def test_fft2_ifft2_roundtrip(self, s):
        """R-POL23-01-06: fft2 then ifft2 recovers [1 2; 3 4]."""
        s.eval("M = [1 2; 3 4]")
        s.eval("R = real(ifft2(fft2(M)))")
        R = _unwrap(s.workspace.get("R"))
        np.testing.assert_allclose(R, [[1, 2], [3, 4]], atol=1e-10)

    def test_imread_imwrite_roundtrip(self, s, tmp_path):
        """R-POL23-01-07: imwrite then imread preserves image shape."""
        fpath = str(tmp_path / "test_img.png").replace("\\", "/")
        s.eval("test_img = uint8(ones(8, 8, 3) * 128)")
        s.eval(f'imwrite(test_img, "{fpath}")')
        s.eval(f'img_back = imread("{fpath}")')
        img_back = _unwrap(s.workspace.get("img_back"))
        assert img_back.shape[0] == 8
        assert img_back.shape[1] == 8


# ── Data manipulation ─────────────────────────────────────────────


class TestDataManipulation:
    """R-POL23-02: Forge SHALL provide data manipulation functions (accumarray,
    sortrows, unique, intersect, union, setdiff, ismember) that produce results
    matching MATLAB/Octave reference values for vectors, matrices, and row-wise
    operations.

    Model-user argument: A scientist porting statistical analysis or data
    cleaning scripts from Octave uses accumarray for grouped sums, sortrows for
    tabular ordering, unique/intersect/union/setdiff for set operations on
    measurement IDs, and ismember for lookup matching. Incorrect results would
    silently assign data to wrong groups or omit valid entries.

    Decomposition:
        R-POL23-02-01: accumarray sums groups correctly
        R-POL23-02-02: sortrows sorts by first column
        R-POL23-02-03: unique removes duplicates from vector
        R-POL23-02-04: unique('rows') removes duplicate rows
        R-POL23-02-05: intersect returns common elements
        R-POL23-02-06: union returns combined unique elements
        R-POL23-02-07: setdiff returns elements in A but not B
        R-POL23-02-08: ismember returns membership flags

    Consistency: Sub-requirements cover grouping (01), sorting (02), uniqueness
    for vectors and rows (03-04), and all four set operations (05-08). Together
    they validate the full data manipulation API.
    """

    def test_accumarray(self, s):
        """R-POL23-02-01: accumarray sums groups to [30, 70, 50]."""
        s.eval("r = accumarray([1;1;2;2;3], [10;20;30;40;50])")
        r = _unwrap(s.workspace.get("r"))
        np.testing.assert_allclose(r.ravel(), [30, 70, 50])

    def test_sortrows(self, s):
        """R-POL23-02-02: sortrows orders by first column ascending."""
        s.eval("r = sortrows([3 1; 1 3; 2 2])")
        r = _unwrap(s.workspace.get("r"))
        expected = [[1, 3], [2, 2], [3, 1]]
        np.testing.assert_allclose(r, expected)

    def test_unique_vector(self, s):
        """R-POL23-02-03: unique([3 1 2 1 3]) returns [1 2 3]."""
        s.eval("r = unique([3 1 2 1 3])")
        r = _unwrap(s.workspace.get("r"))
        np.testing.assert_allclose(r.ravel(), [1, 2, 3])

    def test_unique_rows(self, s):
        """R-POL23-02-04: unique with 'rows' removes duplicate rows."""
        s.eval('r = unique([1 2; 3 4; 1 2], "rows")')
        r = _unwrap(s.workspace.get("r"))
        expected = [[1, 2], [3, 4]]
        np.testing.assert_allclose(r, expected)

    def test_intersect(self, s):
        """R-POL23-02-05: intersect([1 2 3],[2 3 4]) returns [2 3]."""
        s.eval("r = intersect([1 2 3], [2 3 4])")
        r = _unwrap(s.workspace.get("r"))
        np.testing.assert_allclose(r.ravel(), [2, 3])

    def test_union(self, s):
        """R-POL23-02-06: union([1 2 3],[3 4 5]) returns [1 2 3 4 5]."""
        s.eval("r = union([1 2 3], [3 4 5])")
        r = _unwrap(s.workspace.get("r"))
        np.testing.assert_allclose(r.ravel(), [1, 2, 3, 4, 5])

    def test_setdiff(self, s):
        """R-POL23-02-07: setdiff([1 2 3 4],[2 4]) returns [1 3]."""
        s.eval("r = setdiff([1 2 3 4], [2 4])")
        r = _unwrap(s.workspace.get("r"))
        np.testing.assert_allclose(r.ravel(), [1, 3])

    def test_ismember(self, s):
        """R-POL23-02-08: ismember([1 2 3 4 5],[2 4]) returns [0 1 0 1 0]."""
        s.eval("r = ismember([1 2 3 4 5], [2 4])")
        r = _unwrap(s.workspace.get("r"))
        np.testing.assert_allclose(r.ravel(), [0, 1, 0, 1, 0])


# ── Advanced array ops ────────────────────────────────────────────


class TestAdvancedArrayOps:
    """R-POL23-03: Forge SHALL provide advanced array construction functions
    (circshift, repelem, kron, blkdiag) that produce results matching
    MATLAB/Octave reference values.

    Model-user argument: An engineer porting signal processing or FEM code from
    Octave uses circshift for circular buffer simulation, repelem for
    upsampling, kron for Kronecker products in tensor math, and blkdiag for
    assembling block-diagonal stiffness matrices. Incorrect results would
    corrupt system matrices and invalidate simulation output.

    Decomposition:
        R-POL23-03-01: circshift([1 2 3 4 5], 2) yields [4 5 1 2 3]
        R-POL23-03-02: repelem([1 2 3], 2) yields [1 1 2 2 3 3]
        R-POL23-03-03: kron(eye(2), [1 2; 3 4]) yields 4x4 block matrix
        R-POL23-03-04: blkdiag assembles block diagonal matrix

    Consistency: Each sub-requirement tests one array construction function with
    a known result. Together they cover circular shifting, element replication,
    Kronecker product, and block diagonal assembly.
    """

    def test_circshift(self, s):
        """R-POL23-03-01: circshift([1 2 3 4 5], 2) yields [4 5 1 2 3]."""
        s.eval("r = circshift([1 2 3 4 5], 2)")
        r = _unwrap(s.workspace.get("r"))
        np.testing.assert_allclose(r.ravel(), [4, 5, 1, 2, 3])

    def test_repelem(self, s):
        """R-POL23-03-02: repelem([1 2 3], 2) yields [1 1 2 2 3 3]."""
        s.eval("r = repelem([1 2 3], 2)")
        r = _unwrap(s.workspace.get("r"))
        np.testing.assert_allclose(r.ravel(), [1, 1, 2, 2, 3, 3])

    def test_kron(self, s):
        """R-POL23-03-03: kron(eye(2), [1 2; 3 4]) yields correct 4x4 block."""
        s.eval("r = kron(eye(2), [1 2; 3 4])")
        r = _unwrap(s.workspace.get("r"))
        expected = [[1, 2, 0, 0], [3, 4, 0, 0],
                    [0, 0, 1, 2], [0, 0, 3, 4]]
        np.testing.assert_allclose(r, expected)

    def test_blkdiag(self, s):
        """R-POL23-03-04: blkdiag assembles correct block diagonal matrix."""
        s.eval("r = blkdiag([1 2; 3 4], [5 6; 7 8])")
        r = _unwrap(s.workspace.get("r"))
        expected = [[1, 2, 0, 0], [3, 4, 0, 0],
                    [0, 0, 5, 6], [0, 0, 7, 8]]
        np.testing.assert_allclose(r, expected)
