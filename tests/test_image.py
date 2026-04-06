# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""V&V tests for Forge Image Processing Toolbox.

SRS trace: SRS-FUNC-001, SRS-VAL-001
Test method: Comparison against known image properties and round-trip identities.
"""
import pytest
import numpy as np
import tempfile
import os


class TestColormaps:
    """R-IMG-01: Forge SHALL provide colormap generation functions (jet,
    hot) that return Nx3 matrices with values in [0, 1] and correct
    row counts.

    Model-user argument: The engineer uses colormaps to visualize thermal
    images, stress distributions, and measurement heatmaps. In Octave,
    jet(N) and hot(N) return Nx3 RGB lookup tables that plotting functions
    consume. Incorrect dimensions or out-of-range values would produce
    corrupted visualizations, making measurement interpretation unreliable.

    Decomposition:
      R-IMG-01a: jet(256) returns a (256, 3) matrix.
      R-IMG-01b: hot(64) returns a (64, 3) matrix.
      R-IMG-01c: All jet(256) values are in [0.0, 1.0].
      R-IMG-01d: iscolormap returns True for a valid Nx3 matrix in [0, 1].
      R-IMG-01e: iscolormap returns False for an Nx4 matrix.

    Consistency argument: R-IMG-01a and R-IMG-01b verify shape correctness
    for two different colormaps and sizes. R-IMG-01c verifies the value
    range contract. R-IMG-01d and R-IMG-01e verify the validation utility
    for valid and invalid inputs. Together they confirm colormap generation
    and validation are correct.
    """

    def test_colormap_jet_length(self):
        """R-IMG-01a: jet(256) returns shape (256, 3)."""
        from forge.engine.builtins.image import jet
        from forge.engine.types import _unwrap
        cmap = _unwrap(jet(256))
        assert cmap.shape == (256, 3)

    def test_colormap_hot_shape(self):
        """R-IMG-01b: hot(64) returns shape (64, 3)."""
        from forge.engine.builtins.image import hot
        from forge.engine.types import _unwrap
        cmap = _unwrap(hot(64))
        assert cmap.shape == (64, 3)

    def test_colormap_values_range(self):
        """R-IMG-01c: All jet(256) values are in [0.0, 1.0]."""
        from forge.engine.builtins.image import jet
        from forge.engine.types import _unwrap
        cmap = _unwrap(jet(256))
        assert np.all(cmap >= 0.0)
        assert np.all(cmap <= 1.0)

    def test_iscolormap_valid(self):
        """R-IMG-01d: iscolormap returns True for valid Nx3 in [0, 1]."""
        from forge.engine.builtins.image import iscolormap
        from forge.engine.types import _unwrap
        cmap = np.random.rand(64, 3)
        cmap = np.clip(cmap, 0, 1)
        result = _unwrap(iscolormap(cmap))
        assert bool(result)

    def test_iscolormap_invalid(self):
        """R-IMG-01e: iscolormap returns False for Nx4 matrix."""
        from forge.engine.builtins.image import iscolormap
        from forge.engine.types import _unwrap
        bad = np.random.rand(64, 4)  # wrong number of columns
        result = _unwrap(iscolormap(bad))
        assert not bool(result)


class TestColorConversion:
    """R-IMG-02: Forge SHALL provide color space conversion functions
    (rgb2gray, rgb2hsv, hsv2rgb) that produce correct grayscale values
    for known inputs and achieve lossless round-trip for RGB to HSV to RGB.

    Model-user argument: The engineer converts sensor imagery between
    color spaces for measurement extraction. For example, rgb2gray is
    used before edge detection on thermal camera output, and HSV
    conversions isolate specific hue ranges for object segmentation.
    In Octave, these conversions are exact (lossless round-trip for
    HSV). Incorrect grayscale weights or lossy HSV conversion would
    introduce measurement errors.

    Decomposition:
      R-IMG-02a: rgb2gray of pure white returns 1.0.
      R-IMG-02b: rgb2gray of pure black returns 0.0.
      R-IMG-02c: RGB to HSV to RGB round-trip is lossless (atol 1e-10).
      R-IMG-02d: rgb2gray of (H, W, 3) input returns (H, W) output.

    Consistency argument: R-IMG-02a and R-IMG-02b verify grayscale
    boundary values (white, black). R-IMG-02c verifies lossless HSV
    round-trip on random data. R-IMG-02d verifies the dimension reduction
    contract. Together they confirm correctness at boundaries, round-trip
    fidelity, and shape handling.
    """

    def test_rgb2gray_known(self):
        """R-IMG-02a: rgb2gray of pure white returns 1.0."""
        from forge.engine.builtins.image import rgb2gray
        from forge.engine.types import _unwrap
        # Pure white should give 1.0
        white = np.ones((1, 1, 3))
        g = _unwrap(rgb2gray(white))
        assert abs(g.ravel()[0] - 1.0) < 0.01

    def test_rgb2gray_black(self):
        """R-IMG-02b: rgb2gray of pure black returns 0.0."""
        from forge.engine.builtins.image import rgb2gray
        from forge.engine.types import _unwrap
        black = np.zeros((1, 1, 3))
        g = _unwrap(rgb2gray(black))
        assert abs(g.ravel()[0]) < 0.01

    def test_hsv_rgb_roundtrip(self):
        """R-IMG-02c: RGB to HSV to RGB round-trip is lossless."""
        from forge.engine.builtins.image import hsv2rgb, rgb2hsv
        from forge.engine.types import _unwrap
        rgb = np.random.rand(10, 10, 3)
        hsv = rgb2hsv(rgb)
        recovered = hsv2rgb(_unwrap(hsv))
        np.testing.assert_allclose(_unwrap(recovered), rgb, atol=1e-10)

    def test_rgb2gray_shape(self):
        """R-IMG-02d: rgb2gray of (50, 40, 3) returns (50, 40)."""
        from forge.engine.builtins.image import rgb2gray
        from forge.engine.types import _unwrap
        img = np.random.rand(50, 40, 3)
        g = _unwrap(rgb2gray(img))
        assert g.shape == (50, 40)


class TestImageConversion:
    """R-IMG-03: Forge SHALL convert images between data types (im2double)
    such that uint8 [0, 255] maps to float64 [0.0, 1.0] and float inputs
    pass through unchanged.

    Model-user argument: Sensor and camera data arrives as uint8 from
    hardware drivers. The engineer converts to double for numerical
    processing (filtering, FFT, thresholding). In Octave, im2double
    handles this transparently. Incorrect scaling would corrupt all
    downstream measurements; float pass-through prevents double-scaling
    when the image is already normalized.

    Decomposition:
      R-IMG-03a: im2double maps uint8 0 to 0.0, 128 to ~0.5, 255 to 1.0.
      R-IMG-03b: im2double of float input passes through unchanged.

    Consistency argument: R-IMG-03a tests the primary uint8-to-float
    conversion with boundary and midpoint values. R-IMG-03b tests the
    float pass-through path. Together they cover both input-type branches.
    """

    def test_im2double_uint8(self):
        """R-IMG-03a: im2double maps uint8 0/128/255 to 0.0/~0.5/1.0."""
        from forge.engine.builtins.image import im2double
        from forge.engine.types import _unwrap
        img = np.array([[0, 128, 255]], dtype=np.uint8)
        d = _unwrap(im2double(img)).ravel()
        assert abs(d[0] - 0.0) < 1e-10
        assert abs(d[2] - 1.0) < 1e-10
        assert d[1] > 0.4 and d[1] < 0.6

    def test_im2double_already_float(self):
        """R-IMG-03b: im2double of float input passes through unchanged."""
        from forge.engine.builtins.image import im2double
        from forge.engine.types import _unwrap
        img = np.array([[0.0, 0.5, 1.0]])
        d = _unwrap(im2double(img)).ravel()
        np.testing.assert_allclose(d, [0.0, 0.5, 1.0], atol=1e-10)


class TestIO:
    """R-IMG-04: Forge SHALL read and write images in standard formats
    (PNG) with lossless round-trip fidelity for both color and grayscale
    images.

    Model-user argument: The engineer saves processed imagery (e.g.,
    filtered thermal frames, annotated microscopy captures) and reloads
    them for batch analysis. In Octave, imread/imwrite handle PNG
    losslessly. Data corruption on save/load would invalidate any
    measurement extracted from the reloaded image. Grayscale support is
    required because many sensor outputs are single-channel.

    Decomposition:
      R-IMG-04a: imwrite then imread of a color PNG recovers identical
                 pixel values.
      R-IMG-04b: imwrite then imread of a grayscale PNG preserves
                 dimensions.

    Consistency argument: R-IMG-04a tests lossless round-trip for the
    3-channel (color) case with exact pixel comparison. R-IMG-04b tests
    the single-channel (grayscale) case for dimensional correctness.
    Together they cover both channel configurations for the I/O pipeline.
    """

    def test_imread_imwrite_roundtrip(self):
        """R-IMG-04a: Color PNG round-trip recovers identical pixels."""
        from forge.engine.builtins.image import imread, imwrite
        from forge.engine.types import _unwrap
        # Create a simple test image
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            tmppath = f.name
        try:
            imwrite(tmppath, img)
            loaded = _unwrap(imread(tmppath))
            assert loaded.shape == img.shape
            # PNG is lossless so values should match
            np.testing.assert_array_equal(loaded, img)
        finally:
            if os.path.exists(tmppath):
                os.unlink(tmppath)

    def test_imread_grayscale(self):
        """R-IMG-04b: Grayscale PNG round-trip preserves dimensions."""
        from forge.engine.builtins.image import imread, imwrite
        from forge.engine.types import _unwrap
        img = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            tmppath = f.name
        try:
            imwrite(tmppath, img)
            loaded = _unwrap(imread(tmppath))
            assert loaded.shape[0] == 16
            assert loaded.shape[1] == 16
        finally:
            if os.path.exists(tmppath):
                os.unlink(tmppath)
