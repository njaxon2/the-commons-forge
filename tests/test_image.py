# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
import pytest
import numpy as np
import tempfile
import os


class TestColormaps:
    def test_colormap_jet_length(self):
        from forge.engine.builtins.image import jet
        from forge.engine.types import _unwrap
        cmap = _unwrap(jet(256))
        assert cmap.shape == (256, 3)

    def test_colormap_hot_shape(self):
        from forge.engine.builtins.image import hot
        from forge.engine.types import _unwrap
        cmap = _unwrap(hot(64))
        assert cmap.shape == (64, 3)

    def test_colormap_values_range(self):
        from forge.engine.builtins.image import jet
        from forge.engine.types import _unwrap
        cmap = _unwrap(jet(256))
        assert np.all(cmap >= 0.0)
        assert np.all(cmap <= 1.0)

    def test_iscolormap_valid(self):
        from forge.engine.builtins.image import iscolormap
        from forge.engine.types import _unwrap
        cmap = np.random.rand(64, 3)
        cmap = np.clip(cmap, 0, 1)
        result = _unwrap(iscolormap(cmap))
        assert bool(result)

    def test_iscolormap_invalid(self):
        from forge.engine.builtins.image import iscolormap
        from forge.engine.types import _unwrap
        bad = np.random.rand(64, 4)  # wrong number of columns
        result = _unwrap(iscolormap(bad))
        assert not bool(result)


class TestColorConversion:
    def test_rgb2gray_known(self):
        from forge.engine.builtins.image import rgb2gray
        from forge.engine.types import _unwrap
        # Pure white should give 1.0
        white = np.ones((1, 1, 3))
        g = _unwrap(rgb2gray(white))
        assert abs(g.ravel()[0] - 1.0) < 0.01

    def test_rgb2gray_black(self):
        from forge.engine.builtins.image import rgb2gray
        from forge.engine.types import _unwrap
        black = np.zeros((1, 1, 3))
        g = _unwrap(rgb2gray(black))
        assert abs(g.ravel()[0]) < 0.01

    def test_hsv_rgb_roundtrip(self):
        from forge.engine.builtins.image import hsv2rgb, rgb2hsv
        from forge.engine.types import _unwrap
        rgb = np.random.rand(10, 10, 3)
        hsv = rgb2hsv(rgb)
        recovered = hsv2rgb(_unwrap(hsv))
        np.testing.assert_allclose(_unwrap(recovered), rgb, atol=1e-10)

    def test_rgb2gray_shape(self):
        from forge.engine.builtins.image import rgb2gray
        from forge.engine.types import _unwrap
        img = np.random.rand(50, 40, 3)
        g = _unwrap(rgb2gray(img))
        assert g.shape == (50, 40)


class TestImageConversion:
    def test_im2double_uint8(self):
        from forge.engine.builtins.image import im2double
        from forge.engine.types import _unwrap
        img = np.array([[0, 128, 255]], dtype=np.uint8)
        d = _unwrap(im2double(img)).ravel()
        assert abs(d[0] - 0.0) < 1e-10
        assert abs(d[2] - 1.0) < 1e-10
        assert d[1] > 0.4 and d[1] < 0.6

    def test_im2double_already_float(self):
        from forge.engine.builtins.image import im2double
        from forge.engine.types import _unwrap
        img = np.array([[0.0, 0.5, 1.0]])
        d = _unwrap(im2double(img)).ravel()
        np.testing.assert_allclose(d, [0.0, 0.5, 1.0], atol=1e-10)


class TestIO:
    def test_imread_imwrite_roundtrip(self):
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
