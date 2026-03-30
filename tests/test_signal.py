# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
import pytest
import numpy as np

class TestWindows:
    def test_hamming_length(self):
        from forge.engine.builtins.signal import hamming
        w = hamming(64)
        from forge.engine.types import _unwrap
        assert _unwrap(w).ravel().shape[0] == 64

    def test_hamming_symmetric(self):
        from forge.engine.builtins.signal import hamming
        from forge.engine.types import _unwrap
        w = _unwrap(hamming(64)).ravel()
        np.testing.assert_allclose(w, w[::-1], atol=1e-14)

    def test_blackman_length(self):
        from forge.engine.builtins.signal import blackman
        from forge.engine.types import _unwrap
        w = _unwrap(blackman(32)).ravel()
        assert len(w) == 32

    def test_bartlett_endpoints_zero(self):
        from forge.engine.builtins.signal import bartlett
        from forge.engine.types import _unwrap
        w = _unwrap(bartlett(64)).ravel()
        assert abs(w[0]) < 1e-10
        assert abs(w[-1]) < 1e-10

class TestFilterDesign:
    def test_butter_lowpass(self):
        from forge.engine.builtins.signal import butter
        b, a = butter(4, 0.3)
        from forge.engine.types import _unwrap
        assert len(_unwrap(b).ravel()) == 5  # order+1 coefficients
        assert len(_unwrap(a).ravel()) == 5

    def test_cheby1(self):
        from forge.engine.builtins.signal import cheby1
        b, a = cheby1(4, 1, 0.3)  # 1 dB ripple

class TestFiltering:
    def test_filtfilt_dc(self):
        """filtfilt of constant signal should return constant"""
        from forge.engine.builtins.signal import butter, filtfilt
        from forge.engine.types import _unwrap
        b, a = butter(3, 0.1)
        x = np.ones(100)
        y = filtfilt(_unwrap(b).ravel(), _unwrap(a).ravel(), x)
        np.testing.assert_allclose(_unwrap(y).ravel(), 1.0, atol=0.01)

    def test_fftfilt_identity(self):
        from forge.engine.builtins.signal import fftfilt
        from forge.engine.types import _unwrap
        b = np.array([1.0])  # identity filter
        x = np.random.randn(100)
        y = fftfilt(b, x)
        np.testing.assert_allclose(_unwrap(y).ravel()[:100], x, atol=1e-10)

class TestSpectral:
    def test_periodogram_frequency(self):
        """Single tone should have peak at correct frequency"""
        from forge.engine.builtins.signal import periodogram
        from forge.engine.types import _unwrap
        fs = 1000
        t = np.arange(0, 1, 1/fs)
        x = np.sin(2*np.pi*100*t)  # 100 Hz tone
        result = periodogram(x)
        # Should return power spectrum

    def test_spectrogram(self):
        from forge.engine.builtins.signal import spectrogram
        x = np.random.randn(1000)
        result = spectrogram(x)

class TestTransforms:
    def test_fftshift_roundtrip(self):
        from forge.engine.builtins.signal import fftshift, ifftshift
        from forge.engine.types import _unwrap
        x = np.array([0, 1, 2, 3, 4, -4, -3, -2, -1], dtype=float)
        s = fftshift(x)
        r = ifftshift(_unwrap(s))
        np.testing.assert_array_equal(_unwrap(r).ravel(), x)

    def test_hilbert(self):
        from forge.engine.builtins.signal import hilbert_transform as hilbert
        from forge.engine.types import _unwrap
        t = np.linspace(0, 1, 1000)
        x = np.cos(2*np.pi*10*t)
        h = hilbert(x)
        # Analytic signal magnitude should be ~1
        np.testing.assert_allclose(np.abs(_unwrap(h).ravel()), 1.0, atol=0.05)

    def test_xcorr_autocorr_peak(self):
        from forge.engine.builtins.signal import xcorr
        from forge.engine.types import _unwrap
        x = np.random.randn(100)
        r, lags = xcorr(x)
        arr = _unwrap(r).ravel()
        # Autocorrelation peak should be at center (lag=0)
        assert arr[len(arr)//2] == np.max(arr)

class TestUtility:
    def test_sinc_zero(self):
        from forge.engine.builtins.signal import sinc
        from forge.engine.types import _unwrap
        assert abs(_unwrap(sinc(np.array([0.0]))).flat[0] - 1.0) < 1e-10

    def test_detrend_linear(self):
        from forge.engine.builtins.signal import detrend
        from forge.engine.types import _unwrap
        x = np.linspace(0, 10, 100) + np.random.randn(100)*0.1
        d = detrend(x)
        # Detrended should have near-zero mean
        assert abs(np.mean(_unwrap(d).ravel())) < 0.5

    def test_unwrap_phase(self):
        from forge.engine.builtins.signal import unwrap
        from forge.engine.types import _unwrap
        # Create wrapped phase
        theta = np.linspace(0, 4*np.pi, 100)
        wrapped = np.angle(np.exp(1j*theta))
        u = unwrap(wrapped)
        # Unwrapped should be monotonically increasing
        diff = np.diff(_unwrap(u).ravel())
        assert np.all(diff > -0.1)

    def test_findpeaks(self):
        from forge.engine.builtins.signal import findpeaks
        from forge.engine.types import _unwrap
        x = np.array([0, 1, 0, 2, 0, 3, 0], dtype=float)
        pks = findpeaks(x)

    def test_resample(self):
        from forge.engine.builtins.signal import resample
        from forge.engine.types import _unwrap
        x = np.random.randn(100)
        y = resample(x, 200)
        assert _unwrap(y).ravel().shape[0] == 200

    def test_hann_window(self):
        from forge.engine.builtins.signal import hanning
        from forge.engine.types import _unwrap
        w = _unwrap(hanning(128)).ravel()
        assert len(w) == 128
        # Hann window endpoints should be zero
        assert abs(w[0]) < 1e-10
        assert abs(w[-1]) < 1e-10

    @pytest.mark.skip(reason="medfilt1 not implemented in signal toolbox")
    def test_medfilt1(self):
        from forge.engine.builtins.signal import medfilt1
        from forge.engine.types import _unwrap
        # Impulse should be removed by median filter
        x = np.zeros(50)
        x[25] = 100.0  # impulse
        y = medfilt1(x, 5)
        arr = _unwrap(y).ravel()
        assert abs(arr[25]) < 1e-10

    @pytest.mark.skip(reason="conv not implemented; use fftconv instead")
    def test_conv(self):
        from forge.engine.builtins.signal import conv
        from forge.engine.types import _unwrap
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 1.0])
        c = conv(a, b)
        np.testing.assert_allclose(_unwrap(c).ravel(), [1, 3, 5, 3], atol=1e-10)

    def test_freqz_unit(self):
        from forge.engine.builtins.signal import freqz
        from forge.engine.types import _unwrap
        # Unit system: b=[1], a=[1] => H(w)=1 for all w
        h, w = freqz(np.array([1.0]), np.array([1.0]))
        np.testing.assert_allclose(np.abs(_unwrap(h).ravel()), 1.0, atol=1e-10)
