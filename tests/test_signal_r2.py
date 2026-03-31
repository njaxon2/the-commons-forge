# Copyright 2026 The Commons
# SPDX-License-Identifier: Apache-2.0
"""Signal processing round-2 tests — FFT, filters, spectral, windows."""

import numpy as np
import pytest

from forge.engine.builtins.linalg import (
    _forge_fft, _forge_ifft, _forge_fft2, _forge_ifft2,
)
from forge.engine.builtins.signal import (
    butter, cheby1, freqz, grpdelay, filtfilt, lfilter,
    fftshift, ifftshift, spectrogram, stft,
    hamming, hanning, blackman, kaiser_win,
    xcorr, hilbert_transform, resample, detrend,
)
from forge.engine.types import ForgeArray


def _r(x):
    """Extract real 1-D numpy array from ForgeArray."""
    arr = x.data if hasattr(x, "data") else np.asarray(x)
    return np.real(arr).ravel()


# ── FFT round-trips ─────────────────────────────────────────────

class TestFFT:
    def test_fft_ifft_roundtrip(self):
        x = ForgeArray(np.array([1.0, 2.0, 3.0, 4.0]))
        result = _r(_forge_ifft(_forge_fft(x)))
        np.testing.assert_allclose(result, [1, 2, 3, 4], atol=1e-12)

    def test_fft_ifft_longer(self):
        x = ForgeArray(np.random.randn(128))
        result = _r(_forge_ifft(_forge_fft(x)))
        np.testing.assert_allclose(result, x.data.ravel(), atol=1e-10)

    def test_fft2_ifft2_roundtrip(self):
        m = ForgeArray(np.array([[1.0, 2.0], [3.0, 4.0]]))
        result = np.real(_forge_ifft2(_forge_fft2(m)).data)
        np.testing.assert_allclose(result, [[1, 2], [3, 4]], atol=1e-12)

    def test_fft_with_nfft(self):
        """fft(x, N) should zero-pad or truncate to length N."""
        x = ForgeArray(np.array([1.0, 2.0, 3.0]))
        result = _forge_fft(x, 8)
        assert result.data.ravel().shape[0] == 8


# ── fftshift / ifftshift ────────────────────────────────────────

class TestFFTShift:
    def test_fftshift_ifftshift_roundtrip(self):
        x = np.array([0, 1, 2, 3, 4, -4, -3, -2, -1])
        shifted = fftshift(x)
        back = _r(ifftshift(shifted))
        np.testing.assert_array_equal(back, x)

    def test_fftshift_even(self):
        x = np.array([0, 1, 2, 3])
        expected = np.array([2, 3, 0, 1])
        np.testing.assert_array_equal(_r(fftshift(x)), expected)


# ── Butterworth filter design ───────────────────────────────────

class TestButter:
    def test_butter_lowpass_order(self):
        b, a = butter(4, 0.3)
        # 4th order => 5 coefficients
        assert len(_r(b)) == 5
        assert len(_r(a)) == 5

    def test_butter_dc_gain_unity(self):
        """Lowpass Butterworth: DC gain should be 1."""
        b, a = butter(4, 0.3)
        dc_gain = np.sum(_r(b)) / np.sum(_r(a))
        np.testing.assert_allclose(dc_gain, 1.0, atol=1e-10)


# ── freqz ────────────────────────────────────────────────────────

class TestFreqz:
    def test_freqz_butter_dc(self):
        """Frequency response at DC for lowpass should be ~1."""
        b, a = butter(4, 0.3)
        h, w = freqz(b, a, 256)
        dc = abs(_r(h)[0])  # h is complex
        # Actually _r takes real part; use data directly
        h_arr = h.data.ravel() if hasattr(h, "data") else np.asarray(h).ravel()
        np.testing.assert_allclose(abs(h_arr[0]), 1.0, atol=1e-6)

    def test_freqz_butter_stopband(self):
        """At Nyquist, 4th-order LPF with Wn=0.3 should attenuate > 20 dB."""
        b, a = butter(4, 0.3)
        h, w = freqz(b, a, 256)
        h_arr = h.data.ravel() if hasattr(h, "data") else np.asarray(h).ravel()
        atten_dB = 20 * np.log10(abs(h_arr[-1]) + 1e-30)
        assert atten_dB < -20

    def test_freqz_scalar_a(self):
        """freqz with scalar a=1 (FIR filter)."""
        b = np.array([0.25, 0.25, 0.25, 0.25])
        h, w = freqz(b, 1, 64)
        h_arr = h.data.ravel() if hasattr(h, "data") else np.asarray(h).ravel()
        np.testing.assert_allclose(abs(h_arr[0]), 1.0, atol=1e-10)


# ── filter / lfilter ────────────────────────────────────────────

class TestFilter:
    def test_filter_exponential_growth(self):
        """filter([1], [1 -0.9], ones(1,10)) should grow exponentially."""
        y = lfilter(np.array([1.0]), np.array([1.0, -0.9]), np.ones(10))
        yr = _r(y)
        assert yr[-1] > yr[0]
        # Expected: cumulative geometric series approaching 10
        np.testing.assert_allclose(yr[0], 1.0)
        np.testing.assert_allclose(yr[1], 1.9, atol=1e-10)

    def test_filtfilt_zero_phase(self):
        """filtfilt should produce zero-phase output (symmetric impulse response)."""
        b, a = butter(2, 0.2)
        # Create a signal with a single impulse
        x = np.zeros(100)
        x[50] = 1.0
        y = filtfilt(b, a, x)
        yr = _r(y)
        # Zero-phase: peak should be at or very near index 50
        peak_idx = np.argmax(yr)
        assert abs(peak_idx - 50) <= 1


# ── spectrogram / stft ──────────────────────────────────────────

class TestSpectral:
    def test_spectrogram_shape(self):
        x = np.sin(2 * np.pi * 50 * np.arange(1000) / 1000.0)
        S, f, t = spectrogram(x, 1000.0)
        # S should have frequency bins x time frames
        assert S.shape[0] > 1
        assert S.shape[1] > 1

    def test_stft_shape(self):
        x = np.sin(2 * np.pi * 50 * np.arange(1000) / 1000.0)
        S, f, t = stft(x, fs=1000.0)
        assert S.shape[0] > 1
        assert S.shape[1] > 1

    def test_spectrogram_energy_at_50hz(self):
        """A 50 Hz sine at fs=1000 should have peak energy near 50 Hz."""
        fs = 1000.0
        x = np.sin(2 * np.pi * 50 * np.arange(2000) / fs)
        S, f, t = spectrogram(x, fs)
        f_arr = _r(f)
        S_arr = S.data if hasattr(S, "data") else np.asarray(S)
        # Average power across time
        avg_power = np.mean(np.real(S_arr), axis=1).ravel()
        peak_freq = f_arr[np.argmax(avg_power)]
        assert abs(peak_freq - 50.0) < 10.0


# ── Window functions ─────────────────────────────────────────────

class TestWindows:
    def test_hamming_symmetric(self):
        w = _r(hamming(64))
        np.testing.assert_allclose(w, w[::-1], atol=1e-12)

    def test_hanning_endpoints_near_zero(self):
        w = _r(hanning(64))
        assert w[0] < 0.1
        assert w[-1] < 0.1

    def test_blackman_peak_at_center(self):
        w = _r(blackman(65))
        assert np.argmax(w) == 32

    def test_kaiser_symmetric(self):
        w = _r(kaiser_win(64, 8.0))
        np.testing.assert_allclose(w, w[::-1], atol=1e-12)

    def test_kaiser_beta_effect(self):
        """Higher beta => narrower mainlobe, more sidelobe suppression."""
        w_low = _r(kaiser_win(64, 2.0))
        w_high = _r(kaiser_win(64, 14.0))
        # Higher beta: endpoints should be smaller relative to center
        ratio_low = w_low[0] / w_low[32]
        ratio_high = w_high[0] / w_high[32]
        assert ratio_high < ratio_low
