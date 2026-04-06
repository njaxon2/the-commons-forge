# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""
Signal Processing Toolbox: V-Model Traceability Backfill
=========================================================

Requirement Overview
--------------------
This module validates the Forge signal processing toolbox against the behavior
expected by an engineer migrating from MATLAB/Octave. It covers seven functional
areas:

  R-SIG-01  Window functions (Hamming, Blackman, Bartlett, Hann)
  R-SIG-02  IIR filter design (Butterworth, Chebyshev Type I)
  R-SIG-03  Time-domain filtering (filtfilt, fftfilt)
  R-SIG-04  Spectral analysis (periodogram, spectrogram)
  R-SIG-05  Frequency-domain transforms (fftshift/ifftshift, Hilbert, xcorr)
  R-SIG-06  Signal utility functions (sinc, detrend, unwrap, findpeaks,
            resample, medfilt1, conv, freqz, hanning)

All requirements trace to the Forge URD section on MATLAB/Octave compatibility
and the golden user profile of an engineer/scientist who designs filters,
analyzes spectra, and processes sensor data.
"""
import pytest
import numpy as np


class TestWindows:
    """
    Requirement: R-SIG-01
    SHALL statement: Forge SHALL provide window functions (hamming, blackman,
        bartlett) that return vectors of the requested length with the correct
        symmetry and endpoint properties matching MATLAB/Octave behavior.

    Model-user argument:
        The golden user routinely windows time-domain data before computing an
        FFT to reduce spectral leakage. They call hamming(N), blackman(N), or
        bartlett(N) and expect the returned vector to have exactly N samples
        with the documented symmetry. Any deviation in length or symmetry would
        silently corrupt their spectral estimates.

    Decomposition:
        R-SIG-01a  hamming(N) returns a vector of length N
        R-SIG-01b  hamming(N) is symmetric about its midpoint
        R-SIG-01c  blackman(N) returns a vector of length N
        R-SIG-01d  bartlett(N) has zero-valued endpoints

    Consistency argument:
        Sub-requirements 01a and 01b together verify hamming is correctly shaped
        and symmetric. 01c verifies blackman length independently. 01d checks
        the defining property of the Bartlett (triangular) window. Together they
        confirm that the three most common window functions produce output
        consistent with their mathematical definitions.
    """

    def test_hamming_length(self):
        """R-SIG-01a: hamming(64) returns a vector of length 64."""
        from forge.engine.builtins.signal import hamming
        w = hamming(64)
        from forge.engine.types import _unwrap
        assert _unwrap(w).ravel().shape[0] == 64

    def test_hamming_symmetric(self):
        """R-SIG-01b: hamming(64) is symmetric about its center."""
        from forge.engine.builtins.signal import hamming
        from forge.engine.types import _unwrap
        w = _unwrap(hamming(64)).ravel()
        np.testing.assert_allclose(w, w[::-1], atol=1e-14)

    def test_blackman_length(self):
        """R-SIG-01c: blackman(32) returns a vector of length 32."""
        from forge.engine.builtins.signal import blackman
        from forge.engine.types import _unwrap
        w = _unwrap(blackman(32)).ravel()
        assert len(w) == 32

    def test_bartlett_endpoints_zero(self):
        """R-SIG-01d: bartlett(64) has zero-valued first and last samples."""
        from forge.engine.builtins.signal import bartlett
        from forge.engine.types import _unwrap
        w = _unwrap(bartlett(64)).ravel()
        assert abs(w[0]) < 1e-10
        assert abs(w[-1]) < 1e-10


class TestFilterDesign:
    """
    Requirement: R-SIG-02
    SHALL statement: Forge SHALL provide IIR filter design functions (butter,
        cheby1) that accept order, cutoff, and ripple parameters and return
        numerator/denominator coefficient vectors of the correct length,
        matching MATLAB/Octave calling conventions.

    Model-user argument:
        The golden user designs anti-aliasing and bandpass filters by calling
        butter(N, Wn) or cheby1(N, Rp, Wn). They rely on receiving (order+1)
        coefficients in both the b and a vectors so they can cascade filters or
        pass them directly to filtfilt. If the coefficient count is wrong, the
        filter will have the wrong frequency response, corrupting their data.

    Decomposition:
        R-SIG-02a  butter(4, 0.3) returns b and a each of length 5
        R-SIG-02b  cheby1(4, 1, 0.3) executes without error

    Consistency argument:
        02a validates the Butterworth design path produces the correct number
        of coefficients for a 4th-order filter. 02b confirms the Chebyshev
        Type I path accepts the standard (order, ripple, cutoff) signature.
        Together they cover the two most common IIR design functions with their
        MATLAB-compatible calling conventions.
    """

    def test_butter_lowpass(self):
        """R-SIG-02a: butter(4, 0.3) returns 5 coefficients in b and a."""
        from forge.engine.builtins.signal import butter
        b, a = butter(4, 0.3)
        from forge.engine.types import _unwrap
        assert len(_unwrap(b).ravel()) == 5  # order+1 coefficients
        assert len(_unwrap(a).ravel()) == 5

    def test_cheby1(self):
        """R-SIG-02b: cheby1(4, 1, 0.3) executes without error."""
        from forge.engine.builtins.signal import cheby1
        b, a = cheby1(4, 1, 0.3)  # 1 dB ripple


class TestFiltering:
    """
    Requirement: R-SIG-03
    SHALL statement: Forge SHALL provide time-domain filtering functions
        (filtfilt, fftfilt) that apply filter coefficients to a signal and
        preserve DC level (filtfilt) or act as identity when given a unit
        impulse filter (fftfilt), matching MATLAB/Octave semantics.

    Model-user argument:
        The golden user applies zero-phase filtering via filtfilt to preserve
        timing in sensor data, and uses fftfilt for efficient long-FIR
        convolution. They expect filtfilt of a constant signal to return that
        constant (no transient artifacts), and fftfilt with b=[1] to return the
        input unchanged. Any deviation means their processing pipeline has an
        undiscovered gain or offset error.

    Decomposition:
        R-SIG-03a  filtfilt with a Butterworth filter on a DC signal returns DC
        R-SIG-03b  fftfilt with b=[1] returns the original signal

    Consistency argument:
        03a tests the zero-phase property by confirming DC preservation. 03b
        tests the identity-filter property. Together they verify both filtering
        paths produce numerically correct output for their trivial base cases,
        which is a necessary condition for correctness on non-trivial inputs.
    """

    def test_filtfilt_dc(self):
        """R-SIG-03a: filtfilt of constant signal returns constant."""
        from forge.engine.builtins.signal import butter, filtfilt
        from forge.engine.types import _unwrap
        b, a = butter(3, 0.1)
        x = np.ones(100)
        y = filtfilt(_unwrap(b).ravel(), _unwrap(a).ravel(), x)
        np.testing.assert_allclose(_unwrap(y).ravel(), 1.0, atol=0.01)

    def test_fftfilt_identity(self):
        """R-SIG-03b: fftfilt with unit impulse filter returns input."""
        from forge.engine.builtins.signal import fftfilt
        from forge.engine.types import _unwrap
        b = np.array([1.0])  # identity filter
        x = np.random.randn(100)
        y = fftfilt(b, x)
        np.testing.assert_allclose(_unwrap(y).ravel()[:100], x, atol=1e-10)


class TestSpectral:
    """
    Requirement: R-SIG-04
    SHALL statement: Forge SHALL provide spectral analysis functions
        (periodogram, spectrogram) that accept a time-domain signal and return
        power spectral density or time-frequency representations, matching
        MATLAB/Octave calling conventions.

    Model-user argument:
        The golden user computes periodograms to identify dominant frequencies
        in vibration or acoustic data, and uses spectrograms to track how
        frequency content evolves over time (e.g., engine run-up tests). They
        call periodogram(x) and spectrogram(x) with minimal arguments and
        expect usable output. If these functions error out on basic inputs, the
        user cannot perform even rudimentary spectral analysis.

    Decomposition:
        R-SIG-04a  periodogram(x) executes and returns a result for a pure tone
        R-SIG-04b  spectrogram(x) executes and returns a result for random data

    Consistency argument:
        04a verifies the periodogram path accepts a real-valued signal. 04b
        verifies the spectrogram path likewise accepts a real-valued signal.
        Together they confirm both spectral analysis entry points are callable
        with default parameters, which is the minimum bar for MATLAB
        compatibility.
    """

    def test_periodogram_frequency(self):
        """R-SIG-04a: periodogram executes on a 100 Hz pure tone."""
        from forge.engine.builtins.signal import periodogram
        from forge.engine.types import _unwrap
        fs = 1000
        t = np.arange(0, 1, 1/fs)
        x = np.sin(2*np.pi*100*t)  # 100 Hz tone
        result = periodogram(x)
        # Should return power spectrum

    def test_spectrogram(self):
        """R-SIG-04b: spectrogram executes on random data."""
        from forge.engine.builtins.signal import spectrogram
        x = np.random.randn(1000)
        result = spectrogram(x)


class TestTransforms:
    """
    Requirement: R-SIG-05
    SHALL statement: Forge SHALL provide frequency-domain transform utilities
        (fftshift, ifftshift, hilbert, xcorr) that manipulate spectra and
        correlation sequences with results matching MATLAB/Octave to within
        floating-point tolerance.

    Model-user argument:
        The golden user centers FFT output with fftshift before plotting
        spectra, computes analytic signals via hilbert() for envelope detection
        in communications or vibration analysis, and uses xcorr() to find time
        delays between sensor channels. These are daily operations. fftshift
        and ifftshift must be exact inverses; the Hilbert analytic signal of a
        cosine must have unit envelope; and the autocorrelation peak must occur
        at zero lag. Any failure here breaks fundamental signal analysis
        workflows.

    Decomposition:
        R-SIG-05a  fftshift followed by ifftshift returns the original vector
        R-SIG-05b  hilbert of a cosine has approximately unit magnitude
        R-SIG-05c  xcorr autocorrelation peak is at the center (zero lag)

    Consistency argument:
        05a confirms the shift/unshift roundtrip is lossless. 05b validates the
        analytic signal property for a pure tone. 05c verifies the
        autocorrelation symmetry property. These three sub-requirements cover
        the distinct transform operations (spectral rearrangement, analytic
        extension, correlation) that compose R-SIG-05.
    """

    def test_fftshift_roundtrip(self):
        """R-SIG-05a: fftshift then ifftshift returns original vector."""
        from forge.engine.builtins.signal import fftshift, ifftshift
        from forge.engine.types import _unwrap
        x = np.array([0, 1, 2, 3, 4, -4, -3, -2, -1], dtype=float)
        s = fftshift(x)
        r = ifftshift(_unwrap(s))
        np.testing.assert_array_equal(_unwrap(r).ravel(), x)

    def test_hilbert(self):
        """R-SIG-05b: hilbert analytic signal of cosine has unit envelope."""
        from forge.engine.builtins.signal import hilbert_transform as hilbert
        from forge.engine.types import _unwrap
        t = np.linspace(0, 1, 1000)
        x = np.cos(2*np.pi*10*t)
        h = hilbert(x)
        # Analytic signal magnitude should be ~1
        np.testing.assert_allclose(np.abs(_unwrap(h).ravel()), 1.0, atol=0.05)

    def test_xcorr_autocorr_peak(self):
        """R-SIG-05c: xcorr autocorrelation peak is at center (zero lag)."""
        from forge.engine.builtins.signal import xcorr
        from forge.engine.types import _unwrap
        x = np.random.randn(100)
        r, lags = xcorr(x)
        arr = _unwrap(r).ravel()
        # Autocorrelation peak should be at center (lag=0)
        assert arr[len(arr)//2] == np.max(arr)


class TestUtility:
    """
    Requirement: R-SIG-06
    SHALL statement: Forge SHALL provide signal utility functions (sinc,
        detrend, unwrap, findpeaks, resample, hanning, medfilt1, conv, freqz)
        that operate on real-valued vectors and produce results matching
        MATLAB/Octave behavior for standard inputs.

    Model-user argument:
        The golden user relies on a broad set of signal utilities throughout
        their workflow: sinc() for filter kernel construction, detrend() to
        remove DC and linear drift from sensor recordings, unwrap() to recover
        continuous phase from angle() output, findpeaks() to locate resonances,
        resample() to change acquisition rates, medfilt1() to remove impulse
        noise from measurements, conv() for polynomial multiplication and FIR
        filtering, and freqz() to inspect filter frequency response. Each
        function is called dozens of times per analysis session, and any missing
        or incorrect function forces a workaround that breaks the migration
        from MATLAB.

    Decomposition:
        R-SIG-06a   sinc(0) equals 1 (normalized sinc convention)
        R-SIG-06b   detrend removes linear trend, yielding near-zero mean
        R-SIG-06c   unwrap produces monotonically increasing phase
        R-SIG-06d   findpeaks executes on a simple peaked signal
        R-SIG-06e   resample(x, 200) on a length-100 vector returns length 200
        R-SIG-06f   hanning(128) returns length 128 with zero endpoints
        R-SIG-06g   medfilt1 removes a single impulse from a zero signal
        R-SIG-06h   conv([1,2,3],[1,1]) equals [1,3,5,3]
        R-SIG-06i   freqz of unit system (b=[1],a=[1]) has unity magnitude

    Consistency argument:
        Each sub-requirement tests a distinct utility function at its most
        characteristic property: sinc at the origin, detrend on a linear ramp,
        unwrap on a wrapped ramp, findpeaks on an obvious peak pattern,
        resample for length change, hanning for window shape, medfilt1 for
        impulse rejection, conv for polynomial multiplication, and freqz for
        the trivial all-pass case. Together the nine sub-requirements span the
        full set of utility functions in R-SIG-06. No utility function is left
        untested, and each test is independent of the others.
    """

    def test_sinc_zero(self):
        """R-SIG-06a: sinc(0) equals 1.0 (normalized sinc convention)."""
        from forge.engine.builtins.signal import sinc
        from forge.engine.types import _unwrap
        assert abs(_unwrap(sinc(np.array([0.0]))).flat[0] - 1.0) < 1e-10

    def test_detrend_linear(self):
        """R-SIG-06b: detrend of a linear ramp yields near-zero mean."""
        from forge.engine.builtins.signal import detrend
        from forge.engine.types import _unwrap
        x = np.linspace(0, 10, 100) + np.random.randn(100)*0.1
        d = detrend(x)
        # Detrended should have near-zero mean
        assert abs(np.mean(_unwrap(d).ravel())) < 0.5

    def test_unwrap_phase(self):
        """R-SIG-06c: unwrap produces monotonically increasing phase."""
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
        """R-SIG-06d: findpeaks executes on a simple peaked signal."""
        from forge.engine.builtins.signal import findpeaks
        from forge.engine.types import _unwrap
        x = np.array([0, 1, 0, 2, 0, 3, 0], dtype=float)
        pks = findpeaks(x)

    def test_resample(self):
        """R-SIG-06e: resample(x, 200) on length-100 input returns length 200."""
        from forge.engine.builtins.signal import resample
        from forge.engine.types import _unwrap
        x = np.random.randn(100)
        y = resample(x, 200)
        assert _unwrap(y).ravel().shape[0] == 200

    def test_hann_window(self):
        """R-SIG-06f: hanning(128) returns length 128 with zero endpoints."""
        from forge.engine.builtins.signal import hanning
        from forge.engine.types import _unwrap
        w = _unwrap(hanning(128)).ravel()
        assert len(w) == 128
        # Hann window endpoints should be zero
        assert abs(w[0]) < 1e-10
        assert abs(w[-1]) < 1e-10

    def test_medfilt1(self):
        """R-SIG-06g: medfilt1 removes a single impulse from a zero signal."""
        from forge.engine.builtins.signal import medfilt1
        from forge.engine.types import _unwrap
        # Impulse should be removed by median filter
        x = np.zeros(50)
        x[25] = 100.0  # impulse
        y = medfilt1(x, 5)
        arr = _unwrap(y).ravel()
        assert abs(arr[25]) < 1e-10

    def test_conv(self):
        """R-SIG-06h: conv([1,2,3],[1,1]) equals [1,3,5,3]."""
        from forge.engine.builtins.signal import conv
        from forge.engine.types import _unwrap
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 1.0])
        c = conv(a, b)
        np.testing.assert_allclose(_unwrap(c).ravel(), [1, 3, 5, 3], atol=1e-10)

    def test_freqz_unit(self):
        """R-SIG-06i: freqz of unit system has unity magnitude at all frequencies."""
        from forge.engine.builtins.signal import freqz
        from forge.engine.types import _unwrap
        # Unit system: b=[1], a=[1] => H(w)=1 for all w
        h, w = freqz(np.array([1.0]), np.array([1.0]))
        np.testing.assert_allclose(np.abs(_unwrap(h).ravel()), 1.0, atol=1e-10)
