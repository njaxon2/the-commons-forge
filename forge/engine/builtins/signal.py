"""Signal Processing Toolbox for Forge — Octave-compatible functions.

Implements the 39 Octave signal functions plus ~30 additional common
signal processing functions for MATLAB/Octave compatibility.

Backend: NumPy + SciPy (scipy.signal, scipy.fft).
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sig
from scipy import fft as spfft

# ── ForgeArray interop ──────────────────────────────────────────
try:
    from forge.engine.types import ForgeArray, _unwrap
except ImportError:
    # Fallback: ForgeArray = ndarray, _unwrap = np.asarray
    ForgeArray = np.ndarray

    def _unwrap(x):
        """Extract underlying ndarray from ForgeArray or convert to array."""
        if isinstance(x, np.ndarray):
            return x
        return np.asarray(x, dtype=np.float64)


def _fa(x):
    """Wrap result as ForgeArray (or plain ndarray if ForgeArray unavailable)."""
    arr = np.asarray(x)
    if ForgeArray is np.ndarray:
        return arr
    return ForgeArray(arr)


def _ensure_float(x):
    """Convert input to float64 ndarray."""
    return np.asarray(_unwrap(x), dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════
# 1. WINDOW FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def bartlett(N):
    """Bartlett (triangular) window.

    W = bartlett(N)
    Returns an N-point Bartlett window in a column vector.
    """
    return _fa(sig.windows.bartlett(int(N)))


def blackman(N):
    """Blackman window.

    W = blackman(N)
    Returns an N-point Blackman window.
    """
    return _fa(sig.windows.blackman(int(N)))


def hamming(N):
    """Hamming window.

    W = hamming(N)
    Returns an N-point symmetric Hamming window.
    """
    return _fa(sig.windows.hamming(int(N), sym=True))


def hanning(N):
    """Hanning window.

    W = hanning(N)
    Returns an N-point Hanning (Hann) window. Uses symmetric convention.
    """
    return _fa(sig.windows.hann(int(N), sym=True))


def kaiser_win(N, beta):
    """Kaiser window.

    W = kaiser(N, BETA)
    Returns an N-point Kaiser window with parameter BETA.
    """
    return _fa(sig.windows.kaiser(int(N), float(beta)))


def flattop(N):
    """Flat top window.

    W = flattop(N)
    Returns an N-point flat top window. Good for amplitude measurement.
    """
    return _fa(sig.windows.flattop(int(N)))


# ═══════════════════════════════════════════════════════════════════
# 2. FILTER DESIGN (IIR)
# ═══════════════════════════════════════════════════════════════════

def butter(N, Wn, btype='low', analog=False, fs=None):
    """Butterworth filter design.

    [B, A] = butter(N, Wn)
    [B, A] = butter(N, Wn, 'high')
    [B, A] = butter(N, [Wl, Wh], 'bandpass')

    Parameters
    ----------
    N : int — Filter order.
    Wn : float or (float, float) — Cutoff frequency (normalized 0-1 or Hz if fs given).
    btype : str — 'low', 'high', 'bandpass', 'bandstop'.
    analog : bool — If True, design analog filter.
    fs : float or None — Sampling frequency.

    Returns
    -------
    b, a : ForgeArray — Numerator and denominator coefficients.
    """
    b, a = sig.butter(int(N), Wn, btype=btype, analog=analog,
                       output='ba', fs=fs)
    return _fa(b), _fa(a)


def cheby1(N, Rp, Wn, btype='low', analog=False, fs=None):
    """Chebyshev Type I filter design.

    [B, A] = cheby1(N, Rp, Wn)

    Parameters
    ----------
    N : int — Filter order.
    Rp : float — Maximum ripple in passband (dB).
    Wn : float — Cutoff frequency.
    """
    b, a = sig.cheby1(int(N), float(Rp), Wn, btype=btype,
                       analog=analog, output='ba', fs=fs)
    return _fa(b), _fa(a)


def cheby2(N, Rs, Wn, btype='low', analog=False, fs=None):
    """Chebyshev Type II filter design.

    [B, A] = cheby2(N, Rs, Wn)

    Parameters
    ----------
    N : int — Filter order.
    Rs : float — Minimum stopband attenuation (dB).
    Wn : float — Cutoff frequency.
    """
    b, a = sig.cheby2(int(N), float(Rs), Wn, btype=btype,
                       analog=analog, output='ba', fs=fs)
    return _fa(b), _fa(a)


def ellip(N, Rp, Rs, Wn, btype='low', analog=False, fs=None):
    """Elliptic (Cauer) filter design.

    [B, A] = ellip(N, Rp, Rs, Wn)

    Parameters
    ----------
    N : int — Filter order.
    Rp : float — Passband ripple (dB).
    Rs : float — Stopband attenuation (dB).
    Wn : float — Cutoff frequency.
    """
    b, a = sig.ellip(int(N), float(Rp), float(Rs), Wn, btype=btype,
                      analog=analog, output='ba', fs=fs)
    return _fa(b), _fa(a)


def besself(N, Wn, btype='low', analog=True, fs=None):
    """Bessel/Thomson filter design.

    [B, A] = besself(N, Wn)

    Parameters
    ----------
    N : int — Filter order.
    Wn : float — Cutoff frequency (rad/s for analog, normalized for digital).
    analog : bool — Default True (Bessel filters are typically analog).
    """
    if not analog and fs is None:
        # Digital Bessel: Wn is normalized 0-1
        norm = 'phase'
    else:
        norm = 'phase'
    b, a = sig.bessel(int(N), Wn, btype=btype, analog=analog,
                       output='ba', norm=norm, fs=fs)
    return _fa(b), _fa(a)


# ── FIR design ──────────────────────────────────────────────────

def firwin(numtaps, cutoff, width=None, window='hamming', pass_zero=True, fs=None):
    """FIR filter design using the window method.

    B = firwin(NUMTAPS, CUTOFF)

    Parameters
    ----------
    numtaps : int — Length of the filter (number of coefficients, must be odd for Type I).
    cutoff : float or array — Cutoff frequency/frequencies.
    """
    b = sig.firwin(int(numtaps), cutoff, width=width, window=window,
                    pass_zero=pass_zero, fs=fs)
    return _fa(b)


def firwin2(numtaps, freq, gain, window='hamming', antisymmetric=False, fs=None):
    """FIR filter design using the window method with arbitrary response.

    B = firwin2(NUMTAPS, FREQ, GAIN)
    """
    b = sig.firwin2(int(numtaps), freq, gain, window=window,
                     antisymmetric=antisymmetric, fs=fs)
    return _fa(b)


def kaiserord(ripple, width):
    """Estimate Kaiser window FIR filter order.

    [N, BETA] = kaiserord(RIPPLE, WIDTH)

    Parameters
    ----------
    ripple : float — Maximum deviation (ripple) in dB.
    width : float — Transition width (normalized frequency).
    """
    N, beta = sig.kaiserord(float(ripple), float(width))
    return int(N), float(beta)


def remez(numtaps, bands, desired, weight=None, type='bandpass', fs=None):
    """Parks-McClellan optimal FIR filter design.

    B = remez(NUMTAPS, BANDS, DESIRED)

    Parameters
    ----------
    numtaps : int — Filter length.
    bands : array-like — Frequency band edges (pairs).
    desired : array-like — Desired gain in each band.
    """
    kwargs = {}
    if weight is not None:
        kwargs['weight'] = weight
    if fs is not None:
        kwargs['fs'] = fs
    b = sig.remez(int(numtaps), bands, desired, type=type, **kwargs)
    return _fa(b)


# ═══════════════════════════════════════════════════════════════════
# 3. FILTERING
# ═══════════════════════════════════════════════════════════════════

def fftfilt(b, x, n=None):
    """FFT-based FIR filtering (overlap-add).

    Y = fftfilt(B, X)
    Y = fftfilt(B, X, N) — use FFT of length N.

    Equivalent to filter(B, 1, X) but uses FFT for efficiency.
    """
    b = _ensure_float(b)
    x = _ensure_float(x)
    nb = len(b)
    nx = len(x)
    if n is None:
        # Choose FFT length as next power of 2 >= nb + block - 1
        n = max(256, 2 ** int(np.ceil(np.log2(nb + nb - 1))))
    n = int(n)
    nfft = max(n, nb)
    L = nfft - nb + 1  # block length
    B = np.fft.fft(b, nfft)
    y = np.zeros(nx)
    pos = 0
    while pos < nx:
        block_end = min(pos + L, nx)
        xblock = np.zeros(nfft)
        xblock[:block_end - pos] = x[pos:block_end]
        yblock = np.real(np.fft.ifft(np.fft.fft(xblock, nfft) * B))
        valid = min(L, nx - pos)
        y[pos:pos + valid] += yblock[:valid]
        pos += L
    return _fa(y[:nx])


def fftconv(a, b, n=None):
    """FFT-based convolution.

    C = fftconv(A, B)
    C = fftconv(A, B, N) — use FFT of length N.
    """
    a = _ensure_float(a)
    b = _ensure_float(b)
    na, nb = len(a), len(b)
    outlen = na + nb - 1
    if n is None:
        n = 2 ** int(np.ceil(np.log2(outlen)))
    else:
        n = int(n)
    A = np.fft.fft(a, n)
    B = np.fft.fft(b, n)
    c = np.real(np.fft.ifft(A * B))
    return _fa(c[:outlen])


def filtfilt(b, a, x, padtype='odd', padlen=None):
    """Zero-phase digital filtering.

    Y = filtfilt(B, A, X)
    Applies filter forward and backward to eliminate phase distortion.
    """
    b = _ensure_float(b)
    a = _ensure_float(a)
    x = _ensure_float(x)
    y = sig.filtfilt(b, a, x, padtype=padtype, padlen=padlen)
    return _fa(y)


def lfilter(b, a, x, zi=None):
    """Filter data with an IIR or FIR filter.

    Y = lfilter(B, A, X)
    [Y, ZF] = lfilter(B, A, X, ZI)
    """
    b = _ensure_float(b)
    a = _ensure_float(a)
    x = _ensure_float(x)
    if zi is not None:
        zi = _ensure_float(zi)
        y, zf = sig.lfilter(b, a, x, zi=zi)
        return _fa(y), _fa(zf)
    return _fa(sig.lfilter(b, a, x))


def sosfilt(sos, x, zi=None):
    """Filter data with second-order sections.

    Y = sosfilt(SOS, X)
    """
    sos = np.asarray(_unwrap(sos), dtype=np.float64)
    x = _ensure_float(x)
    if zi is not None:
        zi = np.asarray(_unwrap(zi), dtype=np.float64)
        y, zf = sig.sosfilt(sos, x, zi=zi)
        return _fa(y), _fa(zf)
    return _fa(sig.sosfilt(sos, x))


def filter2(b, x, shape='same'):
    """Two-dimensional digital filter.

    Y = filter2(B, X)
    Y = filter2(B, X, SHAPE) — SHAPE is 'same', 'full', or 'valid'.
    """
    from scipy.signal import convolve2d
    b = np.asarray(_unwrap(b), dtype=np.float64)
    x = np.asarray(_unwrap(x), dtype=np.float64)
    # filter2 correlates (not convolves), equivalent to convolve2d with flipped kernel
    y = convolve2d(x, b[::-1, ::-1], mode=shape)
    return _fa(y)


# ═══════════════════════════════════════════════════════════════════
# 4. FREQUENCY RESPONSE
# ═══════════════════════════════════════════════════════════════════

def freqz(b, a=1, n=512, whole=False, fs=None):
    """Digital filter frequency response.

    [H, W] = freqz(B, A, N)
    Returns the N-point frequency response vector H and the angular
    frequency vector W (in rad/sample, or Hz if fs given).
    """
    b = _ensure_float(b)
    a = _ensure_float(np.atleast_1d(a))
    kwargs = {}
    if fs is not None:
        kwargs['fs'] = float(fs)
    w, h = sig.freqz(b, a, worN=int(n), whole=whole, **kwargs)
    return _fa(h), _fa(w)


def freqz_plot(w, h):
    """Plot frequency response from freqz output.

    freqz_plot(W, H)
    Returns (w, mag_dB, phase_deg) for non-interactive use.
    Magnitude in dB and phase in degrees.
    """
    w = _ensure_float(w)
    h = np.asarray(_unwrap(h))
    mag_dB = 20 * np.log10(np.maximum(np.abs(h), 1e-15))
    phase_deg = np.degrees(np.unwrap(np.angle(h)))
    return _fa(w), _fa(mag_dB), _fa(phase_deg)


def freqs(b, a, w=None):
    """Analog filter frequency response.

    [H, W] = freqs(B, A, W)
    """
    b = _ensure_float(b)
    a = _ensure_float(a)
    if w is None:
        w = np.logspace(-1, 3, 512)
    else:
        w = _ensure_float(w)
    wout, h = sig.freqs(b, a, worN=w)
    return _fa(h), _fa(wout)


def grpdelay(b, a=1, n=512, whole=False, fs=None):
    """Group delay of a digital filter.

    [GD, W] = grpdelay(B, A, N)
    Returns group delay in samples.
    """
    b = _ensure_float(b)
    a = _ensure_float(np.atleast_1d(a))
    kwargs = {}
    if fs is not None:
        kwargs['fs'] = float(fs)
    w, gd = sig.group_delay((b, a), w=int(n), whole=whole, **kwargs)
    return _fa(gd), _fa(w)


# ═══════════════════════════════════════════════════════════════════
# 5. SYSTEM CONVERSIONS
# ═══════════════════════════════════════════════════════════════════

def tf2zpk(b, a):
    """Transfer function to zero-pole-gain form.

    [Z, P, K] = tf2zpk(B, A)
    """
    b = _ensure_float(b)
    a = _ensure_float(a)
    z, p, k = sig.tf2zpk(b, a)
    return _fa(z), _fa(p), float(k)


def zpk2tf(z, p, k):
    """Zero-pole-gain to transfer function form.

    [B, A] = zpk2tf(Z, P, K)
    """
    z = np.asarray(_unwrap(z))
    p = np.asarray(_unwrap(p))
    b, a = sig.zpk2tf(z, p, float(k))
    return _fa(np.real(b)), _fa(np.real(a))


def tf2ss(b, a):
    """Transfer function to state-space form.

    [A, B, C, D] = tf2ss(NUM, DEN)
    """
    b_arr = _ensure_float(b)
    a_arr = _ensure_float(a)
    A, B, C, D = sig.tf2ss(b_arr, a_arr)
    return _fa(A), _fa(B), _fa(C), _fa(D)


def ss2tf(A, B, C, D, input_idx=0):
    """State-space to transfer function form.

    [NUM, DEN] = ss2tf(A, B, C, D)
    """
    A = np.asarray(_unwrap(A), dtype=np.float64)
    B = np.asarray(_unwrap(B), dtype=np.float64)
    C = np.asarray(_unwrap(C), dtype=np.float64)
    D = np.asarray(_unwrap(D), dtype=np.float64)
    num, den = sig.ss2tf(A, B, C, D, input=input_idx)
    return _fa(num.squeeze()), _fa(den)


# ═══════════════════════════════════════════════════════════════════
# 6. SPECTRAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def periodogram(x, fs=1.0, window=None, nfft=None, sides='onesided'):
    """Periodogram power spectral density estimate.

    [PXX, F] = periodogram(X, FS)
    [PXX, F] = periodogram(X, FS, WINDOW, NFFT)
    """
    x = _ensure_float(x)
    N = len(x)
    if window is not None:
        if isinstance(window, str):
            w = sig.get_window(window, N)
        else:
            w = _ensure_float(window)
    else:
        w = np.ones(N)
    if nfft is None:
        nfft = N
    nfft = int(nfft)
    xw = x * w
    X = np.fft.fft(xw, nfft)
    Pxx = (np.abs(X) ** 2) / (float(fs) * np.sum(w ** 2))
    if sides == 'onesided':
        n_out = nfft // 2 + 1
        Pxx = Pxx[:n_out]
        Pxx[1:-1] *= 2
        f = np.linspace(0, float(fs) / 2, n_out)
    else:
        f = np.linspace(0, float(fs), nfft, endpoint=False)
    return _fa(Pxx), _fa(f)


def spectrogram(x, fs=1.0, window=None, noverlap=None, nfft=None):
    """STFT-based spectrogram.

    [S, F, T] = spectrogram(X, FS)
    Returns the magnitude squared of the STFT.
    """
    x = _ensure_float(x)
    N = len(x)
    if window is None:
        nperseg = min(256, N)
        window = sig.windows.hann(nperseg, sym=False)
    elif isinstance(window, (int, float)):
        nperseg = int(window)
        window = sig.windows.hann(nperseg, sym=False)
    else:
        window = _ensure_float(window)
        nperseg = len(window)
    if noverlap is None:
        noverlap = nperseg // 2
    if nfft is None:
        nfft = nperseg
    f, t, Sxx = sig.spectrogram(x, fs=float(fs), window=window,
                                  noverlap=int(noverlap), nfft=int(nfft))
    return _fa(Sxx), _fa(f), _fa(t)


def stft(x, window=None, noverlap=None, nfft=None, fs=1.0):
    """Short-time Fourier transform.

    [S, F, T] = stft(X)
    [S, F, T] = stft(X, WINDOW, NOVERLAP, NFFT, FS)
    """
    x = _ensure_float(x)
    N = len(x)
    if window is None:
        nperseg = min(256, N)
        win = sig.windows.hann(nperseg, sym=False)
    elif isinstance(window, (int, float)):
        nperseg = int(window)
        win = sig.windows.hann(nperseg, sym=False)
    else:
        win = _ensure_float(window)
        nperseg = len(win)
    if noverlap is None:
        noverlap = nperseg // 2
    if nfft is None:
        nfft = nperseg
    f, t, Zxx = sig.stft(x, fs=float(fs), window=win,
                          nperseg=nperseg, noverlap=int(noverlap),
                          nfft=int(nfft))
    return _fa(Zxx), _fa(f), _fa(t)


def synthesis(S, window=None, noverlap=None, nfft=None, fs=1.0):
    """Inverse STFT — reconstruct time signal from STFT output.

    X = synthesis(S, WINDOW, NOVERLAP, NFFT, FS)

    Parameters
    ----------
    S : complex array — STFT matrix (freq x time).
    """
    S = np.asarray(_unwrap(S))
    if window is None:
        # Infer nperseg from frequency axis
        nfreqs = S.shape[0]
        nperseg = 2 * (nfreqs - 1)
        win = sig.windows.hann(nperseg, sym=False)
    elif isinstance(window, (int, float)):
        nperseg = int(window)
        win = sig.windows.hann(nperseg, sym=False)
    else:
        win = _ensure_float(window)
        nperseg = len(win)
    if noverlap is None:
        noverlap = nperseg // 2
    if nfft is None:
        nfft = nperseg
    _, x = sig.istft(S, fs=float(fs), window=win,
                      nperseg=nperseg, noverlap=int(noverlap),
                      nfft=int(nfft))
    return _fa(x)


def spectral_adf(c, n=256):
    """Spectral density from autocovariance via FFT (auto density function).

    SXX = spectral_adf(C, N)

    Parameters
    ----------
    c : array — Autocovariance sequence (symmetric or one-sided).
    n : int — Number of FFT points.
    """
    c = _ensure_float(c)
    n = int(n)
    # Compute PSD from autocovariance: S(f) = FFT(c)
    C = np.fft.fft(c, n)
    return _fa(np.real(C))


def spectral_xdf(c, n=256):
    """Cross spectral density from cross-covariance via FFT.

    SXY = spectral_xdf(C, N)

    Parameters
    ----------
    c : array — Cross-covariance sequence.
    n : int — Number of FFT points.
    """
    c = _ensure_float(c)
    n = int(n)
    C = np.fft.fft(c, n)
    return _fa(C)


# ═══════════════════════════════════════════════════════════════════
# 7. TIME SERIES / ARMA
# ═══════════════════════════════════════════════════════════════════

def autoreg_matrix(y, k):
    """Autoregression matrix.

    X = autoreg_matrix(Y, K)
    Given a column vector Y, returns an (N-K) x K matrix where row i
    contains [y(i+k-1), y(i+k-2), ..., y(i)].
    """
    y = _ensure_float(y).ravel()
    k = int(k)
    N = len(y)
    if k >= N:
        raise ValueError("k must be less than length of y")
    nrows = N - k
    X = np.zeros((nrows, k))
    for j in range(k):
        X[:, j] = y[k - 1 - j:N - 1 - j]
    return _fa(X)


def yulewalker(c):
    """Yule-Walker method for AR parameter estimation.

    [A, V] = yulewalker(C)

    Parameters
    ----------
    c : array — Autocorrelation sequence (c[0] = lag 0, c[1] = lag 1, ...).

    Returns
    -------
    a : ForgeArray — AR coefficients (excluding leading 1).
    v : float — Prediction error variance.
    """
    c = _ensure_float(c).ravel()
    p = len(c) - 1
    if p < 1:
        return _fa(np.array([])), float(c[0])
    # Build Toeplitz matrix
    from scipy.linalg import solve_toeplitz
    r = c[:p]
    rhs = c[1:p + 1]
    a = solve_toeplitz(r, rhs)
    v = c[0] - np.dot(a, c[1:p + 1])
    return _fa(a), float(v)


def durbinlevinson(c, p=None):
    """Durbin-Levinson algorithm for AR estimation.

    [A, V] = durbinlevinson(C)
    [A, V] = durbinlevinson(C, P)

    Uses the Levinson-Durbin recursion on autocovariance C.

    Parameters
    ----------
    c : array — Autocovariance sequence starting at lag 0.
    p : int or None — AR order. Default: len(c) - 1.
    """
    c = _ensure_float(c).ravel()
    if p is None:
        p = len(c) - 1
    p = int(p)
    if p < 1:
        return _fa(np.array([])), float(c[0])

    # Levinson-Durbin recursion
    a = np.zeros(p)
    v = c[0]
    a[0] = c[1] / v
    v = v * (1 - a[0] ** 2)

    for i in range(1, p):
        # Reflection coefficient
        lam = (c[i + 1] - np.dot(a[:i], c[i:0:-1])) / v
        # Update coefficients
        a_new = np.copy(a[:i + 1])
        a_new[i] = lam
        for j in range(i):
            a_new[j] = a[j] - lam * a[i - 1 - j]
        a[:i + 1] = a_new
        v = v * (1 - lam ** 2)
    return _fa(a), float(v)


def arma_rnd(a, b, v, t, n=1):
    """Simulate ARMA(p,q) process.

    X = arma_rnd(A, B, V, T, N)

    Parameters
    ----------
    a : array — AR coefficients [a1, a2, ..., ap] (not including leading 1).
    b : array — MA coefficients [b0, b1, ..., bq] (b0 is usually 1).
    v : float — Innovation variance.
    t : int — Number of time steps to simulate.
    n : int — Number of realizations (columns).
    """
    a = _ensure_float(a).ravel()
    b = _ensure_float(b).ravel()
    v = float(v)
    t = int(t)
    n = int(n)
    rng = np.random.default_rng()
    # AR polynomial: A(z) = 1 - a1*z^{-1} - ... (convention: y(t) = a1*y(t-1)+...+e(t))
    # Denominator polynomial for scipy: [1, -a1, -a2, ...]
    den = np.concatenate(([1.0], -a))
    num = b if len(b) > 0 else np.array([1.0])
    result = np.zeros((t, n))
    for col in range(n):
        e = rng.normal(0, np.sqrt(v), t + max(len(den), len(num)))
        y = sig.lfilter(num, den, e)
        result[:, col] = y[-t:]
    if n == 1:
        return _fa(result.ravel())
    return _fa(result)


def arch_rnd(a, b, t):
    """Simulate ARCH(p) process.

    X = arch_rnd(A, B, T)

    Parameters
    ----------
    a : float — Constant variance term.
    b : array — ARCH coefficients [b1, ..., bp].
    t : int — Number of time steps.
    """
    a = float(a)
    b = _ensure_float(b).ravel()
    t = int(t)
    p = len(b)
    rng = np.random.default_rng()
    y = np.zeros(t)
    h = np.zeros(t)
    e = rng.standard_normal(t)
    for i in range(t):
        h[i] = a
        for j in range(min(i, p)):
            h[i] += b[j] * y[i - 1 - j] ** 2
        h[i] = max(h[i], 1e-12)
        y[i] = e[i] * np.sqrt(h[i])
    return _fa(y)


def arch_fit(y, p=1):
    """Fit ARCH(p) model to data using OLS.

    [A, B] = arch_fit(Y, P)

    Parameters
    ----------
    y : array — Time series data.
    p : int — ARCH order.

    Returns
    -------
    a : float — Constant term.
    b : ForgeArray — ARCH coefficients.
    """
    y = _ensure_float(y).ravel()
    t = len(y)
    p = int(p)
    # Squared residuals (assume mean-zero series)
    e2 = y ** 2
    # Build regressor matrix
    if t <= p:
        raise ValueError("Series too short for ARCH(p) estimation")
    X = np.ones((t - p, p + 1))
    for j in range(p):
        X[:, j + 1] = e2[p - 1 - j:t - 1 - j]
    dep = e2[p:]
    # OLS
    theta = np.linalg.lstsq(X, dep, rcond=None)[0]
    return float(theta[0]), _fa(theta[1:])


def arch_test(y, lags=12):
    """Engle's ARCH test for conditional heteroscedasticity.

    [STAT, PVALUE] = arch_test(Y, LAGS)

    Tests H0: no ARCH effects.

    Returns
    -------
    stat : float — Test statistic (T * R^2).
    pvalue : float — p-value from chi-squared distribution.
    """
    from scipy.stats import chi2
    y = _ensure_float(y).ravel()
    e2 = y ** 2
    t = len(e2)
    lags = int(lags)
    # Regress e2 on lagged e2
    X = np.ones((t - lags, lags + 1))
    for j in range(lags):
        X[:, j + 1] = e2[lags - 1 - j:t - 1 - j]
    dep = e2[lags:]
    # OLS
    theta = np.linalg.lstsq(X, dep, rcond=None)[0]
    fitted = X @ theta
    ss_res = np.sum((dep - fitted) ** 2)
    ss_tot = np.sum((dep - np.mean(dep)) ** 2)
    if ss_tot == 0:
        return 0.0, 1.0
    R2 = 1 - ss_res / ss_tot
    stat = (t - lags) * R2
    pvalue = 1 - chi2.cdf(stat, lags)
    return float(stat), float(pvalue)


def diffpara(x, d_range=None):
    """Estimate fractional differencing parameter.

    D = diffpara(X)
    [D, DD] = diffpara(X, D_RANGE)

    Uses the GPH (Geweke-Porter-Hudak) estimator.

    Parameters
    ----------
    x : array — Time series.
    d_range : array or None — Grid of d values to search.

    Returns
    -------
    d : float — Estimated fractional differencing parameter.
    """
    x = _ensure_float(x).ravel()
    n = len(x)
    # GPH estimator using periodogram regression
    m = int(np.floor(np.sqrt(n)))
    # Periodogram at Fourier frequencies
    freqs = np.arange(1, m + 1) * 2 * np.pi / n
    I = np.abs(np.fft.fft(x - np.mean(x), n)[:m]) ** 2 / (2 * np.pi * n)
    I = np.maximum(I[: m], 1e-30)
    # Regress log(I) on log(4 sin^2(freq/2))
    logI = np.log(I)
    logS = np.log(4 * np.sin(freqs / 2) ** 2)
    # OLS
    X = np.column_stack([np.ones(m), logS])
    beta = np.linalg.lstsq(X, logI, rcond=None)[0]
    d = -beta[1]
    return float(d)


def fractdiff(x, d):
    """Fractionally difference a time series.

    Y = fractdiff(X, D)

    Applies the fractional differencing operator (1-L)^d to X using
    the binomial expansion truncated at the series length.
    """
    x = _ensure_float(x).ravel()
    d = float(d)
    n = len(x)
    # Compute weights from binomial expansion of (1-L)^d
    weights = np.zeros(n)
    weights[0] = 1.0
    for k in range(1, n):
        weights[k] = weights[k - 1] * (d - k + 1) / k
    # Apply filter
    y = np.convolve(x, weights[:n], mode='full')[:n]
    return _fa(y)


def hurst(x):
    """Estimate the Hurst exponent of a time series.

    H = hurst(X)

    Uses the rescaled range (R/S) method.
    """
    x = _ensure_float(x).ravel()
    n = len(x)
    if n < 20:
        raise ValueError("Series too short for Hurst estimation")
    # Use multiple sub-series lengths
    max_k = int(np.floor(np.log2(n)))
    ns = []
    rs = []
    for k in range(2, max_k + 1):
        size = 2 ** k
        nblocks = n // size
        if nblocks < 1:
            continue
        rs_vals = []
        for b in range(nblocks):
            block = x[b * size:(b + 1) * size]
            mean_b = np.mean(block)
            cumdev = np.cumsum(block - mean_b)
            R = np.max(cumdev) - np.min(cumdev)
            S = np.std(block, ddof=1)
            if S > 0:
                rs_vals.append(R / S)
        if rs_vals:
            ns.append(size)
            rs.append(np.mean(rs_vals))
    if len(ns) < 2:
        return 0.5
    log_n = np.log(ns)
    log_rs = np.log(rs)
    # Linear regression
    H = np.polyfit(log_n, log_rs, 1)[0]
    return float(H)


def spencer(x):
    """Spencer's 15-point moving average.

    Y = spencer(X)

    Applies Spencer's 15-point weighted moving average filter.
    Weights: [-3, -6, -5, 3, 21, 46, 67, 74, 67, 46, 21, 3, -5, -6, -3] / 320.
    """
    x = _ensure_float(x).ravel()
    weights = np.array([-3, -6, -5, 3, 21, 46, 67, 74,
                         67, 46, 21, 3, -5, -6, -3], dtype=np.float64) / 320.0
    n = len(x)
    nw = len(weights)
    if n < nw:
        raise ValueError("Series must be at least 15 points for Spencer filter")
    y = np.convolve(x, weights, mode='valid')
    # Pad with NaN to match input length (Octave convention)
    pad = (nw - 1) // 2
    result = np.full(n, np.nan)
    result[pad:pad + len(y)] = y
    return _fa(result)


# ═══════════════════════════════════════════════════════════════════
# 8. TRANSFORMS & CORRELATION
# ═══════════════════════════════════════════════════════════════════

def fftshift(x, dim=None):
    """Shift zero-frequency component to center.

    Y = fftshift(X)
    Y = fftshift(X, DIM)
    """
    x = np.asarray(_unwrap(x))
    if dim is None:
        return _fa(np.fft.fftshift(x))
    return _fa(np.fft.fftshift(x, axes=int(dim)))


def ifftshift(x, dim=None):
    """Inverse of fftshift.

    Y = ifftshift(X)
    """
    x = np.asarray(_unwrap(x))
    if dim is None:
        return _fa(np.fft.ifftshift(x))
    return _fa(np.fft.ifftshift(x, axes=int(dim)))


def hilbert_transform(x, n=None):
    """Discrete-time analytic signal via Hilbert transform.

    XA = hilbert(X)
    Returns the analytic signal whose real part is X and imaginary part
    is the Hilbert transform of X.
    """
    x = _ensure_float(x)
    if n is not None:
        n = int(n)
    return _fa(sig.hilbert(x, N=n))


def xcorr(x, y=None, maxlag=None, scale='none'):
    """Cross-correlation estimate.

    [R, LAGS] = xcorr(X, Y, MAXLAG, SCALE)
    [R, LAGS] = xcorr(X, MAXLAG, SCALE) — autocorrelation.

    Parameters
    ----------
    x : array — Input signal.
    y : array or None — Second signal. If None, computes autocorrelation.
    maxlag : int or None — Maximum lag. Default: len(x) - 1.
    scale : str — 'none', 'biased', 'unbiased', 'coeff'.
    """
    x = _ensure_float(x).ravel()
    if y is None:
        y = x.copy()
    else:
        y = _ensure_float(y).ravel()
    N = max(len(x), len(y))
    if maxlag is None:
        maxlag = N - 1
    maxlag = int(maxlag)
    # Full cross-correlation via FFT
    nfft = 2 ** int(np.ceil(np.log2(2 * N - 1)))
    X = np.fft.fft(x, nfft)
    Y = np.fft.fft(y, nfft)
    R_full = np.real(np.fft.ifft(X * np.conj(Y)))
    # Rearrange: negative lags then positive
    R = np.concatenate([R_full[-(N - 1):], R_full[:N]])
    lags = np.arange(-(N - 1), N)
    # Trim to maxlag
    center = N - 1
    R = R[center - maxlag:center + maxlag + 1]
    lags = lags[center - maxlag:center + maxlag + 1]
    # Scaling
    if scale == 'biased':
        R = R / N
    elif scale == 'unbiased':
        denom = N - np.abs(lags)
        denom[denom <= 0] = 1
        R = R / denom
    elif scale == 'coeff':
        R = R / R[maxlag]  # Normalize by zero-lag
    return _fa(R), _fa(lags)


def xcov(x, y=None, maxlag=None, scale='none'):
    """Cross-covariance (mean-removed cross-correlation).

    [C, LAGS] = xcov(X, Y, MAXLAG, SCALE)
    """
    x = _ensure_float(x).ravel()
    x = x - np.mean(x)
    if y is not None:
        y = _ensure_float(y).ravel()
        y = y - np.mean(y)
    return xcorr(x, y, maxlag=maxlag, scale=scale)


# ═══════════════════════════════════════════════════════════════════
# 9. UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def sinc(x):
    """Sinc function: sin(pi*x) / (pi*x).

    Y = sinc(X)
    """
    return _fa(np.sinc(_ensure_float(x)))


def unwrap(p, tol=np.pi, dim=None):
    """Unwrap radian phase angles.

    Q = unwrap(P)
    Q = unwrap(P, TOL) — use TOL as jump tolerance (default pi).
    """
    p = _ensure_float(p)
    if dim is not None:
        return _fa(np.unwrap(p, discont=float(tol), axis=int(dim)))
    return _fa(np.unwrap(p, discont=float(tol)))


def detrend(x, p=1):
    """Remove trend from data.

    Y = detrend(X)       — remove linear trend (p=1).
    Y = detrend(X, 0)    — remove mean (constant detrend).
    Y = detrend(X, P)    — remove polynomial of order P.

    For p=0 or p=1, delegates to scipy.signal.detrend.
    For p>1, fits and removes polynomial of degree p.
    """
    x = _ensure_float(x)
    p = int(p)
    if p == 0:
        return _fa(sig.detrend(x, type='constant'))
    elif p == 1:
        return _fa(sig.detrend(x, type='linear'))
    else:
        # Polynomial detrend
        n = len(x)
        t = np.arange(n, dtype=np.float64)
        coeffs = np.polyfit(t, x, p)
        trend = np.polyval(coeffs, t)
        return _fa(x - trend)


def resample(x, num, t=None):
    """Resample signal to NUM samples using polyphase method.

    Y = resample(X, NUM)
    [Y, TN] = resample(X, NUM, T)
    """
    x = _ensure_float(x)
    if t is not None:
        t = _ensure_float(t)
        y, tn = sig.resample(x, int(num), t=t)
        return _fa(y), _fa(tn)
    return _fa(sig.resample(x, int(num)))


def decimate(x, q, n=None, ftype='iir'):
    """Decrease sampling rate by integer factor.

    Y = decimate(X, Q)
    Y = decimate(X, Q, N, FTYPE)
    """
    x = _ensure_float(x)
    q = int(q)
    kwargs = {}
    if n is not None:
        kwargs['n'] = int(n)
    kwargs['ftype'] = ftype
    return _fa(sig.decimate(x, q, **kwargs))


def interp(x, r, l=4, alpha=0.5):
    """Increase sampling rate by integer factor.

    Y = interp(X, R)
    Y = interp(X, R, L, ALPHA)

    Parameters
    ----------
    x : array — Input signal.
    r : int — Interpolation factor.
    l : int — Filter half-length (default 4).
    alpha : float — Kaiser window parameter (default 0.5).
    """
    x = _ensure_float(x)
    r = int(r)
    n = len(x)
    # Upsample by inserting zeros
    y_up = np.zeros(n * r)
    y_up[::r] = x
    # Design lowpass interpolation filter
    nfilt = 2 * l * r + 1
    b = sig.firwin(nfilt, 1.0 / r, window=('kaiser', alpha * r))
    b *= r  # Compensate for energy loss from zero insertion
    y = np.convolve(y_up, b, mode='same')
    return _fa(y)


def findpeaks(x, height=None, distance=None, prominence=None, width=None):
    """Find local maxima in a signal.

    [PKS, LOCS] = findpeaks(X)
    [PKS, LOCS] = findpeaks(X, 'MinPeakHeight', H, ...)

    Returns
    -------
    pks : ForgeArray — Peak values.
    locs : ForgeArray — Peak locations (0-indexed).
    """
    x = _ensure_float(x)
    kwargs = {}
    if height is not None:
        kwargs['height'] = float(height)
    if distance is not None:
        kwargs['distance'] = int(distance)
    if prominence is not None:
        kwargs['prominence'] = float(prominence)
    if width is not None:
        kwargs['width'] = float(width)
    peaks, properties = sig.find_peaks(x, **kwargs)
    return _fa(x[peaks]), _fa(peaks)


def movfun(f, x, wlen):
    """Apply function F over a moving window of length WLEN.

    Y = movfun(F, X, WLEN)

    Parameters
    ----------
    f : callable — Function to apply (e.g., np.mean).
    x : array — Input data.
    wlen : int — Window length.
    """
    x = _ensure_float(x).ravel()
    wlen = int(wlen)
    n = len(x)
    y = np.zeros(n)
    hw = wlen // 2
    for i in range(n):
        lo = max(0, i - hw)
        hi = min(n, i + hw + 1)
        y[i] = f(x[lo:hi])
    return _fa(y)


def movslice(n, wlen):
    """Compute start/end indices for a moving window.

    [IB, IE] = movslice(N, WLEN)

    Parameters
    ----------
    n : int — Length of data.
    wlen : int — Window length.

    Returns
    -------
    ib : ForgeArray — Beginning indices (0-based).
    ie : ForgeArray — Ending indices (0-based, exclusive).
    """
    n = int(n)
    wlen = int(wlen)
    hw = wlen // 2
    ib = np.maximum(0, np.arange(n) - hw)
    ie = np.minimum(n, np.arange(n) + wlen - hw)
    return _fa(ib), _fa(ie)


def sinetone(freq, rate=1.0, dur=1.0, amp=1.0):
    """Generate a sine tone.

    Y = sinetone(FREQ, RATE, DUR, AMP)

    Parameters
    ----------
    freq : float — Frequency in Hz.
    rate : float — Sample rate in Hz.
    dur : float — Duration in seconds.
    amp : float — Amplitude.
    """
    freq = float(freq)
    rate = float(rate)
    dur = float(dur)
    amp = float(amp)
    t = np.arange(0, dur, 1.0 / rate)
    return _fa(amp * np.sin(2 * np.pi * freq * t))


def sinewave(m, n=1, d=0):
    """Generate a sine wave of M samples.

    Y = sinewave(M, N, D)

    Parameters
    ----------
    m : int — Number of samples.
    n : int — Frequency divider (period = m/n samples).
    d : int — Phase delay in samples.
    """
    m = int(m)
    n = int(n)
    d = int(d)
    t = np.arange(m)
    return _fa(np.sin(2 * np.pi * n * (t - d) / m))


# ═══════════════════════════════════════════════════════════════════
# 10. WAVELET (basic, via manual implementation)
# ═══════════════════════════════════════════════════════════════════

def cwt_transform(x, scales, wavelet='morl'):
    """Continuous wavelet transform.

    COEFFS = cwt(X, SCALES, WAVELET)

    Parameters
    ----------
    x : array — Input signal.
    scales : array — Array of scales.
    wavelet : str — Wavelet name ('morl' for Morlet).

    Returns
    -------
    coeffs : ForgeArray — CWT coefficient matrix (scales x time).
    """
    x = _ensure_float(x).ravel()
    scales = _ensure_float(scales).ravel()
    n = len(x)
    coeffs = np.zeros((len(scales), n))
    for i, s in enumerate(scales):
        # Morlet wavelet
        t_wav = np.arange(-4 * s, 4 * s + 1) / s
        if wavelet == 'morl' or wavelet == 'morlet':
            psi = np.exp(-0.5 * t_wav ** 2) * np.cos(5 * t_wav)
        elif wavelet == 'mexh':
            psi = (1 - t_wav ** 2) * np.exp(-0.5 * t_wav ** 2)
        else:
            psi = np.exp(-0.5 * t_wav ** 2) * np.cos(5 * t_wav)
        psi = psi / np.sqrt(s)
        c = np.convolve(x, psi[::-1], mode='same')
        coeffs[i, :] = c
    return _fa(coeffs)


def dwt(x, wavelet='db4', mode='symmetric'):
    """Discrete wavelet transform (single level).

    [CA, CD] = dwt(X, WAVELET)

    Parameters
    ----------
    x : array — Input signal.
    wavelet : str — Wavelet name (e.g., 'db4', 'haar').
    mode : str — Signal extension mode.

    Returns
    -------
    cA : ForgeArray — Approximation coefficients.
    cD : ForgeArray — Detail coefficients.
    """
    try:
        import pywt
        x = _ensure_float(x).ravel()
        cA, cD = pywt.dwt(x, wavelet, mode=mode)
        return _fa(cA), _fa(cD)
    except ImportError:
        # Manual Haar wavelet as fallback
        x = _ensure_float(x).ravel()
        if wavelet != 'haar':
            raise ImportError("pywt required for wavelets other than 'haar'")
        n = len(x)
        if n % 2 != 0:
            x = np.append(x, 0)
            n += 1
        cA = (x[0::2] + x[1::2]) / np.sqrt(2)
        cD = (x[0::2] - x[1::2]) / np.sqrt(2)
        return _fa(cA), _fa(cD)


def idwt(cA, cD, wavelet='db4', mode='symmetric'):
    """Inverse discrete wavelet transform (single level).

    X = idwt(CA, CD, WAVELET)

    Parameters
    ----------
    cA : array — Approximation coefficients.
    cD : array — Detail coefficients.
    wavelet : str — Wavelet name.

    Returns
    -------
    x : ForgeArray — Reconstructed signal.
    """
    try:
        import pywt
        cA = _ensure_float(cA).ravel()
        cD = _ensure_float(cD).ravel()
        x = pywt.idwt(cA, cD, wavelet, mode=mode)
        return _fa(x)
    except ImportError:
        cA = _ensure_float(cA).ravel()
        cD = _ensure_float(cD).ravel()
        if wavelet != 'haar':
            raise ImportError("pywt required for wavelets other than 'haar'")
        n = len(cA)
        x = np.zeros(2 * n)
        x[0::2] = (cA + cD) / np.sqrt(2)
        x[1::2] = (cA - cD) / np.sqrt(2)
        return _fa(x)




def rectwin(N):
    """Rectangular window.

    W = rectwin(N)
    Returns an N-point rectangular window (all ones) as a column vector.
    """
    return _fa(np.ones((int(N), 1)))


def zerocrossing(x):
    """Find zero-crossing indices of a signal.

    idx = zerocrossing(x)
    Returns the indices where the signal *x* crosses zero.
    Uses linear interpolation to estimate fractional crossing points.
    """
    x = _ensure_float(x).ravel()
    signs = np.sign(x)
    # Find where sign changes (ignore zeros)
    sign_changes = np.where(np.diff(signs) != 0)[0]
    crossings = []
    for i in sign_changes:
        if x[i] == 0:
            crossings.append(float(i))
        elif x[i + 1] == 0:
            crossings.append(float(i + 1))
        else:
            # Linear interpolation
            frac = -x[i] / (x[i + 1] - x[i])
            crossings.append(i + frac)
    return _fa(np.array(crossings)) if crossings else _fa(np.array([]))


# ═══════════════════════════════════════════════════════════════════
# SIGNAL_REGISTRY — maps Octave/MATLAB function names to implementations
# ═══════════════════════════════════════════════════════════════════

SIGNAL_REGISTRY: dict[str, callable] = {
    "flattop": flattop,
    # ── Windows ──────────────────────────────────────────────────
    'bartlett':         bartlett,
    'blackman':         blackman,
    'hamming':          hamming,
    'hanning':          hanning,
    'kaiser':           kaiser_win,
    'rectwin':          rectwin,
    # ── Filter design (IIR) ─────────────────────────────────────
    'butter':           butter,
    'cheby1':           cheby1,
    'cheby2':           cheby2,
    'ellip':            ellip,
    'besself':          besself,
    # ── Filter design (FIR) ─────────────────────────────────────
    'firwin':           firwin,
    'firwin2':          firwin2,
    'kaiserord':        kaiserord,
    'remez':            remez,
    # ── Filtering ────────────────────────────────────────────────
    'fftfilt':          fftfilt,
    'fftconv':          fftconv,
    'filtfilt':         filtfilt,
    'lfilter':          lfilter,
    'sosfilt':          sosfilt,
    'filter2':          filter2,
    # ── Frequency response ──────────────────────────────────────
    'freqz':            freqz,
    'freqz_plot':       freqz_plot,
    'freqs':            freqs,
    'grpdelay':         grpdelay,
    # ── System conversions ──────────────────────────────────────
    'tf2zpk':           tf2zpk,
    'zpk2tf':           zpk2tf,
    'tf2ss':            tf2ss,
    'ss2tf':            ss2tf,
    # ── Spectral analysis ───────────────────────────────────────
    'periodogram':      periodogram,
    'spectrogram':      spectrogram,
    'stft':             stft,
    'synthesis':        synthesis,
    'spectral_adf':     spectral_adf,
    'spectral_xdf':     spectral_xdf,
    # ── Time series / ARMA ──────────────────────────────────────
    'autoreg_matrix':   autoreg_matrix,
    'yulewalker':       yulewalker,
    'durbinlevinson':   durbinlevinson,
    'arma_rnd':         arma_rnd,
    'arch_rnd':         arch_rnd,
    'arch_fit':         arch_fit,
    'arch_test':        arch_test,
    'diffpara':         diffpara,
    'fractdiff':        fractdiff,
    'hurst':            hurst,
    'spencer':          spencer,
    # ── Transforms & correlation ────────────────────────────────
    'fftshift':         fftshift,
    'ifftshift':        ifftshift,
    'hilbert':          hilbert_transform,
    'xcorr':            xcorr,
    'xcov':             xcov,
    # ── Utility ─────────────────────────────────────────────────
    'sinc':             sinc,
    'unwrap':           unwrap,
    'detrend':          detrend,
    'resample':         resample,
    'decimate':         decimate,
    'interp':           interp,
    'findpeaks':        findpeaks,
    'movfun':           movfun,
    'movslice':         movslice,
    'sinetone':         sinetone,
    'sinewave':         sinewave,
    'zerocrossing':     zerocrossing,
    # ── Wavelets ────────────────────────────────────────────────
    'cwt':              cwt_transform,
    'dwt':              dwt,
    'idwt':             idwt,
}
