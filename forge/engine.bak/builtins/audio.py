# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Audio Toolbox for Forge — Octave-compatible functions.

Implements 8 Octave audio functions: mu-law encoding/decoding, sound playback
stubs, and audioplayer/audiorecorder stubs.

Backend: NumPy for mu-law encoding.  Playback functions are stubs (print
warnings) since Forge runs headless.

SRS trace: SRS-FUNC-AUDIO
"""

from __future__ import annotations

import warnings
import numpy as np

# ── ForgeArray interop ───────────────────────────────────────────
try:
    from forge.engine.types import ForgeArray, _unwrap
except ImportError:
    ForgeArray = np.ndarray

    def _unwrap(x):
        if isinstance(x, np.ndarray):
            return x
        return np.asarray(x, dtype=np.float64)

try:
    from forge.engine.containers import ForgeChar
except ImportError:
    ForgeChar = str


def _fa(x):
    """Wrap result as ForgeArray."""
    arr = np.asarray(x)
    if ForgeArray is np.ndarray:
        return arr
    return ForgeArray(arr)


def _ensure_float(x):
    return np.asarray(_unwrap(x), dtype=np.float64)


def _scalar(x):
    if isinstance(x, ForgeArray):
        d = x.data if hasattr(x, 'data') else np.asarray(x)
        return d.flat[0].item() if d.size == 1 else d
    if isinstance(x, np.ndarray) and x.size == 1:
        return x.flat[0].item()
    return x


# ═══════════════════════════════════════════════════════════════════
# 1. MU-LAW ENCODING / DECODING
# ═══════════════════════════════════════════════════════════════════

def lin2mu(x, n=8):
    """Convert linear audio signal to mu-law encoding.

    MU = lin2mu(X)
    MU = lin2mu(X, N)

    Parameters
    ----------
    x : array — Linear audio signal, values in [-1, 1].
    n : int — Number of bits for mu-law encoding (default 8).
        Determines mu = 2^N - 1.

    Returns
    -------
    mu : ForgeArray — Mu-law encoded signal (integer values 0 to 2^N - 1).
    """
    data = _ensure_float(x)
    n = int(_scalar(n))
    mu = 2 ** n - 1

    # Clip to [-1, 1]
    data = np.clip(data, -1.0, 1.0)

    # Mu-law compression: F(x) = sgn(x) * ln(1 + mu*|x|) / ln(1 + mu)
    magnitude = np.log1p(mu * np.abs(data)) / np.log1p(mu)
    compressed = np.sign(data) * magnitude

    # Quantize to N-bit unsigned integer range [0, 2^N - 1]
    # Map [-1, 1] -> [0, 2^N - 1]
    quantized = np.round((compressed + 1.0) / 2.0 * mu).astype(np.float64)
    quantized = np.clip(quantized, 0, mu)

    return _fa(quantized)


def mu2lin(mu_data, n=8):
    """Convert mu-law encoded signal back to linear.

    X = mu2lin(MU)
    X = mu2lin(MU, N)

    Parameters
    ----------
    mu_data : array — Mu-law encoded signal (integer values 0 to 2^N - 1).
    n : int — Number of bits (default 8).

    Returns
    -------
    x : ForgeArray — Linear audio signal, values in [-1, 1].
    """
    data = _ensure_float(mu_data)
    n = int(_scalar(n))
    mu = 2 ** n - 1

    # Map [0, 2^N - 1] -> [-1, 1]
    y = data / mu * 2.0 - 1.0

    # Mu-law expansion: F^-1(y) = sgn(y) * (1/mu) * ((1 + mu)^|y| - 1)
    expanded = np.sign(y) * (1.0 / mu) * ((1.0 + mu) ** np.abs(y) - 1.0)

    return _fa(expanded)


# ═══════════════════════════════════════════════════════════════════
# 2. SOUND PLAYBACK (STUBS)
# ═══════════════════════════════════════════════════════════════════

def sound(y, fs=8000, *args):
    """Play audio signal through speakers (stub).

    sound(Y)
    sound(Y, FS)
    sound(Y, FS, NBITS)

    Parameters
    ----------
    y : array — Audio signal (mono: column vector, stereo: Nx2 matrix).
    fs : float — Sample rate in Hz (default 8000).

    Note: Audio playback is not available in headless mode.
    This function prints signal info instead.
    """
    data = _ensure_float(y)
    fs = float(_scalar(fs))
    nbits = int(_scalar(args[0])) if args else 16
    duration = data.shape[0] / fs if data.ndim >= 1 else 0

    channels = data.shape[1] if data.ndim == 2 else 1
    warnings.warn(
        f"sound: playback not available in headless mode. "
        f"Signal: {data.shape[0]} samples, {channels} channel(s), "
        f"{fs:.0f} Hz, {duration:.2f}s, {nbits}-bit"
    )


def soundsc(y, fs=8000, *args):
    """Play audio signal scaled to full range (stub).

    soundsc(Y)
    soundsc(Y, FS)
    soundsc(Y, FS, NBITS)
    soundsc(Y, FS, NBITS, [YMIN YMAX])

    Scales signal to [-1, 1] before playback.
    Note: Audio playback is not available in headless mode.
    """
    data = _ensure_float(y)
    fs = float(_scalar(fs))

    # Parse optional args
    nbits = 16
    ymin, ymax = None, None
    if len(args) >= 1:
        nbits = int(_scalar(args[0]))
    if len(args) >= 2:
        lims = _ensure_float(args[1]).ravel()
        ymin, ymax = lims[0], lims[1]

    # Scale to [-1, 1]
    if ymin is not None and ymax is not None:
        scaled = 2.0 * (data - ymin) / (ymax - ymin) - 1.0
    else:
        dmin, dmax = data.min(), data.max()
        if dmax - dmin > 0:
            scaled = 2.0 * (data - dmin) / (dmax - dmin) - 1.0
        else:
            scaled = np.zeros_like(data)

    scaled = np.clip(scaled, -1.0, 1.0)
    channels = scaled.shape[1] if scaled.ndim == 2 else 1
    duration = scaled.shape[0] / fs if scaled.ndim >= 1 else 0

    warnings.warn(
        f"soundsc: playback not available in headless mode. "
        f"Signal: {scaled.shape[0]} samples, {channels} channel(s), "
        f"{fs:.0f} Hz, {duration:.2f}s, {nbits}-bit (scaled)"
    )


def record(seconds, fs=8000, *args):
    """Record audio from microphone (stub).

    Y = record(SECONDS)
    Y = record(SECONDS, FS)
    Y = record(SECONDS, FS, NBITS)

    Returns silence (zeros) since recording is not available in headless mode.
    """
    seconds = float(_scalar(seconds))
    fs = float(_scalar(fs))
    nbits = int(_scalar(args[0])) if args else 16

    n_samples = int(seconds * fs)
    warnings.warn(
        f"record: audio recording not available in headless mode. "
        f"Returning {n_samples} samples of silence at {fs:.0f} Hz."
    )
    return _fa(np.zeros(n_samples, dtype=np.float64))


# ═══════════════════════════════════════════════════════════════════
# 3. AUDIOPLAYER / AUDIORECORDER (STUBS)
# ═══════════════════════════════════════════════════════════════════

class _AudioPlayerStub:
    """Stub audioplayer object.

    Stores audio data and parameters but cannot play back.
    """

    def __init__(self, y, fs=8000, nbits=16):
        self.data = _ensure_float(y)
        self.SampleRate = float(fs)
        self.BitsPerSample = int(nbits)
        self.NumChannels = self.data.shape[1] if self.data.ndim == 2 else 1
        self.TotalSamples = self.data.shape[0]
        self.CurrentSample = 0
        self.Running = 'off'

    def play(self, *args):
        warnings.warn("audioplayer.play: playback not available in headless mode")

    def playblocking(self, *args):
        warnings.warn("audioplayer.playblocking: playback not available in headless mode")

    def pause(self):
        warnings.warn("audioplayer.pause: playback not available in headless mode")

    def resume(self):
        warnings.warn("audioplayer.resume: playback not available in headless mode")

    def stop(self):
        self.CurrentSample = 0
        self.Running = 'off'

    def isplaying(self):
        return False

    def __repr__(self):
        return (f"audioplayer({self.TotalSamples} samples, "
                f"{self.NumChannels} ch, {self.SampleRate} Hz, "
                f"{self.BitsPerSample}-bit) [stub]")


class _AudioRecorderStub:
    """Stub audiorecorder object.

    Cannot actually record; stores parameters only.
    """

    def __init__(self, fs=8000, nbits=16, nchannels=1):
        self.SampleRate = float(fs)
        self.BitsPerSample = int(nbits)
        self.NumChannels = int(nchannels)
        self.TotalSamples = 0
        self.CurrentSample = 0
        self.Running = 'off'
        self._data = np.array([], dtype=np.float64)

    def record(self, *args):
        warnings.warn("audiorecorder.record: recording not available in headless mode")

    def recordblocking(self, duration):
        n = int(float(duration) * self.SampleRate)
        warnings.warn(
            f"audiorecorder.recordblocking: recording not available. "
            f"Generating {n} samples of silence."
        )
        self._data = np.zeros((n, self.NumChannels), dtype=np.float64)
        self.TotalSamples = n

    def pause(self):
        pass

    def resume(self):
        pass

    def stop(self):
        self.Running = 'off'

    def getaudiodata(self):
        return _fa(self._data)

    def isrecording(self):
        return False

    def __repr__(self):
        return (f"audiorecorder({self.SampleRate} Hz, "
                f"{self.BitsPerSample}-bit, {self.NumChannels} ch) [stub]")


def audioplayer(y, fs=8000, nbits=16):
    """Create an audioplayer object (stub).

    P = audioplayer(Y, FS)
    P = audioplayer(Y, FS, NBITS)

    Returns a stub object that stores audio data but cannot play.
    """
    return _AudioPlayerStub(y, fs, nbits)


def audiorecorder(fs=8000, nbits=16, nchannels=1):
    """Create an audiorecorder object (stub).

    R = audiorecorder()
    R = audiorecorder(FS, NBITS, NCHANNELS)

    Returns a stub object that cannot actually record.
    """
    return _AudioRecorderStub(fs, nbits, nchannels)


# ═══════════════════════════════════════════════════════════════════
# REGISTRY
# ═══════════════════════════════════════════════════════════════════

AUDIO_REGISTRY = {
    # ── Mu-law encoding ───────────────────────────────────────────
    'lin2mu':           lin2mu,
    'mu2lin':           mu2lin,
    # ── Sound playback (stubs) ────────────────────────────────────
    'sound':            sound,
    'soundsc':          soundsc,
    'record':           record,
    # ── Audio objects (stubs) ─────────────────────────────────────
    'audioplayer':      audioplayer,
    'audiorecorder':    audiorecorder,
}
