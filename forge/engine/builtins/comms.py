# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""
Forge Communications Toolbox
==============================
Digital modulation/demodulation, channel models, convolutional
coding / Viterbi decoding, error metrics, and source coding.

Backend: numpy
"""

import numpy as np
from numpy import ndarray
from typing import Union, Optional, Dict, Any, Tuple, List

Numeric = Union[int, float, np.number]


# ---------------------------------------------------------------------------
# Modulation / Demodulation
# ---------------------------------------------------------------------------

def _gray_encode(n: int, bits: int) -> int:
    """Convert integer to Gray code."""
    return n ^ (n >> 1)


def _gray_decode(g: int, bits: int) -> int:
    """Convert Gray code to integer."""
    n = g
    mask = g >> 1
    while mask:
        n ^= mask
        mask >>= 1
    return n


def qammod(data, M: int, gray: bool = True) -> ndarray:
    """QAM modulation.

    Parameters
    ----------
    data : array of integers in [0, M-1]
    M    : constellation size (must be a perfect square, e.g. 16, 64, 256)
    gray : use Gray coding (default True)

    Returns
    -------
    complex ndarray of modulated symbols
    """
    data = np.asarray(data, dtype=int).ravel()
    k = int(np.sqrt(M))
    if k * k != M:
        raise ValueError("M must be a perfect square for QAM.")
    bits_per_sym = int(np.log2(M))

    # Build constellation
    axis = np.arange(k) - (k - 1) / 2.0
    constellation = np.empty(M, dtype=complex)
    for i in range(M):
        idx = _gray_encode(i, bits_per_sym) if gray else i
        row = idx // k
        col = idx % k
        constellation[i] = axis[col] + 1j * axis[row]

    # Normalize average power to 1
    avg_power = np.mean(np.abs(constellation) ** 2)
    constellation /= np.sqrt(avg_power)

    return constellation[data]


def qamdemod(signal, M: int, gray: bool = True) -> ndarray:
    """QAM demodulation (hard decision, minimum distance)."""
    signal = np.asarray(signal, dtype=complex).ravel()
    k = int(np.sqrt(M))
    bits_per_sym = int(np.log2(M))

    axis = np.arange(k) - (k - 1) / 2.0
    constellation = np.empty(M, dtype=complex)
    for i in range(M):
        idx = _gray_encode(i, bits_per_sym) if gray else i
        row = idx // k
        col = idx % k
        constellation[i] = axis[col] + 1j * axis[row]
    avg_power = np.mean(np.abs(constellation) ** 2)
    constellation /= np.sqrt(avg_power)

    result = np.empty(len(signal), dtype=int)
    for i, s in enumerate(signal):
        dists = np.abs(constellation - s)
        result[i] = np.argmin(dists)
    return result


def pskmod(data, M: int, phase_offset: float = 0) -> ndarray:
    """PSK modulation.

    Parameters
    ----------
    data         : array of integers in [0, M-1]
    M            : number of phases (e.g. 2, 4, 8)
    phase_offset : initial phase offset in radians
    """
    data = np.asarray(data, dtype=int).ravel()
    phases = 2 * np.pi * data / M + phase_offset
    return np.exp(1j * phases)


def pskdemod(signal, M: int, phase_offset: float = 0) -> ndarray:
    """PSK demodulation (hard decision)."""
    signal = np.asarray(signal, dtype=complex).ravel()
    angles = np.angle(signal) - phase_offset
    angles = np.mod(angles, 2 * np.pi)
    return np.round(angles * M / (2 * np.pi)).astype(int) % M


def fskmod(data, M: int, freq_sep: float = 1.0,
           n_samples: int = 8, fs: float = 1.0) -> ndarray:
    """FSK modulation.

    Parameters
    ----------
    data      : array of integers in [0, M-1]
    M         : number of frequencies
    freq_sep  : frequency separation (Hz)
    n_samples : samples per symbol
    fs        : sample rate
    """
    data = np.asarray(data, dtype=int).ravel()
    n_sym = len(data)
    t = np.arange(n_samples) / fs
    signal = np.empty(n_sym * n_samples)
    for i, d in enumerate(data):
        freq = (d - (M - 1) / 2.0) * freq_sep
        signal[i * n_samples:(i + 1) * n_samples] = np.cos(2 * np.pi * freq * t)
    return signal


def fskdemod(signal, M: int, freq_sep: float = 1.0,
             n_samples: int = 8, fs: float = 1.0) -> ndarray:
    """FSK demodulation (correlation-based)."""
    signal = np.asarray(signal, dtype=float).ravel()
    n_sym = len(signal) // n_samples
    t = np.arange(n_samples) / fs
    result = np.empty(n_sym, dtype=int)
    for i in range(n_sym):
        seg = signal[i * n_samples:(i + 1) * n_samples]
        best_corr = -np.inf
        best_idx = 0
        for d in range(M):
            freq = (d - (M - 1) / 2.0) * freq_sep
            ref = np.cos(2 * np.pi * freq * t)
            corr = np.sum(seg * ref)
            if corr > best_corr:
                best_corr = corr
                best_idx = d
        result[i] = best_idx
    return result


# ---------------------------------------------------------------------------
# Channel Models
# ---------------------------------------------------------------------------

def awgn(signal, snr_db: Numeric, signal_power: Optional[Numeric] = None) -> ndarray:
    """Add white Gaussian noise to a signal.

    Parameters
    ----------
    signal       : input signal (real or complex)
    snr_db       : signal-to-noise ratio in dB
    signal_power : measured/assumed signal power (default: computed from signal)
    """
    sig = np.asarray(signal)
    is_complex = np.iscomplexobj(sig)
    if signal_power is None:
        sp = np.mean(np.abs(sig) ** 2)
    else:
        sp = float(signal_power)
    snr_lin = 10 ** (float(snr_db) / 10)
    noise_power = sp / snr_lin
    rng = np.random.default_rng()
    if is_complex:
        noise = np.sqrt(noise_power / 2) * (
            rng.standard_normal(sig.shape) + 1j * rng.standard_normal(sig.shape)
        )
    else:
        noise = np.sqrt(noise_power) * rng.standard_normal(sig.shape)
    return sig + noise


def rayleighchan(signal, max_doppler: float = 0.01) -> Tuple[ndarray, ndarray]:
    """Simple flat Rayleigh fading channel.

    Returns (output_signal, channel_gains).
    """
    sig = np.asarray(signal)
    n = sig.shape[0] if sig.ndim > 0 else 1
    rng = np.random.default_rng()
    h = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    # Apply simple time-correlation via low-pass filter
    if max_doppler > 0 and n > 1:
        alpha = np.exp(-2 * np.pi * max_doppler)
        for i in range(1, n):
            h[i] = alpha * h[i - 1] + np.sqrt(1 - alpha ** 2) * h[i]
    out = sig * h
    return out, h


def ricianchan(signal, K_factor: Numeric = 3.0,
               max_doppler: float = 0.01) -> Tuple[ndarray, ndarray]:
    """Simple flat Rician fading channel.

    Parameters
    ----------
    K_factor : Rician K-factor (ratio of LOS to scattered power, linear)
    """
    sig = np.asarray(signal)
    n = sig.shape[0] if sig.ndim > 0 else 1
    K = float(K_factor)
    rng = np.random.default_rng()

    # LOS component
    los = np.sqrt(K / (K + 1))
    # Scattered component
    scatter_std = np.sqrt(1 / (2 * (K + 1)))
    h_scatter = scatter_std * (rng.standard_normal(n) + 1j * rng.standard_normal(n))

    if max_doppler > 0 and n > 1:
        alpha = np.exp(-2 * np.pi * max_doppler)
        for i in range(1, n):
            h_scatter[i] = alpha * h_scatter[i - 1] + np.sqrt(1 - alpha ** 2) * h_scatter[i]

    h = los + h_scatter
    return sig * h, h


# ---------------------------------------------------------------------------
# Convolutional Coding / Viterbi Decoding
# ---------------------------------------------------------------------------

def convenc(data, trellis: Dict[str, Any]) -> ndarray:
    """Convolutional encoder.

    Parameters
    ----------
    data    : binary array (0/1)
    trellis : dict with keys:
              'n_states'     : number of states
              'n_inputs'     : number of input bits (typically 1)
              'n_outputs'    : number of output bits per input
              'next_states'  : array [n_states, n_inputs] -> next state
              'outputs'      : array [n_states, n_inputs] -> output (decimal)

    Returns
    -------
    encoded binary array
    """
    data = np.asarray(data, dtype=int).ravel()
    ns = trellis['next_states']
    outs = trellis['outputs']
    n_out = trellis['n_outputs']

    state = 0
    encoded = []
    for bit in data:
        out_val = outs[state, bit]
        # Convert to binary
        for i in range(n_out - 1, -1, -1):
            encoded.append((out_val >> i) & 1)
        state = ns[state, bit]
    return np.array(encoded, dtype=int)


def vitdec(data, trellis: Dict[str, Any], tblen: int = 30) -> ndarray:
    """Viterbi decoder (hard decision).

    Parameters
    ----------
    data   : received binary array
    trellis: same structure as convenc
    tblen  : traceback length (default 30)
    """
    data = np.asarray(data, dtype=int).ravel()
    n_out = trellis['n_outputs']
    n_states = trellis['n_states']
    n_inputs = trellis['n_inputs']
    ns = trellis['next_states']
    outs = trellis['outputs']

    n_syms = len(data) // n_out

    INF = 1e9
    path_metric = np.full(n_states, INF)
    path_metric[0] = 0
    survivor = np.zeros((n_syms, n_states), dtype=int)
    decisions = np.zeros((n_syms, n_states), dtype=int)

    for t in range(n_syms):
        rx_bits = data[t * n_out:(t + 1) * n_out]
        new_metric = np.full(n_states, INF)

        for s in range(n_states):
            if path_metric[s] >= INF:
                continue
            for inp in range(n_inputs):
                next_s = ns[s, inp]
                out_val = outs[s, inp]
                # Compute Hamming distance
                expected = np.array([(out_val >> i) & 1
                                     for i in range(n_out - 1, -1, -1)])
                dist = np.sum(rx_bits != expected)
                candidate = path_metric[s] + dist
                if candidate < new_metric[next_s]:
                    new_metric[next_s] = candidate
                    survivor[t, next_s] = s
                    decisions[t, next_s] = inp

        path_metric = new_metric

    # Traceback
    state = np.argmin(path_metric)
    decoded = np.empty(n_syms, dtype=int)
    for t in range(n_syms - 1, -1, -1):
        decoded[t] = decisions[t, state]
        state = survivor[t, state]

    return decoded


# ---------------------------------------------------------------------------
# Error Metrics
# ---------------------------------------------------------------------------

def biterr(x, y) -> Tuple[int, float, int]:
    """Bit error count and rate.

    Returns (n_errors, ber, n_bits).
    """
    a = np.asarray(x, dtype=int).ravel()
    b = np.asarray(y, dtype=int).ravel()
    n = min(len(a), len(b))
    errors = int(np.sum(a[:n] != b[:n]))
    return errors, errors / n if n > 0 else 0.0, n


def symerr(x, y) -> Tuple[int, float, int]:
    """Symbol error count and rate.

    Returns (n_errors, ser, n_symbols).
    """
    a = np.asarray(x).ravel()
    b = np.asarray(y).ravel()
    n = min(len(a), len(b))
    errors = int(np.sum(a[:n] != b[:n]))
    return errors, errors / n if n > 0 else 0.0, n


def berawgn(snr_db, modtype: str = 'psk', M: int = 2) -> float:
    """Theoretical BER for AWGN channel.

    Parameters
    ----------
    snr_db  : Eb/N0 in dB
    modtype : 'psk' or 'qam'
    M       : constellation size
    """
    from scipy.special import erfc
    snr_lin = 10 ** (float(snr_db) / 10)
    k = int(np.log2(M))

    if modtype.lower() == 'psk':
        if M == 2:
            return 0.5 * erfc(np.sqrt(snr_lin))
        else:
            return (1 / k) * erfc(np.sqrt(k * snr_lin) * np.sin(np.pi / M))
    elif modtype.lower() == 'qam':
        if M == 4:
            return 0.5 * erfc(np.sqrt(snr_lin))
        else:
            sqrtM = int(np.sqrt(M))
            return (2 * (sqrtM - 1) / (sqrtM * k)) * erfc(
                np.sqrt(3 * k * snr_lin / (2 * (M - 1)))
            )
    else:
        raise ValueError(f"Unknown modtype: {modtype}")


def eyediagram(signal, n_samples_per_sym: int, offset: int = 0) -> ndarray:
    """Prepare data for eye diagram (returns 2-D array of overlapped traces).

    Each row is one symbol period of samples.
    """
    sig = np.asarray(signal, dtype=float).ravel()
    total = len(sig) - offset
    n_traces = total // n_samples_per_sym
    trimmed = sig[offset:offset + n_traces * n_samples_per_sym]
    return trimmed.reshape(n_traces, n_samples_per_sym)


def scatterplot(signal) -> Tuple[ndarray, ndarray]:
    """Extract I/Q data for scatter plot.

    Returns (real_parts, imag_parts).
    """
    sig = np.asarray(signal, dtype=complex).ravel()
    return sig.real.copy(), sig.imag.copy()


# ---------------------------------------------------------------------------
# Source Coding (Huffman)
# ---------------------------------------------------------------------------

def huffmandict(symbols, probabilities) -> Dict[Any, str]:
    """Build a Huffman dictionary.

    Parameters
    ----------
    symbols       : list of symbols
    probabilities : list of probabilities (must sum to ~1)

    Returns
    -------
    dict mapping symbol -> binary string code
    """
    import heapq

    syms = list(symbols)
    probs = list(probabilities)

    if len(syms) == 1:
        return {syms[0]: '0'}

    # Build tree using heap
    heap = []
    for i, (s, p) in enumerate(zip(syms, probs)):
        heapq.heappush(heap, (p, i, s))

    counter = len(syms)
    nodes = {}  # id -> (left_id, right_id) or symbol

    for i, s in enumerate(syms):
        nodes[i] = s

    while len(heap) > 1:
        p1, id1, _ = heapq.heappop(heap)
        p2, id2, _ = heapq.heappop(heap)
        new_id = counter
        counter += 1
        nodes[new_id] = (id1, id2)
        heapq.heappush(heap, (p1 + p2, new_id, None))

    # Traverse tree to get codes
    _, root_id, _ = heap[0]
    codes: Dict[Any, str] = {}

    def _traverse(node_id, prefix):
        node = nodes[node_id]
        if isinstance(node, tuple):
            left_id, right_id = node
            _traverse(left_id, prefix + '0')
            _traverse(right_id, prefix + '1')
        else:
            codes[node] = prefix if prefix else '0'

    _traverse(root_id, '')
    return codes


def huffmanenco(data, hdict: Dict[Any, str]) -> ndarray:
    """Huffman encode data using a dictionary.

    Returns binary array (0/1).
    """
    bits = []
    for sym in data:
        code = hdict[sym]
        bits.extend(int(b) for b in code)
    return np.array(bits, dtype=int)


def huffmandeco(encoded, hdict: Dict[Any, str]):
    """Huffman decode binary data using a dictionary.

    Returns list of decoded symbols.
    """
    # Build reverse lookup trie
    inv = {v: k for k, v in hdict.items()}
    encoded = np.asarray(encoded, dtype=int).ravel()

    decoded = []
    current = ''
    for bit in encoded:
        current += str(bit)
        if current in inv:
            decoded.append(inv[current])
            current = ''
    return decoded


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

COMMS_REGISTRY: Dict[str, Any] = {
    'qammod': qammod,
    'qamdemod': qamdemod,
    'pskmod': pskmod,
    'pskdemod': pskdemod,
    'fskmod': fskmod,
    'fskdemod': fskdemod,
    'awgn': awgn,
    'rayleighchan': rayleighchan,
    'ricianchan': ricianchan,
    'convenc': convenc,
    'vitdec': vitdec,
    'biterr': biterr,
    'symerr': symerr,
    'berawgn': berawgn,
    'eyediagram': eyediagram,
    'scatterplot': scatterplot,
    'huffmandict': huffmandict,
    'huffmanenco': huffmanenco,
    'huffmandeco': huffmandeco,
}
