"""Tests for Forge Communications Toolbox (15 tests).

Uses COMMS_REGISTRY dict to call functions by their registry name.
"""

import pytest
import numpy as np

from forge.engine.builtins.comms import COMMS_REGISTRY


def _fn(name):
    return COMMS_REGISTRY[name]


# ===========================================================================
# QAM Modulation / Demodulation
# ===========================================================================

class TestQAM:
    def test_qpsk_roundtrip(self):
        """QPSK (4-QAM) mod then demod should recover original symbols."""
        qammod = _fn('qammod')
        qamdemod = _fn('qamdemod')
        data = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        modulated = qammod(data, 4)
        demodulated = qamdemod(modulated, 4)
        np.testing.assert_array_equal(demodulated, data)

    def test_16qam_constellation_size(self):
        """16-QAM should have 16 unique constellation points."""
        qammod = _fn('qammod')
        data = np.arange(16)
        symbols = qammod(data, 16)
        unique = np.unique(np.round(symbols, 10))
        assert len(unique) == 16


# ===========================================================================
# PSK Modulation / Demodulation
# ===========================================================================

class TestPSK:
    def test_psk_mod_demod_roundtrip(self):
        """PSK mod then demod should recover original data (M=4)."""
        pskmod = _fn('pskmod')
        pskdemod = _fn('pskdemod')
        data = np.array([0, 1, 2, 3, 0, 2, 1, 3])
        modulated = pskmod(data, 4)
        demodulated = pskdemod(modulated, 4)
        np.testing.assert_array_equal(demodulated, data)

    def test_psk_unit_circle(self):
        """PSK symbols should lie on the unit circle."""
        pskmod = _fn('pskmod')
        data = np.arange(8)
        symbols = pskmod(data, 8)
        magnitudes = np.abs(symbols)
        np.testing.assert_allclose(magnitudes, 1.0, atol=1e-10)


# ===========================================================================
# Channel Models
# ===========================================================================

class TestChannel:
    def test_awgn_adds_noise(self):
        """AWGN output should differ from input."""
        awgn = _fn('awgn')
        signal = np.ones(1000)
        noisy = awgn(signal, 10)  # 10 dB SNR
        assert not np.allclose(noisy, signal)

    def test_awgn_preserves_shape(self):
        """AWGN output should have same shape as input."""
        awgn = _fn('awgn')
        signal = np.sin(2 * np.pi * np.linspace(0, 1, 500))
        noisy = awgn(signal, 20)
        assert noisy.shape == signal.shape


# ===========================================================================
# Error Metrics
# ===========================================================================

class TestErrorMetrics:
    def test_biterr_counts_correctly(self):
        """biterr should count known bit differences."""
        biterr = _fn('biterr')
        x = np.array([0, 0, 1, 1, 0])
        y = np.array([0, 1, 1, 0, 0])
        num_err, ber, total = biterr(x, y)
        assert num_err == 2
        assert total == 5
        np.testing.assert_allclose(ber, 2.0 / 5.0)

    def test_biterr_identical(self):
        """biterr of identical sequences should be 0."""
        biterr = _fn('biterr')
        x = np.array([1, 0, 1, 0, 1])
        num_err, ber, total = biterr(x, x)
        assert num_err == 0
        assert ber == 0.0

    def test_symerr_known(self):
        """symerr should count symbol-level differences."""
        symerr = _fn('symerr')
        x = np.array([0, 1, 2, 3])
        y = np.array([0, 1, 3, 3])
        num_err, ser, total = symerr(x, y)
        assert num_err == 1
        assert total == 4

    def test_berawgn_reasonable_ber(self):
        """berawgn at 10 dB for BPSK should give BER < 1e-3."""
        berawgn = _fn('berawgn')
        ber = berawgn(10, 'psk', 2)
        assert 0 < ber < 1e-3


# ===========================================================================
# Source Coding (Huffman)
# ===========================================================================

class TestHuffman:
    def test_huffman_encode_decode_roundtrip(self):
        """Huffman encode then decode should recover original data."""
        huffmandict = _fn('huffmandict')
        huffmanenco = _fn('huffmanenco')
        huffmandeco = _fn('huffmandeco')
        symbols = [0, 1, 2, 3]
        probs = [0.5, 0.25, 0.15, 0.10]
        hdict = huffmandict(symbols, probs)
        data = [0, 0, 1, 2, 3, 0, 1, 0]
        encoded = huffmanenco(data, hdict)
        decoded = huffmandeco(encoded, hdict)
        np.testing.assert_array_equal(np.asarray(decoded), np.asarray(data))

    def test_huffman_compression(self):
        """Huffman coding of skewed distribution should compress."""
        huffmandict = _fn('huffmandict')
        huffmanenco = _fn('huffmanenco')
        symbols = [0, 1, 2, 3]
        probs = [0.7, 0.15, 0.10, 0.05]
        hdict = huffmandict(symbols, probs)
        # 100 symbols with mostly 0s
        data = [0] * 70 + [1] * 15 + [2] * 10 + [3] * 5
        encoded = huffmanenco(data, hdict)
        # Compressed bits should be fewer than 2 bits/symbol * 100
        assert len(encoded) < 200
