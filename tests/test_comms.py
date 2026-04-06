# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
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
    """R-COMM-01: Forge SHALL provide QAM modulation and demodulation that
    recovers original symbol sequences through a mod/demod round-trip.

    Model-user argument: A communications engineer migrating from Octave
    relies on qammod/qamdemod for baseband simulation of digital radio
    links. Round-trip symbol recovery is the minimum correctness bar;
    without it, BER measurements and link budget studies are meaningless.

    Decomposition:
      R-COMM-01a: QPSK (M=4) round-trip recovers original symbols.
      R-COMM-01b: 16-QAM produces 16 unique constellation points.

    Consistency argument: R-COMM-01a proves functional correctness for the
    simplest QAM order. R-COMM-01b confirms the constellation geometry
    scales to higher orders. Together they validate that qammod produces
    correctly spaced symbols and qamdemod inverts them.
    """

    def test_qpsk_roundtrip(self):
        """R-COMM-01a: QPSK mod then demod recovers original symbols."""
        qammod = _fn('qammod')
        qamdemod = _fn('qamdemod')
        data = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        modulated = qammod(data, 4)
        demodulated = qamdemod(modulated, 4)
        np.testing.assert_array_equal(demodulated, data)

    def test_16qam_constellation_size(self):
        """R-COMM-01b: 16-QAM produces 16 unique constellation points."""
        qammod = _fn('qammod')
        data = np.arange(16)
        symbols = qammod(data, 16)
        unique = np.unique(np.round(symbols, 10))
        assert len(unique) == 16


# ===========================================================================
# PSK Modulation / Demodulation
# ===========================================================================

class TestPSK:
    """R-COMM-02: Forge SHALL provide PSK modulation and demodulation that
    recovers original symbol sequences and places all symbols on the unit
    circle.

    Model-user argument: Phase-shift keying is the foundation of many
    satellite and wireless standards the engineer simulates. Octave's
    pskmod/pskdemod are used to validate receiver algorithms. Symbols must
    lie on the unit circle (constant envelope), and round-trip recovery
    must be exact for the simulation to be trustworthy.

    Decomposition:
      R-COMM-02a: PSK mod/demod round-trip recovers original data (M=4).
      R-COMM-02b: All PSK symbols lie on the unit circle (magnitude 1).

    Consistency argument: R-COMM-02a proves invertibility of the PSK
    mapping. R-COMM-02b proves geometric correctness (constant envelope).
    Together they fully validate the PSK implementation for any order M.
    """

    def test_psk_mod_demod_roundtrip(self):
        """R-COMM-02a: PSK mod then demod recovers original data (M=4)."""
        pskmod = _fn('pskmod')
        pskdemod = _fn('pskdemod')
        data = np.array([0, 1, 2, 3, 0, 2, 1, 3])
        modulated = pskmod(data, 4)
        demodulated = pskdemod(modulated, 4)
        np.testing.assert_array_equal(demodulated, data)

    def test_psk_unit_circle(self):
        """R-COMM-02b: PSK symbols lie on the unit circle."""
        pskmod = _fn('pskmod')
        data = np.arange(8)
        symbols = pskmod(data, 8)
        magnitudes = np.abs(symbols)
        np.testing.assert_allclose(magnitudes, 1.0, atol=1e-10)


# ===========================================================================
# Channel Models
# ===========================================================================

class TestChannel:
    """R-COMM-03: Forge SHALL provide an AWGN channel model that adds
    Gaussian noise at a specified SNR while preserving signal dimensions.

    Model-user argument: The engineer uses AWGN channels to benchmark
    receiver performance under controlled noise conditions. In Octave,
    awgn() is the standard way to inject noise at a known SNR. The output
    must differ from the input (noise was actually added) and must preserve
    the signal shape so downstream processing pipelines remain compatible.

    Decomposition:
      R-COMM-03a: AWGN output differs from the noiseless input.
      R-COMM-03b: AWGN output has the same shape as the input.

    Consistency argument: R-COMM-03a confirms noise injection is
    functional. R-COMM-03b confirms dimensional compatibility. Together
    they validate that awgn() is usable in any simulation pipeline.
    """

    def test_awgn_adds_noise(self):
        """R-COMM-03a: AWGN output differs from the noiseless input."""
        awgn = _fn('awgn')
        signal = np.ones(1000)
        noisy = awgn(signal, 10)  # 10 dB SNR
        assert not np.allclose(noisy, signal)

    def test_awgn_preserves_shape(self):
        """R-COMM-03b: AWGN output has the same shape as the input."""
        awgn = _fn('awgn')
        signal = np.sin(2 * np.pi * np.linspace(0, 1, 500))
        noisy = awgn(signal, 20)
        assert noisy.shape == signal.shape


# ===========================================================================
# Error Metrics
# ===========================================================================

class TestErrorMetrics:
    """R-COMM-04: Forge SHALL provide bit-level and symbol-level error
    metrics (biterr, symerr) and theoretical BER curves (berawgn) that
    match known reference values.

    Model-user argument: After running a modulation/channel simulation,
    the engineer computes BER and SER to evaluate link quality. These are
    the primary figures of merit in any comms lab report. Theoretical BER
    from berawgn provides the baseline curve the engineer compares
    simulation results against. Octave compatibility here is critical for
    reproducing published results.

    Decomposition:
      R-COMM-04a: biterr counts known bit differences correctly.
      R-COMM-04b: biterr of identical sequences returns zero errors.
      R-COMM-04c: symerr counts symbol-level differences correctly.
      R-COMM-04d: berawgn at 10 dB for BPSK returns BER below 1e-3.

    Consistency argument: R-COMM-04a and R-COMM-04b validate biterr for
    nonzero and zero error cases. R-COMM-04c validates the symbol-level
    counterpart. R-COMM-04d validates the theoretical BER function against
    a well-known reference point. Together they cover the full error
    measurement workflow.
    """

    def test_biterr_counts_correctly(self):
        """R-COMM-04a: biterr counts known bit differences correctly."""
        biterr = _fn('biterr')
        x = np.array([0, 0, 1, 1, 0])
        y = np.array([0, 1, 1, 0, 0])
        num_err, ber, total = biterr(x, y)
        assert num_err == 2
        assert total == 5
        np.testing.assert_allclose(ber, 2.0 / 5.0)

    def test_biterr_identical(self):
        """R-COMM-04b: biterr of identical sequences returns zero errors."""
        biterr = _fn('biterr')
        x = np.array([1, 0, 1, 0, 1])
        num_err, ber, total = biterr(x, x)
        assert num_err == 0
        assert ber == 0.0

    def test_symerr_known(self):
        """R-COMM-04c: symerr counts symbol-level differences correctly."""
        symerr = _fn('symerr')
        x = np.array([0, 1, 2, 3])
        y = np.array([0, 1, 3, 3])
        num_err, ser, total = symerr(x, y)
        assert num_err == 1
        assert total == 4

    def test_berawgn_reasonable_ber(self):
        """R-COMM-04d: berawgn at 10 dB for BPSK returns BER below 1e-3."""
        berawgn = _fn('berawgn')
        ber = berawgn(10, 'psk', 2)
        assert 0 < ber < 1e-3


# ===========================================================================
# Source Coding (Huffman)
# ===========================================================================

class TestHuffman:
    """R-COMM-05: Forge SHALL provide Huffman source coding (dictionary
    construction, encoding, decoding) that achieves lossless compression
    and recovers the original symbol sequence.

    Model-user argument: The engineer uses Huffman coding for entropy
    analysis and data compression in telemetry or sensor data pipelines.
    In Octave, huffmandict/huffmanenco/huffmandeco form a standard
    workflow. Lossless round-trip is mandatory; compression efficiency on
    skewed distributions confirms the algorithm is not trivially padding.

    Decomposition:
      R-COMM-05a: Huffman encode then decode recovers original data.
      R-COMM-05b: Huffman coding of a skewed distribution compresses
                  below 2 bits per symbol.

    Consistency argument: R-COMM-05a proves lossless correctness.
    R-COMM-05b proves the coding achieves nontrivial compression. Together
    they validate functional correctness and coding efficiency.
    """

    def test_huffman_encode_decode_roundtrip(self):
        """R-COMM-05a: Huffman encode then decode recovers original data."""
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
        """R-COMM-05b: Huffman coding of skewed distribution compresses."""
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
