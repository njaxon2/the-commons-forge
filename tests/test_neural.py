# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Tests for Forge Neural Network (Lite) Toolbox (10 tests).

Uses NEURAL_REGISTRY dict to call functions by their registry name.
"""

import pytest
import numpy as np

from forge.engine.builtins.neural import NEURAL_REGISTRY


def _fn(name):
    return NEURAL_REGISTRY[name]


# ===========================================================================
# Network Creation
# ===========================================================================

class TestNetworkCreation:
    def test_feedforwardnet_creation(self):
        """feedforwardnet([5]) should create a network object."""
        feedforwardnet = _fn('feedforwardnet')
        net = feedforwardnet([5])
        assert net is not None
        assert hasattr(net, 'layer_sizes')

    def test_network_has_layers(self):
        """Network created with [10, 5] should have hidden layers."""
        feedforwardnet = _fn('feedforwardnet')
        net = feedforwardnet([10, 5])
        # Should have hidden + output layers
        assert net.n_layers >= 2


# ===========================================================================
# Configuration
# ===========================================================================

class TestConfiguration:
    def test_configure_sets_input_output_sizes(self):
        """configure() should set input and output dimensions from data."""
        feedforwardnet = _fn('feedforwardnet')
        configure = _fn('configure')
        net = feedforwardnet([5])
        X = np.random.randn(2, 50)  # 2 inputs, 50 samples
        T = np.random.randn(1, 50)  # 1 output
        net = configure(net, X, T)
        assert net.input_size == 2 or net.is_configured
        assert net.layer_sizes[-1] == 1


# ===========================================================================
# Training & Simulation
# ===========================================================================

class TestTrainingSim:
    def test_train_on_xor(self):
        """Train a small net on XOR -- loss should decrease."""
        feedforwardnet = _fn('feedforwardnet')
        configure = _fn('configure')
        train = _fn('train')
        sim = _fn('sim')
        # XOR data: 2 inputs, 1 output
        X = np.array([[0, 0, 1, 1],
                       [0, 1, 0, 1]], dtype=float)
        T = np.array([[0, 1, 1, 0]], dtype=float)
        net = feedforwardnet([4])
        net = configure(net, X, T)
        np.random.seed(42)
        net, info = train(net, X, T)
        # Loss should have decreased from initial
        assert info['perf'][-1] < info['perf'][0]

    def test_sim_forward_pass_shape(self):
        """sim() output should have shape (output_size, n_samples)."""
        feedforwardnet = _fn('feedforwardnet')
        configure = _fn('configure')
        init_net = _fn('init_net')
        sim = _fn('sim')
        X = np.random.randn(3, 20)  # 3 inputs, 20 samples
        T = np.random.randn(2, 20)  # 2 outputs
        net = feedforwardnet([5])
        net = configure(net, X, T)
        net = init_net(net)
        Y = sim(net, X)
        assert np.asarray(Y).shape == (2, 20)


# ===========================================================================
# Metrics
# ===========================================================================

class TestMetrics:
    def test_mse_metric_computation(self):
        """mse_metric([1,2,3], [1,2,3]) should be 0."""
        mse_metric = _fn('mse_metric')
        result = mse_metric(np.array([1, 2, 3]), np.array([1, 2, 3]))
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_mse_metric_known_value(self):
        """mse_metric([0,0], [1,1]) should be 1.0."""
        mse_metric = _fn('mse_metric')
        result = mse_metric(np.array([0, 0]), np.array([1, 1]))
        np.testing.assert_allclose(result, 1.0, atol=1e-10)


# ===========================================================================
# Activation Functions
# ===========================================================================

class TestActivations:
    def test_tansig_range(self):
        """tansig output should be in [-1, 1]."""
        tansig = _fn('tansig')
        x = np.linspace(-10, 10, 100)
        y = tansig(x)
        assert np.all(y >= -1.0)
        assert np.all(y <= 1.0)

    def test_logsig_range(self):
        """logsig output should be in [0, 1]."""
        logsig = _fn('logsig')
        x = np.linspace(-10, 10, 100)
        y = logsig(x)
        assert np.all(y >= 0.0)
        assert np.all(y <= 1.0)
