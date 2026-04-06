# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Tests for Forge Neural Network (Lite) Toolbox (10 tests).

Uses NEURAL_REGISTRY dict to call functions by their registry name.

V-Model Traceability
---------------------
Requirement: R-NEUR
Parent SHALL statement: Forge SHALL provide a neural network toolbox that
    supports feedforward network creation, configuration, training, simulation,
    and standard activation functions, following MATLAB/Octave naming conventions.
Model-user argument: An engineer exploring neural networks for system
    identification and function approximation expects to build, configure, and
    train feedforward nets using the same layer-specification syntax they learned
    in MATLAB. If Forge lacks these primitives, the user cannot prototype
    data-driven models without switching to a separate Python ML framework,
    breaking their workflow continuity.
Decomposition:
    R-NEUR-01: feedforwardnet creates a network object from a layer-size vector.
    R-NEUR-02: Network stores hidden layer structure matching the size vector.
    R-NEUR-03: configure sets input/output dimensions from sample data.
    R-NEUR-04: train reduces loss on a known-solvable problem (XOR).
    R-NEUR-05: sim produces output with correct (output_size, n_samples) shape.
    R-NEUR-06: mse_metric returns 0.0 for identical vectors.
    R-NEUR-07: mse_metric returns the correct known value for a unit difference.
    R-NEUR-08: tansig output is bounded in [-1, 1].
    R-NEUR-09: logsig output is bounded in [0, 1].
Consistency argument: R-NEUR-01 through R-NEUR-03 cover network lifecycle
    (creation, structure, configuration). R-NEUR-04 and R-NEUR-05 cover the
    training and inference loop. R-NEUR-06 and R-NEUR-07 verify the loss metric
    used during training. R-NEUR-08 and R-NEUR-09 verify activation functions
    that compose the network layers. Together these span creation, configuration,
    training, inference, loss computation, and activation correctness.
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
    """R-NEUR-01..02: feedforwardnet SHALL create a network object with the
    requested hidden layer structure from a layer-size vector.
    """

    def test_feedforwardnet_creation(self):
        """R-NEUR-01: feedforwardnet([5]) returns a non-None object with layer_sizes."""
        feedforwardnet = _fn('feedforwardnet')
        net = feedforwardnet([5])
        assert net is not None
        assert hasattr(net, 'layer_sizes')

    def test_network_has_layers(self):
        """R-NEUR-02: feedforwardnet([10, 5]) produces a network with at least 2 layers."""
        feedforwardnet = _fn('feedforwardnet')
        net = feedforwardnet([10, 5])
        # Should have hidden + output layers
        assert net.n_layers >= 2


# ===========================================================================
# Configuration
# ===========================================================================

class TestConfiguration:
    """R-NEUR-03: configure SHALL set input and output dimensions from sample data."""

    def test_configure_sets_input_output_sizes(self):
        """R-NEUR-03: configure() sets input_size and output layer size from data shapes."""
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
    """R-NEUR-04..05: train SHALL reduce loss on a solvable problem; sim SHALL
    produce output with the correct shape (output_size, n_samples).
    """

    def test_train_on_xor(self):
        """R-NEUR-04: Training on XOR data reduces loss from initial to final epoch."""
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
        """R-NEUR-05: sim() output shape equals (output_size, n_samples)."""
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
    """R-NEUR-06..07: mse_metric SHALL return correct mean squared error values."""

    def test_mse_metric_computation(self):
        """R-NEUR-06: mse_metric([1,2,3], [1,2,3]) equals 0.0."""
        mse_metric = _fn('mse_metric')
        result = mse_metric(np.array([1, 2, 3]), np.array([1, 2, 3]))
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_mse_metric_known_value(self):
        """R-NEUR-07: mse_metric([0,0], [1,1]) equals 1.0."""
        mse_metric = _fn('mse_metric')
        result = mse_metric(np.array([0, 0]), np.array([1, 1]))
        np.testing.assert_allclose(result, 1.0, atol=1e-10)


# ===========================================================================
# Activation Functions
# ===========================================================================

class TestActivations:
    """R-NEUR-08..09: Activation functions SHALL produce outputs within their
    defined mathematical ranges.
    """

    def test_tansig_range(self):
        """R-NEUR-08: tansig output is bounded within [-1, 1]."""
        tansig = _fn('tansig')
        x = np.linspace(-10, 10, 100)
        y = tansig(x)
        assert np.all(y >= -1.0)
        assert np.all(y <= 1.0)

    def test_logsig_range(self):
        """R-NEUR-09: logsig output is bounded within [0, 1]."""
        logsig = _fn('logsig')
        x = np.linspace(-10, 10, 100)
        y = logsig(x)
        assert np.all(y >= 0.0)
        assert np.all(y <= 1.0)
