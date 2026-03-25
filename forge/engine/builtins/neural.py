"""Neural Network (Lite) Toolbox for Forge.

Simple feedforward neural networks using only NumPy for training and
inference. Provides network construction, backpropagation training (SGD),
forward pass, and performance metrics.

Target location: forge/engine/builtins/neural.py

Backend: NumPy only (no PyTorch, TensorFlow, or other ML framework dependency).
"""

from __future__ import annotations

import numpy as np
from typing import Any


# ── Toolbox function registry ────────────────────────────────────
_FUNCTIONS: dict[str, callable] = {}


def _tb(name: str | None = None):
    """Local decorator to register a toolbox function."""
    def decorator(func):
        fn_name = name or func.__name__
        _FUNCTIONS[fn_name] = func
        return func
    return decorator


# =====================================================================
# Activation Functions
# =====================================================================

@_tb('forge_tansig')
def forge_tansig(n: Any) -> np.ndarray:
    """Hyperbolic tangent sigmoid transfer function.

    tansig(n) = 2 / (1 + exp(-2*n)) - 1

    Equivalent to tanh(n) but named for MATLAB compatibility.

    Parameters
    ----------
    n : array_like
        Input values.

    Returns
    -------
    ndarray
        Output in (-1, 1).
    """
    return np.tanh(np.asarray(n, dtype=np.float64))


def _tansig_deriv(a: np.ndarray) -> np.ndarray:
    """Derivative of tansig given output a = tansig(n)."""
    return 1.0 - a ** 2


@_tb('forge_logsig')
def forge_logsig(n: Any) -> np.ndarray:
    """Log-sigmoid transfer function.

    logsig(n) = 1 / (1 + exp(-n))

    Parameters
    ----------
    n : array_like
        Input values.

    Returns
    -------
    ndarray
        Output in (0, 1).
    """
    n = np.asarray(n, dtype=np.float64)
    # Numerically stable sigmoid
    return np.where(n >= 0,
                    1.0 / (1.0 + np.exp(-n)),
                    np.exp(n) / (1.0 + np.exp(n)))


def _logsig_deriv(a: np.ndarray) -> np.ndarray:
    """Derivative of logsig given output a = logsig(n)."""
    return a * (1.0 - a)


@_tb('forge_purelin')
def forge_purelin(n: Any) -> np.ndarray:
    """Linear transfer function.

    purelin(n) = n

    Parameters
    ----------
    n : array_like
        Input values.

    Returns
    -------
    ndarray
        Same as input.
    """
    return np.asarray(n, dtype=np.float64)


def _purelin_deriv(a: np.ndarray) -> np.ndarray:
    """Derivative of purelin (always 1)."""
    return np.ones_like(a)


@_tb('forge_softmax')
def forge_softmax(n: Any) -> np.ndarray:
    """Softmax transfer function.

    Parameters
    ----------
    n : array_like
        Input values, shape (n_classes,) or (n_classes, n_samples).

    Returns
    -------
    ndarray
        Probability distribution (columns sum to 1).
    """
    n = np.asarray(n, dtype=np.float64)
    if n.ndim == 1:
        n = n.reshape(-1, 1)
    # Numerically stable softmax
    e = np.exp(n - np.max(n, axis=0, keepdims=True))
    return e / np.sum(e, axis=0, keepdims=True)


@_tb('forge_relu')
def forge_relu(n: Any) -> np.ndarray:
    """Rectified Linear Unit transfer function.

    Parameters
    ----------
    n : array_like
        Input values.

    Returns
    -------
    ndarray
        max(0, n).
    """
    return np.maximum(0, np.asarray(n, dtype=np.float64))


def _relu_deriv(a: np.ndarray) -> np.ndarray:
    """Derivative of ReLU given output a."""
    return np.where(a > 0, 1.0, 0.0)


# Activation function registry
_ACTIVATIONS = {
    'tansig': (forge_tansig, _tansig_deriv),
    'logsig': (forge_logsig, _logsig_deriv),
    'purelin': (forge_purelin, _purelin_deriv),
    'relu': (forge_relu, _relu_deriv),
}


# =====================================================================
# Loss / Performance Functions
# =====================================================================

@_tb('forge_mse_metric')
def forge_mse_metric(targets: Any, outputs: Any) -> float:
    """Mean squared error performance metric.

    Parameters
    ----------
    targets : array_like
        Target values.
    outputs : array_like
        Network output values.

    Returns
    -------
    float
        Mean squared error.
    """
    T = np.asarray(targets, dtype=np.float64)
    Y = np.asarray(outputs, dtype=np.float64)
    return float(np.mean((T - Y) ** 2))


@_tb('forge_crossentropy')
def forge_crossentropy(targets: Any, outputs: Any) -> float:
    """Cross-entropy performance metric.

    Parameters
    ----------
    targets : array_like
        Target values (one-hot or probabilities).
    outputs : array_like
        Network output values (probabilities).

    Returns
    -------
    float
        Cross-entropy loss.
    """
    T = np.asarray(targets, dtype=np.float64)
    Y = np.asarray(outputs, dtype=np.float64)
    Y = np.clip(Y, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(T * np.log(Y) + (1.0 - T) * np.log(1.0 - Y)))


# =====================================================================
# Neural Network Data Structure
# =====================================================================

class ForgeNetwork:
    """Simple feedforward neural network.

    Stores layer sizes, weights, biases, activation functions, and
    training configuration.
    """

    def __init__(self, layer_sizes: list[int],
                 activations: list[str],
                 net_type: str = 'feedforward'):
        """
        Parameters
        ----------
        layer_sizes : list of int
            Number of neurons in each layer (input size is set during
            configure). E.g. [10, 5] means one hidden layer of 10 and
            output layer of 5.
        activations : list of str
            Activation function names for each layer.
        net_type : str
            'feedforward', 'pattern', or 'fitnet'.
        """
        self.layer_sizes = list(layer_sizes)
        self.activations = list(activations)
        self.net_type = net_type
        self.n_layers = len(layer_sizes)

        # Weights and biases (initialized during configure/init)
        self.weights: list[np.ndarray] = []  # W[l]: (n_l, n_{l-1})
        self.biases: list[np.ndarray] = []   # b[l]: (n_l, 1)

        # Training parameters
        self.lr = 0.01           # learning rate
        self.epochs = 1000       # max epochs
        self.goal = 0.0          # target performance
        self.min_grad = 1e-7     # minimum gradient
        self.momentum = 0.9      # momentum coefficient
        self.batch_size = None   # None = full batch
        self.loss_fn = 'mse'     # 'mse' or 'crossentropy'
        self.show_interval = 100 # display interval (0 = silent)

        # Training state
        self.is_configured = False
        self.input_size: int | None = None
        self.train_record: dict[str, list] = {'epoch': [], 'perf': []}

        # Input/output normalization
        self._input_offset: np.ndarray | None = None
        self._input_scale: np.ndarray | None = None
        self._output_offset: np.ndarray | None = None
        self._output_scale: np.ndarray | None = None

    def __repr__(self) -> str:
        if self.is_configured:
            arch = f"[{self.input_size}] -> " + " -> ".join(
                f"[{s}]" for s in self.layer_sizes)
        else:
            arch = " -> ".join(f"[{s}]" for s in self.layer_sizes)
        return f"ForgeNetwork({self.net_type}, {arch})"


# =====================================================================
# Network Construction
# =====================================================================

@_tb('forge_feedforwardnet')
def forge_feedforwardnet(layers: Any) -> ForgeNetwork:
    """Create a feedforward neural network.

    Parameters
    ----------
    layers : array_like
        Hidden layer sizes. E.g. [10] for one hidden layer of 10 neurons.
        The output layer is added during configuration.

    Returns
    -------
    ForgeNetwork
        Unconfigured network.
    """
    layers = np.asarray(layers, dtype=np.int64).ravel().tolist()
    # Default: tansig for hidden, purelin for output
    activations = ['tansig'] * len(layers) + ['purelin']
    # Output size added during configure; use placeholder
    return ForgeNetwork(layers + [1], activations, net_type='feedforward')


@_tb('forge_patternnet')
def forge_patternnet(layers: Any) -> ForgeNetwork:
    """Create a pattern recognition (classification) network.

    Parameters
    ----------
    layers : array_like
        Hidden layer sizes.

    Returns
    -------
    ForgeNetwork
        Network with softmax output and crossentropy loss.
    """
    layers = np.asarray(layers, dtype=np.int64).ravel().tolist()
    activations = ['tansig'] * len(layers) + ['logsig']
    net = ForgeNetwork(layers + [1], activations, net_type='pattern')
    net.loss_fn = 'crossentropy'
    return net


@_tb('forge_fitnet')
def forge_fitnet(layers: Any) -> ForgeNetwork:
    """Create a function fitting (regression) network.

    Parameters
    ----------
    layers : array_like
        Hidden layer sizes.

    Returns
    -------
    ForgeNetwork
        Network with linear output for regression.
    """
    layers = np.asarray(layers, dtype=np.int64).ravel().tolist()
    activations = ['tansig'] * len(layers) + ['purelin']
    net = ForgeNetwork(layers + [1], activations, net_type='fitnet')
    net.loss_fn = 'mse'
    return net


# =====================================================================
# Network Configuration & Initialization
# =====================================================================

@_tb('forge_configure')
def forge_configure(net: ForgeNetwork, X: Any,
                    T: Any) -> ForgeNetwork:
    """Auto-configure network from data dimensions.

    Sets input size, output size, and computes normalization parameters.

    Parameters
    ----------
    net : ForgeNetwork
        Network to configure.
    X : array_like
        Input data, shape (n_features, n_samples) or (n_samples,).
    T : array_like
        Target data, shape (n_outputs, n_samples) or (n_samples,).

    Returns
    -------
    ForgeNetwork
        Configured network (same object).
    """
    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    # Ensure column-major layout: (features, samples)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if T.ndim == 1:
        T = T.reshape(1, -1)

    n_inputs = X.shape[0]
    n_outputs = T.shape[0]

    net.input_size = n_inputs
    net.layer_sizes[-1] = n_outputs  # set output layer size

    # Compute normalization (map to [-1, 1])
    x_min = np.min(X, axis=1, keepdims=True)
    x_max = np.max(X, axis=1, keepdims=True)
    x_range = x_max - x_min
    x_range[x_range < 1e-15] = 1.0
    net._input_offset = -x_min
    net._input_scale = 2.0 / x_range

    t_min = np.min(T, axis=1, keepdims=True)
    t_max = np.max(T, axis=1, keepdims=True)
    t_range = t_max - t_min
    t_range[t_range < 1e-15] = 1.0
    net._output_offset = -t_min
    net._output_scale = 2.0 / t_range

    net.is_configured = True

    # Initialize weights
    forge_init_net(net)

    return net


@_tb('forge_init_net')
def forge_init_net(net: ForgeNetwork) -> ForgeNetwork:
    """Initialize (or reinitialize) network weights using Xavier/Glorot.

    Parameters
    ----------
    net : ForgeNetwork
        Network to initialize.

    Returns
    -------
    ForgeNetwork
        Network with freshly initialized weights.
    """
    net.weights = []
    net.biases = []

    prev_size = net.input_size if net.input_size else 1

    for l_idx in range(net.n_layers):
        n_neurons = net.layer_sizes[l_idx]
        # Xavier initialization
        limit = np.sqrt(6.0 / (prev_size + n_neurons))
        W = np.random.uniform(-limit, limit, (n_neurons, prev_size))
        b = np.zeros((n_neurons, 1))
        net.weights.append(W)
        net.biases.append(b)
        prev_size = n_neurons

    net.train_record = {'epoch': [], 'perf': []}
    return net


# =====================================================================
# Forward Pass
# =====================================================================

def _forward(net: ForgeNetwork, X: np.ndarray) -> tuple[list, list]:
    """Internal forward pass returning all layer activations.

    Parameters
    ----------
    net : ForgeNetwork
    X : ndarray, shape (n_inputs, n_samples)

    Returns
    -------
    (list of ndarray, list of ndarray)
        (pre_activations, post_activations) for each layer.
    """
    a = X  # input layer output
    pre_acts = []
    post_acts = [a]

    for l_idx in range(net.n_layers):
        n = net.weights[l_idx] @ a + net.biases[l_idx]
        pre_acts.append(n)

        act_name = net.activations[l_idx]
        act_fn, _ = _ACTIVATIONS.get(act_name, (forge_purelin, _purelin_deriv))
        a = act_fn(n)
        post_acts.append(a)

    return pre_acts, post_acts


@_tb('forge_sim')
def forge_sim(net: ForgeNetwork, X: Any) -> np.ndarray:
    """Simulate (forward pass) the network.

    Parameters
    ----------
    net : ForgeNetwork
        A configured and optionally trained network.
    X : array_like
        Input data, shape (n_features, n_samples) or (n_samples,).

    Returns
    -------
    ndarray
        Network output, shape (n_outputs, n_samples).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    # Normalize inputs
    if net._input_offset is not None:
        X = (X + net._input_offset) * net._input_scale - 1.0

    _, post_acts = _forward(net, X)
    Y = post_acts[-1]

    # Denormalize outputs
    if net._output_offset is not None:
        Y = (Y + 1.0) / net._output_scale - net._output_offset

    # Squeeze if single output / single sample
    if Y.shape[0] == 1:
        Y = Y.ravel()
    return Y


# =====================================================================
# Training (Backpropagation with SGD + Momentum)
# =====================================================================

@_tb('forge_train')
def forge_train(net: ForgeNetwork, X: Any,
                T: Any, **kwargs) -> tuple[ForgeNetwork, dict]:
    """Train the network using backpropagation with SGD + momentum.

    Parameters
    ----------
    net : ForgeNetwork
        Network to train.
    X : array_like
        Input data, shape (n_features, n_samples).
    T : array_like
        Target data, shape (n_outputs, n_samples).
    **kwargs
        Override training parameters: lr, epochs, goal, momentum,
        batch_size, show_interval.

    Returns
    -------
    (ForgeNetwork, dict)
        Trained network and training record {'epoch': [...], 'perf': [...]}.
    """
    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if T.ndim == 1:
        T = T.reshape(1, -1)

    # Auto-configure if needed
    if not net.is_configured:
        forge_configure(net, X, T)

    # Apply kwargs overrides
    for key in ('lr', 'epochs', 'goal', 'momentum', 'batch_size',
                'show_interval', 'min_grad'):
        if key in kwargs:
            setattr(net, key, kwargs[key])

    n_samples = X.shape[1]

    # Normalize data
    X_norm = X.copy()
    T_norm = T.copy()
    if net._input_offset is not None:
        X_norm = (X_norm + net._input_offset) * net._input_scale - 1.0
    if net._output_offset is not None:
        T_norm = (T_norm + net._output_offset) * net._output_scale - 1.0

    # Initialize momentum velocity
    vel_W = [np.zeros_like(W) for W in net.weights]
    vel_b = [np.zeros_like(b) for b in net.biases]

    net.train_record = {'epoch': [], 'perf': []}

    for epoch in range(1, net.epochs + 1):
        # Mini-batch or full batch
        if net.batch_size and net.batch_size < n_samples:
            indices = np.random.permutation(n_samples)
            batch_starts = range(0, n_samples, net.batch_size)
        else:
            indices = np.arange(n_samples)
            batch_starts = [0]
            net.batch_size = n_samples

        total_loss = 0.0
        n_batches = 0

        for start in batch_starts:
            end = min(start + (net.batch_size or n_samples), n_samples)
            batch_idx = indices[start:end]
            X_batch = X_norm[:, batch_idx]
            T_batch = T_norm[:, batch_idx]
            bs = X_batch.shape[1]

            # Forward pass
            pre_acts, post_acts = _forward(net, X_batch)
            Y = post_acts[-1]

            # Compute loss
            if net.loss_fn == 'crossentropy':
                Y_clip = np.clip(Y, 1e-15, 1.0 - 1e-15)
                loss = -np.mean(T_batch * np.log(Y_clip) +
                                (1.0 - T_batch) * np.log(1.0 - Y_clip))
            else:
                loss = np.mean((T_batch - Y) ** 2)
            total_loss += loss
            n_batches += 1

            # Backward pass
            # Output layer delta
            act_name = net.activations[-1]
            _, deriv_fn = _ACTIVATIONS.get(act_name,
                                            (forge_purelin, _purelin_deriv))
            if net.loss_fn == 'crossentropy' and act_name == 'logsig':
                # Simplified gradient for cross-entropy + sigmoid
                delta = Y - T_batch
            else:
                error = Y - T_batch
                delta = error * deriv_fn(Y)

            # Backpropagate through layers
            deltas = [None] * net.n_layers
            deltas[-1] = delta

            for l_idx in range(net.n_layers - 2, -1, -1):
                act_name = net.activations[l_idx]
                _, deriv_fn = _ACTIVATIONS.get(act_name,
                                                (forge_purelin, _purelin_deriv))
                a = post_acts[l_idx + 1]
                delta_prop = net.weights[l_idx + 1].T @ deltas[l_idx + 1]
                deltas[l_idx] = delta_prop * deriv_fn(a)

            # Update weights with momentum
            for l_idx in range(net.n_layers):
                a_prev = post_acts[l_idx]
                grad_W = (deltas[l_idx] @ a_prev.T) / bs
                grad_b = np.mean(deltas[l_idx], axis=1, keepdims=True)

                vel_W[l_idx] = net.momentum * vel_W[l_idx] - net.lr * grad_W
                vel_b[l_idx] = net.momentum * vel_b[l_idx] - net.lr * grad_b

                net.weights[l_idx] += vel_W[l_idx]
                net.biases[l_idx] += vel_b[l_idx]

        avg_loss = total_loss / max(n_batches, 1)
        net.train_record['epoch'].append(epoch)
        net.train_record['perf'].append(avg_loss)

        # Check stopping criteria
        if avg_loss <= net.goal:
            break
        if epoch > 1 and len(net.train_record['perf']) >= 2:
            grad_est = abs(net.train_record['perf'][-1] -
                           net.train_record['perf'][-2])
            if grad_est < net.min_grad:
                break

    return net, net.train_record


# =====================================================================
# Performance Evaluation
# =====================================================================

@_tb('forge_perform')
def forge_perform(net: ForgeNetwork, T: Any, Y: Any) -> float:
    """Compute network performance (loss) between targets and outputs.

    Parameters
    ----------
    net : ForgeNetwork
        Network (used to determine loss function type).
    T : array_like
        Target values.
    Y : array_like
        Network output values.

    Returns
    -------
    float
        Performance metric value.
    """
    T = np.asarray(T, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if net.loss_fn == 'crossentropy':
        return forge_crossentropy(T, Y)
    return forge_mse_metric(T, Y)


# =====================================================================
# Utility Functions
# =====================================================================

@_tb('forge_net_info')
def forge_net_info(net: ForgeNetwork) -> str:
    """Return a human-readable summary of the network architecture.

    Parameters
    ----------
    net : ForgeNetwork

    Returns
    -------
    str
        Network summary string.
    """
    lines = [
        f"Network type: {net.net_type}",
        f"Configured: {net.is_configured}",
    ]
    if net.is_configured:
        lines.append(f"Input size: {net.input_size}")
    lines.append(f"Layers: {net.n_layers}")
    for i, (size, act) in enumerate(zip(net.layer_sizes, net.activations)):
        n_params = 0
        if i < len(net.weights):
            n_params = net.weights[i].size + net.biases[i].size
        lines.append(f"  Layer {i + 1}: {size} neurons, "
                      f"activation={act}, params={n_params}")
    total = sum(W.size + b.size for W, b in zip(net.weights, net.biases))
    lines.append(f"Total parameters: {total}")
    lines.append(f"Loss function: {net.loss_fn}")
    lines.append(f"Learning rate: {net.lr}")
    lines.append(f"Momentum: {net.momentum}")
    lines.append(f"Max epochs: {net.epochs}")
    if net.train_record['epoch']:
        lines.append(f"Last training: {len(net.train_record['epoch'])} epochs, "
                      f"final perf={net.train_record['perf'][-1]:.6e}")
    return '\n'.join(lines)


@_tb('forge_getwb')
def forge_getwb(net: ForgeNetwork) -> np.ndarray:
    """Get all weights and biases as a single vector.

    Parameters
    ----------
    net : ForgeNetwork

    Returns
    -------
    ndarray
        Concatenated weight/bias vector.
    """
    parts = []
    for W, b in zip(net.weights, net.biases):
        parts.append(W.ravel())
        parts.append(b.ravel())
    return np.concatenate(parts)


@_tb('forge_setwb')
def forge_setwb(net: ForgeNetwork, wb: Any) -> ForgeNetwork:
    """Set all weights and biases from a single vector.

    Parameters
    ----------
    net : ForgeNetwork
    wb : array_like
        Weight/bias vector (same length as forge_getwb output).

    Returns
    -------
    ForgeNetwork
        Network with updated weights.
    """
    wb = np.asarray(wb, dtype=np.float64).ravel()
    idx = 0
    for l_idx in range(net.n_layers):
        W_shape = net.weights[l_idx].shape
        W_size = net.weights[l_idx].size
        net.weights[l_idx] = wb[idx:idx + W_size].reshape(W_shape)
        idx += W_size

        b_shape = net.biases[l_idx].shape
        b_size = net.biases[l_idx].size
        net.biases[l_idx] = wb[idx:idx + b_size].reshape(b_shape)
        idx += b_size

    return net


# =====================================================================
# Neural Network Registry
# =====================================================================

NEURAL_REGISTRY: dict[str, callable] = dict(_FUNCTIONS)


# ── Registration ─────────────────────────────────────────────────
def _load() -> dict[str, callable]:
    return dict(_FUNCTIONS)


