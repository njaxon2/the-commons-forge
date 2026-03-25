"""Fuzzy Logic Toolbox for Forge.

Provides Mamdani and Sugeno fuzzy inference systems, membership functions,
rule management, defuzzification methods, and FIS evaluation.

Target location: forge/engine/builtins/fuzzy.py

Backend: NumPy only.
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
# Internal Data Structures
# =====================================================================

class MembershipFunction:
    """A single membership function."""
    __slots__ = ('name', 'mf_type', 'params')

    def __init__(self, name: str, mf_type: str, params: np.ndarray):
        self.name = name
        self.mf_type = mf_type
        self.params = np.asarray(params, dtype=np.float64)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate membership degree for input x."""
        return _evaluate_mf(x, self.mf_type, self.params)

    def __repr__(self) -> str:
        return f"MF({self.name!r}, {self.mf_type}, {self.params.tolist()})"


class FuzzyVariable:
    """An input or output fuzzy variable."""
    __slots__ = ('name', 'var_type', 'range', 'mfs')

    def __init__(self, name: str, var_type: str,
                 var_range: np.ndarray):
        self.name = name
        self.var_type = var_type  # 'input' or 'output'
        self.range = np.asarray(var_range, dtype=np.float64)
        self.mfs: list[MembershipFunction] = []

    def __repr__(self) -> str:
        return (f"FuzzyVar({self.name!r}, {self.var_type}, "
                f"range={self.range.tolist()}, mfs={len(self.mfs)})")


class FuzzyRule:
    """A fuzzy rule: IF antecedent THEN consequent.

    antecedent : list of (var_index, mf_index) tuples for inputs.
                 Use -1 for mf_index to mean "don't care".
    consequent : list of (var_index, mf_index) tuples for outputs.
                 For Sugeno: mf_index references output MF params.
    weight     : rule weight (0 to 1).
    connection : 'and' or 'or' for combining antecedent conditions.
    """
    __slots__ = ('antecedent', 'consequent', 'weight', 'connection')

    def __init__(self, antecedent: list, consequent: list,
                 weight: float = 1.0, connection: str = 'and'):
        self.antecedent = antecedent
        self.consequent = consequent
        self.weight = float(weight)
        self.connection = connection.lower()


class FuzzyInferenceSystem:
    """Fuzzy Inference System container."""

    def __init__(self, fis_type: str = 'mamdani', name: str = 'fis'):
        self.name = name
        self.type = fis_type.lower()  # 'mamdani' or 'sugeno'
        self.inputs: list[FuzzyVariable] = []
        self.outputs: list[FuzzyVariable] = []
        self.rules: list[FuzzyRule] = []
        self.and_method = 'min'      # 'min' or 'prod'
        self.or_method = 'max'       # 'max' or 'probor'
        self.implication = 'min'     # 'min' or 'prod'
        self.aggregation = 'max'     # 'max' or 'sum'
        self.defuzz_method = 'centroid'  # default defuzzification

    def __repr__(self) -> str:
        return (f"FIS({self.name!r}, type={self.type}, "
                f"inputs={len(self.inputs)}, outputs={len(self.outputs)}, "
                f"rules={len(self.rules)})")


# =====================================================================
# Membership Function Evaluation
# =====================================================================

def _evaluate_mf(x: np.ndarray, mf_type: str,
                 params: np.ndarray) -> np.ndarray:
    """Evaluate a membership function at points x."""
    x = np.asarray(x, dtype=np.float64)
    t = mf_type.lower()

    if t == 'trimf':
        a, b, c = params[0], params[1], params[2]
        return np.maximum(0.0, np.minimum(
            np.where(b - a > 0, (x - a) / (b - a), np.where(x >= b, 1.0, 0.0)),
            np.where(c - b > 0, (c - x) / (c - b), np.where(x <= b, 1.0, 0.0))
        ))

    elif t == 'trapmf':
        a, b, c, d = params[0], params[1], params[2], params[3]
        return np.maximum(0.0, np.minimum(1.0, np.minimum(
            np.where(b - a > 0, (x - a) / (b - a), np.where(x >= a, 1.0, 0.0)),
            np.where(d - c > 0, (d - x) / (d - c), np.where(x <= d, 1.0, 0.0))
        )))

    elif t == 'gaussmf':
        sigma, c = params[0], params[1]
        return np.exp(-0.5 * ((x - c) / sigma) ** 2)

    elif t == 'gbellmf' or t == 'bellmf':
        a, b, c = params[0], params[1], params[2]
        return 1.0 / (1.0 + np.abs((x - c) / a) ** (2.0 * b))

    elif t == 'sigmf':
        a, c = params[0], params[1]
        return 1.0 / (1.0 + np.exp(-a * (x - c)))

    elif t == 'zmf':
        a, b = params[0], params[1]
        mid = (a + b) / 2.0
        result = np.ones_like(x)
        mask1 = (x >= a) & (x <= mid)
        mask2 = (x > mid) & (x < b)
        mask3 = x >= b
        result[mask1] = 1.0 - 2.0 * ((x[mask1] - a) / (b - a)) ** 2
        result[mask2] = 2.0 * ((x[mask2] - b) / (b - a)) ** 2
        result[mask3] = 0.0
        return result

    elif t == 'smf':
        a, b = params[0], params[1]
        mid = (a + b) / 2.0
        result = np.zeros_like(x)
        mask1 = (x >= a) & (x <= mid)
        mask2 = (x > mid) & (x < b)
        mask3 = x >= b
        result[mask1] = 2.0 * ((x[mask1] - a) / (b - a)) ** 2
        result[mask2] = 1.0 - 2.0 * ((x[mask2] - b) / (b - a)) ** 2
        result[mask3] = 1.0
        return result

    elif t == 'pimf':
        a, b, c, d = params[0], params[1], params[2], params[3]
        s_val = _evaluate_mf(x, 'smf', np.array([a, b]))
        z_val = _evaluate_mf(x, 'zmf', np.array([c, d]))
        return s_val * z_val

    elif t == 'gauss2mf':
        sigma1, c1, sigma2, c2 = params[0], params[1], params[2], params[3]
        left = np.exp(-0.5 * ((x - c1) / sigma1) ** 2)
        right = np.exp(-0.5 * ((x - c2) / sigma2) ** 2)
        # Left side of c1, right side of c2
        return np.where(x <= c1, left, np.where(x >= c2, right, 1.0))

    elif t == 'constant':
        # For Sugeno: constant output
        return np.full_like(x, params[0])

    elif t == 'linear':
        # For Sugeno: linear combination (params = [c1, c2, ..., c0])
        # x should be the input vector in this case
        return params  # handled specially in evalfis

    else:
        raise ValueError(f"Unknown membership function type: {mf_type}")


# =====================================================================
# Standalone MF Functions
# =====================================================================

@_tb()
def trimf(x, params):
    """Triangular membership function.

    Parameters
    ----------
    x : array_like
        Input values.
    params : array_like
        [a, b, c] — left foot, peak, right foot.

    Returns
    -------
    ndarray
        Membership degrees in [0, 1].
    """
    return _evaluate_mf(np.asarray(x, dtype=np.float64), 'trimf',
                        np.asarray(params, dtype=np.float64))


@_tb()
def trapmf(x, params):
    """Trapezoidal membership function.

    Parameters
    ----------
    x : array_like
        Input values.
    params : array_like
        [a, b, c, d] — left foot, left shoulder, right shoulder, right foot.
    """
    return _evaluate_mf(np.asarray(x, dtype=np.float64), 'trapmf',
                        np.asarray(params, dtype=np.float64))


@_tb()
def gaussmf(x, params):
    """Gaussian membership function.

    Parameters
    ----------
    x : array_like
        Input values.
    params : array_like
        [sigma, c] — standard deviation, center.
    """
    return _evaluate_mf(np.asarray(x, dtype=np.float64), 'gaussmf',
                        np.asarray(params, dtype=np.float64))


@_tb()
def gbellmf(x, params):
    """Generalized bell membership function.

    Parameters
    ----------
    x : array_like
        Input values.
    params : array_like
        [a, b, c] — width, slope, center.
    """
    return _evaluate_mf(np.asarray(x, dtype=np.float64), 'gbellmf',
                        np.asarray(params, dtype=np.float64))


@_tb()
def sigmf(x, params):
    """Sigmoidal membership function.

    Parameters
    ----------
    x : array_like
        Input values.
    params : array_like
        [a, c] — slope, crossover point.
    """
    return _evaluate_mf(np.asarray(x, dtype=np.float64), 'sigmf',
                        np.asarray(params, dtype=np.float64))


@_tb()
def zmf(x, params):
    """Z-shaped membership function.

    Parameters
    ----------
    x : array_like
        Input values.
    params : array_like
        [a, b] — start of transition, end of transition.
    """
    return _evaluate_mf(np.asarray(x, dtype=np.float64), 'zmf',
                        np.asarray(params, dtype=np.float64))


@_tb()
def smf(x, params):
    """S-shaped membership function.

    Parameters
    ----------
    x : array_like
        Input values.
    params : array_like
        [a, b] — start of transition, end of transition.
    """
    return _evaluate_mf(np.asarray(x, dtype=np.float64), 'smf',
                        np.asarray(params, dtype=np.float64))


@_tb()
def pimf(x, params):
    """Pi-shaped membership function (product of S and Z).

    Parameters
    ----------
    x : array_like
        Input values.
    params : array_like
        [a, b, c, d] — S-start, S-end, Z-start, Z-end.
    """
    return _evaluate_mf(np.asarray(x, dtype=np.float64), 'pimf',
                        np.asarray(params, dtype=np.float64))


# =====================================================================
# FIS Construction
# =====================================================================

@_tb('forge_mamfis')
def forge_mamfis(name: str = 'mamdani_fis') -> FuzzyInferenceSystem:
    """Create a Mamdani fuzzy inference system.

    Parameters
    ----------
    name : str
        Name for the FIS.

    Returns
    -------
    FuzzyInferenceSystem
        Empty Mamdani FIS ready for variable/rule addition.
    """
    return FuzzyInferenceSystem(fis_type='mamdani', name=name)


@_tb('forge_sugfis')
def forge_sugfis(name: str = 'sugeno_fis') -> FuzzyInferenceSystem:
    """Create a Sugeno fuzzy inference system.

    Parameters
    ----------
    name : str
        Name for the FIS.

    Returns
    -------
    FuzzyInferenceSystem
        Empty Sugeno FIS.
    """
    return FuzzyInferenceSystem(fis_type='sugeno', name=name)


@_tb('forge_addvar')
def forge_addvar(fis: FuzzyInferenceSystem, var_type: str,
                 name: str, var_range: Any) -> FuzzyInferenceSystem:
    """Add an input or output variable to a FIS.

    Parameters
    ----------
    fis : FuzzyInferenceSystem
        The FIS to modify.
    var_type : str
        'input' or 'output'.
    name : str
        Variable name.
    var_range : array_like
        [min, max] range for the variable.

    Returns
    -------
    FuzzyInferenceSystem
        The modified FIS (same object).
    """
    var = FuzzyVariable(name, var_type.lower(),
                        np.asarray(var_range, dtype=np.float64))
    if var_type.lower() == 'input':
        fis.inputs.append(var)
    elif var_type.lower() == 'output':
        fis.outputs.append(var)
    else:
        raise ValueError(f"var_type must be 'input' or 'output', got {var_type!r}")
    return fis


@_tb('forge_addmf')
def forge_addmf(fis: FuzzyInferenceSystem, var_name: str,
                mf_name: str, mf_type: str,
                params: Any) -> FuzzyInferenceSystem:
    """Add a membership function to a variable.

    Parameters
    ----------
    fis : FuzzyInferenceSystem
        The FIS to modify.
    var_name : str
        Name of the variable (must already exist).
    mf_name : str
        Name for the membership function.
    mf_type : str
        MF type: 'trimf', 'trapmf', 'gaussmf', 'gbellmf',
        'sigmf', 'zmf', 'smf', 'pimf', 'constant', 'linear'.
    params : array_like
        Parameters for the membership function.

    Returns
    -------
    FuzzyInferenceSystem
        The modified FIS.
    """
    # Find variable
    var = None
    for v in fis.inputs + fis.outputs:
        if v.name == var_name:
            var = v
            break
    if var is None:
        raise ValueError(f"Variable {var_name!r} not found in FIS")

    mf = MembershipFunction(mf_name, mf_type,
                            np.asarray(params, dtype=np.float64))
    var.mfs.append(mf)
    return fis


@_tb('forge_addrule')
def forge_addrule(fis: FuzzyInferenceSystem,
                  rules: Any) -> FuzzyInferenceSystem:
    """Add rules to a FIS.

    Parameters
    ----------
    fis : FuzzyInferenceSystem
        The FIS to modify.
    rules : array_like or list of lists
        Rule matrix where each row is:
        [in1_mf, in2_mf, ..., out1_mf, out2_mf, ..., weight, connection]
        MF indices are 1-based (0 = don't care). Connection: 1=AND, 2=OR.

    Returns
    -------
    FuzzyInferenceSystem
        The modified FIS.
    """
    rules_arr = np.asarray(rules, dtype=np.float64)
    if rules_arr.ndim == 1:
        rules_arr = rules_arr.reshape(1, -1)

    n_in = len(fis.inputs)
    n_out = len(fis.outputs)

    for row in rules_arr:
        # Parse rule row
        antecedent = []
        for i in range(n_in):
            mf_idx = int(row[i]) - 1  # convert to 0-based (-1 = don't care)
            antecedent.append((i, mf_idx))

        consequent = []
        for j in range(n_out):
            mf_idx = int(row[n_in + j]) - 1
            consequent.append((j, mf_idx))

        weight = float(row[n_in + n_out]) if len(row) > n_in + n_out else 1.0
        conn_val = int(row[n_in + n_out + 1]) if len(row) > n_in + n_out + 1 else 1
        connection = 'and' if conn_val == 1 else 'or'

        fis.rules.append(FuzzyRule(antecedent, consequent, weight, connection))

    return fis


# =====================================================================
# Defuzzification Methods
# =====================================================================

def _defuzz_centroid(x: np.ndarray, mf: np.ndarray) -> float:
    """Centroid defuzzification (center of area)."""
    total_area = np.trapz(mf, x)
    if abs(total_area) < 1e-30:
        return float(np.mean(x))
    return float(np.trapz(x * mf, x) / total_area)


def _defuzz_bisector(x: np.ndarray, mf: np.ndarray) -> float:
    """Bisector defuzzification (divides area in half)."""
    total_area = np.trapz(mf, x)
    if abs(total_area) < 1e-30:
        return float(np.mean(x))
    half = total_area / 2.0
    cumulative = np.cumsum((mf[:-1] + mf[1:]) / 2.0 * np.diff(x))
    idx = np.searchsorted(cumulative, half)
    idx = min(idx, len(x) - 1)
    return float(x[idx])


def _defuzz_mom(x: np.ndarray, mf: np.ndarray) -> float:
    """Mean of Maximum defuzzification."""
    max_val = np.max(mf)
    if max_val < 1e-30:
        return float(np.mean(x))
    max_indices = np.where(np.abs(mf - max_val) < 1e-10)[0]
    return float(np.mean(x[max_indices]))


def _defuzz_som(x: np.ndarray, mf: np.ndarray) -> float:
    """Smallest of Maximum defuzzification."""
    max_val = np.max(mf)
    if max_val < 1e-30:
        return float(x[0])
    max_indices = np.where(np.abs(mf - max_val) < 1e-10)[0]
    return float(x[max_indices[0]])


def _defuzz_lom(x: np.ndarray, mf: np.ndarray) -> float:
    """Largest of Maximum defuzzification."""
    max_val = np.max(mf)
    if max_val < 1e-30:
        return float(x[-1])
    max_indices = np.where(np.abs(mf - max_val) < 1e-10)[0]
    return float(x[max_indices[-1]])


_DEFUZZ_METHODS = {
    'centroid': _defuzz_centroid,
    'bisector': _defuzz_bisector,
    'mom': _defuzz_mom,
    'som': _defuzz_som,
    'lom': _defuzz_lom,
}


@_tb()
def defuzz(x, mf_values, method='centroid'):
    """Defuzzify a fuzzy output using the specified method.

    Parameters
    ----------
    x : array_like
        Universe of discourse points.
    mf_values : array_like
        Aggregated membership function values.
    method : str
        'centroid', 'bisector', 'mom', 'som', or 'lom'.

    Returns
    -------
    float
        Crisp output value.
    """
    x = np.asarray(x, dtype=np.float64)
    mf = np.asarray(mf_values, dtype=np.float64)
    fn = _DEFUZZ_METHODS.get(method.lower())
    if fn is None:
        raise ValueError(
            f"Unknown defuzzification method: {method!r}. "
            f"Available: {list(_DEFUZZ_METHODS.keys())}")
    return fn(x, mf)


# =====================================================================
# FIS Evaluation
# =====================================================================

@_tb('forge_evalfis')
def forge_evalfis(fis: FuzzyInferenceSystem,
                  inputs: Any,
                  n_points: int = 201) -> np.ndarray:
    """Evaluate a fuzzy inference system.

    Parameters
    ----------
    fis : FuzzyInferenceSystem
        A configured FIS with variables, MFs, and rules.
    inputs : array_like
        Input values. Shape (n_inputs,) for single evaluation, or
        (n_samples, n_inputs) for batch evaluation.
    n_points : int
        Number of discretization points for Mamdani defuzzification.

    Returns
    -------
    ndarray
        Output values. Shape (n_outputs,) or (n_samples, n_outputs).
    """
    inputs_arr = np.asarray(inputs, dtype=np.float64)
    if inputs_arr.ndim == 1:
        inputs_arr = inputs_arr.reshape(1, -1)

    n_samples = inputs_arr.shape[0]
    n_outputs = len(fis.outputs)
    results = np.zeros((n_samples, n_outputs))

    for s in range(n_samples):
        x_in = inputs_arr[s, :]

        # 1. Fuzzify inputs: compute membership degrees for each rule
        rule_strengths = []
        for rule in fis.rules:
            degrees = []
            for var_idx, mf_idx in rule.antecedent:
                if mf_idx < 0:
                    degrees.append(1.0)  # don't care
                    continue
                var = fis.inputs[var_idx]
                if mf_idx < len(var.mfs):
                    deg = float(var.mfs[mf_idx].evaluate(
                        np.array([x_in[var_idx]]))[0])
                else:
                    deg = 0.0
                degrees.append(deg)

            # Combine antecedent degrees
            if rule.connection == 'and':
                if fis.and_method == 'min':
                    strength = min(degrees) if degrees else 0.0
                else:  # prod
                    strength = float(np.prod(degrees)) if degrees else 0.0
            else:  # or
                if fis.or_method == 'max':
                    strength = max(degrees) if degrees else 0.0
                else:  # probor
                    strength = 1.0 - float(np.prod(
                        [1.0 - d for d in degrees])) if degrees else 0.0

            strength *= rule.weight
            rule_strengths.append(strength)

        # 2. Evaluate outputs
        if fis.type == 'mamdani':
            for out_idx in range(n_outputs):
                out_var = fis.outputs[out_idx]
                x_universe = np.linspace(
                    float(out_var.range[0]), float(out_var.range[1]),
                    n_points)
                aggregated = np.zeros(n_points)

                for r_idx, rule in enumerate(fis.rules):
                    rs = rule_strengths[r_idx]
                    if rs < 1e-15:
                        continue
                    for v_idx, mf_idx in rule.consequent:
                        if v_idx != out_idx or mf_idx < 0:
                            continue
                        if mf_idx >= len(out_var.mfs):
                            continue
                        mf_vals = out_var.mfs[mf_idx].evaluate(x_universe)
                        # Implication
                        if fis.implication == 'min':
                            implied = np.minimum(rs, mf_vals)
                        else:  # prod
                            implied = rs * mf_vals
                        # Aggregation
                        if fis.aggregation == 'max':
                            aggregated = np.maximum(aggregated, implied)
                        else:  # sum
                            aggregated = aggregated + implied

                # Defuzzify
                results[s, out_idx] = defuzz(
                    x_universe, aggregated, fis.defuzz_method)

        elif fis.type == 'sugeno':
            for out_idx in range(n_outputs):
                out_var = fis.outputs[out_idx]
                weighted_sum = 0.0
                weight_total = 0.0

                for r_idx, rule in enumerate(fis.rules):
                    rs = rule_strengths[r_idx]
                    if rs < 1e-15:
                        continue
                    for v_idx, mf_idx in rule.consequent:
                        if v_idx != out_idx or mf_idx < 0:
                            continue
                        if mf_idx >= len(out_var.mfs):
                            continue
                        mf = out_var.mfs[mf_idx]
                        if mf.mf_type == 'constant':
                            z = float(mf.params[0])
                        elif mf.mf_type == 'linear':
                            # Linear: z = c0 + c1*x1 + c2*x2 + ...
                            coeffs = mf.params
                            z = float(coeffs[0])  # constant term
                            for k in range(min(len(x_in), len(coeffs) - 1)):
                                z += float(coeffs[k + 1]) * float(x_in[k])
                        else:
                            z = 0.0
                        weighted_sum += rs * z
                        weight_total += rs

                if weight_total > 1e-15:
                    results[s, out_idx] = weighted_sum / weight_total
                else:
                    results[s, out_idx] = float(np.mean(out_var.range))

    # Squeeze single-sample and single-output dimensions
    if results.shape[0] == 1 and results.shape[1] == 1:
        return float(results[0, 0])
    elif results.shape[0] == 1:
        return results[0, :]
    elif results.shape[1] == 1:
        return results[:, 0]
    return results


# =====================================================================
# FIS Utility Functions
# =====================================================================

@_tb('forge_plotmf')
def forge_plotmf(fis: FuzzyInferenceSystem, var_type: str,
                 var_index: int, n_points: int = 201) -> dict:
    """Compute membership function curves for plotting.

    Parameters
    ----------
    fis : FuzzyInferenceSystem
        The FIS.
    var_type : str
        'input' or 'output'.
    var_index : int
        1-based index of the variable.

    Returns
    -------
    dict
        {'x': universe array, 'curves': {mf_name: values}, 'var_name': str}
    """
    idx = int(var_index) - 1  # convert to 0-based
    if var_type.lower() == 'input':
        var = fis.inputs[idx]
    else:
        var = fis.outputs[idx]

    x = np.linspace(float(var.range[0]), float(var.range[1]), n_points)
    curves = {}
    for mf in var.mfs:
        curves[mf.name] = mf.evaluate(x)

    return {'x': x, 'curves': curves, 'var_name': var.name}


@_tb('forge_showfis')
def forge_showfis(fis: FuzzyInferenceSystem) -> str:
    """Return a text summary of a FIS.

    Parameters
    ----------
    fis : FuzzyInferenceSystem
        The FIS to describe.

    Returns
    -------
    str
        Human-readable FIS summary.
    """
    lines = [
        f"Name: {fis.name}",
        f"Type: {fis.type}",
        f"Inputs: {len(fis.inputs)}",
        f"Outputs: {len(fis.outputs)}",
        f"Rules: {len(fis.rules)}",
        f"And method: {fis.and_method}",
        f"Or method: {fis.or_method}",
        f"Implication: {fis.implication}",
        f"Aggregation: {fis.aggregation}",
        f"Defuzzification: {fis.defuzz_method}",
        "",
    ]
    for i, var in enumerate(fis.inputs):
        lines.append(f"  Input {i + 1}: {var.name} [{var.range[0]:.2f}, {var.range[1]:.2f}]")
        for mf in var.mfs:
            lines.append(f"    MF: {mf.name} ({mf.mf_type}) params={mf.params.tolist()}")
    for i, var in enumerate(fis.outputs):
        lines.append(f"  Output {i + 1}: {var.name} [{var.range[0]:.2f}, {var.range[1]:.2f}]")
        for mf in var.mfs:
            lines.append(f"    MF: {mf.name} ({mf.mf_type}) params={mf.params.tolist()}")

    lines.append("")
    for i, rule in enumerate(fis.rules):
        ant_str = ', '.join(
            f"In{vi + 1}=MF{mi + 1}" for vi, mi in rule.antecedent if mi >= 0)
        con_str = ', '.join(
            f"Out{vi + 1}=MF{mi + 1}" for vi, mi in rule.consequent if mi >= 0)
        lines.append(
            f"  Rule {i + 1}: IF {ant_str} ({rule.connection.upper()}) "
            f"THEN {con_str} (w={rule.weight})")

    return '\n'.join(lines)


@_tb('forge_genfis')
def forge_genfis(input_data: Any, output_data: Any,
                 n_mfs: int = 3,
                 mf_type: str = 'gaussmf') -> FuzzyInferenceSystem:
    """Generate a FIS from data using grid partitioning.

    Parameters
    ----------
    input_data : array_like
        Input data, shape (n_samples, n_inputs).
    output_data : array_like
        Output data, shape (n_samples,) or (n_samples, n_outputs).
    n_mfs : int
        Number of membership functions per variable.
    mf_type : str
        MF type for all variables.

    Returns
    -------
    FuzzyInferenceSystem
        A Mamdani FIS with auto-generated MFs and rules.
    """
    X = np.asarray(input_data, dtype=np.float64)
    Y = np.asarray(output_data, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    n_in = X.shape[1]
    n_out = Y.shape[1]

    fis = FuzzyInferenceSystem(fis_type='mamdani', name='genfis_result')

    # Add input variables with evenly-spaced MFs
    for i in range(n_in):
        lo, hi = float(np.min(X[:, i])), float(np.max(X[:, i]))
        margin = (hi - lo) * 0.1
        fis = forge_addvar(fis, 'input', f'in{i + 1}', [lo - margin, hi + margin])
        centers = np.linspace(lo, hi, n_mfs)
        sigma = (hi - lo) / (2.0 * (n_mfs - 1)) if n_mfs > 1 else (hi - lo) / 2.0
        for j, c in enumerate(centers):
            if mf_type == 'gaussmf':
                params = [sigma, c]
            elif mf_type == 'trimf':
                left = c - sigma * 2
                right = c + sigma * 2
                params = [left, c, right]
            else:
                params = [sigma, c]
            fis = forge_addmf(fis, f'in{i + 1}', f'mf{j + 1}', mf_type, params)

    # Add output variables
    for k in range(n_out):
        lo, hi = float(np.min(Y[:, k])), float(np.max(Y[:, k]))
        margin = (hi - lo) * 0.1
        fis = forge_addvar(fis, 'output', f'out{k + 1}', [lo - margin, hi + margin])
        centers = np.linspace(lo, hi, n_mfs)
        sigma = (hi - lo) / (2.0 * (n_mfs - 1)) if n_mfs > 1 else (hi - lo) / 2.0
        for j, c in enumerate(centers):
            if mf_type == 'gaussmf':
                params = [sigma, c]
            elif mf_type == 'trimf':
                left = c - sigma * 2
                right = c + sigma * 2
                params = [left, c, right]
            else:
                params = [sigma, c]
            fis = forge_addmf(fis, f'out{k + 1}', f'mf{j + 1}', mf_type, params)

    # Generate rules (simple grid: map input MF combos to nearest output MF)
    # For single-input case: map MF i -> MF i
    if n_in == 1:
        for j in range(n_mfs):
            rule_row = [j + 1]  # input MF (1-based)
            for k in range(n_out):
                rule_row.append(min(j + 1, n_mfs))  # output MF
            rule_row.extend([1.0, 1])  # weight, AND
            fis = forge_addrule(fis, [rule_row])
    else:
        # Multi-input: create diagonal rules as starting point
        for j in range(n_mfs):
            rule_row = [j + 1] * n_in  # all inputs use MF j+1
            for k in range(n_out):
                rule_row.append(min(j + 1, n_mfs))
            rule_row.extend([1.0, 1])
            fis = forge_addrule(fis, [rule_row])

    return fis


@_tb('forge_writefis')
def forge_writefis(fis: FuzzyInferenceSystem) -> str:
    """Serialize a FIS to a string (Octave .fis format compatible).

    Parameters
    ----------
    fis : FuzzyInferenceSystem
        The FIS to serialize.

    Returns
    -------
    str
        FIS file content as string.
    """
    lines = [
        '[System]',
        f'Name=\'{fis.name}\'',
        f'Type=\'{fis.type}\'',
        f'Version=2.0',
        f'NumInputs={len(fis.inputs)}',
        f'NumOutputs={len(fis.outputs)}',
        f'NumRules={len(fis.rules)}',
        f'AndMethod=\'{fis.and_method}\'',
        f'OrMethod=\'{fis.or_method}\'',
        f'ImpMethod=\'{fis.implication}\'',
        f'AggMethod=\'{fis.aggregation}\'',
        f'DefuzzMethod=\'{fis.defuzz_method}\'',
        '',
    ]

    for i, var in enumerate(fis.inputs):
        lines.append(f'[Input{i + 1}]')
        lines.append(f'Name=\'{var.name}\'')
        lines.append(f'Range=[{var.range[0]} {var.range[1]}]')
        lines.append(f'NumMFs={len(var.mfs)}')
        for j, mf in enumerate(var.mfs):
            params_str = ' '.join(f'{p}' for p in mf.params)
            lines.append(f'MF{j + 1}=\'{mf.name}\':\'{mf.mf_type}\',[{params_str}]')
        lines.append('')

    for i, var in enumerate(fis.outputs):
        lines.append(f'[Output{i + 1}]')
        lines.append(f'Name=\'{var.name}\'')
        lines.append(f'Range=[{var.range[0]} {var.range[1]}]')
        lines.append(f'NumMFs={len(var.mfs)}')
        for j, mf in enumerate(var.mfs):
            params_str = ' '.join(f'{p}' for p in mf.params)
            lines.append(f'MF{j + 1}=\'{mf.name}\':\'{mf.mf_type}\',[{params_str}]')
        lines.append('')

    lines.append('[Rules]')
    for rule in fis.rules:
        parts = []
        for _, mf_idx in rule.antecedent:
            parts.append(str(mf_idx + 1))
        parts.append(',')
        for _, mf_idx in rule.consequent:
            parts.append(str(mf_idx + 1))
        conn = '1' if rule.connection == 'and' else '2'
        parts.append(f'({rule.weight}) : {conn}')
        lines.append(' '.join(parts))

    return '\n'.join(lines)


@_tb('forge_readfis')
def forge_readfis(fis_string: str) -> FuzzyInferenceSystem:
    """Parse a FIS from string (.fis format).

    Parameters
    ----------
    fis_string : str
        FIS content in .fis format.

    Returns
    -------
    FuzzyInferenceSystem
        Parsed FIS object.
    """
    fis = FuzzyInferenceSystem()
    current_section = None
    current_var = None

    for line in fis_string.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        # Section headers
        if line.startswith('[') and line.endswith(']'):
            current_section = line[1:-1]
            if current_section.startswith('Input'):
                idx = int(current_section[5:]) - 1
                while len(fis.inputs) <= idx:
                    fis.inputs.append(FuzzyVariable(f'in{len(fis.inputs) + 1}',
                                                     'input', np.array([0, 1])))
                current_var = fis.inputs[idx]
            elif current_section.startswith('Output'):
                idx = int(current_section[6:]) - 1
                while len(fis.outputs) <= idx:
                    fis.outputs.append(FuzzyVariable(f'out{len(fis.outputs) + 1}',
                                                      'output', np.array([0, 1])))
                current_var = fis.outputs[idx]
            else:
                current_var = None
            continue

        # Key=Value parsing
        if '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip("'")

        if current_section == 'System':
            if key == 'Name':
                fis.name = value
            elif key == 'Type':
                fis.type = value.lower()
            elif key == 'AndMethod':
                fis.and_method = value
            elif key == 'OrMethod':
                fis.or_method = value
            elif key == 'ImpMethod':
                fis.implication = value
            elif key == 'AggMethod':
                fis.aggregation = value
            elif key == 'DefuzzMethod':
                fis.defuzz_method = value

        elif current_var is not None:
            if key == 'Name':
                current_var.name = value
            elif key == 'Range':
                nums = value.strip('[]').split()
                current_var.range = np.array([float(n) for n in nums])
            elif key.startswith('MF'):
                # Format: 'name':'type',[params]
                parts = value.split(':')
                mf_name = parts[0].strip("'")
                rest = ':'.join(parts[1:])
                type_and_params = rest.split(',', 1)
                mf_type = type_and_params[0].strip("'")
                params_str = type_and_params[1].strip(' []') if len(type_and_params) > 1 else ''
                params = [float(p) for p in params_str.split() if p]
                current_var.mfs.append(
                    MembershipFunction(mf_name, mf_type, np.array(params)))

    return fis


# =====================================================================
# Fuzzy Registry
# =====================================================================

FUZZY_REGISTRY: dict[str, callable] = dict(_FUNCTIONS)


# ── Registration ─────────────────────────────────────────────────
def _load() -> dict[str, callable]:
    return dict(_FUNCTIONS)


