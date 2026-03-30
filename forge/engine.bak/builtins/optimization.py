# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Optimization Toolbox for Forge.

Provides 13 optimization and root-finding functions compatible with
Octave/MATLAB optimization conventions.

Backend: scipy.optimize + numpy.
"""

from __future__ import annotations

import numpy as np

from forge.engine.types import ForgeArray, _unwrap
from forge.engine.containers import ForgeStruct


# ── Helpers ──────────────────────────────────────────────────────

def _wrap(value):
    """Wrap a numpy array as a ForgeArray."""
    if isinstance(value, np.ndarray):
        return ForgeArray(value)
    return value


def _scalar(value):
    """Extract scalar from 0-d or single-element array."""
    if isinstance(value, np.ndarray):
        if value.ndim == 0 or value.size == 1:
            return value.item()
    return value


def _extract_optim_options(opts):
    """Extract common optimization options from a ForgeStruct or dict."""
    if opts is None:
        return {}
    if isinstance(opts, ForgeStruct):
        return {k: v for k, v in opts.__dict__.items() if not k.startswith('_') and v is not None}
    if isinstance(opts, dict):
        return {k: v for k, v in opts.items() if v is not None}
    return {}


# ── Toolbox function registry ───────────────────────────────────
OPTIMIZATION_REGISTRY: dict[str, callable] = {}


def _tb(name: str | None = None):
    """Local decorator to register a toolbox function."""
    def decorator(func):
        fn_name = name or func.__name__
        OPTIMIZATION_REGISTRY[fn_name] = func
        return func
    return decorator


# =====================================================================
# Root Finding
# =====================================================================

@_tb("fzero")
def forge_fzero(fun, x0, opts=None):
    """Find a zero of a univariate function.

    x = fzero(fun, x0)
    x = fzero(fun, [a, b])
    [x, fval] = fzero(fun, x0)
    [x, fval, info] = fzero(fun, x0, opts)

    If x0 is scalar, uses a search starting from x0.
    If x0 is [a, b], uses Brent's method on the bracket.

    Returns (x, fval, info) where info > 0 indicates convergence.
    """
    from scipy.optimize import brentq, brent
    from scipy.optimize import fsolve as _fsolve

    x0_arr = np.asarray(x0, dtype=np.float64).ravel()
    scipy_kwargs = {}

    raw_opts = _extract_optim_options(opts)
    if raw_opts.get('TolX') is not None:
        scipy_kwargs['xtol'] = float(raw_opts['TolX'])

    if len(x0_arr) == 2:
        # Bracket mode
        a, b = float(x0_arr[0]), float(x0_arr[1])
        root = brentq(fun, a, b, **scipy_kwargs)
    else:
        # Initial guess mode
        result = _fsolve(fun, float(x0_arr[0]), full_output=True)
        root = float(result[0][0])

    fval = float(fun(root))
    info = 1  # converged
    return (root, fval, info)


@_tb("fsolve")
def forge_fsolve(fun, x0, opts=None):
    """Solve system of nonlinear equations.

    x = fsolve(fun, x0)
    [x, fval] = fsolve(fun, x0)
    [x, fval, info] = fsolve(fun, x0, opts)

    fun: function handle f(x) = 0
    x0: initial guess (vector)

    Returns (x, fval, info).
    """
    from scipy.optimize import fsolve as _fsolve

    x0_arr = np.asarray(x0, dtype=np.float64).ravel()
    kwargs = {'full_output': True}

    raw_opts = _extract_optim_options(opts)
    if raw_opts.get('TolFun') is not None:
        # fsolve doesn't have a direct ftol, but we can set xtol
        pass
    if raw_opts.get('TolX') is not None:
        kwargs['xtol'] = float(raw_opts['TolX'])
    if raw_opts.get('MaxFunEvals') is not None:
        kwargs['maxfev'] = int(raw_opts['MaxFunEvals'])

    result = _fsolve(fun, x0_arr, **kwargs)
    x_sol = result[0]
    info_dict = result[1]
    ier = result[2]  # 1 = converged

    fval = np.asarray(fun(x_sol), dtype=np.float64).ravel()

    if x_sol.size == 1:
        x_sol = float(x_sol[0])
    else:
        x_sol = _wrap(x_sol)

    return (x_sol, _wrap(fval), ier)


# =====================================================================
# Minimization
# =====================================================================

@_tb("fminbnd")
def forge_fminbnd(fun, x1, x2, opts=None):
    """Find minimum of single-variable function on bounded interval.

    x = fminbnd(fun, x1, x2)
    [x, fval] = fminbnd(fun, x1, x2)
    [x, fval, info] = fminbnd(fun, x1, x2, opts)
    """
    from scipy.optimize import minimize_scalar

    kwargs = {}
    raw_opts = _extract_optim_options(opts)
    if raw_opts.get('TolX') is not None:
        kwargs['options'] = {'xatol': float(raw_opts['TolX'])}

    result = minimize_scalar(fun, bounds=(float(x1), float(x2)),
                             method='bounded', **kwargs)
    return (result.x, result.fun, 1 if result.success else 0)


@_tb("fminsearch")
def forge_fminsearch(fun, x0, opts=None):
    """Find minimum of unconstrained multivariable function (Nelder-Mead).

    x = fminsearch(fun, x0)
    [x, fval] = fminsearch(fun, x0, opts)
    [x, fval, info] = fminsearch(fun, x0, opts)
    """
    from scipy.optimize import minimize

    x0_arr = np.asarray(x0, dtype=np.float64).ravel()
    kwargs = {'method': 'Nelder-Mead'}

    raw_opts = _extract_optim_options(opts)
    nm_options = {}
    if raw_opts.get('TolX') is not None:
        nm_options['xatol'] = float(raw_opts['TolX'])
    if raw_opts.get('TolFun') is not None:
        nm_options['fatol'] = float(raw_opts['TolFun'])
    if raw_opts.get('MaxFunEvals') is not None:
        nm_options['maxfev'] = int(raw_opts['MaxFunEvals'])
    if raw_opts.get('MaxIter') is not None:
        nm_options['maxiter'] = int(raw_opts['MaxIter'])
    if nm_options:
        kwargs['options'] = nm_options

    result = minimize(fun, x0_arr, **kwargs)
    x_sol = result.x
    if x_sol.size == 1:
        x_sol = float(x_sol[0])
    else:
        x_sol = _wrap(x_sol)

    return (x_sol, result.fun, 1 if result.success else 0)


@_tb("fminunc")
def forge_fminunc(fun, x0, opts=None):
    """Find minimum of unconstrained multivariable function (BFGS).

    x = fminunc(fun, x0)
    [x, fval] = fminunc(fun, x0, opts)
    [x, fval, info] = fminunc(fun, x0, opts)

    Uses BFGS quasi-Newton method with numerical gradients.
    """
    from scipy.optimize import minimize

    x0_arr = np.asarray(x0, dtype=np.float64).ravel()
    kwargs = {'method': 'BFGS'}

    raw_opts = _extract_optim_options(opts)
    bfgs_options = {}
    if raw_opts.get('TolFun') is not None:
        bfgs_options['gtol'] = float(raw_opts['TolFun'])
    if raw_opts.get('MaxIter') is not None:
        bfgs_options['maxiter'] = int(raw_opts['MaxIter'])
    if raw_opts.get('MaxFunEvals') is not None:
        bfgs_options['maxfun'] = int(raw_opts['MaxFunEvals'])
    if bfgs_options:
        kwargs['options'] = bfgs_options

    # Check if gradient is provided
    grad = raw_opts.get('GradObj') or raw_opts.get('Gradient')
    if grad is not None and callable(grad):
        kwargs['jac'] = grad

    result = minimize(fun, x0_arr, **kwargs)
    x_sol = result.x
    if x_sol.size == 1:
        x_sol = float(x_sol[0])
    else:
        x_sol = _wrap(x_sol)

    return (x_sol, result.fun, 1 if result.success else 0)


@_tb("lsqnonneg")
def forge_lsqnonneg(C, d, opts=None):
    """Linear least squares with nonnegativity constraints.

    x = lsqnonneg(C, d)
    [x, resnorm] = lsqnonneg(C, d)
    [x, resnorm, residual] = lsqnonneg(C, d)

    Solves min || C*x - d ||^2 subject to x >= 0.
    """
    from scipy.optimize import nnls

    C_arr = np.asarray(C, dtype=np.float64)
    d_arr = np.asarray(d, dtype=np.float64).ravel()

    x_sol, resnorm = nnls(C_arr, d_arr)
    residual = d_arr - C_arr @ x_sol

    return (_wrap(x_sol), resnorm, _wrap(residual))


# =====================================================================
# Optimization Options
# =====================================================================

@_tb("optimset")
def forge_optimset(*args, **kwargs):
    """Create or modify optimization options structure.

    opts = optimset('TolX', 1e-6, 'TolFun', 1e-8)
    opts = optimset(oldopts, 'MaxIter', 1000)

    Supported options:
      Display       - 'off', 'iter', 'final', 'notify'
      MaxFunEvals   - Maximum function evaluations
      MaxIter       - Maximum iterations
      TolFun        - Function value tolerance
      TolX          - Variable tolerance
      GradObj       - Gradient function ('on'/'off' or callable)
      Algorithm     - Algorithm name
      OutputFcn     - Output function handle

    Returns a ForgeStruct with the option fields.
    """
    defaults = {
        'Display': 'off',
        'MaxFunEvals': None,
        'MaxIter': None,
        'TolFun': 1e-6,
        'TolX': 1e-6,
        'GradObj': None,
        'Algorithm': None,
        'OutputFcn': None,
    }

    i = 0
    if args and isinstance(args[0], (dict, ForgeStruct)):
        old = args[0]
        if isinstance(old, ForgeStruct):
            old = {k: v for k, v in old.__dict__.items() if not k.startswith('_')}
        defaults.update(old)
        i = 1

    # Check for merge of two option structs
    if i < len(args) and isinstance(args[i], (dict, ForgeStruct)):
        opts2 = args[i]
        if isinstance(opts2, ForgeStruct):
            opts2 = {k: v for k, v in opts2.__dict__.items() if not k.startswith('_')}
        for key, val in opts2.items():
            if val is not None:
                defaults[key] = val
        i += 1

    while i + 1 < len(args):
        defaults[str(args[i])] = args[i + 1]
        i += 2

    defaults.update(kwargs)
    return ForgeStruct(**defaults)


@_tb("optimget")
def forge_optimget(options, name, default=None):
    """Get optimization option value.

    val = optimget(opts, 'TolX')
    val = optimget(opts, 'TolX', 1e-6)
    """
    if options is None:
        return default
    name = str(name)
    if isinstance(options, ForgeStruct):
        return getattr(options, name, default)
    elif isinstance(options, dict):
        return options.get(name, default)
    return default


# =====================================================================
# Test Functions
# =====================================================================

@_tb("humps")
def forge_humps(x=None):
    """Humps test function for optimization and integration.

    y = humps(x)

    humps(x) = 1./((x-0.3).^2 + 0.01) + 1./((x-0.9).^2 + 0.04) - 6

    If called with no arguments, returns values for x = linspace(0, 1, 101).
    """
    if x is None:
        x = np.linspace(0, 1, 101)
    x = np.asarray(x, dtype=np.float64)
    y = 1.0 / ((x - 0.3) ** 2 + 0.01) + 1.0 / ((x - 0.9) ** 2 + 0.04) - 6.0
    return _scalar(_wrap(y))


# =====================================================================
# Quadratic / Linear Programming
# =====================================================================

@_tb("qp")
def forge_qp(H, f, A=None, b=None, Aeq=None, beq=None, lb=None, ub=None, x0=None, opts=None):
    """Solve quadratic programming problem.

    x = qp(H, f)
    x = qp(H, f, A, b)
    x = qp(H, f, A, b, Aeq, beq)
    x = qp(H, f, A, b, Aeq, beq, lb, ub)
    [x, fval, info] = qp(...)

    Minimize: 0.5 * x' * H * x + f' * x
    Subject to: A*x <= b, Aeq*x = beq, lb <= x <= ub
    """
    from scipy.optimize import minimize

    H = np.asarray(H, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64).ravel()
    n = len(f)

    def objective(x):
        return 0.5 * x @ H @ x + f @ x

    def gradient(x):
        return H @ x + f

    constraints = []
    if A is not None and b is not None:
        A_arr = np.asarray(A, dtype=np.float64)
        b_arr = np.asarray(b, dtype=np.float64).ravel()
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, A=A_arr, b=b_arr: b - A @ x,
            'jac': lambda x, A=A_arr: -A,
        })
    if Aeq is not None and beq is not None:
        Aeq_arr = np.asarray(Aeq, dtype=np.float64)
        beq_arr = np.asarray(beq, dtype=np.float64).ravel()
        constraints.append({
            'type': 'eq',
            'fun': lambda x, A=Aeq_arr, b=beq_arr: A @ x - b,
            'jac': lambda x, A=Aeq_arr: A,
        })

    bounds = None
    if lb is not None or ub is not None:
        lb_arr = np.full(n, -np.inf) if lb is None else np.asarray(lb, dtype=np.float64).ravel()
        ub_arr = np.full(n, np.inf) if ub is None else np.asarray(ub, dtype=np.float64).ravel()
        bounds = list(zip(lb_arr, ub_arr))

    if x0 is None:
        x0 = np.zeros(n)
    else:
        x0 = np.asarray(x0, dtype=np.float64).ravel()

    result = minimize(objective, x0, jac=gradient, method='SLSQP',
                      constraints=constraints if constraints else (),
                      bounds=bounds)

    info = 1 if result.success else 0
    return (_wrap(result.x), result.fun, info)


@_tb("sqp")
def forge_sqp(x0, phi, g=None, h=None, lb=None, ub=None):
    """Solve nonlinear programming problem using SQP.

    x = sqp(x0, phi)
    x = sqp(x0, phi, g, h)
    x = sqp(x0, phi, g, h, lb, ub)
    [x, fval, info] = sqp(...)

    phi: objective function (and optionally gradient)
    g: inequality constraint function g(x) >= 0
    h: equality constraint function h(x) = 0
    """
    from scipy.optimize import minimize

    x0_arr = np.asarray(x0, dtype=np.float64).ravel()
    n = len(x0_arr)

    constraints = []
    if g is not None:
        constraints.append({
            'type': 'ineq',
            'fun': lambda x: np.asarray(g(x), dtype=np.float64).ravel(),
        })
    if h is not None:
        constraints.append({
            'type': 'eq',
            'fun': lambda x: np.asarray(h(x), dtype=np.float64).ravel(),
        })

    bounds = None
    if lb is not None or ub is not None:
        lb_arr = np.full(n, -np.inf) if lb is None else np.asarray(lb, dtype=np.float64).ravel()
        ub_arr = np.full(n, np.inf) if ub is None else np.asarray(ub, dtype=np.float64).ravel()
        bounds = list(zip(lb_arr, ub_arr))

    result = minimize(phi, x0_arr, method='SLSQP',
                      constraints=constraints if constraints else (),
                      bounds=bounds)

    info = 1 if result.success else 0
    return (_wrap(result.x), result.fun, info)


@_tb("glpk")
def forge_glpk(c, A, b, lb=None, ub=None, ctype=None, vartype=None, sense=1):
    """Solve linear programming problem using scipy.

    [x, fval, status] = glpk(c, A, b, lb, ub, ctype, vartype, sense)

    Minimize (sense=1) or Maximize (sense=-1):  c' * x
    Subject to:
      A * x {<=, =, >=} b  (determined by ctype)
      lb <= x <= ub
      vartype: 'C' continuous, 'I' integer, 'B' binary

    ctype: string of constraint types, each char is:
      'U' -> A*x <= b  (upper bound)
      'S' -> A*x = b   (equality / fixed)
      'L' -> A*x >= b  (lower bound)

    Returns (x, fval, status) where status=0 is optimal.
    """
    from scipy.optimize import linprog

    c_arr = np.asarray(c, dtype=np.float64).ravel()
    A_arr = np.asarray(A, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64).ravel()
    n = len(c_arr)
    m = len(b_arr)

    # Apply sense (1 = minimize, -1 = maximize)
    obj = c_arr * float(sense)

    # Parse constraint types
    A_ub_list = []
    b_ub_list = []
    A_eq_list = []
    b_eq_list = []

    if ctype is None:
        # Default: all upper bounds (<=)
        ctype = "U" * m

    ctype_str = str(ctype)
    for i in range(m):
        ct = ctype_str[i] if i < len(ctype_str) else 'U'
        if ct in ('U', 'u'):
            # A*x <= b
            A_ub_list.append(A_arr[i])
            b_ub_list.append(b_arr[i])
        elif ct in ('S', 's', 'E', 'e', 'F', 'f'):
            # A*x = b
            A_eq_list.append(A_arr[i])
            b_eq_list.append(b_arr[i])
        elif ct in ('L', 'l'):
            # A*x >= b  ->  -A*x <= -b
            A_ub_list.append(-A_arr[i])
            b_ub_list.append(-b_arr[i])
        else:
            # Default to <=
            A_ub_list.append(A_arr[i])
            b_ub_list.append(b_arr[i])

    A_ub = np.array(A_ub_list) if A_ub_list else None
    b_ub = np.array(b_ub_list) if b_ub_list else None
    A_eq = np.array(A_eq_list) if A_eq_list else None
    b_eq = np.array(b_eq_list) if b_eq_list else None

    # Bounds
    if lb is None and ub is None:
        bounds_list = [(None, None)] * n
    else:
        lb_arr = np.full(n, -np.inf) if lb is None else np.asarray(lb, dtype=np.float64).ravel()
        ub_arr = np.full(n, np.inf) if ub is None else np.asarray(ub, dtype=np.float64).ravel()
        bounds_list = list(zip(lb_arr, ub_arr))

    # Handle integer/binary variables (scipy linprog only supports continuous)
    integrality = None
    if vartype is not None:
        vartype_str = str(vartype)
        int_arr = np.zeros(n, dtype=int)
        for i in range(min(n, len(vartype_str))):
            if vartype_str[i] in ('I', 'i', 'B', 'b'):
                int_arr[i] = 1
            if vartype_str[i] in ('B', 'b'):
                bounds_list[i] = (0, 1)
        if np.any(int_arr):
            integrality = int_arr

    kwargs = {}
    if integrality is not None:
        kwargs['integrality'] = integrality

    result = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds_list, **kwargs)

    if result.success:
        x_sol = result.x
        fval = float(sense) * result.fun  # undo sense transformation
        status = 0  # optimal
    else:
        x_sol = np.zeros(n)
        fval = np.inf
        status = 1  # failed

    return (_wrap(x_sol), fval, status)


@_tb("pqpnonneg")
def forge_pqpnonneg(C, d, x0=None):
    """Solve nonneg least squares using an active-set QP approach.

    x = pqpnonneg(C, d)
    [x, resnorm] = pqpnonneg(C, d)

    Equivalent to lsqnonneg but using QP formulation.
    min || C*x - d ||^2  subject to x >= 0.
    """
    from scipy.optimize import nnls

    C_arr = np.asarray(C, dtype=np.float64)
    d_arr = np.asarray(d, dtype=np.float64).ravel()

    x_sol, resnorm = nnls(C_arr, d_arr)
    return (_wrap(x_sol), resnorm)
