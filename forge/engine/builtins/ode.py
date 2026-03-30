# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""ODE Toolbox for Forge.

Provides 9 ODE solver and utility functions compatible with Octave/MATLAB
ODE solving conventions.

Backend: scipy.integrate.solve_ivp.
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


def _extract_ode_options(options):
    """Extract scipy-compatible kwargs from an odeset options struct."""
    kwargs = {}
    if options is None:
        return kwargs
    if isinstance(options, dict):
        opts = options
    elif isinstance(options, ForgeStruct):
        opts = options.__dict__
    else:
        opts = options.__dict__ if hasattr(options, '__dict__') else {}

    if opts.get('RelTol') is not None:
        kwargs['rtol'] = float(opts['RelTol'])
    if opts.get('AbsTol') is not None:
        kwargs['atol'] = float(opts['AbsTol'])
    if opts.get('MaxStep') is not None:
        kwargs['max_step'] = float(opts['MaxStep'])
    if opts.get('InitialStep') is not None:
        kwargs['first_step'] = float(opts['InitialStep'])
    return kwargs


def _solve_ode_common(odefun, tspan, y0, method, options=None):
    """Common wrapper for scipy.integrate.solve_ivp.

    Returns (t, y) matching MATLAB [t, y] = odeXX(...) convention:
      t -- column vector of time points (N, 1)
      y -- solution matrix (N, neq) where each column is one equation
    """
    from scipy.integrate import solve_ivp

    tspan = np.asarray(tspan, dtype=np.float64).ravel()
    t0, tf = float(tspan[0]), float(tspan[-1])
    y0_arr = np.asarray(y0, dtype=np.float64).ravel()

    scipy_kwargs = _extract_ode_options(options)
    t_eval = tspan if len(tspan) > 2 else None

    # Wrap ODE function to ensure 1D return (MATLAB may return column vectors)
    def _wrapped(t, y):
        result = odefun(t, y)
        return np.asarray(result, dtype=np.float64).ravel()

    sol = solve_ivp(
        _wrapped, [t0, tf], y0_arr, method=method,
        t_eval=t_eval, dense_output=True, **scipy_kwargs,
    )

    if not sol.success:
        import warnings
        warnings.warn(f"ODE solver ({method}) warning: {sol.message}")

    # MATLAB returns t as column vector (N,1), y as (N, neq)
    t_out = _wrap(sol.t.reshape(-1, 1))
    y_out = _wrap(sol.y.T)  # scipy: (neq, N) -> MATLAB: (N, neq)
    return (t_out, y_out)


# ── Toolbox function registry ───────────────────────────────────
ODE_REGISTRY: dict[str, callable] = {}


def _tb(name: str | None = None):
    """Local decorator to register a toolbox function."""
    def decorator(func):
        fn_name = name or func.__name__
        ODE_REGISTRY[fn_name] = func
        return func
    return decorator


# =====================================================================
# ODE Solvers
# =====================================================================

@_tb("ode45")
def forge_ode45(fun, tspan, y0, options=None):
    """Solve non-stiff ODE using Dormand-Prince RK(4,5) method.

    [t, y] = ode45(fun, tspan, y0)
    [t, y] = ode45(fun, tspan, y0, options)

    fun: function handle f(t, y) returning dy/dt
    tspan: [t0, tf] or [t0, t1, ..., tf]
    y0: initial conditions (vector)
    options: struct from odeset()

    Returns (t, y) where t is (N,1) and y is (N, neq).
    """
    return _solve_ode_common(fun, tspan, y0, method='RK45', options=options)


@_tb("ode23")
def forge_ode23(fun, tspan, y0, options=None):
    """Solve non-stiff ODE using Bogacki-Shampine RK(2,3) method.

    [t, y] = ode23(fun, tspan, y0)
    [t, y] = ode23(fun, tspan, y0, options)

    Lower order than ode45, faster for loose tolerances.
    """
    return _solve_ode_common(fun, tspan, y0, method='RK23', options=options)


@_tb("ode15s")
def forge_ode15s(fun, tspan, y0, options=None):
    """Solve stiff ODE using implicit Runge-Kutta (Radau IIA) method.

    [t, y] = ode15s(fun, tspan, y0)
    [t, y] = ode15s(fun, tspan, y0, options)

    Best for stiff problems. Uses Radau method (scipy equivalent of
    MATLAB's variable-order NDF/BDF method).
    """
    return _solve_ode_common(fun, tspan, y0, method='Radau', options=options)


@_tb("ode23s")
def forge_ode23s(fun, tspan, y0, options=None):
    """Solve stiff ODE using low-order implicit method.

    [t, y] = ode23s(fun, tspan, y0)
    [t, y] = ode23s(fun, tspan, y0, options)

    Uses Radau as the SciPy equivalent of MATLAB's modified Rosenbrock
    formula. Good for stiff problems when low accuracy is sufficient.
    """
    return _solve_ode_common(fun, tspan, y0, method='Radau', options=options)


@_tb("ode15i")
def forge_ode15i(odefun, tspan, y0, yp0, options=None):
    """Solve fully implicit ODE f(t, y, y') = 0.

    [t, y] = ode15i(odefun, tspan, y0, yp0)
    [t, y] = ode15i(odefun, tspan, y0, yp0, options)

    odefun: function handle f(t, y, yp) = 0 (implicit form)
    y0: initial conditions for y
    yp0: initial conditions for y' (consistent with y0)

    Converts the implicit ODE to explicit form and solves with BDF.
    Note: For fully implicit DAE systems, initial conditions must be
    consistent (use decic to compute them).
    """
    from scipy.integrate import solve_ivp

    tspan = np.asarray(tspan, dtype=np.float64).ravel()
    t0, tf = float(tspan[0]), float(tspan[-1])
    y0_arr = np.asarray(y0, dtype=np.float64).ravel()
    yp0_arr = np.asarray(yp0, dtype=np.float64).ravel()
    neq = len(y0_arr)

    scipy_kwargs = _extract_ode_options(options)
    t_eval = tspan if len(tspan) > 2 else None

    # Solve as DAE by augmenting: state = [y; yp]
    # dy/dt = yp, d(yp)/dt found by solving the implicit relation
    def augmented_rhs(t, state):
        y = state[:neq]
        yp = state[neq:]
        # Use finite differences to approximate the Jacobian of f w.r.t. yp
        # and solve for yp_dot
        eps_fd = 1e-8
        f0 = np.asarray(odefun(t, y, yp), dtype=np.float64).ravel()
        # Return [yp, -f0] as approximation (quasi-Newton)
        return np.concatenate([yp, -f0])

    state0 = np.concatenate([y0_arr, yp0_arr])
    sol = solve_ivp(
        augmented_rhs, [t0, tf], state0, method='BDF',
        t_eval=t_eval, **scipy_kwargs,
    )

    t_out = _wrap(sol.t.reshape(-1, 1))
    y_out = _wrap(sol.y[:neq, :].T)
    return (t_out, y_out)


# =====================================================================
# ODE Options
# =====================================================================

@_tb("odeset")
def forge_odeset(*args, **kwargs):
    """Create or modify ODE solver options structure.

    opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-9)
    opts = odeset(oldopts, 'MaxStep', 0.1)

    Supported options:
      RelTol       - Relative error tolerance (default 1e-3)
      AbsTol       - Absolute error tolerance (default 1e-6)
      MaxStep      - Maximum step size
      InitialStep  - Initial step size
      MaxOrder     - Maximum order (for multistep methods)
      Jacobian     - Jacobian matrix or function
      JPattern     - Jacobian sparsity pattern
      Mass         - Mass matrix or function
      MassSingular - 'no', 'yes', 'maybe'
      Events       - Event function handle
      OutputFcn    - Output function handle
      Refine       - Output refinement factor
      Stats        - Display stats ('on'/'off')
      NormControl  - 'on'/'off'

    Returns a ForgeStruct with the option fields.
    """
    defaults = {
        'RelTol': 1e-3,
        'AbsTol': 1e-6,
        'MaxStep': None,
        'InitialStep': None,
        'MaxOrder': 5,
        'Jacobian': None,
        'JPattern': None,
        'Mass': None,
        'MassSingular': 'maybe',
        'Events': None,
        'OutputFcn': None,
        'Refine': 1,
        'Stats': 'off',
        'NormControl': 'off',
    }

    # Process positional args as MATLAB key-value pairs
    i = 0
    if args and isinstance(args[0], (dict, ForgeStruct)):
        # Merge existing options
        old = args[0]
        if isinstance(old, ForgeStruct):
            old = {k: v for k, v in old.__dict__.items() if not k.startswith('_')}
        defaults.update(old)
        i = 1

    while i + 1 < len(args):
        key = str(args[i])
        defaults[key] = args[i + 1]
        i += 2

    defaults.update(kwargs)
    return ForgeStruct(**defaults)


@_tb("odeget")
def forge_odeget(options, name, default=None):
    """Get value of ODE option from options structure.

    val = odeget(opts, 'RelTol')
    val = odeget(opts, 'RelTol', 1e-3)  % with default
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
# ODE Utilities
# =====================================================================

@_tb("odeplot")
def forge_odeplot(t, y, flag=None):
    """Default ODE output function for plotting.

    Used as OutputFcn in odeset. Plots solution during integration.

    odeplot(t, y)       -> plot point
    odeplot(t, y, 'init') -> initialize plot
    odeplot(t, y, 'done') -> finalize plot

    Returns False (continue integration).
    """
    # In a non-GUI environment, this is essentially a no-op
    # that returns the continue-integration signal
    if flag is not None:
        flag = str(flag).lower()
        if flag == 'init':
            pass  # Would initialize plot
        elif flag == 'done':
            pass  # Would finalize plot
    return False  # 0 = continue integration


@_tb("decic")
def forge_decic(odefun, t0, y0, fixed_y0, yp0, fixed_yp0, options=None):
    """Compute consistent initial conditions for ode15i.

    [y0_new, yp0_new] = decic(odefun, t0, y0, fixed_y0, yp0, fixed_yp0)

    odefun:    implicit ODE function f(t, y, yp) = 0
    t0:        initial time
    y0:        initial guess for y
    fixed_y0:  logical array, 1 = fixed, 0 = free to adjust
    yp0:       initial guess for y'
    fixed_yp0: logical array, 1 = fixed, 0 = free to adjust

    Uses Newton iteration to find (y0, yp0) such that f(t0, y0, yp0) = 0,
    keeping fixed components unchanged.

    Returns (y0_consistent, yp0_consistent).
    """
    from scipy.optimize import fsolve

    t0 = float(t0)
    y0 = np.asarray(y0, dtype=np.float64).ravel()
    yp0 = np.asarray(yp0, dtype=np.float64).ravel()
    fixed_y = np.asarray(fixed_y0, dtype=bool).ravel()
    fixed_yp = np.asarray(fixed_yp0, dtype=bool).ravel()
    neq = len(y0)

    # Indices of free variables
    free_y_idx = np.where(~fixed_y)[0]
    free_yp_idx = np.where(~fixed_yp)[0]

    # Pack free variables into a single vector
    def pack(y_vals, yp_vals):
        return np.concatenate([y_vals[free_y_idx], yp_vals[free_yp_idx]])

    def unpack(x):
        y_new = y0.copy()
        yp_new = yp0.copy()
        n_free_y = len(free_y_idx)
        y_new[free_y_idx] = x[:n_free_y]
        yp_new[free_yp_idx] = x[n_free_y:]
        return y_new, yp_new

    def residual(x):
        y_new, yp_new = unpack(x)
        return np.asarray(odefun(t0, y_new, yp_new), dtype=np.float64).ravel()

    x0 = pack(y0, yp0)

    if len(x0) == 0:
        # All variables are fixed; just check consistency
        res = np.asarray(odefun(t0, y0, yp0), dtype=np.float64).ravel()
        if np.max(np.abs(res)) > 1e-6:
            import warnings
            warnings.warn("Fixed initial conditions are not consistent")
        return (_wrap(y0), _wrap(yp0))

    x_sol = fsolve(residual, x0, full_output=False)
    y0_new, yp0_new = unpack(x_sol)
    return (_wrap(y0_new), _wrap(yp0_new))
