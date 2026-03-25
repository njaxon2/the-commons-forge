"""
Forge Symbolic Math Toolbox (lite)
===================================
Thin wrappers around SymPy providing an Octave/MATLAB-style interface
for symbolic computation.

Backend: sympy
"""

from typing import Union, Optional, Dict, Any, List
import numpy as np


def _import_sympy():
    import sympy
    return sympy


# ---------------------------------------------------------------------------
# Variable creation
# ---------------------------------------------------------------------------

def sym(name: str):
    """Create a single symbolic variable.

    Parameters
    ----------
    name : str – variable name (e.g. 'x')

    Returns
    -------
    sympy.Symbol
    """
    sp = _import_sympy()
    return sp.Symbol(name)


def syms(*names: str):
    """Create multiple symbolic variables.

    Parameters
    ----------
    *names : str – variable names (e.g. syms('x', 'y', 'z'))
              A single space-separated string is also accepted: syms('x y z')

    Returns
    -------
    tuple of sympy.Symbol
    """
    sp = _import_sympy()
    if len(names) == 1 and ' ' in names[0]:
        names = tuple(names[0].split())
    return sp.symbols(names)


# ---------------------------------------------------------------------------
# Calculus
# ---------------------------------------------------------------------------

def diff_sym(expr, var=None, n: int = 1):
    """Symbolic differentiation.

    Parameters
    ----------
    expr : sympy expression
    var  : symbol to differentiate with respect to (auto-detected if univariate)
    n    : order of derivative (default 1)
    """
    sp = _import_sympy()
    if var is None:
        free = list(expr.free_symbols)
        if len(free) != 1:
            raise ValueError("Ambiguous variable; pass var explicitly.")
        var = free[0]
    return sp.diff(expr, var, n)


def int_sym(expr, var=None, a=None, b=None):
    """Symbolic integration (indefinite or definite).

    Parameters
    ----------
    expr : sympy expression
    var  : integration variable (auto-detected if univariate)
    a, b : limits for definite integral (omit for indefinite)
    """
    sp = _import_sympy()
    if var is None:
        free = list(expr.free_symbols)
        if len(free) != 1:
            raise ValueError("Ambiguous variable; pass var explicitly.")
        var = free[0]
    if a is not None and b is not None:
        return sp.integrate(expr, (var, a, b))
    return sp.integrate(expr, var)


def limit_sym(expr, var, point, direction: str = '+-'):
    """Symbolic limit.

    Parameters
    ----------
    expr      : sympy expression
    var       : variable approaching the point
    point     : the limit point (can be sympy.oo for infinity)
    direction : '+-' (both), '+' (right), '-' (left)
    """
    sp = _import_sympy()
    dir_map = {'+-': '+-', '+': '+', '-': '-'}
    return sp.limit(expr, var, point, dir_map.get(direction, '+-'))


def taylor_sym(expr, var=None, point=0, order: int = 6):
    """Taylor / Maclaurin series expansion.

    Parameters
    ----------
    expr  : sympy expression
    var   : expansion variable
    point : expansion point (default 0)
    order : number of terms (default 6)
    """
    sp = _import_sympy()
    if var is None:
        free = list(expr.free_symbols)
        if len(free) != 1:
            raise ValueError("Ambiguous variable; pass var explicitly.")
        var = free[0]
    return sp.series(expr, var, point, order)


# ---------------------------------------------------------------------------
# Algebraic manipulation
# ---------------------------------------------------------------------------

def simplify_sym(expr):
    """Simplify a symbolic expression."""
    sp = _import_sympy()
    return sp.simplify(expr)


def expand_sym(expr):
    """Expand products and powers."""
    sp = _import_sympy()
    return sp.expand(expr)


def factor_sym(expr):
    """Factor a polynomial expression."""
    sp = _import_sympy()
    return sp.factor(expr)


def collect_sym(expr, var):
    """Collect terms with respect to a variable."""
    sp = _import_sympy()
    return sp.collect(expr, var)


# ---------------------------------------------------------------------------
# Solving
# ---------------------------------------------------------------------------

def solve_sym(expr, var=None):
    """Solve algebraic equation(s).

    Parameters
    ----------
    expr : expression (set equal to zero) or list of expressions
    var  : variable(s) to solve for
    """
    sp = _import_sympy()
    if var is None:
        if hasattr(expr, 'free_symbols'):
            free = list(expr.free_symbols)
        else:
            free = list(set().union(*(e.free_symbols for e in expr)))
        if len(free) == 1:
            var = free[0]
    return sp.solve(expr, var)


def dsolve_sym(ode, func=None):
    """Solve an ordinary differential equation.

    Parameters
    ----------
    ode  : sympy ODE expression (set equal to zero)
    func : the unknown function, e.g. sympy.Function('y')(x)
    """
    sp = _import_sympy()
    return sp.dsolve(ode, func)


# ---------------------------------------------------------------------------
# Substitution
# ---------------------------------------------------------------------------

def subs_sym(expr, old, new):
    """Substitute old -> new in expression.

    Parameters
    ----------
    expr : sympy expression
    old  : symbol or expression to replace (or dict of {old: new})
    new  : replacement (ignored if old is a dict)
    """
    if isinstance(old, dict):
        return expr.subs(old)
    return expr.subs(old, new)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def pretty_sym(expr) -> str:
    """Pretty-print a symbolic expression (Unicode text)."""
    sp = _import_sympy()
    return sp.pretty(expr, use_unicode=True)


def latex_sym(expr) -> str:
    """LaTeX representation of a symbolic expression."""
    sp = _import_sympy()
    return sp.latex(expr)


# ---------------------------------------------------------------------------
# Matrix calculus
# ---------------------------------------------------------------------------

def jacobian_sym(exprs, vars_list):
    """Jacobian matrix of vector-valued expression.

    Parameters
    ----------
    exprs     : list of sympy expressions (the vector function)
    vars_list : list of sympy symbols (variables)
    """
    sp = _import_sympy()
    F = sp.Matrix(exprs)
    V = sp.Matrix(vars_list)
    return F.jacobian(V)


def hessian_sym(expr, vars_list):
    """Hessian matrix of a scalar expression.

    Parameters
    ----------
    expr      : sympy expression
    vars_list : list of sympy symbols
    """
    sp = _import_sympy()
    return sp.hessian(expr, vars_list)


# ---------------------------------------------------------------------------
# Numeric conversion
# ---------------------------------------------------------------------------

def double_sym(expr) -> Union[float, np.ndarray]:
    """Convert symbolic expression to numeric float (or numpy array for Matrix)."""
    sp = _import_sympy()
    if isinstance(expr, sp.matrices.MatrixBase):
        return np.array(expr.tolist(), dtype=float)
    return float(expr.evalf())


def vpa_sym(expr, digits: int = 32):
    """Variable-precision arithmetic evaluation.

    Parameters
    ----------
    expr   : sympy expression
    digits : number of significant digits (default 32)
    """
    sp = _import_sympy()
    if isinstance(expr, sp.matrices.MatrixBase):
        return expr.evalf(digits)
    return expr.evalf(digits)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SYMBOLIC_REGISTRY: Dict[str, Dict[str, Any]] = {
    'sym':          {'func': sym,          'section': 'symbolic', 'desc': 'Create symbolic variable'},
    'syms':         {'func': syms,         'section': 'symbolic', 'desc': 'Create multiple symbolic variables'},
    'diff_sym':     {'func': diff_sym,     'section': 'symbolic', 'desc': 'Symbolic differentiation'},
    'int_sym':      {'func': int_sym,      'section': 'symbolic', 'desc': 'Symbolic integration'},
    'limit_sym':    {'func': limit_sym,    'section': 'symbolic', 'desc': 'Symbolic limit'},
    'taylor_sym':   {'func': taylor_sym,   'section': 'symbolic', 'desc': 'Taylor series expansion'},
    'simplify_sym': {'func': simplify_sym, 'section': 'symbolic', 'desc': 'Simplify expression'},
    'expand_sym':   {'func': expand_sym,   'section': 'symbolic', 'desc': 'Expand expression'},
    'factor_sym':   {'func': factor_sym,   'section': 'symbolic', 'desc': 'Factor polynomial'},
    'collect_sym':  {'func': collect_sym,  'section': 'symbolic', 'desc': 'Collect terms by variable'},
    'solve_sym':    {'func': solve_sym,    'section': 'symbolic', 'desc': 'Solve algebraic equations'},
    'dsolve_sym':   {'func': dsolve_sym,   'section': 'symbolic', 'desc': 'Solve ODE'},
    'subs_sym':     {'func': subs_sym,     'section': 'symbolic', 'desc': 'Substitute in expression'},
    'pretty_sym':   {'func': pretty_sym,   'section': 'symbolic', 'desc': 'Pretty-print expression'},
    'latex_sym':    {'func': latex_sym,     'section': 'symbolic', 'desc': 'LaTeX representation'},
    'jacobian_sym': {'func': jacobian_sym, 'section': 'symbolic', 'desc': 'Jacobian matrix'},
    'hessian_sym':  {'func': hessian_sym,  'section': 'symbolic', 'desc': 'Hessian matrix'},
    'double_sym':   {'func': double_sym,   'section': 'symbolic', 'desc': 'Convert to numeric float'},
    'vpa_sym':      {'func': vpa_sym,      'section': 'symbolic', 'desc': 'Variable-precision arithmetic'},
}
