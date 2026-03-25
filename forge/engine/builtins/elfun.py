"""Elementary functions (elfun toolbox).

Implements: acosd, acot, acotd, acoth, acsc, acscd, acsch, asec, asecd, asech,
asind, atan2d, atand, cosd, cospi, cot, cotd, coth, csc, cscd, csch,
sec, secd, sech, sind, sinpi, tand

SRS trace: SRS-FUNC-001 (Octave-compatible function library)
"""
import numpy as np
from forge.engine.types import ForgeArray, _unwrap


def _wrap(x):
    return ForgeArray(np.asarray(x, dtype=float))


def _deg2rad(x):
    return np.deg2rad(np.asarray(x, dtype=float))


def _rad2deg(x):
    return np.rad2deg(np.asarray(x, dtype=float))


# ── Trig in degrees ──────────────────────────────────────────────

def forge_sind(x):
    """Sine of argument in degrees."""
    return _wrap(np.sin(_deg2rad(_unwrap(x))))

def forge_cosd(x):
    """Cosine of argument in degrees."""
    return _wrap(np.cos(_deg2rad(_unwrap(x))))

def forge_tand(x):
    """Tangent of argument in degrees."""
    return _wrap(np.tan(_deg2rad(_unwrap(x))))

def forge_asind(x):
    """Inverse sine, result in degrees."""
    return _wrap(_rad2deg(np.arcsin(_unwrap(x))))

def forge_acosd(x):
    """Inverse cosine, result in degrees."""
    return _wrap(_rad2deg(np.arccos(_unwrap(x))))

def forge_atand(x):
    """Inverse tangent, result in degrees."""
    return _wrap(_rad2deg(np.arctan(_unwrap(x))))

def forge_atan2d(y, x):
    """Two-argument inverse tangent, result in degrees."""
    return _wrap(_rad2deg(np.arctan2(_unwrap(y), _unwrap(x))))


# ── Reciprocal trig ──────────────────────────────────────────────

def forge_sec(x):
    """Secant."""
    return _wrap(1.0 / np.cos(_unwrap(x)))

def forge_csc(x):
    """Cosecant."""
    return _wrap(1.0 / np.sin(_unwrap(x)))

def forge_cot(x):
    """Cotangent."""
    return _wrap(1.0 / np.tan(_unwrap(x)))

def forge_secd(x):
    """Secant of argument in degrees."""
    return _wrap(1.0 / np.cos(_deg2rad(_unwrap(x))))

def forge_cscd(x):
    """Cosecant of argument in degrees."""
    return _wrap(1.0 / np.sin(_deg2rad(_unwrap(x))))

def forge_cotd(x):
    """Cotangent of argument in degrees."""
    return _wrap(1.0 / np.tan(_deg2rad(_unwrap(x))))


# ── Reciprocal trig (inverse) in degrees ─────────────────────────

def forge_asecd(x):
    """Inverse secant, result in degrees."""
    return _wrap(_rad2deg(np.arccos(1.0 / _unwrap(x))))

def forge_acscd(x):
    """Inverse cosecant, result in degrees."""
    return _wrap(_rad2deg(np.arcsin(1.0 / _unwrap(x))))

def forge_acotd(x):
    """Inverse cotangent, result in degrees."""
    return _wrap(_rad2deg(np.arctan(1.0 / _unwrap(x))))


# ── Reciprocal trig (inverse) in radians ─────────────────────────

def forge_asec(x):
    """Inverse secant."""
    return _wrap(np.arccos(1.0 / _unwrap(x)))

def forge_acsc(x):
    """Inverse cosecant."""
    return _wrap(np.arcsin(1.0 / _unwrap(x)))

def forge_acot(x):
    """Inverse cotangent."""
    return _wrap(np.arctan(1.0 / _unwrap(x)))


# ── Hyperbolic reciprocal ────────────────────────────────────────

def forge_sech(x):
    """Hyperbolic secant."""
    return _wrap(1.0 / np.cosh(_unwrap(x)))

def forge_csch(x):
    """Hyperbolic cosecant."""
    return _wrap(1.0 / np.sinh(_unwrap(x)))

def forge_coth(x):
    """Hyperbolic cotangent."""
    return _wrap(1.0 / np.tanh(_unwrap(x)))

def forge_acsch(x):
    """Inverse hyperbolic cosecant."""
    v = _unwrap(x)
    return _wrap(np.arcsinh(1.0 / v))

def forge_asech(x):
    """Inverse hyperbolic secant."""
    v = _unwrap(x)
    return _wrap(np.arccosh(1.0 / v))

def forge_acoth(x):
    """Inverse hyperbolic cotangent."""
    v = _unwrap(x)
    return _wrap(np.arctanh(1.0 / v))


# ── Pi-scaled trig ───────────────────────────────────────────────

def forge_sinpi(x):
    """sin(pi * x), exact at integer multiples."""
    v = _unwrap(x)
    return _wrap(np.where(v == np.floor(v), 0.0, np.sin(np.pi * v)))

def forge_cospi(x):
    """cos(pi * x), exact at half-integer multiples."""
    v = _unwrap(x)
    half = v - 0.5
    return _wrap(np.where(half == np.floor(half), 0.0, np.cos(np.pi * v)))


# ── Registry for evaluator ───────────────────────────────────────

ELFUN_REGISTRY = {
    "sind": forge_sind, "cosd": forge_cosd, "tand": forge_tand,
    "asind": forge_asind, "acosd": forge_acosd, "atand": forge_atand,
    "atan2d": forge_atan2d,
    "sec": forge_sec, "csc": forge_csc, "cot": forge_cot,
    "secd": forge_secd, "cscd": forge_cscd, "cotd": forge_cotd,
    "asecd": forge_asecd, "acscd": forge_acscd, "acotd": forge_acotd,
    "asec": forge_asec, "acsc": forge_acsc, "acot": forge_acot,
    "sech": forge_sech, "csch": forge_csch, "coth": forge_coth,
    "acsch": forge_acsch, "asech": forge_asech, "acoth": forge_acoth,
    "sinpi": forge_sinpi, "cospi": forge_cospi,
}
