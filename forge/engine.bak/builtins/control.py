# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""
Forge Control Systems Toolbox
MATLAB/Octave-compatible control systems functions using scipy.signal as backend.
"""

import numpy as np
from numpy import array, zeros, eye, diag, real, imag, conj, sqrt, log, log10, pi
import scipy.signal as sig
import scipy.linalg as la


# ============================================================================
# Helper utilities
# ============================================================================

def _ensure_tf(sys):
    """Convert any system representation to transfer function form."""
    if sys['type'] == 'tf':
        return sys
    elif sys['type'] == 'ss':
        return forge_ss2tf(sys)
    elif sys['type'] == 'zpk':
        return forge_zpk2tf(sys)
    else:
        raise ValueError(f"Unknown system type: {sys['type']}")


def _ensure_ss(sys):
    """Convert any system representation to state-space form."""
    if sys['type'] == 'ss':
        return sys
    elif sys['type'] == 'tf':
        return forge_tf2ss(sys)
    elif sys['type'] == 'zpk':
        return forge_zpk2ss(sys)
    else:
        raise ValueError(f"Unknown system type: {sys['type']}")


def _ensure_zpk(sys):
    """Convert any system representation to zero-pole-gain form."""
    if sys['type'] == 'zpk':
        return sys
    elif sys['type'] == 'tf':
        return forge_tf2zpk(sys)
    elif sys['type'] == 'ss':
        return forge_ss2zpk(sys)
    else:
        raise ValueError(f"Unknown system type: {sys['type']}")


def _default_time_vector(sys, n_points=1000):
    """Generate a default time vector based on system dynamics."""
    ss = _ensure_ss(sys)
    poles = np.linalg.eigvals(ss['A'])
    if len(poles) == 0:
        return np.linspace(0, 10, n_points)
    real_parts = np.abs(np.real(poles))
    real_parts = real_parts[real_parts > 0]
    if len(real_parts) == 0:
        t_max = 10.0
    else:
        t_max = 5.0 / np.min(real_parts)
    t_max = min(max(t_max, 1.0), 100.0)
    return np.linspace(0, t_max, n_points)


def _default_freq_vector(sys, n_points=500):
    """Generate a default frequency vector (rad/s) based on system dynamics."""
    ss = _ensure_ss(sys)
    poles = np.linalg.eigvals(ss['A'])
    if len(poles) == 0:
        return np.logspace(-2, 2, n_points)
    freqs = np.abs(poles)
    freqs = freqs[freqs > 0]
    if len(freqs) == 0:
        return np.logspace(-2, 2, n_points)
    w_min = np.min(freqs) / 10.0
    w_max = np.max(freqs) * 10.0
    return np.logspace(np.log10(w_min), np.log10(w_max), n_points)


# ============================================================================
# 1. System Representations (~15 functions)
# ============================================================================

def forge_tf(num, den, dt=None):
    """Create a transfer function representation.

    Parameters
    ----------
    num : array_like
        Numerator polynomial coefficients (descending powers of s).
    den : array_like
        Denominator polynomial coefficients (descending powers of s).
    dt : float or None
        Sampling time. None for continuous-time, float for discrete-time.

    Returns
    -------
    dict : Transfer function system {'type', 'num', 'den', 'dt'}.
    """
    num = np.atleast_1d(np.asarray(num, dtype=float))
    den = np.atleast_1d(np.asarray(den, dtype=float))
    return {'type': 'tf', 'num': num, 'den': den, 'dt': dt}


def forge_zpk(z, p, k, dt=None):
    """Create a zero-pole-gain representation.

    Parameters
    ----------
    z : array_like
        Zeros of the system.
    p : array_like
        Poles of the system.
    k : float
        System gain.
    dt : float or None
        Sampling time.

    Returns
    -------
    dict : ZPK system {'type', 'z', 'p', 'k', 'dt'}.
    """
    z = np.atleast_1d(np.asarray(z, dtype=complex))
    p = np.atleast_1d(np.asarray(p, dtype=complex))
    k = float(k)
    return {'type': 'zpk', 'z': z, 'p': p, 'k': k, 'dt': dt}


def forge_ss(A, B, C, D, dt=None):
    """Create a state-space representation.

    Parameters
    ----------
    A, B, C, D : array_like
        State-space matrices.
    dt : float or None
        Sampling time.

    Returns
    -------
    dict : State-space system {'type', 'A', 'B', 'C', 'D', 'dt'}.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    C = np.atleast_2d(np.asarray(C, dtype=float))
    D = np.atleast_2d(np.asarray(D, dtype=float))
    return {'type': 'ss', 'A': A, 'B': B, 'C': C, 'D': D, 'dt': dt}


def forge_frd(response, freq, dt=None):
    """Create a frequency response data model.

    Parameters
    ----------
    response : array_like
        Complex frequency response values.
    freq : array_like
        Frequency points (rad/s).
    dt : float or None
        Sampling time.

    Returns
    -------
    dict : FRD system {'type', 'response', 'freq', 'dt'}.
    """
    response = np.atleast_1d(np.asarray(response, dtype=complex))
    freq = np.atleast_1d(np.asarray(freq, dtype=float))
    return {'type': 'frd', 'response': response, 'freq': freq, 'dt': dt}


def forge_tf2ss(sys):
    """Convert transfer function to state-space representation."""
    num = np.atleast_1d(sys['num'])
    den = np.atleast_1d(sys['den'])
    A, B, C, D = sig.tf2ss(num, den)
    return forge_ss(A, B, C, D, dt=sys.get('dt'))


def forge_ss2tf(sys):
    """Convert state-space to transfer function representation."""
    A, B, C, D = sys['A'], sys['B'], sys['C'], sys['D']
    num, den = sig.ss2tf(A, B, C, D)
    # ss2tf may return 2-D num; take first row for SISO
    if num.ndim > 1:
        num = num[0]
    return forge_tf(num, den, dt=sys.get('dt'))


def forge_tf2zpk(sys):
    """Convert transfer function to zero-pole-gain representation."""
    num = np.atleast_1d(sys['num'])
    den = np.atleast_1d(sys['den'])
    z, p, k = sig.tf2zpk(num, den)
    return forge_zpk(z, p, k, dt=sys.get('dt'))


def forge_zpk2tf(sys):
    """Convert zero-pole-gain to transfer function representation."""
    z, p, k = sys['z'], sys['p'], sys['k']
    num, den = sig.zpk2tf(z, p, k)
    return forge_tf(num, den, dt=sys.get('dt'))


def forge_ss2zpk(sys):
    """Convert state-space to zero-pole-gain representation."""
    tf = forge_ss2tf(sys)
    return forge_tf2zpk(tf)


def forge_zpk2ss(sys):
    """Convert zero-pole-gain to state-space representation."""
    tf = forge_zpk2tf(sys)
    return forge_tf2ss(tf)


def forge_tfdata(sys):
    """Extract transfer function data from a system.

    Returns
    -------
    tuple : (num, den, dt)
    """
    tf = _ensure_tf(sys)
    return tf['num'].copy(), tf['den'].copy(), tf.get('dt')


def forge_ssdata(sys):
    """Extract state-space data from a system.

    Returns
    -------
    tuple : (A, B, C, D, dt)
    """
    ss = _ensure_ss(sys)
    return ss['A'].copy(), ss['B'].copy(), ss['C'].copy(), ss['D'].copy(), ss.get('dt')


def forge_zpkdata(sys):
    """Extract zero-pole-gain data from a system.

    Returns
    -------
    tuple : (z, p, k, dt)
    """
    zpk = _ensure_zpk(sys)
    return zpk['z'].copy(), zpk['p'].copy(), zpk['k'], zpk.get('dt')


def forge_minreal(sys, tol=1e-8):
    """Minimal realization via pole-zero cancellation.

    Parameters
    ----------
    sys : dict
        System representation.
    tol : float
        Tolerance for cancellation.

    Returns
    -------
    dict : Reduced system in the same representation type.
    """
    zpk = _ensure_zpk(sys)
    z = list(zpk['z'])
    p = list(zpk['p'])
    # Cancel matching pole-zero pairs
    z_remaining = []
    p_cancelled = list(p)
    for zi in z:
        cancelled = False
        for j, pj in enumerate(p_cancelled):
            if abs(zi - pj) < tol:
                p_cancelled.pop(j)
                cancelled = True
                break
        if not cancelled:
            z_remaining.append(zi)
    result_zpk = forge_zpk(np.array(z_remaining) if z_remaining else np.array([]),
                           np.array(p_cancelled) if p_cancelled else np.array([]),
                           zpk['k'], dt=zpk.get('dt'))
    if sys['type'] == 'tf':
        return forge_zpk2tf(result_zpk)
    elif sys['type'] == 'ss':
        return forge_zpk2ss(result_zpk)
    return result_zpk


# ============================================================================
# 2. System Interconnection (~8 functions)
# ============================================================================

def forge_series(sys1, sys2):
    """Cascade (series) connection: sys2 * sys1.

    Transfer function of the result is G2(s) * G1(s).

    Parameters
    ----------
    sys1, sys2 : dict
        System representations.

    Returns
    -------
    dict : Series-connected system (transfer function form).
    """
    tf1 = _ensure_tf(sys1)
    tf2 = _ensure_tf(sys2)
    num = np.polymul(tf1['num'], tf2['num'])
    den = np.polymul(tf1['den'], tf2['den'])
    dt = tf1.get('dt') or tf2.get('dt')
    return forge_tf(num, den, dt=dt)


def forge_parallel(sys1, sys2):
    """Parallel connection: sys1 + sys2.

    Parameters
    ----------
    sys1, sys2 : dict
        System representations.

    Returns
    -------
    dict : Parallel-connected system (transfer function form).
    """
    tf1 = _ensure_tf(sys1)
    tf2 = _ensure_tf(sys2)
    # G1/D1 + G2/D2 = (G1*D2 + G2*D1) / (D1*D2)
    num = np.polyadd(np.polymul(tf1['num'], tf2['den']),
                     np.polymul(tf2['num'], tf1['den']))
    den = np.polymul(tf1['den'], tf2['den'])
    dt = tf1.get('dt') or tf2.get('dt')
    return forge_tf(num, den, dt=dt)


def forge_feedback(sys1, sys2=None, sign=-1):
    """Feedback connection.

    Computes sys1 / (1 - sign * sys1 * sys2).
    Default is negative feedback (sign=-1).

    Parameters
    ----------
    sys1 : dict
        Forward path system.
    sys2 : dict or None
        Feedback path system. If None, unity feedback.
    sign : int
        -1 for negative feedback, +1 for positive feedback.

    Returns
    -------
    dict : Closed-loop system (transfer function form).
    """
    tf1 = _ensure_tf(sys1)
    if sys2 is None:
        tf2_num = np.array([1.0])
        tf2_den = np.array([1.0])
    else:
        tf2 = _ensure_tf(sys2)
        tf2_num = tf2['num']
        tf2_den = tf2['den']

    # CL = (N1*D2) / (D1*D2 - sign*N1*N2)
    num = np.polymul(tf1['num'], tf2_den)
    den_term1 = np.polymul(tf1['den'], tf2_den)
    den_term2 = np.polymul(tf1['num'], tf2_num)
    den = np.polysub(den_term1, sign * den_term2)
    dt = tf1.get('dt')
    return forge_tf(num, den, dt=dt)


def forge_connect(systems, connections, inputs, outputs):
    """General interconnection of systems.

    Parameters
    ----------
    systems : list of dict
        List of system representations.
    connections : list of tuples
        Each tuple (from_sys, from_out, to_sys, to_in) defining signal flow.
    inputs : list of tuples
        External input assignments (ext_in, to_sys, to_in).
    outputs : list of tuples
        External output assignments (from_sys, from_out, ext_out).

    Returns
    -------
    dict : Interconnected state-space system.
    """
    # Convert all to state-space
    ss_list = [_ensure_ss(s) for s in systems]

    # Build block-diagonal composite
    n_states = sum(s['A'].shape[0] for s in ss_list)
    n_inputs_total = sum(s['B'].shape[1] for s in ss_list)
    n_outputs_total = sum(s['C'].shape[0] for s in ss_list)

    A_big = np.zeros((n_states, n_states))
    B_big = np.zeros((n_states, n_inputs_total))
    C_big = np.zeros((n_outputs_total, n_states))
    D_big = np.zeros((n_outputs_total, n_inputs_total))

    r, c, ro, co = 0, 0, 0, 0
    offsets_state = []
    offsets_in = []
    offsets_out = []
    for s in ss_list:
        ns = s['A'].shape[0]
        ni = s['B'].shape[1]
        no = s['C'].shape[0]
        offsets_state.append(r)
        offsets_in.append(c)
        offsets_out.append(ro)
        A_big[r:r+ns, r:r+ns] = s['A']
        B_big[r:r+ns, c:c+ni] = s['B']
        C_big[ro:ro+no, r:r+ns] = s['C']
        D_big[ro:ro+no, c:c+ni] = s['D']
        r += ns
        c += ni
        ro += no

    # Apply connections by modifying the composite matrices
    for (fs, fo, ts, ti) in connections:
        out_idx = offsets_out[fs] + fo
        in_idx = offsets_in[ts] + ti
        # Route output of fs to input of ts through state feedback
        A_big[:, offsets_state[ts]:offsets_state[ts]+ss_list[ts]['A'].shape[0]] += (
            B_big[:, in_idx:in_idx+1] @ C_big[out_idx:out_idx+1, :])

    n_ext_in = max(e[0] for e in inputs) + 1 if inputs else 0
    n_ext_out = max(e[2] for e in outputs) + 1 if outputs else 0
    B_ext = np.zeros((n_states, n_ext_in))
    D_ext = np.zeros((n_ext_out, n_ext_in))
    C_ext = np.zeros((n_ext_out, n_states))

    for (ei, ts, ti) in inputs:
        in_idx = offsets_in[ts] + ti
        B_ext[:, ei] += B_big[:, in_idx]

    for (fs, fo, eo) in outputs:
        out_idx = offsets_out[fs] + fo
        C_ext[eo, :] += C_big[out_idx, :]

    return forge_ss(A_big, B_ext, C_ext, D_ext)


def forge_append(sys1, sys2):
    """Diagonal append of two systems.

    Parameters
    ----------
    sys1, sys2 : dict
        System representations.

    Returns
    -------
    dict : Block-diagonal state-space system.
    """
    ss1 = _ensure_ss(sys1)
    ss2 = _ensure_ss(sys2)
    n1 = ss1['A'].shape[0]
    n2 = ss2['A'].shape[0]
    m1 = ss1['B'].shape[1]
    m2 = ss2['B'].shape[1]
    p1 = ss1['C'].shape[0]
    p2 = ss2['C'].shape[0]

    A = np.zeros((n1+n2, n1+n2))
    B = np.zeros((n1+n2, m1+m2))
    C = np.zeros((p1+p2, n1+n2))
    D = np.zeros((p1+p2, m1+m2))

    A[:n1, :n1] = ss1['A']
    A[n1:, n1:] = ss2['A']
    B[:n1, :m1] = ss1['B']
    B[n1:, m1:] = ss2['B']
    C[:p1, :n1] = ss1['C']
    C[p1:, n1:] = ss2['C']
    D[:p1, :m1] = ss1['D']
    D[p1:, m1:] = ss2['D']

    dt = ss1.get('dt') or ss2.get('dt')
    return forge_ss(A, B, C, D, dt=dt)


def forge_blkdiag_sys(*systems):
    """Block diagonal combination of multiple systems.

    Parameters
    ----------
    *systems : dict
        Variable number of system representations.

    Returns
    -------
    dict : Block-diagonal state-space system.
    """
    if len(systems) == 0:
        return forge_ss(np.zeros((0, 0)), np.zeros((0, 0)),
                        np.zeros((0, 0)), np.zeros((0, 0)))
    result = systems[0]
    for s in systems[1:]:
        result = forge_append(result, s)
    return result


# ============================================================================
# 3. Time Response (~8 functions)
# ============================================================================

def forge_step(sys, t=None):
    """Step response of a system.

    Parameters
    ----------
    sys : dict
        System representation.
    t : array_like or None
        Time vector. Auto-generated if None.

    Returns
    -------
    tuple : (t, y) time and output arrays.
    """
    ss = _ensure_ss(sys)
    if t is None:
        t = _default_time_vector(sys)
    t = np.asarray(t, dtype=float)
    sys_tuple = (ss['A'], ss['B'], ss['C'], ss['D'])
    if ss.get('dt') is not None:
        tout, yout = sig.dstep((*sys_tuple, ss['dt']), t=t)
        yout = np.squeeze(np.array(yout))
        return np.asarray(tout), yout
    else:
        tout, yout = sig.step(sys_tuple, T=t)
        return tout, yout


def forge_impulse(sys, t=None):
    """Impulse response of a system.

    Parameters
    ----------
    sys : dict
        System representation.
    t : array_like or None
        Time vector.

    Returns
    -------
    tuple : (t, y) time and output arrays.
    """
    ss = _ensure_ss(sys)
    if t is None:
        t = _default_time_vector(sys)
    t = np.asarray(t, dtype=float)
    sys_tuple = (ss['A'], ss['B'], ss['C'], ss['D'])
    if ss.get('dt') is not None:
        tout, yout = sig.dimpulse((*sys_tuple, ss['dt']), t=t)
        yout = np.squeeze(np.array(yout))
        return np.asarray(tout), yout
    else:
        tout, yout = sig.impulse(sys_tuple, T=t)
        return tout, yout


def forge_lsim(sys, u, t, x0=None):
    """Simulate system response to arbitrary input.

    Parameters
    ----------
    sys : dict
        System representation.
    u : array_like
        Input signal array.
    t : array_like
        Time vector.
    x0 : array_like or None
        Initial state vector.

    Returns
    -------
    tuple : (t, y, x) time, output, and state arrays.
    """
    ss = _ensure_ss(sys)
    t = np.asarray(t, dtype=float)
    u = np.asarray(u, dtype=float)
    sys_tuple = (ss['A'], ss['B'], ss['C'], ss['D'])
    if x0 is not None:
        x0 = np.asarray(x0, dtype=float)
    tout, yout, xout = sig.lsim(sys_tuple, U=u, T=t, X0=x0)
    return tout, yout, xout


def forge_initial(sys, x0, t=None):
    """Initial condition response (zero input).

    Parameters
    ----------
    sys : dict
        System representation (state-space).
    x0 : array_like
        Initial state vector.
    t : array_like or None
        Time vector.

    Returns
    -------
    tuple : (t, y) time and output arrays.
    """
    ss = _ensure_ss(sys)
    if t is None:
        t = _default_time_vector(sys)
    t = np.asarray(t, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    n = len(t)
    u = np.zeros(n)
    sys_tuple = (ss['A'], ss['B'], ss['C'], ss['D'])
    tout, yout, _ = sig.lsim(sys_tuple, U=u, T=t, X0=x0)
    return tout, yout


def forge_stepinfo(sys, t=None):
    """Compute step response characteristics.

    Parameters
    ----------
    sys : dict
        System representation.
    t : array_like or None
        Time vector.

    Returns
    -------
    dict : Step response info with keys:
        'RiseTime', 'SettlingTime', 'SettlingMin', 'SettlingMax',
        'Overshoot', 'Undershoot', 'Peak', 'PeakTime', 'SteadyState'.
    """
    t_out, y = forge_step(sys, t)
    if len(y) == 0:
        return {}

    y_final = y[-1]
    y_init = 0.0

    info = {}
    info['SteadyState'] = y_final

    if abs(y_final - y_init) < 1e-12:
        return info

    # Normalize for analysis
    y_norm = (y - y_init) / (y_final - y_init)

    # Rise time: 10% to 90%
    idx_10 = np.where(y_norm >= 0.1)[0]
    idx_90 = np.where(y_norm >= 0.9)[0]
    if len(idx_10) > 0 and len(idx_90) > 0:
        info['RiseTime'] = t_out[idx_90[0]] - t_out[idx_10[0]]
    else:
        info['RiseTime'] = float('inf')

    # Settling time: last time outside 2% band
    tol = 0.02
    settled = np.abs(y_norm - 1.0) <= tol
    not_settled = np.where(~settled)[0]
    if len(not_settled) > 0:
        info['SettlingTime'] = t_out[not_settled[-1]]
    else:
        info['SettlingTime'] = t_out[0]

    # Peak
    peak_idx = np.argmax(np.abs(y))
    info['Peak'] = float(y[peak_idx])
    info['PeakTime'] = float(t_out[peak_idx])

    # Overshoot / Undershoot
    if y_final > y_init:
        overshoot = (np.max(y) - y_final) / (y_final - y_init) * 100.0
        undershoot = (y_init - np.min(y)) / (y_final - y_init) * 100.0
    else:
        overshoot = (y_final - np.min(y)) / (y_init - y_final) * 100.0
        undershoot = (np.max(y) - y_init) / (y_init - y_final) * 100.0
    info['Overshoot'] = max(0.0, overshoot)
    info['Undershoot'] = max(0.0, undershoot)

    # Settling min/max
    info['SettlingMin'] = float(np.min(y))
    info['SettlingMax'] = float(np.max(y))

    return info


# ============================================================================
# 4. Frequency Response (~10 functions)
# ============================================================================

def forge_bode(sys, w=None):
    """Bode plot data (magnitude and phase).

    Parameters
    ----------
    sys : dict
        System representation.
    w : array_like or None
        Frequency vector (rad/s).

    Returns
    -------
    tuple : (w, mag, phase) where mag is in dB and phase in degrees.
    """
    tf = _ensure_tf(sys)
    if w is None:
        w = _default_freq_vector(sys)
    w = np.asarray(w, dtype=float)
    w_out, H = sig.freqs(tf['num'], tf['den'], worN=w)
    mag_db = 20.0 * np.log10(np.maximum(np.abs(H), 1e-30))
    phase_deg = np.degrees(np.unwrap(np.angle(H)))
    return w_out, mag_db, phase_deg


def forge_nyquist(sys, w=None):
    """Nyquist plot data.

    Parameters
    ----------
    sys : dict
        System representation.
    w : array_like or None
        Frequency vector (rad/s).

    Returns
    -------
    tuple : (real, imag, w) real and imaginary parts of frequency response.
    """
    tf = _ensure_tf(sys)
    if w is None:
        w = _default_freq_vector(sys)
    w = np.asarray(w, dtype=float)
    _, H = sig.freqs(tf['num'], tf['den'], worN=w)
    return np.real(H), np.imag(H), w


def forge_nichols(sys, w=None):
    """Nichols chart data.

    Parameters
    ----------
    sys : dict
        System representation.
    w : array_like or None
        Frequency vector (rad/s).

    Returns
    -------
    tuple : (phase_deg, mag_db, w) open-loop phase and magnitude.
    """
    w_out, mag_db, phase_deg = forge_bode(sys, w)
    return phase_deg, mag_db, w_out


def forge_margin(sys):
    """Compute gain and phase margins.

    Parameters
    ----------
    sys : dict
        System representation.

    Returns
    -------
    dict : {'GainMargin', 'PhaseMargin', 'GainCrossoverFreq', 'PhaseCrossoverFreq'}
        GainMargin in dB, PhaseMargin in degrees.
    """
    tf = _ensure_tf(sys)
    w = _default_freq_vector(sys, n_points=2000)
    _, H = sig.freqs(tf['num'], tf['den'], worN=w)
    mag = np.abs(H)
    phase = np.degrees(np.unwrap(np.angle(H)))

    result = {
        'GainMargin': float('inf'),
        'PhaseMargin': float('inf'),
        'GainCrossoverFreq': float('nan'),
        'PhaseCrossoverFreq': float('nan'),
    }

    # Gain crossover: |G(jw)| = 1 (0 dB)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-30))
    crossings = np.where(np.diff(np.sign(mag_db)))[0]
    if len(crossings) > 0:
        idx = crossings[0]
        # Linear interpolation
        frac = -mag_db[idx] / (mag_db[idx+1] - mag_db[idx]) if mag_db[idx+1] != mag_db[idx] else 0
        wc = w[idx] + frac * (w[idx+1] - w[idx])
        phase_at_wc = phase[idx] + frac * (phase[idx+1] - phase[idx])
        result['PhaseMargin'] = 180.0 + phase_at_wc
        result['GainCrossoverFreq'] = wc

    # Phase crossover: angle(G(jw)) = -180 deg
    phase_shifted = phase + 180.0
    crossings = np.where(np.diff(np.sign(phase_shifted)))[0]
    if len(crossings) > 0:
        idx = crossings[0]
        frac = -phase_shifted[idx] / (phase_shifted[idx+1] - phase_shifted[idx]) \
            if phase_shifted[idx+1] != phase_shifted[idx] else 0
        wp = w[idx] + frac * (w[idx+1] - w[idx])
        mag_at_wp = mag_db[idx] + frac * (mag_db[idx+1] - mag_db[idx])
        result['GainMargin'] = -mag_at_wp  # in dB
        result['PhaseCrossoverFreq'] = wp

    return result


def forge_bandwidth(sys, dbdrop=-3.0):
    """Compute bandwidth (-3 dB frequency).

    Parameters
    ----------
    sys : dict
        System representation.
    dbdrop : float
        Magnitude drop in dB defining bandwidth (default -3 dB).

    Returns
    -------
    float : Bandwidth in rad/s.
    """
    tf = _ensure_tf(sys)
    w = _default_freq_vector(sys, n_points=2000)
    _, H = sig.freqs(tf['num'], tf['den'], worN=w)
    mag = np.abs(H)
    dc_gain = mag[0] if mag[0] > 0 else 1.0
    mag_db = 20.0 * np.log10(np.maximum(mag / dc_gain, 1e-30))

    below = np.where(mag_db < dbdrop)[0]
    if len(below) == 0:
        return float('inf')
    return float(w[below[0]])


def forge_evalfr(sys, s):
    """Evaluate system frequency response at a complex frequency.

    Parameters
    ----------
    sys : dict
        System representation.
    s : complex
        Complex frequency point.

    Returns
    -------
    complex : System response at the given frequency.
    """
    tf = _ensure_tf(sys)
    num_val = np.polyval(tf['num'], s)
    den_val = np.polyval(tf['den'], s)
    if abs(den_val) < 1e-30:
        return complex(float('inf'), 0)
    return num_val / den_val


def forge_freqresp(sys, w=None):
    """Compute frequency response.

    Parameters
    ----------
    sys : dict
        System representation.
    w : array_like or None
        Frequency vector (rad/s).

    Returns
    -------
    tuple : (w, H) frequency vector and complex response.
    """
    tf = _ensure_tf(sys)
    if w is None:
        w = _default_freq_vector(sys)
    w = np.asarray(w, dtype=float)
    w_out, H = sig.freqs(tf['num'], tf['den'], worN=w)
    return w_out, H


# ============================================================================
# 5. Stability & Analysis (~10 functions)
# ============================================================================

def forge_pole(sys):
    """Compute system poles.

    Parameters
    ----------
    sys : dict
        System representation.

    Returns
    -------
    ndarray : Array of poles (complex).
    """
    if sys['type'] == 'zpk':
        return np.array(sys['p'], dtype=complex)
    if sys['type'] == 'tf':
        return np.roots(sys['den'])
    if sys['type'] == 'ss':
        return np.linalg.eigvals(sys['A'])
    raise ValueError(f"Unknown system type: {sys['type']}")


def forge_zero(sys):
    """Compute system zeros.

    Parameters
    ----------
    sys : dict
        System representation.

    Returns
    -------
    ndarray : Array of zeros (complex).
    """
    if sys['type'] == 'zpk':
        return np.array(sys['z'], dtype=complex)
    if sys['type'] == 'tf':
        return np.roots(sys['num'])
    if sys['type'] == 'ss':
        tf = forge_ss2tf(sys)
        return np.roots(tf['num'])
    raise ValueError(f"Unknown system type: {sys['type']}")


def forge_dcgain(sys):
    """Compute DC gain (steady-state gain) of a system.

    Parameters
    ----------
    sys : dict
        System representation.

    Returns
    -------
    float : DC gain value.
    """
    tf = _ensure_tf(sys)
    num_dc = np.polyval(tf['num'], 0.0)
    den_dc = np.polyval(tf['den'], 0.0)
    if abs(den_dc) < 1e-30:
        return float('inf') if num_dc >= 0 else float('-inf')
    return float(num_dc / den_dc)


def forge_pzmap(sys):
    """Compute pole-zero map data.

    Parameters
    ----------
    sys : dict
        System representation.

    Returns
    -------
    tuple : (poles, zeros) arrays of complex values.
    """
    return forge_pole(sys), forge_zero(sys)


def forge_rlocus(sys, k=None):
    """Root locus data.

    Computes the closed-loop pole locations as gain k varies.

    Parameters
    ----------
    sys : dict
        Open-loop system.
    k : array_like or None
        Gain values. Auto-generated if None.

    Returns
    -------
    tuple : (roots, k) arrays of closed-loop poles and corresponding gains.
    """
    tf = _ensure_tf(sys)
    num = tf['num']
    den = tf['den']

    if k is None:
        k = np.concatenate([
            np.linspace(0, 1, 50),
            np.logspace(0, 3, 200),
        ])
    k = np.asarray(k, dtype=float)

    n_poles = len(den) - 1
    roots_array = np.zeros((len(k), n_poles), dtype=complex)

    for i, ki in enumerate(k):
        cl_den = np.polyadd(den, ki * np.pad(num, (max(0, len(den) - len(num)), 0)))
        r = np.roots(cl_den)
        # Sort for continuity
        r = np.sort_complex(r)
        if len(r) < n_poles:
            r = np.pad(r, (0, n_poles - len(r)), constant_values=np.nan)
        roots_array[i, :] = r[:n_poles]

    return roots_array, k


def forge_isstable(sys):
    """Check if a system is stable.

    For continuous-time: all poles must have negative real parts.
    For discrete-time: all poles must be inside the unit circle.

    Parameters
    ----------
    sys : dict
        System representation.

    Returns
    -------
    bool : True if the system is stable.
    """
    p = forge_pole(sys)
    if len(p) == 0:
        return True
    dt = sys.get('dt')
    if dt is not None and dt > 0:
        # Discrete-time: poles inside unit circle
        return bool(np.all(np.abs(p) < 1.0))
    else:
        # Continuous-time: poles in left half-plane
        return bool(np.all(np.real(p) < 0.0))


def forge_ctrb(A, B):
    """Compute the controllability matrix.

    Parameters
    ----------
    A : array_like
        State matrix (n x n).
    B : array_like
        Input matrix (n x m).

    Returns
    -------
    ndarray : Controllability matrix [B, A*B, A^2*B, ..., A^(n-1)*B].
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    n = A.shape[0]
    C_mat = B.copy()
    AB = B.copy()
    for i in range(1, n):
        AB = A @ AB
        C_mat = np.hstack([C_mat, AB])
    return C_mat


def forge_obsv(A, C):
    """Compute the observability matrix.

    Parameters
    ----------
    A : array_like
        State matrix (n x n).
    C : array_like
        Output matrix (p x n).

    Returns
    -------
    ndarray : Observability matrix [C; C*A; C*A^2; ...; C*A^(n-1)].
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    C = np.atleast_2d(np.asarray(C, dtype=float))
    n = A.shape[0]
    O_mat = C.copy()
    CA = C.copy()
    for i in range(1, n):
        CA = CA @ A
        O_mat = np.vstack([O_mat, CA])
    return O_mat


def forge_gram(sys, gram_type):
    """Compute Gramian matrix.

    Parameters
    ----------
    sys : dict
        State-space system.
    gram_type : str
        'c' for controllability Gramian, 'o' for observability Gramian.

    Returns
    -------
    ndarray : Gramian matrix.
    """
    ss = _ensure_ss(sys)
    A, B, C = ss['A'], ss['B'], ss['C']

    if gram_type.lower() in ('c', 'controllability'):
        # Solve A*Wc + Wc*A' + B*B' = 0
        Q = B @ B.T
        return la.solve_continuous_lyapunov(A, -Q)
    elif gram_type.lower() in ('o', 'observability'):
        # Solve A'*Wo + Wo*A + C'*C = 0
        Q = C.T @ C
        return la.solve_continuous_lyapunov(A.T, -Q)
    else:
        raise ValueError(f"Unknown Gramian type: {gram_type}. Use 'c' or 'o'.")


def forge_lyap(A, Q):
    """Solve continuous Lyapunov equation A*X + X*A' + Q = 0.

    Parameters
    ----------
    A : array_like
        System matrix.
    Q : array_like
        Symmetric positive (semi-)definite matrix.

    Returns
    -------
    ndarray : Solution matrix X.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    return la.solve_continuous_lyapunov(A, -Q)


def forge_dlyap(A, Q):
    """Solve discrete Lyapunov equation A*X*A' - X + Q = 0.

    Parameters
    ----------
    A : array_like
        System matrix.
    Q : array_like
        Symmetric positive (semi-)definite matrix.

    Returns
    -------
    ndarray : Solution matrix X.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    return la.solve_discrete_lyapunov(A, Q)


# ============================================================================
# 6. Controller Design (~8 functions)
# ============================================================================

def forge_pid(Kp, Ki=0.0, Kd=0.0, Tf=0.0, dt=None):
    """Create a PID controller as a transfer function.

    C(s) = Kp + Ki/s + Kd*s/(Tf*s + 1)

    If Tf=0, derivative term is pure: C(s) = Kp + Ki/s + Kd*s
    which becomes (Kd*s^2 + Kp*s + Ki) / s.

    Parameters
    ----------
    Kp : float
        Proportional gain.
    Ki : float
        Integral gain.
    Kd : float
        Derivative gain.
    Tf : float
        Derivative filter time constant.
    dt : float or None
        Sampling time.

    Returns
    -------
    dict : Transfer function of the PID controller.
    """
    if Tf > 0:
        # C(s) = [Kd*s^2 + (Kp*Tf + Kp + Ki*Tf... let's derive properly)]
        # Kp + Ki/s + Kd*s/(Tf*s+1)
        # = [Kp*s*(Tf*s+1) + Ki*(Tf*s+1) + Kd*s^2] / [s*(Tf*s+1)]
        # Num: (Kp*Tf + Kd)*s^2 + (Kp + Ki*Tf)*s + Ki
        # Den: Tf*s^2 + s
        num = np.array([Kp * Tf + Kd, Kp + Ki * Tf, Ki])
        den = np.array([Tf, 1.0, 0.0])
    else:
        # C(s) = (Kd*s^2 + Kp*s + Ki) / s
        num = np.array([Kd, Kp, Ki])
        den = np.array([1.0, 0.0])
    # Remove leading zeros from num
    while len(num) > 1 and abs(num[0]) < 1e-15:
        num = num[1:]
    return forge_tf(num, den, dt=dt)


def forge_pidtune(sys, controller_type='pid'):
    """Auto-tune a PID controller using Ziegler-Nichols-like heuristics.

    Parameters
    ----------
    sys : dict
        Plant system representation.
    controller_type : str
        'p', 'pi', 'pd', or 'pid'.

    Returns
    -------
    dict : Tuned PID controller as transfer function.
    """
    # Find ultimate gain and frequency via gain margin
    margins = forge_margin(sys)
    wc = margins.get('PhaseCrossoverFreq', float('nan'))
    gm = margins.get('GainMargin', float('inf'))

    if np.isnan(wc) or np.isinf(gm) or gm <= 0:
        # Fallback: use bandwidth-based tuning
        bw = forge_bandwidth(sys)
        if np.isinf(bw):
            bw = 1.0
        Ku = 1.0 / max(abs(forge_dcgain(sys)), 1e-10)
        Tu = 2.0 * np.pi / bw
    else:
        Ku = 10.0 ** (gm / 20.0)  # Convert dB to linear
        Tu = 2.0 * np.pi / wc

    ctype = controller_type.lower()
    if ctype == 'p':
        Kp = 0.5 * Ku
        return forge_pid(Kp, 0.0, 0.0)
    elif ctype == 'pi':
        Kp = 0.45 * Ku
        Ki = Kp / (Tu / 1.2)
        return forge_pid(Kp, Ki, 0.0)
    elif ctype == 'pd':
        Kp = 0.8 * Ku
        Kd = Kp * Tu / 8.0
        return forge_pid(Kp, 0.0, Kd)
    elif ctype == 'pid':
        Kp = 0.6 * Ku
        Ki = Kp / (Tu / 2.0)
        Kd = Kp * Tu / 8.0
        return forge_pid(Kp, Ki, Kd)
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")


def forge_place(A, B, p):
    """Pole placement for state feedback.

    Computes gain matrix K such that eig(A - B*K) = p.

    Parameters
    ----------
    A : array_like
        State matrix (n x n).
    B : array_like
        Input matrix (n x m).
    p : array_like
        Desired closed-loop pole locations.

    Returns
    -------
    ndarray : State feedback gain matrix K.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    p = np.atleast_1d(np.asarray(p, dtype=complex))
    result = sig.place_poles(A, B, p)
    return result.gain_matrix


def forge_acker(A, B, p):
    """Ackermann's formula for pole placement (SISO systems only).

    Parameters
    ----------
    A : array_like
        State matrix (n x n).
    B : array_like
        Input matrix (n x 1).
    p : array_like
        Desired closed-loop pole locations (n values).

    Returns
    -------
    ndarray : State feedback gain vector K (1 x n).
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    p = np.atleast_1d(np.asarray(p, dtype=complex))
    n = A.shape[0]

    # Controllability matrix
    Cm = forge_ctrb(A, B)
    if np.linalg.matrix_rank(Cm) < n:
        raise ValueError("System is not controllable; Ackermann's formula cannot be applied.")

    # Desired characteristic polynomial: alpha(A)
    # coeffs of (s - p1)(s - p2)...(s - pn)
    poly_coeffs = np.real(np.poly(p))  # [1, a1, a2, ..., an]
    alpha_A = np.zeros_like(A, dtype=float)
    A_pow = np.eye(n)
    for i in range(n, -1, -1):
        alpha_A += poly_coeffs[n - i] * A_pow
        if i > 0:
            A_pow = A_pow @ A

    # K = [0 0 ... 0 1] * inv(Cm) * alpha(A)
    e_n = np.zeros((1, n))
    e_n[0, -1] = 1.0
    K = e_n @ np.linalg.inv(Cm) @ alpha_A
    return np.real(K)


def forge_lqr(A, B, Q, R, N=None):
    """Linear Quadratic Regulator (continuous-time).

    Minimizes J = integral(x'Qx + u'Ru + 2x'Nu) dt.

    Parameters
    ----------
    A : array_like
        State matrix.
    B : array_like
        Input matrix.
    Q : array_like
        State weighting matrix.
    R : array_like
        Input weighting matrix.
    N : array_like or None
        Cross-weighting matrix.

    Returns
    -------
    tuple : (K, S, E) where K is the gain, S is the Riccati solution,
            and E are the closed-loop eigenvalues.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    R = np.atleast_2d(np.asarray(R, dtype=float))

    if N is not None:
        N = np.atleast_2d(np.asarray(N, dtype=float))
        A_mod = A - B @ np.linalg.inv(R) @ N.T
        Q_mod = Q - N @ np.linalg.inv(R) @ N.T
    else:
        A_mod = A
        Q_mod = Q

    # Solve continuous algebraic Riccati equation
    S = la.solve_continuous_are(A_mod, B, Q_mod, R)
    K = np.linalg.inv(R) @ B.T @ S
    if N is not None:
        K = K + np.linalg.inv(R) @ N.T
    E = np.linalg.eigvals(A - B @ K)
    return K, S, E


def forge_dlqr(A, B, Q, R, N=None):
    """Linear Quadratic Regulator (discrete-time).

    Parameters
    ----------
    A : array_like
        State matrix.
    B : array_like
        Input matrix.
    Q : array_like
        State weighting matrix.
    R : array_like
        Input weighting matrix.
    N : array_like or None
        Cross-weighting matrix.

    Returns
    -------
    tuple : (K, S, E) gain, Riccati solution, closed-loop eigenvalues.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    R = np.atleast_2d(np.asarray(R, dtype=float))

    if N is not None:
        N = np.atleast_2d(np.asarray(N, dtype=float))
        A_mod = A - B @ np.linalg.inv(R) @ N.T
        Q_mod = Q - N @ np.linalg.inv(R) @ N.T
    else:
        A_mod = A
        Q_mod = Q

    S = la.solve_discrete_are(A_mod, B, Q_mod, R)
    K = np.linalg.inv(R + B.T @ S @ B) @ (B.T @ S @ A)
    if N is not None:
        K = K + np.linalg.inv(R) @ N.T
    E = np.linalg.eigvals(A - B @ K)
    return K, S, E


def forge_kalman(A, C, Q, R):
    """Compute Kalman filter gain for continuous-time system.

    Parameters
    ----------
    A : array_like
        State matrix (n x n).
    C : array_like
        Output matrix (p x n).
    Q : array_like
        Process noise covariance (n x n).
    R : array_like
        Measurement noise covariance (p x p).

    Returns
    -------
    tuple : (L, P, E) where L is the Kalman gain, P is the error covariance,
            and E are the estimator eigenvalues.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    C = np.atleast_2d(np.asarray(C, dtype=float))
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    R = np.atleast_2d(np.asarray(R, dtype=float))

    # Dual of LQR: solve Riccati for the observer
    # A*P + P*A' - P*C'*inv(R)*C*P + Q = 0
    P = la.solve_continuous_are(A.T, C.T, Q, R)
    L = P @ C.T @ np.linalg.inv(R)
    E = np.linalg.eigvals(A - L @ C)
    return L, P, E


# ============================================================================
# CONTROL_REGISTRY
# ============================================================================

CONTROL_REGISTRY = {
    # --- System Representations ---
    'tf': {
        'func': forge_tf,
        'args': ['num', 'den'],
        'opts': {'dt': None},
        'desc': 'Create transfer function representation',
        'category': 'system_representation',
    },
    'zpk': {
        'func': forge_zpk,
        'args': ['z', 'p', 'k'],
        'opts': {'dt': None},
        'desc': 'Create zero-pole-gain representation',
        'category': 'system_representation',
    },
    'ss': {
        'func': forge_ss,
        'args': ['A', 'B', 'C', 'D'],
        'opts': {'dt': None},
        'desc': 'Create state-space representation',
        'category': 'system_representation',
    },
    'frd': {
        'func': forge_frd,
        'args': ['response', 'freq'],
        'opts': {'dt': None},
        'desc': 'Create frequency response data model',
        'category': 'system_representation',
    },
    'tf2ss': {
        'func': forge_tf2ss,
        'args': ['sys'],
        'desc': 'Convert transfer function to state-space',
        'category': 'system_representation',
    },
    'ss2tf': {
        'func': forge_ss2tf,
        'args': ['sys'],
        'desc': 'Convert state-space to transfer function',
        'category': 'system_representation',
    },
    'tf2zpk': {
        'func': forge_tf2zpk,
        'args': ['sys'],
        'desc': 'Convert transfer function to zero-pole-gain',
        'category': 'system_representation',
    },
    'zpk2tf': {
        'func': forge_zpk2tf,
        'args': ['sys'],
        'desc': 'Convert zero-pole-gain to transfer function',
        'category': 'system_representation',
    },
    'ss2zpk': {
        'func': forge_ss2zpk,
        'args': ['sys'],
        'desc': 'Convert state-space to zero-pole-gain',
        'category': 'system_representation',
    },
    'zpk2ss': {
        'func': forge_zpk2ss,
        'args': ['sys'],
        'desc': 'Convert zero-pole-gain to state-space',
        'category': 'system_representation',
    },
    'tfdata': {
        'func': forge_tfdata,
        'args': ['sys'],
        'desc': 'Extract transfer function data from system',
        'category': 'system_representation',
    },
    'ssdata': {
        'func': forge_ssdata,
        'args': ['sys'],
        'desc': 'Extract state-space data from system',
        'category': 'system_representation',
    },
    'zpkdata': {
        'func': forge_zpkdata,
        'args': ['sys'],
        'desc': 'Extract zero-pole-gain data from system',
        'category': 'system_representation',
    },
    'minreal': {
        'func': forge_minreal,
        'args': ['sys'],
        'opts': {'tol': 1e-8},
        'desc': 'Minimal realization via pole-zero cancellation',
        'category': 'system_representation',
    },

    # --- System Interconnection ---
    'series': {
        'func': forge_series,
        'args': ['sys1', 'sys2'],
        'desc': 'Cascade (series) connection of two systems',
        'category': 'interconnection',
    },
    'parallel': {
        'func': forge_parallel,
        'args': ['sys1', 'sys2'],
        'desc': 'Parallel connection of two systems',
        'category': 'interconnection',
    },
    'feedback': {
        'func': forge_feedback,
        'args': ['sys1'],
        'opts': {'sys2': None, 'sign': -1},
        'desc': 'Feedback connection (default negative)',
        'category': 'interconnection',
    },
    'connect': {
        'func': forge_connect,
        'args': ['systems', 'connections', 'inputs', 'outputs'],
        'desc': 'General interconnection of systems',
        'category': 'interconnection',
    },
    'append': {
        'func': forge_append,
        'args': ['sys1', 'sys2'],
        'desc': 'Diagonal append of two systems',
        'category': 'interconnection',
    },
    'blkdiag_sys': {
        'func': forge_blkdiag_sys,
        'args': ['*systems'],
        'desc': 'Block diagonal combination of multiple systems',
        'category': 'interconnection',
    },

    # --- Time Response ---
    'step': {
        'func': forge_step,
        'args': ['sys'],
        'opts': {'t': None},
        'desc': 'Step response of a system',
        'category': 'time_response',
    },
    'impulse': {
        'func': forge_impulse,
        'args': ['sys'],
        'opts': {'t': None},
        'desc': 'Impulse response of a system',
        'category': 'time_response',
    },
    'lsim': {
        'func': forge_lsim,
        'args': ['sys', 'u', 't'],
        'opts': {'x0': None},
        'desc': 'Simulate system response to arbitrary input',
        'category': 'time_response',
    },
    'initial': {
        'func': forge_initial,
        'args': ['sys', 'x0'],
        'opts': {'t': None},
        'desc': 'Initial condition response (zero input)',
        'category': 'time_response',
    },
    'stepinfo': {
        'func': forge_stepinfo,
        'args': ['sys'],
        'opts': {'t': None},
        'desc': 'Step response characteristics (rise time, overshoot, etc.)',
        'category': 'time_response',
    },

    # --- Frequency Response ---
    'bode': {
        'func': forge_bode,
        'args': ['sys'],
        'opts': {'w': None},
        'desc': 'Bode plot data (magnitude and phase)',
        'category': 'frequency_response',
    },
    'nyquist': {
        'func': forge_nyquist,
        'args': ['sys'],
        'opts': {'w': None},
        'desc': 'Nyquist plot data',
        'category': 'frequency_response',
    },
    'nichols': {
        'func': forge_nichols,
        'args': ['sys'],
        'opts': {'w': None},
        'desc': 'Nichols chart data',
        'category': 'frequency_response',
    },
    'margin': {
        'func': forge_margin,
        'args': ['sys'],
        'desc': 'Gain and phase margins',
        'category': 'frequency_response',
    },
    'bandwidth': {
        'func': forge_bandwidth,
        'args': ['sys'],
        'opts': {'dbdrop': -3.0},
        'desc': 'Compute -3 dB bandwidth',
        'category': 'frequency_response',
    },
    'evalfr': {
        'func': forge_evalfr,
        'args': ['sys', 's'],
        'desc': 'Evaluate frequency response at complex frequency',
        'category': 'frequency_response',
    },
    'freqresp': {
        'func': forge_freqresp,
        'args': ['sys'],
        'opts': {'w': None},
        'desc': 'Compute frequency response',
        'category': 'frequency_response',
    },

    # --- Stability & Analysis ---
    'pole': {
        'func': forge_pole,
        'args': ['sys'],
        'desc': 'Compute system poles',
        'category': 'stability_analysis',
    },
    'zero': {
        'func': forge_zero,
        'args': ['sys'],
        'desc': 'Compute system zeros',
        'category': 'stability_analysis',
    },
    'dcgain': {
        'func': forge_dcgain,
        'args': ['sys'],
        'desc': 'Compute DC (steady-state) gain',
        'category': 'stability_analysis',
    },
    'pzmap': {
        'func': forge_pzmap,
        'args': ['sys'],
        'desc': 'Pole-zero map data',
        'category': 'stability_analysis',
    },
    'rlocus': {
        'func': forge_rlocus,
        'args': ['sys'],
        'opts': {'k': None},
        'desc': 'Root locus data',
        'category': 'stability_analysis',
    },
    'isstable': {
        'func': forge_isstable,
        'args': ['sys'],
        'desc': 'Check system stability',
        'category': 'stability_analysis',
    },
    'ctrb': {
        'func': forge_ctrb,
        'args': ['A', 'B'],
        'desc': 'Controllability matrix',
        'category': 'stability_analysis',
    },
    'obsv': {
        'func': forge_obsv,
        'args': ['A', 'C'],
        'desc': 'Observability matrix',
        'category': 'stability_analysis',
    },
    'gram': {
        'func': forge_gram,
        'args': ['sys', 'gram_type'],
        'desc': 'Gramian matrix (controllability or observability)',
        'category': 'stability_analysis',
    },
    'lyap': {
        'func': forge_lyap,
        'args': ['A', 'Q'],
        'desc': 'Solve continuous Lyapunov equation',
        'category': 'stability_analysis',
    },
    'dlyap': {
        'func': forge_dlyap,
        'args': ['A', 'Q'],
        'desc': 'Solve discrete Lyapunov equation',
        'category': 'stability_analysis',
    },

    # --- Controller Design ---
    'pid': {
        'func': forge_pid,
        'args': ['Kp'],
        'opts': {'Ki': 0.0, 'Kd': 0.0, 'Tf': 0.0, 'dt': None},
        'desc': 'Create PID controller as transfer function',
        'category': 'controller_design',
    },
    'pidtune': {
        'func': forge_pidtune,
        'args': ['sys'],
        'opts': {'controller_type': 'pid'},
        'desc': 'Auto-tune PID controller',
        'category': 'controller_design',
    },
    'place': {
        'func': forge_place,
        'args': ['A', 'B', 'p'],
        'desc': 'Pole placement for state feedback',
        'category': 'controller_design',
    },
    'acker': {
        'func': forge_acker,
        'args': ['A', 'B', 'p'],
        'desc': "Ackermann's formula for pole placement (SISO)",
        'category': 'controller_design',
    },
    'lqr': {
        'func': forge_lqr,
        'args': ['A', 'B', 'Q', 'R'],
        'opts': {'N': None},
        'desc': 'Linear Quadratic Regulator (continuous)',
        'category': 'controller_design',
    },
    'dlqr': {
        'func': forge_dlqr,
        'args': ['A', 'B', 'Q', 'R'],
        'opts': {'N': None},
        'desc': 'Linear Quadratic Regulator (discrete)',
        'category': 'controller_design',
    },
    'kalman': {
        'func': forge_kalman,
        'args': ['A', 'C', 'Q', 'R'],
        'desc': 'Kalman filter gain (continuous)',
        'category': 'controller_design',
    },
}
