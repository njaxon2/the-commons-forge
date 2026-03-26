#!/usr/bin/env python3
"""Study how arc angle affects the conditioning ratio.

For a circular arc of angle theta, the NURBS weight is w_geo = cos(theta/2).
Question: how does cond(K; w_geo)/cond(K; w=1) depend on theta?
"""
import time, sys, os
import numpy as np
sys.path.insert(0, '/home/ubuntu/forge')
os.environ.setdefault('DISPLAY', ':99')

from forge.engine.session import ForgeSession
s = ForgeSession()
s.eval('addpath("ForgeHome/tiga")')

def compute_arc_cond(nel, R1, R2, w_mid, theta):
    """Assemble stiffness for arc of angle theta with given weight.

    Control points for arc of angle theta:
      P0 = (r, 0), P1 = (r, r*tan(theta/2)), P2 = (r*cos(theta), r*sin(theta))
      Weight: w0=1, w1=w_mid, w2=1

    But we need the knot vector and control points to match.
    For quadratic NURBS with 3 control points, this is the standard conic section.
    """
    # We need a custom version of compute_annulus_cond that takes theta
    # Let's define the control points in Python and pass them

    # For now, use the existing compute_annulus_cond which is hardcoded for quarter circle
    # We modify the control points via a different .m file

    # Actually, let's just evaluate M-code directly
    s.eval(f'_theta = {theta};')
    s.eval(f'_w_mid = {w_mid};')
    s.eval(f'_R1 = {R1};')
    s.eval(f'_R2 = {R2};')
    s.eval(f'_nel_r = {nel};')
    s.eval('_p = 2; _nqp = 4;')

    # Build knot vectors
    s.eval('_interior_r = linspace(0, 1, _nel_r + 1);')
    s.eval('_interior_r = _interior_r(2:end-1);')
    s.eval('_Xi_r = [zeros(1, _p+1), _interior_r, ones(1, _p+1)];')
    s.eval('_n_r = length(_Xi_r) - _p - 1;')
    s.eval('_Xi_t = [0 0 0 1 1 1];')
    s.eval('_n_t = 3;')
    s.eval('_n_2d = _n_r * _n_t;')

    # Control points for arc of angle theta
    s.eval('_r_cp = linspace(_R1, _R2, _n_r);')
    s.eval('_CPx = zeros(_n_r, _n_t);')
    s.eval('_CPy = zeros(_n_r, _n_t);')
    s.eval('_Cw = ones(_n_r, _n_t);')

    # P0 = (r, 0), P1 = (r, r*tan(theta/2)), P2 = (r*cos(theta), r*sin(theta))
    # But for proper NURBS: P1 is the intersection of tangent lines at P0 and P2
    s.eval("""for _i = 1:_n_r
        _r = _r_cp(_i);
        _CPx(_i, 1) = _r;
        _CPy(_i, 1) = 0;
        _Cw(_i, 1) = 1;
        _CPx(_i, 2) = _r;
        _CPy(_i, 2) = _r * tan(_theta/2);
        _Cw(_i, 2) = _w_mid;
        _CPx(_i, 3) = _r * cos(_theta);
        _CPy(_i, 3) = _r * sin(_theta);
        _Cw(_i, 3) = 1;
    end""")

    # Assembly (reuse standard assembly logic)
    s.eval('[_gp, _gw] = gaussQuad(_nqp);')
    s.eval('_knots_r = unique(_Xi_r);')
    s.eval('_knots_t = unique(_Xi_t);')
    s.eval('_nel_rad = length(_knots_r) - 1;')
    s.eval('_nel_cir = length(_knots_t) - 1;')

    s.eval('_K = zeros(_n_2d, _n_2d);')

    # Full assembly loop
    assembly_code = """for _er = 1:_nel_rad
    _xi_a = _knots_r(_er); _xi_b = _knots_r(_er+1);
    if _xi_b - _xi_a < 1e-14; continue; end
    _Jr = (_xi_b - _xi_a)/2;
    for _et = 1:_nel_cir
        _eta_a = _knots_t(_et); _eta_b = _knots_t(_et+1);
        if _eta_b - _eta_a < 1e-14; continue; end
        _Jt = (_eta_b - _eta_a)/2;
        for _qr = 1:_nqp
            _xi = (_xi_a+_xi_b)/2 + _Jr*_gp(_qr);
            _span_r = findspan(_n_r-1, _p, _xi, _Xi_r);
            _ders_r = derbasisfun(_span_r, _xi, _p, 1, _Xi_r);
            _Nr = _ders_r(1,:); _dNr = _ders_r(2,:);
            for _qt = 1:_nqp
                _eta = (_eta_a+_eta_b)/2 + _Jt*_gp(_qt);
                _span_t = findspan(_n_t-1, _p, _eta, _Xi_t);
                _ders_t = derbasisfun(_span_t, _eta, _p, 1, _Xi_t);
                _Nt = _ders_t(1,:); _dNt = _ders_t(2,:);
                _wt_q = _gw(_qr)*_Jr*_gw(_qt)*_Jt;
                _W=0; _dW_dxi=0; _dW_deta=0;
                for _a = 0:_p
                    _ir = _span_r-_p+_a+1;
                    for _b = 0:_p
                        _it = _span_t-_p+_b+1;
                        _ww = _Cw(_ir,_it);
                        _W = _W + _Nr(_a+1)*_Nt(_b+1)*_ww;
                        _dW_dxi = _dW_dxi + _dNr(_a+1)*_Nt(_b+1)*_ww;
                        _dW_deta = _dW_deta + _Nr(_a+1)*_dNt(_b+1)*_ww;
                    end
                end
                _dx_dxi=0; _dx_deta=0; _dy_dxi=0; _dy_deta=0;
                for _a = 0:_p
                    _ir = _span_r-_p+_a+1;
                    for _b = 0:_p
                        _it = _span_t-_p+_b+1;
                        _ww = _Cw(_ir,_it);
                        _dR_dxi = (_dNr(_a+1)*_Nt(_b+1)*_ww*_W - _Nr(_a+1)*_Nt(_b+1)*_ww*_dW_dxi)/_W^2;
                        _dR_deta = (_Nr(_a+1)*_dNt(_b+1)*_ww*_W - _Nr(_a+1)*_Nt(_b+1)*_ww*_dW_deta)/_W^2;
                        _dx_dxi = _dx_dxi + _dR_dxi*_CPx(_ir,_it);
                        _dx_deta = _dx_deta + _dR_deta*_CPx(_ir,_it);
                        _dy_dxi = _dy_dxi + _dR_dxi*_CPy(_ir,_it);
                        _dy_deta = _dy_deta + _dR_deta*_CPy(_ir,_it);
                    end
                end
                _detJ = _dx_dxi*_dy_deta - _dx_deta*_dy_dxi;
                if abs(_detJ) < 1e-15; continue; end
                _inv_J11 = _dy_deta/_detJ;
                _inv_J12 = -_dy_dxi/_detJ;
                _inv_J21 = -_dx_deta/_detJ;
                _inv_J22 = _dx_dxi/_detJ;
                for _a = 0:_p
                    _ir = _span_r-_p+_a+1;
                    for _b = 0:_p
                        _it = _span_t-_p+_b+1;
                        _ww_A = _Cw(_ir,_it);
                        _dR_A_dxi = (_dNr(_a+1)*_Nt(_b+1)*_ww_A*_W - _Nr(_a+1)*_Nt(_b+1)*_ww_A*_dW_dxi)/_W^2;
                        _dR_A_deta = (_Nr(_a+1)*_dNt(_b+1)*_ww_A*_W - _Nr(_a+1)*_Nt(_b+1)*_ww_A*_dW_deta)/_W^2;
                        _dR_A_dx = _inv_J11*_dR_A_dxi + _inv_J12*_dR_A_deta;
                        _dR_A_dy = _inv_J21*_dR_A_dxi + _inv_J22*_dR_A_deta;
                        _glob_A = (_it-1)*_n_r + _ir;
                        for _c = 0:_p
                            _jr = _span_r-_p+_c+1;
                            for _d = 0:_p
                                _jt = _span_t-_p+_d+1;
                                _ww_B = _Cw(_jr,_jt);
                                _dR_B_dxi = (_dNr(_c+1)*_Nt(_d+1)*_ww_B*_W - _Nr(_c+1)*_Nt(_d+1)*_ww_B*_dW_dxi)/_W^2;
                                _dR_B_deta = (_Nr(_c+1)*_dNt(_d+1)*_ww_B*_W - _Nr(_c+1)*_Nt(_d+1)*_ww_B*_dW_deta)/_W^2;
                                _dR_B_dx = _inv_J11*_dR_B_dxi + _inv_J12*_dR_B_deta;
                                _dR_B_dy = _inv_J21*_dR_B_dxi + _inv_J22*_dR_B_deta;
                                _glob_B = (_jt-1)*_n_r + _jr;
                                _K(_glob_A,_glob_B) = _K(_glob_A,_glob_B) + (_dR_A_dx*_dR_B_dx + _dR_A_dy*_dR_B_dy)*abs(_detJ)*_wt_q;
                            end
                        end
                    end
                end
            end
        end
    end
end"""
    s.eval(assembly_code)

    # BCs and eigenvalues
    s.eval("""_bc_dofs = [];
for _j = 1:_n_t
    _bc_dofs = [_bc_dofs, (_j-1)*_n_r+1, (_j-1)*_n_r+_n_r];
end
_bc_dofs = unique(_bc_dofs);
_free_dofs = setdiff(1:_n_2d, _bc_dofs);
_Kf = _K(_free_dofs, _free_dofs);
_ev = sort(real(eig(_Kf)));
_ev_pos = _ev(_ev > 1e-10);
_cond_val = max(_ev_pos) / min(_ev_pos);""")

    c = float(s.eval('_cond_val').strip())
    return c

print("=== Arc Angle vs Conditioning Study ===\n", flush=True)

R1, R2 = 0.5, 1.5
nel = 4

# Test different arc angles
# theta = pi/6 (30°), pi/4 (45°), pi/3 (60°), pi/2 (90°), 2pi/3 (120°)
angles_deg = [30, 45, 60, 90, 120]
angles_rad = [a * np.pi / 180 for a in angles_deg]

print("Arc angle study: cond(w_geo)/cond(w=1) vs theta\n", flush=True)
print(f"{'theta_deg':>10} {'w_geo':>10} {'cond_geo':>12} {'cond_bsp':>12} {'ratio':>10}", flush=True)

for theta_d, theta in zip(angles_deg, angles_rad):
    w_geo = np.cos(theta/2)
    t0 = time.time()
    c_geo = compute_arc_cond(nel, R1, R2, w_geo, theta)
    c_bsp = compute_arc_cond(nel, R1, R2, 1.0, theta)
    ratio = c_geo / c_bsp
    elapsed = time.time() - t0
    print(f"{theta_d:10d} {w_geo:10.6f} {c_geo:12.4e} {c_bsp:12.4e} {ratio:10.4f}  ({elapsed:.1f}s)", flush=True)

# Also check: does the ratio depend on cos(theta/2) directly?
print("\nAnalytical check:", flush=True)
for theta_d, theta in zip(angles_deg, angles_rad):
    w_geo = np.cos(theta/2)
    # Hypothesis: ratio = 1/w_geo^alpha for some alpha
    # From quarter circle: ratio=1.444, w_geo=0.7071
    # log(1.444)/log(1/0.7071) = log(1.444)/log(1.4142) = 0.367/0.346 = 1.06
    # So ratio ≈ 1/w_geo ≈ (1/cos(theta/2))
    print(f"  theta={theta_d:3d}  w_geo={w_geo:.4f}  1/w_geo={1/w_geo:.4f}  1/w_geo^2={1/w_geo**2:.4f}", flush=True)

print("\n=== Arc Angle Study Complete ===", flush=True)
