#!/usr/bin/env python3
"""Test: does the conditioning ratio exponent depend on polynomial degree p?

Hypothesis: cond(K; w_geo)/cond(K; w=1) = (W_max/W_min)^alpha
where alpha might depend on p or on d (spatial dimension).

For p=2 on 2D quarter circle, we found alpha ~ 3.
Test with p=2 but varying mesh size to confirm, using the 2D code.
"""
import sys, os
import numpy as np
sys.path.insert(0, '/home/ubuntu/forge')
os.environ.setdefault('DISPLAY', ':99')

from forge.engine.session import ForgeSession
s = ForgeSession()
s.eval('addpath("ForgeHome/tiga")')

# For the quarter circle: w_geo = 1/sqrt(2), W_max/W_min = 2/(1+1/sqrt(2)) = 1.17157
W_ratio = 2 / (1 + 1/np.sqrt(2))

# Let's test multiple mesh sizes and verify the ratio is constant
# Then compute alpha = log(ratio) / log(W_ratio)

print("=== Conditioning Exponent Study ===\n", flush=True)

# Part 1: Verify mesh independence and compute alpha for standard quarter circle
print("Part 1: Alpha vs mesh size (p=2, quarter circle)", flush=True)
for nel in [2, 4, 8, 16]:
    s.eval(f'[_cg,_,_,_] = compute_annulus_cond(2, {nel}, 4, 0.5, 1.5, 1/sqrt(2));')
    c_geo = float(s.eval('_cg').strip())
    s.eval(f'[_cb,_,_,_] = compute_annulus_cond(2, {nel}, 4, 0.5, 1.5, 1.0);')
    c_bsp = float(s.eval('_cb').strip())
    ratio = c_geo / c_bsp
    alpha = np.log(ratio) / np.log(W_ratio)
    print(f"  nel={nel:2d}  ratio={ratio:.6f}  alpha={alpha:.4f}", flush=True)

# Part 2: Compute alpha for different arc angles
print("\nPart 2: Alpha vs arc angle (p=2, nel=4)", flush=True)
nel = 4
for theta_d in [30, 45, 60, 90, 120]:
    theta = theta_d * np.pi / 180
    w_geo = np.cos(theta/2)
    Wr = 2 / (1 + w_geo)

    # Use the arc_cond approach from run_arc_study.py
    # For theta != 90, we need to modify the control points
    # Let's use inline M-code
    s.eval(f'_theta = {theta};')
    s.eval(f'_w_mid = {w_geo};')
    s.eval('_R1=0.5; _R2=1.5; _p=2; _nqp=4;')
    s.eval(f'_nel_r = {nel};')

    # Build common setup
    setup = """_interior_r = linspace(0, 1, _nel_r + 1);
_interior_r = _interior_r(2:end-1);
_Xi_r = [zeros(1, _p+1), _interior_r, ones(1, _p+1)];
_n_r = length(_Xi_r) - _p - 1;
_Xi_t = [0 0 0 1 1 1];
_n_t = 3;
_n_2d = _n_r * _n_t;
_r_cp = linspace(_R1, _R2, _n_r);
[_gp, _gw] = gaussQuad(_nqp);
_knots_r = unique(_Xi_r);
_knots_t = unique(_Xi_t);
_nel_rad = length(_knots_r) - 1;
_nel_cir = length(_knots_t) - 1;"""
    s.eval(setup)

    def assemble_and_cond(w_val):
        s.eval(f'_ww = {w_val};')
        s.eval("""_CPx = zeros(_n_r, _n_t); _CPy = zeros(_n_r, _n_t); _Cw = ones(_n_r, _n_t);
for _i = 1:_n_r
    _r = _r_cp(_i);
    _CPx(_i,1)=_r; _CPy(_i,1)=0; _Cw(_i,1)=1;
    _CPx(_i,2)=_r; _CPy(_i,2)=_r*tan(_theta/2); _Cw(_i,2)=_ww;
    _CPx(_i,3)=_r*cos(_theta); _CPy(_i,3)=_r*sin(_theta); _Cw(_i,3)=1;
end""")

        assembly = """_K = zeros(_n_2d, _n_2d);
for _er = 1:_nel_rad
    _xi_a=_knots_r(_er); _xi_b=_knots_r(_er+1);
    if _xi_b-_xi_a<1e-14; continue; end
    _Jr=(_xi_b-_xi_a)/2;
    for _et = 1:_nel_cir
        _eta_a=_knots_t(_et); _eta_b=_knots_t(_et+1);
        if _eta_b-_eta_a<1e-14; continue; end
        _Jt=(_eta_b-_eta_a)/2;
        for _qr = 1:_nqp
            _xi=(_xi_a+_xi_b)/2+_Jr*_gp(_qr);
            _span_r=findspan(_n_r-1,_p,_xi,_Xi_r);
            _ders_r=derbasisfun(_span_r,_xi,_p,1,_Xi_r);
            _Nr=_ders_r(1,:); _dNr=_ders_r(2,:);
            for _qt = 1:_nqp
                _eta=(_eta_a+_eta_b)/2+_Jt*_gp(_qt);
                _span_t=findspan(_n_t-1,_p,_eta,_Xi_t);
                _ders_t=derbasisfun(_span_t,_eta,_p,1,_Xi_t);
                _Nt=_ders_t(1,:); _dNt=_ders_t(2,:);
                _wt_q=_gw(_qr)*_Jr*_gw(_qt)*_Jt;
                _W=0; _dW_dxi=0; _dW_deta=0;
                for _a=0:_p
                    _ir=_span_r-_p+_a+1;
                    for _b=0:_p
                        _it=_span_t-_p+_b+1;
                        _wv=_Cw(_ir,_it);
                        _W=_W+_Nr(_a+1)*_Nt(_b+1)*_wv;
                        _dW_dxi=_dW_dxi+_dNr(_a+1)*_Nt(_b+1)*_wv;
                        _dW_deta=_dW_deta+_Nr(_a+1)*_dNt(_b+1)*_wv;
                    end
                end
                _dx_dxi=0;_dx_deta=0;_dy_dxi=0;_dy_deta=0;
                for _a=0:_p
                    _ir=_span_r-_p+_a+1;
                    for _b=0:_p
                        _it=_span_t-_p+_b+1;
                        _wv=_Cw(_ir,_it);
                        _dR_dxi=(_dNr(_a+1)*_Nt(_b+1)*_wv*_W-_Nr(_a+1)*_Nt(_b+1)*_wv*_dW_dxi)/_W^2;
                        _dR_deta=(_Nr(_a+1)*_dNt(_b+1)*_wv*_W-_Nr(_a+1)*_Nt(_b+1)*_wv*_dW_deta)/_W^2;
                        _dx_dxi=_dx_dxi+_dR_dxi*_CPx(_ir,_it);
                        _dx_deta=_dx_deta+_dR_deta*_CPx(_ir,_it);
                        _dy_dxi=_dy_dxi+_dR_dxi*_CPy(_ir,_it);
                        _dy_deta=_dy_deta+_dR_deta*_CPy(_ir,_it);
                    end
                end
                _detJ=_dx_dxi*_dy_deta-_dx_deta*_dy_dxi;
                if abs(_detJ)<1e-15; continue; end
                _inv_J11=_dy_deta/_detJ; _inv_J12=-_dy_dxi/_detJ;
                _inv_J21=-_dx_deta/_detJ; _inv_J22=_dx_dxi/_detJ;
                for _a=0:_p
                    _ir=_span_r-_p+_a+1;
                    for _b=0:_p
                        _it=_span_t-_p+_b+1;
                        _wA=_Cw(_ir,_it);
                        _dRA_dxi=(_dNr(_a+1)*_Nt(_b+1)*_wA*_W-_Nr(_a+1)*_Nt(_b+1)*_wA*_dW_dxi)/_W^2;
                        _dRA_deta=(_Nr(_a+1)*_dNt(_b+1)*_wA*_W-_Nr(_a+1)*_Nt(_b+1)*_wA*_dW_deta)/_W^2;
                        _dRA_dx=_inv_J11*_dRA_dxi+_inv_J12*_dRA_deta;
                        _dRA_dy=_inv_J21*_dRA_dxi+_inv_J22*_dRA_deta;
                        _gA=(_it-1)*_n_r+_ir;
                        for _c=0:_p
                            _jr=_span_r-_p+_c+1;
                            for _d=0:_p
                                _jt=_span_t-_p+_d+1;
                                _wB=_Cw(_jr,_jt);
                                _dRB_dxi=(_dNr(_c+1)*_Nt(_d+1)*_wB*_W-_Nr(_c+1)*_Nt(_d+1)*_wB*_dW_dxi)/_W^2;
                                _dRB_deta=(_Nr(_c+1)*_dNt(_d+1)*_wB*_W-_Nr(_c+1)*_Nt(_d+1)*_wB*_dW_deta)/_W^2;
                                _dRB_dx=_inv_J11*_dRB_dxi+_inv_J12*_dRB_deta;
                                _dRB_dy=_inv_J21*_dRB_dxi+_inv_J22*_dRB_deta;
                                _gB=(_jt-1)*_n_r+_jr;
                                _K(_gA,_gB)=_K(_gA,_gB)+(_dRA_dx*_dRB_dx+_dRA_dy*_dRB_dy)*abs(_detJ)*_wt_q;
                            end
                        end
                    end
                end
            end
        end
    end
end"""
        s.eval(assembly)

        s.eval("""_bc=[];
for _j=1:_n_t
    _bc=[_bc,(_j-1)*_n_r+1,(_j-1)*_n_r+_n_r];
end
_bc=unique(_bc);
_free=setdiff(1:_n_2d,_bc);
_Kf=_K(_free,_free);
_ev=sort(real(eig(_Kf)));
_ev_pos=_ev(_ev>1e-10);
_cond=max(_ev_pos)/min(_ev_pos);""")
        return float(s.eval('_cond').strip())

    c_geo = assemble_and_cond(w_geo)
    c_bsp = assemble_and_cond(1.0)
    ratio = c_geo / c_bsp
    alpha = np.log(ratio) / np.log(Wr)
    print(f"  theta={theta_d:3d}  w_geo={w_geo:.4f}  Wr={Wr:.4f}  ratio={ratio:.4f}  alpha={alpha:.4f}", flush=True)

print("\n=== Exponent Study Complete ===", flush=True)
