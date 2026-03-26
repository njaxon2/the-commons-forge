#!/usr/bin/env python3
"""Generate all thesis figures for the NURBS weight-conditioning study.

Produces 4 publication-quality figures:
  Fig 1: Weight-conditioning tradeoff (main result)
  Fig 2: Diagonal preconditioning effectiveness
  Fig 3: Arc angle dependence
  Fig 4: Elliptical geometry generalization

All computations run in Forge; plots generated via Forge's plotting system.
"""
import sys, os, time
import numpy as np
sys.path.insert(0, '/home/ubuntu/forge')
os.environ.setdefault('DISPLAY', ':99')

from forge.engine.session import ForgeSession
s = ForgeSession()
s.eval('addpath("ForgeHome/tiga")')

output_dir = '/home/ubuntu/forge/thesis_figures'
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# Helper: build stiffness matrix for arbitrary arc on annulus
# ============================================================
ASSEMBLY_CODE = """_K = zeros(_n_2d, _n_2d);
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

BC_AND_COND = """_bc=[];
for _j=1:_n_t
    _bc=[_bc,(_j-1)*_n_r+1,(_j-1)*_n_r+_n_r];
end
_bc=unique(_bc);
_free=setdiff(1:_n_2d,_bc);
_Kf=_K(_free,_free);
_ev=sort(real(eig(_Kf)));
_evp=_ev(_ev>1e-10);
_cond_orig=max(_evp)/min(_evp);"""

PRECOND = """_D=diag(_Kf); _Dinv=diag(1./sqrt(_D)); _Kp=_Dinv*_Kf*_Dinv;
_evp2=sort(real(eig(_Kp))); _evp2=_evp2(_evp2>1e-10);
_cond_prec=max(_evp2)/min(_evp2);"""


def setup_arc_mesh(nel, R1, R2, w_mid, theta):
    """Set up mesh for arc of angle theta on annulus."""
    s.eval(f'_p=2; _R1={R1}; _R2={R2}; _nel_r={nel}; _nqp=4; _w_mid={w_mid}; _theta={theta};')
    s.eval("""_interior_r = linspace(0, 1, _nel_r + 1);
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
_nel_cir = length(_knots_t) - 1;""")

    s.eval("""_CPx = zeros(_n_r, _n_t); _CPy = zeros(_n_r, _n_t); _Cw = ones(_n_r, _n_t);
for _i = 1:_n_r
    _r = _r_cp(_i);
    _CPx(_i,1)=_r; _CPy(_i,1)=0; _Cw(_i,1)=1;
    _CPx(_i,2)=_r; _CPy(_i,2)=_r*tan(_theta/2); _Cw(_i,2)=_w_mid;
    _CPx(_i,3)=_r*cos(_theta); _CPy(_i,3)=_r*sin(_theta); _Cw(_i,3)=1;
end""")


def setup_ellipse_mesh(nel, a_in, b_in, a_out, b_out, w_mid):
    """Set up mesh for quarter elliptical annulus."""
    s.eval(f'_p=2; _nel_r={nel}; _nqp=4; _w={w_mid};')
    s.eval(f'_ai={a_in}; _bi={b_in}; _ao={a_out}; _bo={b_out};')
    s.eval("""_interior_r = linspace(0, 1, _nel_r + 1);
_interior_r = _interior_r(2:end-1);
_Xi_r = [zeros(1, _p+1), _interior_r, ones(1, _p+1)];
_n_r = length(_Xi_r) - _p - 1;
_Xi_t = [0 0 0 1 1 1];
_n_t = 3;
_n_2d = _n_r * _n_t;""")

    s.eval("""_r_frac = linspace(0, 1, _n_r);
_CPx = zeros(_n_r, _n_t); _CPy = zeros(_n_r, _n_t); _Cw = ones(_n_r, _n_t);
for _i = 1:_n_r
    _t = _r_frac(_i);
    _ax = _ai + _t * (_ao - _ai);
    _bx = _bi + _t * (_bo - _bi);
    _CPx(_i,1) = _ax; _CPy(_i,1) = 0;   _Cw(_i,1) = 1;
    _CPx(_i,2) = _ax; _CPy(_i,2) = _bx;  _Cw(_i,2) = _w;
    _CPx(_i,3) = 0;   _CPy(_i,3) = _bx;  _Cw(_i,3) = 1;
end""")

    s.eval("""[_gp, _gw] = gaussQuad(_nqp);
_knots_r = unique(_Xi_r);
_knots_t = unique(_Xi_t);
_nel_rad = length(_knots_r) - 1;
_nel_cir = length(_knots_t) - 1;""")


def compute_cond_and_prec():
    """Assemble K, apply BCs, compute original and preconditioned condition numbers."""
    s.eval(ASSEMBLY_CODE)
    s.eval(BC_AND_COND)
    cond_orig = float(s.eval('_cond_orig').strip())
    s.eval(PRECOND)
    cond_prec = float(s.eval('_cond_prec').strip())
    return cond_orig, cond_prec


def build_arc(nel, R1, R2, w_mid, theta):
    setup_arc_mesh(nel, R1, R2, w_mid, theta)
    return compute_cond_and_prec()


def build_ellipse(nel, ai, bi, ao, bo, w_mid):
    setup_ellipse_mesh(nel, ai, bi, ao, bo, w_mid)
    return compute_cond_and_prec()


# ============================================================
# FIGURE 1: Weight-Conditioning Tradeoff Curve
# ============================================================
print("=== Figure 1: Weight-Conditioning Tradeoff ===", flush=True)
t0 = time.time()

R1, R2 = 0.5, 1.5
nel = 4

w_vals = np.concatenate([
    np.linspace(0.3, 0.6, 8),
    np.linspace(0.6, 0.8, 15),
    np.linspace(0.8, 1.5, 15),
    np.linspace(1.5, 4.0, 15)
])

conds = []
for w in w_vals:
    s.eval(f'[_c,_lm,_lx,_e] = compute_annulus_cond(2, {nel}, 4, {R1}, {R2}, {w});')
    c = float(s.eval('_c').strip())
    conds.append(c)

# B-spline baseline
s.eval(f'[_cb,_lmb,_lxb,_eb] = compute_annulus_cond(2, {nel}, 4, {R1}, {R2}, 1.0);')
c_bsp = float(s.eval('_cb').strip())

ratios = [c / c_bsp for c in conds]

# Upload data to Forge
w_str = "[" + " ".join(f"{w:.6f}" for w in w_vals) + "]"
r_str = "[" + " ".join(f"{r:.6f}" for r in ratios) + "]"
s.eval(f"_w_data = {w_str};")
s.eval(f"_r_data = {r_str};")

w_geo = 1/np.sqrt(2)
w_geo_idx = np.argmin(np.abs(w_vals - w_geo))
r_geo = ratios[w_geo_idx]
w_opt_idx = np.argmin(ratios)
w_opt = w_vals[w_opt_idx]
r_opt = ratios[w_opt_idx]

s.eval("figure(1);")
s.eval('plot(_w_data, _r_data, "b-", "LineWidth", 2);')
s.eval("hold on;")
s.eval(f'plot({w_geo}, {r_geo}, "ro", "MarkerSize", 10, "LineWidth", 2);')
s.eval(f'plot(1.0, 1.0, "ks", "MarkerSize", 10, "LineWidth", 2);')
s.eval(f'plot({w_opt}, {r_opt}, "gd", "MarkerSize", 10, "LineWidth", 2);')
s.eval('plot([0.1 4.0], [1 1], "k--", "LineWidth", 0.5);')
s.eval("hold off;")
s.eval('xlabel("NURBS weight w", "FontSize", 13);')
s.eval('ylabel("cond(K; w) / cond(K; 1)", "FontSize", 13);')
s.eval('title("Weight-Conditioning Tradeoff (Quarter Annulus, p=2, nel=4)", "FontSize", 14);')
s.eval('legend("Conditioning ratio", "Geometric w=1/sqrt(2)", "B-spline w=1", "Optimal w*", "Location", "northeast");')
s.eval('set(gca, "FontSize", 11);')
s.eval("grid on;")
s.eval("xlim([0.2 4.2]);")
s.eval("ylim([0.5 2.5]);")
s.eval(f'saveas(1, "{output_dir}/fig1_weight_tradeoff.png");')
print(f"  Figure 1 done in {time.time()-t0:.1f}s", flush=True)


# ============================================================
# FIGURE 2: Preconditioning Effectiveness (bar chart)
# ============================================================
print("\n=== Figure 2: Preconditioning Effectiveness ===", flush=True)
t0 = time.time()

theta = np.pi/2
nel_vals = [2, 4, 8]
geo_ratios = []
prec_ratios = []

for n in nel_vals:
    c_geo, c_prec = build_arc(n, R1, R2, w_geo, theta)
    c_bsp_ref, _ = build_arc(n, R1, R2, 1.0, theta)
    geo_ratios.append(c_geo / c_bsp_ref)
    prec_ratios.append(c_prec / c_bsp_ref)
    print(f"  nel={n}: geo/bsp={c_geo/c_bsp_ref:.4f}, prec/bsp={c_prec/c_bsp_ref:.4f}", flush=True)

s.eval(f"_geo_bar = [{' '.join(f'{g:.6f}' for g in geo_ratios)}];")
s.eval(f"_prec_bar = [{' '.join(f'{p:.6f}' for p in prec_ratios)}];")

s.eval("figure(2);")
s.eval("bar([1 2 3]-0.15, _geo_bar, 0.3, \"r\");")
s.eval("hold on;")
s.eval("bar([1 2 3]+0.15, _prec_bar, 0.3, \"b\");")
s.eval("plot([0.5 3.5], [1 1], \"k--\", \"LineWidth\", 1);")
s.eval("hold off;")
s.eval('xlabel("Radial elements", "FontSize", 13);')
s.eval('ylabel("Conditioning ratio vs B-spline", "FontSize", 13);')
s.eval('title("Diagonal Preconditioning: Mesh Independence", "FontSize", 14);')
s.eval('legend("Unpreconditioned", "Preconditioned", "B-spline baseline", "Location", "northeast");')
s.eval('set(gca, "FontSize", 11);')
s.eval("grid on;")
s.eval("ylim([0 2]);")
s.eval(f'saveas(2, "{output_dir}/fig2_preconditioning.png");')
print(f"  Figure 2 done in {time.time()-t0:.1f}s", flush=True)


# ============================================================
# FIGURE 3: Arc Angle Dependence
# ============================================================
print("\n=== Figure 3: Arc Angle Dependence ===", flush=True)
t0 = time.time()

nel = 4
angles_deg = [15, 30, 45, 60, 75, 90, 105, 120]
geo_angle = []
prec_angle = []
model_angle = []

for theta_d in angles_deg:
    theta_r = theta_d * np.pi / 180
    w_g = np.cos(theta_r / 2)
    c_geo, c_prec = build_arc(nel, R1, R2, w_g, theta_r)
    c_bsp, _ = build_arc(nel, R1, R2, 1.0, theta_r)
    geo_angle.append(c_geo / c_bsp)
    prec_angle.append(c_prec / c_bsp)
    Wratio = 2.0 / (1.0 + w_g)
    model_angle.append(Wratio ** 2.32)
    print(f"  theta={theta_d}: geo/bsp={c_geo/c_bsp:.4f}, prec/bsp={c_prec/c_bsp:.4f}", flush=True)

s.eval(f"_angles = [{' '.join(str(a) for a in angles_deg)}];")
s.eval(f"_geo_ang = [{' '.join(f'{g:.6f}' for g in geo_angle)}];")
s.eval(f"_prec_ang = [{' '.join(f'{p:.6f}' for p in prec_angle)}];")
s.eval(f"_model_ang = [{' '.join(f'{m:.6f}' for m in model_angle)}];")

s.eval("figure(3);")
s.eval('plot(_angles, _geo_ang, "ro-", "LineWidth", 2, "MarkerSize", 8);')
s.eval("hold on;")
s.eval('plot(_angles, _prec_ang, "bs-", "LineWidth", 2, "MarkerSize", 8);')
s.eval('plot(_angles, _model_ang, "k--", "LineWidth", 1.5);')
s.eval('plot([10 125], [1 1], "k:", "LineWidth", 0.5);')
s.eval("hold off;")
s.eval('xlabel("Arc angle (degrees)", "FontSize", 13);')
s.eval('ylabel("Conditioning ratio vs B-spline", "FontSize", 13);')
s.eval('title("Conditioning Penalty vs Arc Angle (nel=4)", "FontSize", 14);')
s.eval('legend("Unpreconditioned", "Preconditioned", "Scaling model", "Location", "northwest");')
s.eval('set(gca, "FontSize", 11);')
s.eval("grid on;")
s.eval(f'saveas(3, "{output_dir}/fig3_arc_angle.png");')
print(f"  Figure 3 done in {time.time()-t0:.1f}s", flush=True)


# ============================================================
# FIGURE 4: Elliptical Geometry Generalization
# ============================================================
print("\n=== Figure 4: Elliptical Geometry Generalization ===", flush=True)
t0 = time.time()

nel = 4
w_geo = 1/np.sqrt(2)

configs = [
    ("Circle", 0.5, 0.5, 1.5, 1.5),
    ("Ellipse 2:1", 0.5, 0.25, 1.5, 0.75),
    ("Ellipse 4:1", 1.0, 0.25, 2.0, 0.5),
    ("Non-conformal", 0.5, 0.5, 2.0, 1.0),
]

geo_ell = []
prec_ell = []

for name, ai, bi, ao, bo in configs:
    c_geo, c_prec = build_ellipse(nel, ai, bi, ao, bo, w_geo)
    c_bsp, _ = build_ellipse(nel, ai, bi, ao, bo, 1.0)
    geo_ell.append(c_geo / c_bsp)
    prec_ell.append(c_prec / c_bsp)
    print(f"  {name}: geo/bsp={c_geo/c_bsp:.4f}, prec/bsp={c_prec/c_bsp:.4f}", flush=True)

s.eval(f"_geo_ell = [{' '.join(f'{g:.6f}' for g in geo_ell)}];")
s.eval(f"_prec_ell = [{' '.join(f'{p:.6f}' for p in prec_ell)}];")

s.eval("figure(4);")
s.eval("bar([1 2 3 4]-0.15, _geo_ell, 0.3, \"r\");")
s.eval("hold on;")
s.eval("bar([1 2 3 4]+0.15, _prec_ell, 0.3, \"b\");")
s.eval("plot([0.5 4.5], [1 1], \"k--\", \"LineWidth\", 1);")
s.eval("hold off;")
s.eval('xlabel("Geometry", "FontSize", 13);')
s.eval('ylabel("Conditioning ratio vs B-spline", "FontSize", 13);')
s.eval('title("Preconditioning on Elliptical Geometries", "FontSize", 14);')
s.eval('legend("Unpreconditioned", "Preconditioned", "B-spline baseline", "Location", "northeast");')
s.eval('set(gca, "FontSize", 11);')
s.eval("grid on;")
s.eval("ylim([0 2]);")
s.eval(f'saveas(4, "{output_dir}/fig4_ellipse.png");')
print(f"  Figure 4 done in {time.time()-t0:.1f}s", flush=True)

print(f"\n=== All thesis figures saved to {output_dir}/ ===", flush=True)
