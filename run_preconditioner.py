#!/usr/bin/env python3
"""Weight-based preconditioning: can we get exact geometry AND good conditioning?

Idea: Use w=1/sqrt(2) for exact circle geometry, but precondition the
stiffness matrix based on the weight function W(eta).

If the conditioning penalty comes from W(eta) distortion, then a diagonal
preconditioner D where D_{ii} ~ W(eta_i) should improve conditioning while
maintaining exact geometry.

Test: compute cond(D^{-1/2} K D^{-1/2}) where D approximates the weight effect.
"""
import sys, os, time
import numpy as np
sys.path.insert(0, '/home/ubuntu/forge')
os.environ.setdefault('DISPLAY', ':99')

from forge.engine.session import ForgeSession
s = ForgeSession()
s.eval('addpath("ForgeHome/tiga")')

print("=== Weight-Based Preconditioning Study ===\n", flush=True)

# Step 1: Build the stiffness matrix K for w=1/sqrt(2) and extract the free system
R1, R2 = 0.5, 1.5
nel = 8
p = 2
w_geo = 1/np.sqrt(2)

# Use Forge to build K
s.eval(f'p={p}; R1={R1}; R2={R2}; nel_r={nel}; nqp={p+2};')
s.eval(f'w_mid = 1/sqrt(2);')

# Build and extract the full stiffness matrix
s.eval("""interior_r = linspace(0, 1, nel_r + 1);
interior_r = interior_r(2:end - 1);
Xi_r = [zeros(1, p + 1), interior_r, ones(1, p + 1)];
n_r = length(Xi_r) - p - 1;
Xi_t = [0 0 0 1 1 1];
n_t = 3;
n_2d = n_r * n_t;
r_cp = linspace(R1, R2, n_r);
[gp, gw] = gaussQuad(nqp);
knots_r = unique(Xi_r);
knots_t = unique(Xi_t);
nel_rad = length(knots_r) - 1;
nel_cir = length(knots_t) - 1;""")

s.eval("""CPx = zeros(n_r, n_t); CPy = zeros(n_r, n_t); Cw = ones(n_r, n_t);
for i = 1:n_r
    r = r_cp(i);
    CPx(i, 1) = r; CPy(i, 1) = 0; Cw(i, 1) = 1;
    CPx(i, 2) = r; CPy(i, 2) = r; Cw(i, 2) = w_mid;
    CPx(i, 3) = 0; CPy(i, 3) = r; Cw(i, 3) = 1;
end""")

# Full assembly
assembly_code = """K = zeros(n_2d, n_2d);
f = zeros(n_2d, 1);
for er = 1:nel_rad
    xi_a = knots_r(er); xi_b = knots_r(er + 1);
    if xi_b - xi_a < 1e-14; continue; end
    Jr = (xi_b - xi_a) / 2;
    for et = 1:nel_cir
        eta_a = knots_t(et); eta_b = knots_t(et + 1);
        if eta_b - eta_a < 1e-14; continue; end
        Jt = (eta_b - eta_a) / 2;
        for qr = 1:nqp
            xi = (xi_a + xi_b) / 2 + Jr * gp(qr);
            span_r = findspan(n_r - 1, p, xi, Xi_r);
            ders_r = derbasisfun(span_r, xi, p, 1, Xi_r);
            Nr = ders_r(1, :); dNr = ders_r(2, :);
            for qt = 1:nqp
                eta = (eta_a + eta_b) / 2 + Jt * gp(qt);
                span_t = findspan(n_t - 1, p, eta, Xi_t);
                ders_t = derbasisfun(span_t, eta, p, 1, Xi_t);
                Nt = ders_t(1, :); dNt = ders_t(2, :);
                wt_q = gw(qr) * Jr * gw(qt) * Jt;
                W = 0; dW_dxi = 0; dW_deta = 0;
                for a = 0:p
                    ir = span_r - p + a + 1;
                    for b = 0:p
                        it = span_t - p + b + 1;
                        ww = Cw(ir, it);
                        W = W + Nr(a+1) * Nt(b+1) * ww;
                        dW_dxi = dW_dxi + dNr(a+1) * Nt(b+1) * ww;
                        dW_deta = dW_deta + Nr(a+1) * dNt(b+1) * ww;
                    end
                end
                x_phys = 0; y_phys = 0;
                dx_dxi = 0; dx_deta = 0; dy_dxi = 0; dy_deta = 0;
                for a = 0:p
                    ir = span_r - p + a + 1;
                    for b = 0:p
                        it = span_t - p + b + 1;
                        ww = Cw(ir, it);
                        R_val = Nr(a+1) * Nt(b+1) * ww / W;
                        dR_dxi = (dNr(a+1) * Nt(b+1) * ww * W - Nr(a+1) * Nt(b+1) * ww * dW_dxi) / W^2;
                        dR_deta = (Nr(a+1) * dNt(b+1) * ww * W - Nr(a+1) * Nt(b+1) * ww * dW_deta) / W^2;
                        x_phys = x_phys + R_val * CPx(ir, it);
                        y_phys = y_phys + R_val * CPy(ir, it);
                        dx_dxi = dx_dxi + dR_dxi * CPx(ir, it);
                        dx_deta = dx_deta + dR_deta * CPx(ir, it);
                        dy_dxi = dy_dxi + dR_dxi * CPy(ir, it);
                        dy_deta = dy_deta + dR_deta * CPy(ir, it);
                    end
                end
                detJ = dx_dxi * dy_deta - dx_deta * dy_dxi;
                if abs(detJ) < 1e-15; continue; end
                inv_J11 = dy_deta / detJ; inv_J12 = -dy_dxi / detJ;
                inv_J21 = -dx_deta / detJ; inv_J22 = dx_dxi / detJ;
                r_phys = sqrt(x_phys^2 + y_phys^2);
                f_val = 16 * r_phys^2 - 4 * (R1^2 + R2^2);
                for a = 0:p
                    ir = span_r - p + a + 1;
                    for b = 0:p
                        it = span_t - p + b + 1;
                        ww_A = Cw(ir, it);
                        R_A = Nr(a+1) * Nt(b+1) * ww_A / W;
                        dR_A_dxi = (dNr(a+1) * Nt(b+1) * ww_A * W - Nr(a+1) * Nt(b+1) * ww_A * dW_dxi) / W^2;
                        dR_A_deta = (Nr(a+1) * dNt(b+1) * ww_A * W - Nr(a+1) * Nt(b+1) * ww_A * dW_deta) / W^2;
                        dR_A_dx = inv_J11 * dR_A_dxi + inv_J12 * dR_A_deta;
                        dR_A_dy = inv_J21 * dR_A_dxi + inv_J22 * dR_A_deta;
                        glob_A = (it - 1) * n_r + ir;
                        f(glob_A) = f(glob_A) + R_A * f_val * abs(detJ) * wt_q;
                        for c = 0:p
                            jr = span_r - p + c + 1;
                            for d = 0:p
                                jt = span_t - p + d + 1;
                                ww_B = Cw(jr, jt);
                                dR_B_dxi = (dNr(c+1) * Nt(d+1) * ww_B * W - Nr(c+1) * Nt(d+1) * ww_B * dW_dxi) / W^2;
                                dR_B_deta = (Nr(c+1) * dNt(d+1) * ww_B * W - Nr(c+1) * Nt(d+1) * ww_B * dW_deta) / W^2;
                                dR_B_dx = inv_J11 * dR_B_dxi + inv_J12 * dR_B_deta;
                                dR_B_dy = inv_J21 * dR_B_dxi + inv_J22 * dR_B_deta;
                                glob_B = (jt - 1) * n_r + jr;
                                K(glob_A, glob_B) = K(glob_A, glob_B) + (dR_A_dx * dR_B_dx + dR_A_dy * dR_B_dy) * abs(detJ) * wt_q;
                            end
                        end
                    end
                end
            end
        end
    end
end"""
print("Assembling stiffness matrix...", flush=True)
t0 = time.time()
s.eval(assembly_code)
print(f"  Assembly done in {time.time()-t0:.1f}s", flush=True)

# Apply BCs and extract free system
s.eval("""bc_dofs = [];
for j = 1:n_t
    bc_dofs = [bc_dofs, (j - 1) * n_r + 1, (j - 1) * n_r + n_r];
end
bc_dofs = unique(bc_dofs);
free_dofs = setdiff(1:n_2d, bc_dofs);
Kf = K(free_dofs, free_dofs);""")

# Unpreconditioned condition number
s.eval("ev_orig = sort(real(eig(Kf)));")
s.eval("ev_pos = ev_orig(ev_orig > 1e-10);")
s.eval("cond_orig = max(ev_pos) / min(ev_pos);")
cond_orig = float(s.eval('cond_orig').strip())
print(f"\nUnpreconditioned cond(K) = {cond_orig:.4e}", flush=True)

# Diagonal preconditioner: D_ii = diag(K)_ii
s.eval("D_diag = diag(Kf);")
s.eval("D_inv_sqrt = diag(1 ./ sqrt(D_diag));")
s.eval("Kp = D_inv_sqrt * Kf * D_inv_sqrt;")
s.eval("ev_p = sort(real(eig(Kp)));")
s.eval("ev_pp = ev_p(ev_p > 1e-10);")
s.eval("cond_diag = max(ev_pp) / min(ev_pp);")
cond_diag = float(s.eval('cond_diag').strip())
print(f"Diagonal preconditioned cond(D^-1/2 K D^-1/2) = {cond_diag:.4e}", flush=True)

# Compare with B-spline condition number
s.eval('[_cb,_,_,_] = compute_annulus_cond(2, 8, 4, 0.5, 1.5, 1.0);')
cond_bsp = float(s.eval('_cb').strip())
print(f"B-spline (w=1) cond = {cond_bsp:.4e}", flush=True)
print(f"\nRatios:", flush=True)
print(f"  Original geo/bsp = {cond_orig/cond_bsp:.4f}", flush=True)
print(f"  Preconditioned geo/bsp = {cond_diag/cond_bsp:.4f}", flush=True)
print(f"  Improvement from preconditioning: {cond_orig/cond_diag:.2f}x", flush=True)

# Also try: diagonal preconditioner based on row sums
s.eval("D_rowsum = diag(sum(abs(Kf), 2));")
s.eval("D_rs_inv_sqrt = diag(1 ./ sqrt(diag(D_rowsum)));")
s.eval("Kp2 = D_rs_inv_sqrt * Kf * D_rs_inv_sqrt;")
s.eval("ev_p2 = sort(real(eig(Kp2)));")
s.eval("ev_pp2 = ev_p2(ev_p2 > 1e-10);")
s.eval("cond_rowsum = max(ev_pp2) / min(ev_pp2);")
cond_rs = float(s.eval('cond_rowsum').strip())
print(f"Row-sum preconditioned = {cond_rs:.4e} (ratio={cond_rs/cond_bsp:.4f})", flush=True)

print("\n=== Preconditioning Study Complete ===", flush=True)
