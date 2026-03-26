#!/usr/bin/env python3
"""Generate the key summary figure for the weight-conditioning thesis."""
import sys, os
import numpy as np
sys.path.insert(0, '/home/ubuntu/forge')
os.environ.setdefault('DISPLAY', ':99')

from forge.engine.session import ForgeSession
s = ForgeSession()
s.eval('addpath("ForgeHome/tiga")')

# Collect all data
print("Computing summary figure data...", flush=True)

# 1. Fine weight sweep at nel=8 for smooth curve
w_vals = np.logspace(-0.3, 0.6, 25)
cond_vals = []
err_vals = []
lmin_vals = []
lmax_vals = []
for w in w_vals:
    s.eval(f'[_c,_lm,_lx,_e] = compute_annulus_cond(2, 8, 4, 0.5, 1.5, {w});')
    cond_vals.append(float(s.eval('_c').strip()))
    lmin_vals.append(float(s.eval('_lm').strip()))
    lmax_vals.append(float(s.eval('_lx').strip()))
    err_vals.append(float(s.eval('_e').strip()))
    print(f"  [{len(cond_vals)}/{len(w_vals)}] w={w:.3f}", flush=True)

cond_vals = np.array(cond_vals)
err_vals = np.array(err_vals)
lmin_vals = np.array(lmin_vals)
lmax_vals = np.array(lmax_vals)

# Reference values
s.eval('[_c,_lm,_lx,_e] = compute_annulus_cond(2, 8, 4, 0.5, 1.5, 1.0);')
c_bsp = float(s.eval('_c').strip())
e_bsp = float(s.eval('_e').strip())
s.eval('[_c,_lm,_lx,_e] = compute_annulus_cond(2, 8, 4, 0.5, 1.5, 0.70710678);')
c_geo = float(s.eval('_c').strip())
e_geo = float(s.eval('_e').strip())

# Find optimal
idx_opt = np.argmin(cond_vals)
w_opt = float(w_vals[idx_opt])
c_opt = cond_vals[idx_opt]
e_opt = err_vals[idx_opt]

print(f"\nKey values: c_bsp={c_bsp:.2f}, c_geo={c_geo:.2f}, c_opt={c_opt:.2f}, w_opt={w_opt:.3f}")
print(f"  e_bsp={e_bsp:.4e}, e_geo={e_geo:.4e}, e_opt={e_opt:.4e}")

# Transfer to Forge workspace
w_str = '[' + ','.join(f'{x:.6f}' for x in w_vals) + ']'
cn_str = '[' + ','.join(f'{x:.6f}' for x in cond_vals/c_bsp) + ']'
en_str = '[' + ','.join(f'{x:.6e}' for x in err_vals) + ']'
lmn_str = '[' + ','.join(f'{x:.6e}' for x in lmin_vals) + ']'
lxn_str = '[' + ','.join(f'{x:.6e}' for x in lmax_vals) + ']'

s.eval(f'w = {w_str};')
s.eval(f'cn = {cn_str};')
s.eval(f'en = {en_str};')
s.eval(f'lmn = {lmn_str};')
s.eval(f'lxn = {lxn_str};')

# ---- Create 2x3 subplot figure ----
s.eval('figure(1);')

# (1,1): Normalized condition number
s.eval('subplot(2,3,1);')
s.eval("semilogx(w, cn, 'b-', 'LineWidth', 2);")
s.eval('hold on;')
s.eval(f"semilogx(1/sqrt(2), {c_geo/c_bsp}, 'rv', 'MarkerSize', 10, 'MarkerFaceColor', 'r');")
s.eval(f"semilogx(1.0, 1.0, 'ks', 'MarkerSize', 8, 'MarkerFaceColor', 'k');")
s.eval(f"semilogx({w_opt}, {c_opt/c_bsp}, 'g^', 'MarkerSize', 10, 'MarkerFaceColor', 'g');")
s.eval("semilogx([0.4, 5], [1, 1], 'k--');")
s.eval('hold off;')
s.eval("xlabel('Weight w');")
s.eval("ylabel('cond(K)/cond_0');")
s.eval("title('(a) Normalized Conditioning');")

# (1,2): Solution error
s.eval('subplot(2,3,2);')
s.eval("loglog(w, en, 'g-', 'LineWidth', 2);")
s.eval('hold on;')
s.eval(f"loglog(1/sqrt(2), {e_geo}, 'rv', 'MarkerSize', 10, 'MarkerFaceColor', 'r');")
s.eval(f"loglog(1.0, {e_bsp}, 'ks', 'MarkerSize', 8, 'MarkerFaceColor', 'k');")
s.eval('hold off;')
s.eval("xlabel('Weight w');")
s.eval("ylabel('Max Error');")
s.eval("title('(b) Solution Error');")

# (1,3): Eigenvalue extremes (normalized)
s.eval('subplot(2,3,3);')
lmin_bsp = lmin_vals[np.argmin(np.abs(w_vals - 1.0))]
lmax_bsp = lmax_vals[np.argmin(np.abs(w_vals - 1.0))]
lmnn_str = '[' + ','.join(f'{x:.6e}' for x in lmin_vals/lmin_bsp) + ']'
lxnn_str = '[' + ','.join(f'{x:.6e}' for x in lmax_vals/lmax_bsp) + ']'
s.eval(f'lmnn = {lmnn_str};')
s.eval(f'lxnn = {lxnn_str};')
s.eval("semilogx(w, lmnn, 'r-', 'LineWidth', 2);")
s.eval('hold on;')
s.eval("semilogx(w, lxnn, 'b-', 'LineWidth', 2);")
s.eval("semilogx([0.4, 5], [1, 1], 'k--');")
s.eval('hold off;')
s.eval("xlabel('Weight w');")
s.eval("ylabel('Eigenvalue / Eigenvalue_0');")
s.eval("title('(c) Spectral Response');")
s.eval("legend('lambda_{min}', 'lambda_{max}');")

# (2,1): Weight function shape
s.eval('subplot(2,3,4);')
s.eval('eta = linspace(0, 1, 100);')
s.eval('W_geo = 1 + 2*(1/sqrt(2)-1)*eta.*(1-eta);')
s.eval('W_bsp = ones(1, 100);')
s.eval(f'W_opt = 1 + 2*({w_opt}-1)*eta.*(1-eta);')
s.eval("plot(eta, W_geo, 'r-', 'LineWidth', 2);")
s.eval('hold on;')
s.eval("plot(eta, W_bsp, 'k-', 'LineWidth', 2);")
s.eval("plot(eta, W_opt, 'g-', 'LineWidth', 2);")
s.eval('hold off;')
s.eval("xlabel('eta');")
s.eval("ylabel('W(eta)');")
s.eval("title('(d) Weight Function');")
s.eval("legend('w=1/sqrt(2)', 'w=1', 'w=w*');")

# (2,2): Conditioning ratio vs arc angle with model
theta_deg = [30, 45, 60, 90, 120]
ratio = [1.0346, 1.0878, 1.1744, 1.4906, 2.3129]
W_ratio = [2/(1+np.cos(t*np.pi/360)) for t in theta_deg]

t_str = '[' + ','.join(str(t) for t in theta_deg) + ']'
r_str = '[' + ','.join(f'{r:.4f}' for r in ratio) + ']'
wr_str = '[' + ','.join(f'{wr:.4f}' for wr in W_ratio) + ']'

s.eval(f'td = {t_str};')
s.eval(f'rmeas = {r_str};')
s.eval(f'Wr = {wr_str};')

s.eval('subplot(2,3,5);')
s.eval("plot(td, rmeas, 'bo-', 'MarkerSize', 8, 'MarkerFaceColor', 'b', 'LineWidth', 2);")
s.eval('hold on;')
s.eval("plot(td, Wr, 'r--', 'LineWidth', 1.5);")
s.eval("plot(td, Wr.^2, 'g--', 'LineWidth', 1.5);")
s.eval('hold off;')
s.eval("xlabel('Arc Angle (deg)');")
s.eval("ylabel('cond ratio');")
s.eval("title('(e) Angle Dependence');")
s.eval("legend('Measured', 'W_{max}/W_{min}', '(W_{max}/W_{min})^2');")

# (2,3): Mesh independence table as text plot
s.eval('subplot(2,3,6);')
# Use text annotations for the table
s.eval("axis([0, 10, 0, 10]);")
s.eval("axis off;")
s.eval("title('(f) Mesh Independence');")
s.eval("text(0.5, 9, 'nel    w*      opt/bsp  geo/bsp', 'FontSize', 9, 'FontName', 'Monospace');")
s.eval("text(0.5, 7.5, '  2   2.158    0.685    1.513', 'FontSize', 9, 'FontName', 'Monospace');")
s.eval("text(0.5, 6, '  4   2.368    0.672    1.491', 'FontSize', 9, 'FontName', 'Monospace');")
s.eval("text(0.5, 4.5, '  8   2.306    0.686    1.444', 'FontSize', 9, 'FontName', 'Monospace');")
s.eval("text(0.5, 3, ' 16   2.307    0.686    1.444', 'FontSize', 9, 'FontName', 'Monospace');")
s.eval("text(0.5, 1, 'Ratios converge as h -> 0', 'FontSize', 9);")

s.eval("saveas(1, '/tmp/thesis_summary.png');")
print("\nSummary figure saved to /tmp/thesis_summary.png", flush=True)
