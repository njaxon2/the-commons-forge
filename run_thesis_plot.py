#!/usr/bin/env python3
"""Generate thesis-quality plots for the weight-conditioning result."""
import sys, os
import numpy as np
sys.path.insert(0, '/home/ubuntu/forge')
os.environ.setdefault('DISPLAY', ':99')

from forge.engine.session import ForgeSession
s = ForgeSession()
s.eval('addpath("ForgeHome/tiga")')

# Data from anatomy study
w_fine_str = ','.join([f'{x:.6f}' for x in np.logspace(-1, 1, 30)])
# Re-read from the anatomy log
cond_data = {
    'w': [0.1, 0.1292, 0.1668, 0.2154, 0.2783, 0.3594, 0.4642, 0.5995, 0.7743, 1.0,
          1.2915, 1.6681, 2.1544, 2.7826, 3.5938, 4.6416, 5.9948, 7.7426, 10.0],
}

# Actually, let me just compute the key points we need via Forge
print("Computing thesis plot data...", flush=True)

# Fine sweep for smooth curve
w_vals = np.logspace(-0.5, 0.8, 40)
cond_vals = []
for w in w_vals:
    s.eval(f'[_c,_lm,_lx,_e] = compute_annulus_cond(2, 4, 4, 0.5, 1.5, {w});')
    c = float(s.eval('_c').strip())
    cond_vals.append(c)
    print(f"  w={w:.4f}  cond={c:.2f}", flush=True)

cond_vals = np.array(cond_vals)

# Normalize by B-spline value
s.eval('[_c,_lm,_lx,_e] = compute_annulus_cond(2, 4, 4, 0.5, 1.5, 1.0);')
c_bsp = float(s.eval('_c').strip())

# Transfer to Forge for plotting
w_str = '[' + ','.join(f'{x:.6f}' for x in w_vals) + ']'
c_str = '[' + ','.join(f'{x:.6f}' for x in cond_vals) + ']'
cn_str = '[' + ','.join(f'{x:.6f}' for x in cond_vals/c_bsp) + ']'

s.eval(f'w_vals = {w_str};')
s.eval(f'cond_vals = {c_str};')
s.eval(f'cond_norm = {cn_str};')

# Also compute the weight function W(eta) for different weights
s.eval('eta = linspace(0, 1, 100);')
for w_label, w_val in [('geo', 1/np.sqrt(2)), ('bsp', 1.0), ('opt', 2.31)]:
    s.eval(f'W_{w_label} = 1 + 2*({w_val}-1)*eta.*(1-eta);')

# Figure 1: Normalized condition number with annotations
s.eval('figure(1);')
s.eval("semilogx(w_vals, cond_norm, 'b-', 'LineWidth', 2);")
s.eval('hold on;')
# Mark key points
s.eval(f"semilogx(1/sqrt(2), {cond_vals[np.argmin(np.abs(w_vals - 1/np.sqrt(2)))]}/{c_bsp}, 'rv', 'MarkerSize', 12, 'MarkerFaceColor', 'r');")
s.eval(f"semilogx(1.0, 1.0, 'ks', 'MarkerSize', 10, 'MarkerFaceColor', 'k');")
idx_opt = np.argmin(cond_vals)
s.eval(f"semilogx({w_vals[idx_opt]}, {cond_vals[idx_opt]/c_bsp}, 'g^', 'MarkerSize', 12, 'MarkerFaceColor', 'g');")
# Reference line at 1
s.eval("semilogx([0.3, 7], [1, 1], 'k--');")
s.eval('hold off;')
s.eval("xlabel('NURBS Weight w');")
s.eval("ylabel('cond(K) / cond(K)|_{w=1}');")
s.eval("title('Normalized Stiffness Conditioning vs NURBS Weight');")
s.eval("legend('cond(K)/cond_0', 'w = 1/sqrt(2) (exact geometry)', 'w = 1 (B-spline)', 'w* (optimal conditioning)');")
s.eval("grid on;")
s.eval("saveas(1, '/tmp/thesis_fig1_normalized_cond.png');")
print("Figure 1 saved", flush=True)

# Figure 2: Weight function shape comparison
s.eval('figure(2);')
s.eval("plot(eta, W_geo, 'r-', 'LineWidth', 2);")
s.eval('hold on;')
s.eval("plot(eta, W_bsp, 'k-', 'LineWidth', 2);")
s.eval("plot(eta, W_opt, 'g-', 'LineWidth', 2);")
s.eval("plot([0, 1], [1, 1], 'k--');")
s.eval('hold off;')
s.eval("xlabel('Parameter eta');")
s.eval("ylabel('W(eta)');")
s.eval("title('NURBS Weight Function W(eta) = 1 + 2(w-1) eta (1-eta)');")
s.eval("legend('w = 1/sqrt(2) (geometry)', 'w = 1 (B-spline)', 'w = 2.31 (optimal cond.)');")
s.eval("grid on;")
s.eval("saveas(2, '/tmp/thesis_fig2_weight_function.png');")
print("Figure 2 saved", flush=True)

# Figure 3: Arc angle study
theta_deg = [30, 45, 60, 90, 120]
ratio = [1.0346, 1.0878, 1.1744, 1.4906, 2.3129]
W_ratio = [2/(1+np.cos(t*np.pi/360)) for t in theta_deg]
W_ratio_cubed = [wr**3 for wr in W_ratio]

t_str = '[' + ','.join(str(t) for t in theta_deg) + ']'
r_str = '[' + ','.join(f'{r:.4f}' for r in ratio) + ']'
wr_str = '[' + ','.join(f'{wr:.4f}' for wr in W_ratio) + ']'
wr3_str = '[' + ','.join(f'{wr3:.4f}' for wr3 in W_ratio_cubed) + ']'

s.eval(f'theta_deg = {t_str};')
s.eval(f'ratio_meas = {r_str};')
s.eval(f'W_ratio = {wr_str};')
s.eval(f'W_ratio_cubed = {wr3_str};')

s.eval('figure(3);')
s.eval("plot(theta_deg, ratio_meas, 'bo-', 'MarkerSize', 10, 'MarkerFaceColor', 'b', 'LineWidth', 2);")
s.eval('hold on;')
s.eval("plot(theta_deg, W_ratio, 'r--', 'LineWidth', 1.5);")
s.eval("plot(theta_deg, W_ratio_cubed, 'g--', 'LineWidth', 1.5);")
s.eval('hold off;')
s.eval("xlabel('Arc Angle theta (degrees)');")
s.eval("ylabel('cond(K; w_{geo}) / cond(K; w=1)');")
s.eval("title('Conditioning Penalty vs Arc Angle');")
s.eval("legend('Measured', 'W_{max}/W_{min}', '(W_{max}/W_{min})^3');")
s.eval("grid on;")
s.eval("saveas(3, '/tmp/thesis_fig3_arc_angle.png');")
print("Figure 3 saved", flush=True)

print("\n=== Thesis plots complete ===", flush=True)
