#!/usr/bin/env python3
"""Run the weight-conditioning anatomy study via ForgeSession."""
import time, sys, os
sys.path.insert(0, '/home/ubuntu/forge')
os.environ.setdefault('DISPLAY', ':99')

from forge.engine.session import ForgeSession
s = ForgeSession()
s.eval('addpath("ForgeHome/tiga")')

def compute(nel, R1, R2, w):
    """Call compute_annulus_cond and return (cond, lmin, lmax, err)."""
    s.eval(f'[_c,_lm,_lx,_e] = compute_annulus_cond(2, {nel}, 4, {R1}, {R2}, {w});')
    c = float(s.eval('_c').strip())
    lm = float(s.eval('_lm').strip())
    lx = float(s.eval('_lx').strip())
    e = float(s.eval('_e').strip())
    return c, lm, lx, e

import numpy as np

print("=== TIGA Weight-Conditioning Anatomy ===\n", flush=True)

# Part 1: Fine weight sweep
R1, R2 = 0.5, 1.5
nel = 4
w_fine = np.logspace(-1, 1, 30)
cond_f = np.zeros(len(w_fine))
lmin_f = np.zeros(len(w_fine))
lmax_f = np.zeros(len(w_fine))
err_f = np.zeros(len(w_fine))

print(f"Part 1: Fine sweep nel={nel}, {len(w_fine)} points", flush=True)
t0 = time.time()
for i, w in enumerate(w_fine):
    cond_f[i], lmin_f[i], lmax_f[i], err_f[i] = compute(nel, R1, R2, w)
    if i % 5 == 0:
        print(f"  [{i+1}/{len(w_fine)}] w={w:.4f}  cond={cond_f[i]:.4e}  err={err_f[i]:.4e}", flush=True)
print(f"  Sweep done in {time.time()-t0:.1f}s", flush=True)

# Find minimum
idx_min = np.argmin(cond_f)
w_approx = w_fine[idx_min]
print(f"\n  Approximate optimum: w*={w_approx:.4f}, cond={cond_f[idx_min]:.4e}")

# Golden section refinement
if idx_min > 0 and idx_min < len(w_fine)-1:
    a, b = float(w_fine[idx_min-1]), float(w_fine[idx_min+1])
else:
    a, b = w_approx*0.5, w_approx*2.0

gr = (np.sqrt(5)-1)/2
c = b - gr*(b-a)
d = a + gr*(b-a)
fc = compute(nel, R1, R2, c)[0]
fd = compute(nel, R1, R2, d)[0]
for _ in range(30):
    if b-a < 1e-6:
        break
    if fc < fd:
        b, d, fd = d, c, fc
        c = b - gr*(b-a)
        fc = compute(nel, R1, R2, c)[0]
    else:
        a, c, fc = c, d, fd
        d = a + gr*(b-a)
        fd = compute(nel, R1, R2, d)[0]

w_opt = (a+b)/2
cond_opt = compute(nel, R1, R2, w_opt)[0]
print(f"  Golden section: w*={w_opt:.6f}, cond={cond_opt:.4e}")

c_geo, _, _, e_geo = compute(nel, R1, R2, 1/np.sqrt(2))
c_bsp, _, _, e_bsp = compute(nel, R1, R2, 1.0)
print(f"  w=1/sqrt(2): cond={c_geo:.4e}, err={e_geo:.4e}")
print(f"  w=1.0:       cond={c_bsp:.4e}, err={e_bsp:.4e}")
print(f"  Conditioning ratio geo/bsp = {c_geo/c_bsp:.4f}")
print(f"  Conditioning ratio opt/bsp = {cond_opt/c_bsp:.4f}")

# Part 2: Geometry dependence
print(f"\nPart 2: w* vs R2/R1", flush=True)
ratios = [1.2, 1.5, 2.0, 3.0, 5.0, 10.0]
w_opt_geom = []
for ratio in ratios:
    R1g, R2g = 1.0, ratio
    # Coarse scan
    w_scan = np.linspace(0.5, 6.0, 15)
    c_scan = [compute(nel, R1g, R2g, w)[0] for w in w_scan]
    idx = np.argmin(c_scan)
    # Golden section
    a = float(w_scan[max(idx-1,0)])
    b = float(w_scan[min(idx+1,len(w_scan)-1)])
    c = b-gr*(b-a); d = a+gr*(b-a)
    fc = compute(nel, R1g, R2g, c)[0]
    fd = compute(nel, R1g, R2g, d)[0]
    for _ in range(20):
        if b-a < 1e-5: break
        if fc<fd:
            b,d,fd = d,c,fc; c=b-gr*(b-a); fc=compute(nel,R1g,R2g,c)[0]
        else:
            a,c,fc = c,d,fd; d=a+gr*(b-a); fd=compute(nel,R1g,R2g,d)[0]
    wopt = (a+b)/2
    copt = compute(nel, R1g, R2g, wopt)[0]
    cg = compute(nel, R1g, R2g, 1/np.sqrt(2))[0]
    cb = compute(nel, R1g, R2g, 1.0)[0]
    w_opt_geom.append(wopt)
    print(f"  R2/R1={ratio:5.1f}  w*={wopt:.4f}  cond*={copt:.2e}  geo={cg:.2e}  bsp={cb:.2e}", flush=True)

# Part 3: Mesh independence
print(f"\nPart 3: Mesh independence", flush=True)
for nel_test in [2, 4, 8, 16]:
    a, b = 1.0, 5.0
    c = b-gr*(b-a); d = a+gr*(b-a)
    fc = compute(nel_test, R1, R2, c)[0]
    fd = compute(nel_test, R1, R2, d)[0]
    for _ in range(25):
        if b-a<1e-5: break
        if fc<fd:
            b,d,fd=d,c,fc; c=b-gr*(b-a); fc=compute(nel_test,R1,R2,c)[0]
        else:
            a,c,fc=c,d,fd; d=a+gr*(b-a); fd=compute(nel_test,R1,R2,d)[0]
    wm=(a+b)/2
    cm=compute(nel_test,R1,R2,wm)[0]
    cb=compute(nel_test,R1,R2,1.0)[0]
    cg=compute(nel_test,R1,R2,1/np.sqrt(2))[0]
    print(f"  nel={nel_test:2d}  w*={wm:.4f}  cond*={cm:.2e}  opt/bsp={cm/cb:.4f}  geo/bsp={cg/cb:.4f}", flush=True)

# Part 4: Plots via Forge
print("\nPart 4: Generating plots...", flush=True)

# Store arrays in workspace for plotting
s.eval('clear')
s.eval('addpath("ForgeHome/tiga")')

# Transfer data to Forge workspace
w_str = '[' + ','.join(f'{x:.6f}' for x in w_fine) + ']'
c_str = '[' + ','.join(f'{x:.6e}' for x in cond_f) + ']'
lm_str = '[' + ','.join(f'{x:.6e}' for x in lmin_f) + ']'
lx_str = '[' + ','.join(f'{x:.6e}' for x in lmax_f) + ']'
e_str = '[' + ','.join(f'{x:.6e}' for x in err_f) + ']'

s.eval(f'w_fine = {w_str};')
s.eval(f'cond_fine = {c_str};')
s.eval(f'lmin_fine = {lm_str};')
s.eval(f'lmax_fine = {lx_str};')
s.eval(f'err_fine = {e_str};')
s.eval(f'w_opt = {w_opt};')
s.eval(f'cond_opt = {cond_opt};')
s.eval(f'c_geo = {c_geo};')
s.eval(f'c_bsp = {c_bsp};')
s.eval(f'e_geo = {e_geo};')

r_str = '[' + ','.join(f'{x}' for x in ratios) + ']'
wo_str = '[' + ','.join(f'{x:.4f}' for x in w_opt_geom) + ']'
s.eval(f'ratios = {r_str};')
s.eval(f'w_opt_geom = {wo_str};')

# Create 2x2 subplot figure
s.eval('figure(1);')

s.eval('subplot(2,2,1);')
s.eval("loglog(w_fine, cond_fine, 'b-');")
s.eval('hold on;')
s.eval(f"loglog(1/sqrt(2), c_geo, 'rv', 'MarkerSize', 10, 'MarkerFaceColor', 'r');")
s.eval(f"loglog(w_opt, cond_opt, 'g^', 'MarkerSize', 10, 'MarkerFaceColor', 'g');")
s.eval(f"loglog(1.0, c_bsp, 'ks', 'MarkerSize', 8, 'MarkerFaceColor', 'k');")
s.eval('hold off;')
s.eval("xlabel('Weight w');")
s.eval("ylabel('cond(K)');")
s.eval("title('Condition Number vs Weight');")
s.eval("legend('cond(K)', 'w=1/sqrt2', 'w* optimal', 'w=1 B-spline');")

s.eval('subplot(2,2,2);')
s.eval("loglog(w_fine, lmin_fine, 'r-');")
s.eval('hold on;')
s.eval("loglog(w_fine, lmax_fine, 'b-');")
s.eval('hold off;')
s.eval("xlabel('Weight w');")
s.eval("ylabel('Eigenvalue');")
s.eval("title('Extreme Eigenvalues');")
s.eval("legend('lambda_{min}', 'lambda_{max}');")

s.eval('subplot(2,2,3);')
s.eval("semilogx(ratios, w_opt_geom, 'bo-', 'MarkerFaceColor', 'b');")
s.eval('hold on;')
s.eval("semilogx([1, 20], [1/sqrt(2), 1/sqrt(2)], 'r--');")
s.eval("semilogx([1, 20], [1, 1], 'k--');")
s.eval('hold off;')
s.eval("xlabel('R_2/R_1');")
s.eval("ylabel('w*');")
s.eval("title('Optimal Weight vs Geometry');")
s.eval("legend('w*', '1/sqrt(2)', '1.0');")

s.eval('subplot(2,2,4);')
s.eval("loglog(w_fine, err_fine, 'g-');")
s.eval('hold on;')
s.eval(f"loglog(1/sqrt(2), e_geo, 'rv', 'MarkerSize', 10, 'MarkerFaceColor', 'r');")
s.eval('hold off;')
s.eval("xlabel('Weight w');")
s.eval("ylabel('Max Error');")
s.eval("title('Solution Error vs Weight');")

s.eval("saveas(1, '/tmp/weight_anatomy.png');")
print("Plot saved to /tmp/weight_anatomy.png")

elapsed = time.time() - t0
print(f"\nTotal time: {elapsed:.1f}s")
print("=== Anatomy Study Complete ===")
