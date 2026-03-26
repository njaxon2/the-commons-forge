#!/usr/bin/env python3
"""1D NURBS-mapped Laplacian: weight effect on conditioning.

Simple approach: use the weight function effect on the 1D Jacobian
to compute the condition number analytically and numerically.

For a 1D problem on [0,L] mapped by x(xi) with Jacobian J = dx/dxi:
  K_AB = integral dR_A/dx * dR_B/dx dx = integral dR_A/dxi * dR_B/dxi / J dxi

The B-spline basis is the same in all cases; only J changes with w.
So cond(K(w)) / cond(K(w=1)) depends only on max(J)/min(J) where
J is the RATIO J(w)/J(w=1).

For the NURBS weight function W(eta) = 1 + 2(w-1)*eta*(1-eta):
  J(w,eta) = J_base(eta) * correction(W)

For the parametric speed of the NURBS curve:
  ds/deta = |dr/deta| where r is the position on the arc

The key insight is: the NURBS derivatives involve dR/deta which has
the weight function in the denominator via the quotient rule.
The net effect on the Jacobian is:

J(w,eta) ~ J(w=1,eta) * W(eta) / W(eta) (cancellation) ... no, that's wrong.

Actually, for a B-spline curve (w=1), the Jacobian is J_0(eta).
For NURBS with weight w, the control points are the SAME but the
basis functions change. The Jacobian becomes:

J_w(eta) = J_0(eta) * h(w, eta)

where h depends on the weight function. For the specific case of
the 3-control-point quadratic NURBS:

dx/deta = sum_i dR_i/deta * Px_i
dR_i/deta = (dN_i*w_i*W - N_i*w_i*dW) / W^2

For uniform weights (B-spline), W=1 and dW=0, so dR_i/deta = dN_i.
For non-uniform weights: dR_i/deta = dN_i*w_i/W - N_i*w_i*dW/W^2

The modification from weight w on the middle point:
W(eta) = 1 + 2(w-1)*eta*(1-eta)
dW/deta = 2(w-1)*(1-2*eta)

The Jacobian ratio J_w/J_0 can be computed numerically.
"""
import numpy as np

def W_func(eta, w):
    return 1 + 2*(w-1)*eta*(1-eta)

def dW_func(eta, w):
    return 2*(w-1)*(1-2*eta)

# Quadratic B-spline basis on [0,0,0,1,1,1]
def N0(eta): return (1-eta)**2
def N1(eta): return 2*eta*(1-eta)
def N2(eta): return eta**2
def dN0(eta): return -2*(1-eta)
def dN1(eta): return 2*(1-2*eta)
def dN2(eta): return 2*eta

def compute_jacobian_ratio(eta, w, theta):
    """Compute the ratio of NURBS Jacobian to B-spline Jacobian at eta."""
    R = 1.0  # radius doesn't matter for ratio

    # Control points
    Px = np.array([R, R, R*np.cos(theta)])
    Py = np.array([0, R*np.tan(theta/2), R*np.sin(theta)])
    Cw = np.array([1.0, w, 1.0])

    # B-spline (w=1): all weights = 1
    Nv = np.array([N0(eta), N1(eta), N2(eta)])
    dNv = np.array([dN0(eta), dN1(eta), dN2(eta)])

    # B-spline Jacobian
    dx_bsp = np.dot(dNv, Px)
    dy_bsp = np.dot(dNv, Py)
    J_bsp = np.sqrt(dx_bsp**2 + dy_bsp**2)

    # NURBS Jacobian
    W = W_func(eta, w)
    dW = dW_func(eta, w)

    R_vals = Nv * Cw / W
    dR_vals = (dNv * Cw * W - Nv * Cw * dW) / W**2

    dx_nurbs = np.dot(dR_vals, Px)
    dy_nurbs = np.dot(dR_vals, Py)
    J_nurbs = np.sqrt(dx_nurbs**2 + dy_nurbs**2)

    return J_nurbs / J_bsp if J_bsp > 1e-15 else 1.0

print("=== Jacobian Ratio Analysis ===\n")

theta = np.pi/2
etas = np.linspace(0.01, 0.99, 200)

# For quarter circle with w = 1/sqrt(2)
w_geo = 1/np.sqrt(2)
J_ratios = [compute_jacobian_ratio(eta, w_geo, theta) for eta in etas]
J_ratios = np.array(J_ratios)

print(f"Quarter circle (theta=90), w=1/sqrt(2):")
print(f"  J_ratio: min={np.min(J_ratios):.6f}, max={np.max(J_ratios):.6f}")
print(f"  J_ratio range: max/min = {np.max(J_ratios)/np.min(J_ratios):.6f}")
print(f"  W_ratio = 2/(1+w) = {2/(1+w_geo):.6f}")

# Check: does J_ratio = const/W?
# If dR_i = dN_i*w_i/W - N_i*w_i*dW/W^2
# = (dN_i*w_i - N_i*w_i*dW/W) / W
# For small deviations from w=1: dW ≈ 2(w-1)*dN1_part, so
# the second term is small and dR_i ≈ dN_i * w_i / W
# Then J_w ≈ J_0 * w_eff / W where w_eff = sum(dN_i * w_i * Px_i) / sum(dN_i * Px_i)

# Let me just check J_ratio * W(eta) = const?
check = J_ratios * np.array([W_func(eta, w_geo) for eta in etas])
print(f"  J_ratio * W: min={np.min(check):.6f}, max={np.max(check):.6f}")
print(f"  (If constant, J_ratio ∝ 1/W)")

# Also check J_ratio * W^2
check2 = J_ratios * np.array([W_func(eta, w_geo)**2 for eta in etas])
print(f"  J_ratio * W^2: min={np.min(check2):.6f}, max={np.max(check2):.6f}")

# Now let's check for different weights and angles
print(f"\nSystematic check of J_ratio extremes:")
print(f"{'theta':>6} {'w':>8} {'J_max/J_min':>12} {'W_max/W_min':>12} {'ratio_of_ratios':>16}")
for theta_d in [30, 45, 60, 90, 120]:
    theta_r = theta_d * np.pi / 180
    w_g = np.cos(theta_r / 2)
    J_rats = np.array([compute_jacobian_ratio(eta, w_g, theta_r) for eta in etas])
    J_range = np.max(J_rats) / np.min(J_rats)
    W_range = 2 / (1 + w_g)
    print(f"{theta_d:6d} {w_g:8.4f} {J_range:12.6f} {W_range:12.6f} {J_range/W_range:16.6f}")

# So J_max/J_min should relate to cond(K_w)/cond(K_1)
# From the 2D data:
print(f"\nComparing J_range to measured 2D conditioning ratio:")
measured_ratios = {30: 1.0346, 45: 1.0878, 60: 1.1744, 90: 1.4906, 120: 2.3129}
for theta_d in [30, 45, 60, 90, 120]:
    theta_r = theta_d * np.pi / 180
    w_g = np.cos(theta_r / 2)
    J_rats = np.array([compute_jacobian_ratio(eta, w_g, theta_r) for eta in etas])
    J_range = np.max(J_rats) / np.min(J_rats)
    mr = measured_ratios[theta_d]
    print(f"  theta={theta_d:3d}  J_range={J_range:.4f}  J_range^2={J_range**2:.4f}  J_range^3={J_range**3:.4f}  2D_ratio={mr:.4f}")

print("\n=== Analysis Complete ===")
