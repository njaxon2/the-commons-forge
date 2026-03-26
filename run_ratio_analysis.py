#!/usr/bin/env python3
"""Analyze the conditioning ratio vs weight function properties."""
import numpy as np

# Data from the arc study
theta_deg = np.array([30, 45, 60, 90, 120])
theta_rad = theta_deg * np.pi / 180
w_geo = np.cos(theta_rad / 2)
ratio = np.array([1.0346, 1.0878, 1.1744, 1.4906, 2.3129])

# Weight function W(eta) = 1 + 2(w-1)*eta*(1-eta)
# At eta=1/2: W_mid = (1+w)/2
# At eta=0,1: W_end = 1
# So W_max/W_min = max(1, (1+w)/2) / min(1, (1+w)/2)
# For w<1: W_mid < 1, so W_max=1, W_min=(1+w)/2, ratio_W = 2/(1+w)
# For w>1: W_mid > 1, so W_max=(1+w)/2, W_min=1, ratio_W = (1+w)/2

W_ratio = 2 / (1 + w_geo)  # for w<1

print("Conditioning ratio analysis:")
print(f"{'theta':>6} {'w_geo':>8} {'ratio':>8} {'W_ratio':>8} {'ratio/W_ratio':>13} {'W_ratio^2':>10} {'ratio/W^2':>10}")
for i in range(len(theta_deg)):
    wr = W_ratio[i]
    print(f"{theta_deg[i]:6d} {w_geo[i]:8.4f} {ratio[i]:8.4f} {wr:8.4f} {ratio[i]/wr:13.4f} {wr**2:10.4f} {ratio[i]/wr**2:10.4f}")

# Try fitting ratio = a * (W_ratio)^b
log_ratio = np.log(ratio)
log_Wr = np.log(W_ratio)
# Linear fit in log space
from numpy.polynomial import polynomial as P
coeffs = np.polyfit(log_Wr, log_ratio, 1)
b_fit = coeffs[0]
a_fit = np.exp(coeffs[1])
print(f"\nPower law fit: ratio = {a_fit:.4f} * W_ratio^{b_fit:.4f}")
print(f"Predicted vs actual:")
for i in range(len(theta_deg)):
    pred = a_fit * W_ratio[i]**b_fit
    print(f"  theta={theta_deg[i]:3d}: actual={ratio[i]:.4f}, predicted={pred:.4f}, error={abs(ratio[i]-pred)/ratio[i]*100:.1f}%")

# Try other functional forms
# Maybe ratio involves the Jacobian determinant variation
# det(J) for the NURBS map involves W and dW/deta
# dW/deta = 2(w-1)(1-2*eta)
# At eta=0: dW/deta = 2(w-1)
# The Jacobian stretching comes from the denominator W^2 in dR/deta

# For the mapping x(xi,eta), y(xi,eta):
# x = sum R_i * CPx_i, y = sum R_i * CPy_i
# The tangential component involves dR/deta which has W^2 in denominator
# So the tangential metric is ~ 1/W^2
# And the condition of the stiffness matrix involves integral of grad*grad
# which involves J^{-T} * J^{-1} * det(J) = det(J)^{-1} * (J*J^T)^{-1} * det(J)

# Simpler approach: just check if ratio = integral-average of 1/W^2 / integral-average of 1
# The "effective" conditioning multiplier from W is related to max(1/W^2)/min(1/W^2)
# = W_max^2 / W_min^2 = (2/(1+w))^2 for w<1

Wr2 = W_ratio**2
print(f"\nW_ratio^2 model:")
for i in range(len(theta_deg)):
    print(f"  theta={theta_deg[i]:3d}: actual={ratio[i]:.4f}, W_ratio^2={Wr2[i]:.4f}, diff={ratio[i]-Wr2[i]:.4f}")

# Hmm. Let me try: the integral of W^{-2} over [0,1]
# W(eta) = 1 + 2(w-1)*eta*(1-eta)
# integral_0^1 W^{-2} deta
def W_func(eta, w):
    return 1 + 2*(w-1)*eta*(1-eta)

from scipy import integrate
print(f"\nIntegral-based analysis:")
for i in range(len(theta_deg)):
    w = w_geo[i]
    # integral of 1/W^2
    int_Wm2, _ = integrate.quad(lambda eta: 1/W_func(eta,w)**2, 0, 1)
    # For w=1 (B-spline), W=1 everywhere, so int_Wm2 = 1
    # Ratio of integrals
    print(f"  theta={theta_deg[i]:3d}: w={w:.4f}, int(1/W^2)={int_Wm2:.4f}, sqrt(int)={np.sqrt(int_Wm2):.4f}, ratio={ratio[i]:.4f}")

# Try: ratio = (max_eta ||J^{-1}(eta)||) / (min_eta ||J^{-1}(eta)||)
# which is approximately max(1/W) / min(1/W) = W_max/W_min = 2/(1+w)
# But that gives 1.17 for 90 degrees vs 1.49 actual

# The condition number of K is lambda_max/lambda_min
# lambda_max ~ h^{-2} * max(det(J)^{-1} * ||J^{-1}||^2) ~ h^{-2} * max_metric
# lambda_min ~ h^d * min(det(J)^{-1} * ||J^{-1}||^2) * ... more complex

# Actually for 2D: cond(K) ~ (max_metric / min_metric)
# where metric = ||J^{-T}||^2 * |det(J)|
# For the NURBS map, J = J_param * diag(1/W corrections)
# The key insight is that the radial direction is unaffected by w
# Only the circumferential direction changes

# In the circumferential direction:
# the circumferential length element is |dx/deta| * deta
# dx/deta involves dR/deta which has the W^2 denominator
# So the circumferential metric component ~ (r*theta_phys)^2 / W^2(eta)
# And the ratio of max to min circumferential metric ~ W_max^2/W_min^2 = (2/(1+w))^2

# But the stiffness matrix condition depends on the full metric tensor ratio
# which mixes radial and circumferential components
# cond(K) ~ max_{elements} (h_max/h_min) * max_{quad points} (sqrt(g_max/g_min))

# Let me try an integral-weighted approach
print(f"\nJacobian-based analysis:")
for i in range(len(theta_deg)):
    w = w_geo[i]
    theta = theta_rad[i]
    # The physical curve at radius r traces an arc of angle theta
    # Parameterized by eta in [0,1]
    # The physical angular position alpha(eta) satisfies:
    # cos(alpha) = x/r, sin(alpha) = y/r
    # For the NURBS curve: x(eta) = (N0*r + N1*w*r + N2*r*cos(theta)) / W
    #                       y(eta) = (N1*w*r*tan(theta/2) + N2*r*sin(theta)) / W
    # Simplify with r=1:
    # x = ((1-eta)^2 + 2*w*eta*(1-eta) + eta^2*cos(theta)) / W
    # y = (2*w*eta*(1-eta)*tan(theta/2) + eta^2*sin(theta)) / W

    # dalpha/deta gives the angular velocity which relates to the metric
    # The Jacobian determinant in the circumferential direction is ~ dalpha/deta

    N = 100
    etas = np.linspace(0, 1, N)
    dalpha_deta = np.zeros(N)
    for j in range(N):
        eta = etas[j]
        Wv = W_func(eta, w)
        N0 = (1-eta)**2; N1 = 2*eta*(1-eta); N2 = eta**2
        x = (N0*1 + N1*w*1 + N2*np.cos(theta)) / Wv
        y = (N1*w*np.tan(theta/2) + N2*np.sin(theta)) / Wv
        # numerical derivative
        deps = 1e-7
        eta2 = eta + deps
        Wv2 = W_func(eta2, w)
        N0_2 = (1-eta2)**2; N1_2 = 2*eta2*(1-eta2); N2_2 = eta2**2
        x2 = (N0_2*1 + N1_2*w*1 + N2_2*np.cos(theta)) / Wv2
        y2 = (N1_2*w*np.tan(theta/2) + N2_2*np.sin(theta)) / Wv2
        dx = (x2-x)/deps
        dy = (y2-y)/deps
        dalpha_deta[j] = np.sqrt(dx**2 + dy**2)  # arc length speed

    speed_ratio = np.max(dalpha_deta) / np.min(dalpha_deta)
    print(f"  theta={theta_deg[i]:3d}: speed_ratio={speed_ratio:.4f}, speed_ratio^2={speed_ratio**2:.4f}, cond_ratio={ratio[i]:.4f}")
