#!/usr/bin/env python3
"""1D study: how does the NURBS weight affect conditioning of a 1D Laplacian?

Solve -u'' = f on a curved line (circular arc) parameterized by NURBS.
This isolates the weight effect in 1D.
"""
import sys, os
import numpy as np
sys.path.insert(0, '/home/ubuntu/forge')
os.environ.setdefault('DISPLAY', ':99')

from forge.engine.session import ForgeSession
s = ForgeSession()
s.eval('addpath("ForgeHome/tiga")')

# For a 1D NURBS curve of arc angle theta, with weight w on middle control point:
# x(eta) = (N0*r + N1*w*r + N2*r*cos(theta)) / W
# y(eta) = (N1*w*r*tan(theta/2) + N2*r*sin(theta)) / W
# The physical coordinate is the arc length s(eta)
# The Jacobian is ds/deta = sqrt((dx/deta)^2 + (dy/deta)^2)
# The 1D stiffness matrix: K_AB = integral (dR_A/ds)(dR_B/ds) ds
# = integral (dR_A/deta)(dR_B/deta) / (ds/deta)^2 * (ds/deta) deta
# = integral (dR_A/deta)(dR_B/deta) / (ds/deta) deta

# The condition number depends on max/min of 1/(ds/deta)

# Let's compute this numerically for a 1D problem on a circular arc

def compute_1d_cond(nel, p, w_mid, theta, R=1.0):
    """1D NURBS-mapped Laplacian on arc of angle theta, radius R."""
    # Knot vector
    interior = np.linspace(0, 1, nel+1)[1:-1]
    Xi = np.concatenate([np.zeros(p+1), interior, np.ones(p+1)])
    n = len(Xi) - p - 1

    # NURBS control points for arc
    # P0 = (R, 0), P1 = (R, R*tan(theta/2)), P2 = (R*cos(theta), R*sin(theta))
    CPx = np.array([R, R, R*np.cos(theta)])
    CPy = np.array([0, R*np.tan(theta/2), R*np.sin(theta)])
    Cw = np.array([1.0, w_mid, 1.0])

    # Gauss quadrature
    nqp = p + 2
    # Use scipy for Gauss points
    from numpy.polynomial.legendre import leggauss
    gp, gw_q = leggauss(nqp)

    knots = np.unique(Xi)
    nel_actual = len(knots) - 1

    K = np.zeros((n, n))

    for e in range(nel_actual):
        xi_a, xi_b = knots[e], knots[e+1]
        if xi_b - xi_a < 1e-14:
            continue
        Je = (xi_b - xi_a) / 2

        for q in range(nqp):
            xi = (xi_a + xi_b) / 2 + Je * gp[q]

            # B-spline basis and derivatives
            # Simple implementation for p=2
            N, dN = eval_bspline(xi, p, Xi, n)

            # NURBS
            W = np.dot(N, Cw)
            dW = np.dot(dN, Cw)

            R_vals = N * Cw / W
            dR_vals = (dN * Cw * W - N * Cw * dW) / W**2

            # Physical coordinates
            x = np.dot(R_vals, CPx)
            y = np.dot(R_vals, CPy)
            dx_dxi = np.dot(dR_vals, CPx)
            dy_dxi = np.dot(dR_vals, CPy)

            # Jacobian (arc length speed)
            J_phys = np.sqrt(dx_dxi**2 + dy_dxi**2)
            if J_phys < 1e-15:
                continue

            # Stiffness: int (dR/ds)(dR/ds) ds = int (dR/dxi)^2 / J_phys dxi
            wt = gw_q[q] * Je

            for A in range(n):
                for B in range(n):
                    K[A, B] += dR_vals[A] * dR_vals[B] / J_phys * wt

    # BCs: fix first and last DOF
    bc = [0, n-1]
    free = np.setdiff1d(np.arange(n), bc)

    Kf = K[np.ix_(free, free)]
    ev = np.sort(np.real(np.linalg.eigvalsh(Kf)))
    ev_pos = ev[ev > 1e-10]

    if len(ev_pos) == 0:
        return 1e10, 0, 0

    return ev_pos[-1] / ev_pos[0], ev_pos[0], ev_pos[-1]


def eval_bspline(xi, p, Xi, n):
    """Evaluate all B-spline basis functions and derivatives at xi."""
    N = np.zeros(n)
    dN = np.zeros(n)

    # Find span
    if xi >= Xi[-p-1]:
        xi = Xi[-p-1] - 1e-14

    span = p
    for i in range(p, n):
        if Xi[i] <= xi < Xi[i+1]:
            span = i
            break

    # Cox-de Boor for p=2
    # Local basis: N[span-p]..N[span]
    left = np.zeros(p+1)
    right = np.zeros(p+1)
    ndu = np.zeros((p+1, p+1))
    ndu[0, 0] = 1.0

    for j in range(1, p+1):
        left[j] = xi - Xi[span+1-j]
        right[j] = Xi[span+j] - xi
        saved = 0.0
        for r in range(j):
            ndu[j, r] = right[r+1] + left[j-r]
            temp = ndu[r, j-1] / ndu[j, r]
            ndu[r, j] = saved + right[r+1] * temp
            saved = left[j-r] * temp
        ndu[j, j] = saved

    # Basis values
    for j in range(p+1):
        idx = span - p + j
        if 0 <= idx < n:
            N[idx] = ndu[j, p]

    # Derivatives (first order)
    a = np.zeros((2, p+1))
    for r in range(p+1):
        s1, s2 = 0, 1
        a[0, 0] = 1.0
        # k=1 derivative
        d = 0.0
        rr = r
        if rr >= 1:
            a[s2, 0] = a[s1, 0] / ndu[p, rr-1]
            d = a[s2, 0] * ndu[rr-1, p-1]
        if rr <= p-1:
            a[s2, 1] = -a[s1, 0] / ndu[p, rr]
            d += a[s2, 1] * ndu[rr, p-1]

        idx = span - p + r
        if 0 <= idx < n:
            dN[idx] = d * p

        a[s1, :] = 0
        a[s2, :] = 0

    return N, dN


print("=== 1D Weight-Conditioning Study ===\n", flush=True)

nel = 8
p = 2
theta = np.pi / 2  # quarter circle

# Weight sweep
print("1D conditioning vs weight (nel=8, p=2, quarter circle):", flush=True)
w_vals = np.logspace(-0.5, 0.8, 25)
cond_1d = []
for w in w_vals:
    c, lm, lx = compute_1d_cond(nel, p, w, theta)
    cond_1d.append(c)
    print(f"  w={w:.4f}  cond={c:.4e}", flush=True)

cond_1d = np.array(cond_1d)
c_bsp_1d = compute_1d_cond(nel, p, 1.0, theta)[0]

print(f"\n  B-spline (w=1): cond={c_bsp_1d:.4e}")
c_geo_1d = compute_1d_cond(nel, p, 1/np.sqrt(2), theta)[0]
print(f"  Geometric (w=1/sqrt2): cond={c_geo_1d:.4e}")
print(f"  1D geo/bsp ratio: {c_geo_1d/c_bsp_1d:.4f}")
print(f"  (Compare 2D ratio: 1.4906)")

# Arc angle study in 1D
print(f"\n1D conditioning ratio vs arc angle:", flush=True)
for theta_d in [30, 45, 60, 90, 120]:
    theta_r = theta_d * np.pi / 180
    w_g = np.cos(theta_r/2)
    c_g = compute_1d_cond(nel, p, w_g, theta_r)[0]
    c_b = compute_1d_cond(nel, p, 1.0, theta_r)[0]
    W_ratio = 2 / (1 + w_g)
    print(f"  theta={theta_d:3d}  w_geo={w_g:.4f}  ratio={c_g/c_b:.4f}  W_ratio={W_ratio:.4f}  W_ratio^2={W_ratio**2:.4f}")

print("\n=== 1D Study Complete ===", flush=True)
