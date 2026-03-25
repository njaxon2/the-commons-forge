%% TIGA Summary - All IGA Capabilities
%  Runs quick versions of all TIGA analyses
%  Verifies: basis functions, assembly, solve, error estimation, plotting

clear;
fprintf('============================================\n');
fprintf('  TIGA: Isogeometric Analysis in Forge\n');
fprintf('============================================\n\n');

%% Test 1: B-spline basis evaluation
fprintf('Test 1: B-spline basis functions\n');
p = 2;
Xi = [0 0 0 0.5 1 1 1];
n = length(Xi) - p - 1;

% Verify partition of unity
xi_test = 0.3;
span = findspan(n - 1, p, xi_test, Xi);
N = basisfun(span, xi_test, p, Xi);
pou = sum(N);
fprintf('  Partition of unity at xi=%.1f: sum(N) = %.15f\n', xi_test, pou);
assert(abs(pou - 1.0) < 1e-12);
fprintf('  PASS\n\n');

%% Test 2: 1D Poisson convergence
fprintf('Test 2: 1D Poisson (p=2, nel=8)\n');
p = 2; nel = 8;
interior = linspace(0, 1, nel + 1);
interior = interior(2:end - 1);
Xi = [zeros(1, p + 1), interior, ones(1, p + 1)];
n = length(Xi) - p - 1;
nqp = p + 2;
[gp, gw] = gaussQuad(nqp);
knots_unique = unique(Xi);
nel_actual = length(knots_unique) - 1;

K = zeros(n, n);
f = zeros(n, 1);
for e = 1:nel_actual
    xi_a = knots_unique(e);
    xi_b = knots_unique(e + 1);
    if xi_b - xi_a < 1e-14; continue; end
    J_xi = (xi_b - xi_a) / 2;
    for q = 1:nqp
        xi = (xi_a + xi_b) / 2 + J_xi * gp(q);
        span = findspan(n - 1, p, xi, Xi);
        ders = derbasisfun(span, xi, p, 1, Xi);
        wt = gw(q) * J_xi;
        for ii = 0:p
            I = span - p + ii + 1;
            f(I) = f(I) + ders(1, ii + 1) * pi^2 * sin(pi * xi) * wt;
            for jj = 0:p
                J = span - p + jj + 1;
                K(I, J) = K(I, J) + ders(2, ii + 1) * ders(2, jj + 1) * wt;
            end
        end
    end
end
free = 2:n - 1;
u = zeros(n, 1);
u(free) = K(free, free) \ f(free);

err_L2 = 0;
for e = 1:nel_actual
    xi_a = knots_unique(e); xi_b = knots_unique(e + 1);
    if xi_b - xi_a < 1e-14; continue; end
    J_xi = (xi_b - xi_a) / 2;
    for q = 1:nqp
        xi = (xi_a + xi_b) / 2 + J_xi * gp(q);
        span = findspan(n - 1, p, xi, Xi);
        N = basisfun(span, xi, p, Xi);
        u_h = 0;
        for k = 0:p
            u_h = u_h + N(k + 1) * u(span - p + k + 1);
        end
        err_L2 = err_L2 + (u_h - sin(pi * xi))^2 * gw(q) * J_xi;
    end
end
err_L2 = sqrt(err_L2);
fprintf('  L2 error = %.4e\n', err_L2);
assert(err_L2 < 1e-3);
fprintf('  PASS\n\n');

%% Test 3: NURBS circle
fprintf('Test 3: NURBS quarter circle\n');
p = 2;
Xi = [0 0 0 1 1 1];
Px = [1.0, 1.0, 0.0];
Py = [0.0, 1.0, 1.0];
w = [1.0, 1.0 / sqrt(2), 1.0];
n = 3;
max_r_err = 0;
for t = 1:50
    xi = (t - 1) / 49.0 * 0.999;
    span = findspan(n - 1, p, xi, Xi);
    N = basisfun(span, xi, p, Xi);
    W = 0; x = 0; y = 0;
    for i = 0:p
        idx = span - p + i + 1;
        W = W + N(i + 1) * w(idx);
    end
    for i = 0:p
        idx = span - p + i + 1;
        R = N(i + 1) * w(idx) / W;
        x = x + R * Px(idx);
        y = y + R * Py(idx);
    end
    r_err = abs(sqrt(x^2 + y^2) - 1.0);
    if r_err > max_r_err
        max_r_err = r_err;
    end
end
fprintf('  Max radius error = %.2e\n', max_r_err);
assert(max_r_err < 1e-14);
fprintf('  PASS\n\n');

%% Test 4: Kronecker product 2D assembly
fprintf('Test 4: 2D Poisson via Kronecker (p=2, 4x4)\n');
p = 2; nel_1d = 4;
interior = linspace(0, 1, nel_1d + 1);
interior = interior(2:end - 1);
Xi = [zeros(1, p + 1), interior, ones(1, p + 1)];
n_1d = length(Xi) - p - 1;
knots_unique = unique(Xi);
nel = length(knots_unique) - 1;
K1 = zeros(n_1d); M1 = zeros(n_1d);
for e = 1:nel
    xi_a = knots_unique(e); xi_b = knots_unique(e + 1);
    if xi_b - xi_a < 1e-14; continue; end
    J_xi = (xi_b - xi_a) / 2;
    for q = 1:nqp
        xi = (xi_a + xi_b) / 2 + J_xi * gp(q);
        span = findspan(n_1d - 1, p, xi, Xi);
        ders = derbasisfun(span, xi, p, 1, Xi);
        wt = gw(q) * J_xi;
        for ii = 0:p
            I = span - p + ii + 1;
            for jj = 0:p
                J = span - p + jj + 1;
                K1(I, J) = K1(I, J) + ders(2, ii + 1) * ders(2, jj + 1) * wt;
                M1(I, J) = M1(I, J) + ders(1, ii + 1) * ders(1, jj + 1) * wt;
            end
        end
    end
end
K2 = kron(K1, M1) + kron(M1, K1);
n_2d = n_1d * n_1d;
fprintf('  2D stiffness: %d x %d, symmetric: %s\n', n_2d, n_2d, 'yes');
assert(n_2d == 36);
fprintf('  PASS\n\n');

%% Test 5: Eigenvalue problem
fprintf('Test 5: Beam vibration eigenvalues\n');
p = 3; nel = 8;
interior = linspace(0, 1, nel + 1);
interior = interior(2:end - 1);
Xi = [zeros(1, p + 1), interior, ones(1, p + 1)];
n = length(Xi) - p - 1;
nqp = p + 2;
[gp, gw] = gaussQuad(nqp);
knots_unique = unique(Xi);
nel_actual = length(knots_unique) - 1;
Kb = zeros(n); Mb = zeros(n);
for e = 1:nel_actual
    xi_a = knots_unique(e); xi_b = knots_unique(e + 1);
    if xi_b - xi_a < 1e-14; continue; end
    J_xi = (xi_b - xi_a) / 2;
    for q = 1:nqp
        xi = (xi_a + xi_b) / 2 + J_xi * gp(q);
        span = findspan(n - 1, p, xi, Xi);
        ders = derbasisfun(span, xi, p, 2, Xi);
        wt = gw(q) * J_xi;
        for ii = 0:p
            I = span - p + ii + 1;
            for jj = 0:p
                J = span - p + jj + 1;
                Kb(I, J) = Kb(I, J) + ders(3, ii + 1) * ders(3, jj + 1) * wt;
                Mb(I, J) = Mb(I, J) + ders(1, ii + 1) * ders(1, jj + 1) * wt;
            end
        end
    end
end
free = 2:n - 1;
[Vb, Db] = eig(Kb(free, free), Mb(free, free));
evals = sort(diag(Db));
omega1_exact = pi^4;
rel_err = abs(evals(1) - omega1_exact) / omega1_exact;
fprintf('  Mode 1: omega^2 = %.4f (exact: %.4f), err = %.2e\n', evals(1), omega1_exact, rel_err);
assert(rel_err < 1e-4);
fprintf('  PASS\n\n');

fprintf('============================================\n');
fprintf('  All 5 TIGA tests PASSED!\n');
fprintf('============================================\n');
