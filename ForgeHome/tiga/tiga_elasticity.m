%% TIGA 2D Plane Stress Elasticity
%  Solve 2D elasticity on [0,1]x[0,1] with manufactured solution
%  Tests: multi-DOF assembly, plane stress constitutive, vector PDE

clear;
fprintf('=== TIGA 2D Plane Stress Elasticity ===\n\n');

% Material
E = 1.0;
nu = 0.3;

% Plane stress constitutive
C = E / (1 - nu^2) * [1, nu, 0; nu, 1, 0; 0, 0, (1 - nu) / 2];

p = 2;
nel_1d = 6;

% Build 1D knot vector
interior = linspace(0, 1, nel_1d + 1);
interior = interior(2:end - 1);
Xi = [zeros(1, p + 1), interior, ones(1, p + 1)];
n_1d = length(Xi) - p - 1;
n_2d = n_1d * n_1d;
n_dof = 2 * n_2d;

fprintf('  E=%.1f, nu=%.1f\n', E, nu);
fprintf('  p=%d, nel_1d=%d, n_1d=%d\n', p, nel_1d, n_1d);
fprintf('  2D nodes: %d, total DOFs: %d\n', n_2d, n_dof);

% Quadrature
nqp = p + 2;
[gp, gw] = gaussQuad(nqp);
knots_unique = unique(Xi);
nel = length(knots_unique) - 1;

% Manufactured solution:
% u_x(x,y) = sin(pi*x)*sin(pi*y)
% u_y(x,y) = sin(pi*x)*sin(pi*y)
% Body force computed from equilibrium

%% Assemble 1D matrices
K1 = zeros(n_1d, n_1d);
M1 = zeros(n_1d, n_1d);

for e = 1:nel
    xi_a = knots_unique(e);
    xi_b = knots_unique(e + 1);
    if xi_b - xi_a < 1e-14; continue; end
    J_xi = (xi_b - xi_a) / 2;
    for q = 1:nqp
        xi = (xi_a + xi_b) / 2 + J_xi * gp(q);
        span = findspan(n_1d - 1, p, xi, Xi);
        ders = derbasisfun(span, xi, p, 1, Xi);
        N_val = ders(1, :);
        dN_dx = ders(2, :);
        wt = gw(q) * J_xi;
        for ii = 0:p
            I = span - p + ii + 1;
            for jj = 0:p
                J = span - p + jj + 1;
                K1(I, J) = K1(I, J) + dN_dx(ii + 1) * dN_dx(jj + 1) * wt;
                M1(I, J) = M1(I, J) + N_val(ii + 1) * N_val(jj + 1) * wt;
            end
        end
    end
end

%% Build 2D elasticity stiffness
% For plane stress with manufactured solution
% K_xx = C(1,1)*kron(K1,M1) + C(3,3)*kron(M1,K1)
% K_yy = C(3,3)*kron(K1,M1) + C(2,2)*kron(M1,K1)
% K_xy = C(1,2)*kron(K1,K1') + C(3,3)*kron(K1',K1)  (cross terms)
% Note: for uniform mesh, K1 is symmetric, so K1' = K1

K_xx = C(1, 1) * kron(K1, M1) + C(3, 3) * kron(M1, K1);
K_yy = C(3, 3) * kron(K1, M1) + C(2, 2) * kron(M1, K1);
K_xy = (C(1, 2) + C(3, 3)) * kron(K1, K1);

% Assemble global stiffness [K_xx, K_xy; K_xy', K_yy]
K_glob = zeros(n_dof, n_dof);
K_glob(1:n_2d, 1:n_2d) = K_xx;
K_glob(1:n_2d, n_2d + 1:n_dof) = K_xy;
K_glob(n_2d + 1:n_dof, 1:n_2d) = K_xy;
K_glob(n_2d + 1:n_dof, n_2d + 1:n_dof) = K_yy;

fprintf('  Assembled global stiffness: %d x %d\n', n_dof, n_dof);

%% Body force for manufactured solution
% u = sin(pi*x)*sin(pi*y) for both components
% -div(sigma) = f
% f_x = (C11 + C33)*pi^2*sin(pi*x)*sin(pi*y)
% f_y = (C33 + C22)*pi^2*sin(pi*x)*sin(pi*y)

f_glob = zeros(n_dof, 1);

for ex = 1:nel
    xi_a = knots_unique(ex);
    xi_b = knots_unique(ex + 1);
    if xi_b - xi_a < 1e-14; continue; end
    Jx = (xi_b - xi_a) / 2;
    for ey = 1:nel
        eta_a = knots_unique(ey);
        eta_b = knots_unique(ey + 1);
        if eta_b - eta_a < 1e-14; continue; end
        Jy = (eta_b - eta_a) / 2;
        for qx = 1:nqp
            xi = (xi_a + xi_b) / 2 + Jx * gp(qx);
            span_x = findspan(n_1d - 1, p, xi, Xi);
            Nx = basisfun(span_x, xi, p, Xi);
            for qy = 1:nqp
                eta = (eta_a + eta_b) / 2 + Jy * gp(qy);
                span_y = findspan(n_1d - 1, p, eta, Xi);
                Ny = basisfun(span_y, eta, p, Xi);

                fx = (C(1, 1) + C(3, 3)) * pi^2 * sin(pi * xi) * sin(pi * eta);
                fy = (C(3, 3) + C(2, 2)) * pi^2 * sin(pi * xi) * sin(pi * eta);
                wt = gw(qx) * Jx * gw(qy) * Jy;

                for ii = 0:p
                    I = span_x - p + ii + 1;
                    for jj = 0:p
                        J = span_y - p + jj + 1;
                        glob = (J - 1) * n_1d + I;
                        f_glob(glob) = f_glob(glob) + Nx(ii + 1) * Ny(jj + 1) * fx * wt;
                        f_glob(n_2d + glob) = f_glob(n_2d + glob) + Nx(ii + 1) * Ny(jj + 1) * fy * wt;
                    end
                end
            end
        end
    end
end

%% Boundary conditions: u = 0 on all boundaries
bc_dofs_scalar = [];
for i = 1:n_1d
    bc_dofs_scalar = [bc_dofs_scalar, i, (n_1d - 1) * n_1d + i, (i - 1) * n_1d + 1, (i - 1) * n_1d + n_1d];
end
bc_dofs_scalar = unique(bc_dofs_scalar);

% BC dofs for both components
bc_dofs = [bc_dofs_scalar, bc_dofs_scalar + n_2d];
free_dofs = setdiff(1:n_dof, bc_dofs);

fprintf('  BC DOFs: %d, Free DOFs: %d\n', length(bc_dofs), length(free_dofs));

%% Solve
K_f = K_glob(free_dofs, free_dofs);
f_f = f_glob(free_dofs);
u_f = K_f \ f_f;
u_sol = zeros(n_dof, 1);
u_sol(free_dofs) = u_f;

% Extract components
ux = u_sol(1:n_2d);
uy = u_sol(n_2d + 1:n_dof);

fprintf('  max|u_x| = %.6f, max|u_y| = %.6f\n', max(abs(ux)), max(abs(uy)));

%% Compute L2 error
err_L2_x = 0;
err_L2_y = 0;

for ex = 1:nel
    xi_a = knots_unique(ex);
    xi_b = knots_unique(ex + 1);
    if xi_b - xi_a < 1e-14; continue; end
    Jx = (xi_b - xi_a) / 2;
    for ey = 1:nel
        eta_a = knots_unique(ey);
        eta_b = knots_unique(ey + 1);
        if eta_b - eta_a < 1e-14; continue; end
        Jy = (eta_b - eta_a) / 2;
        for qx = 1:nqp
            xi = (xi_a + xi_b) / 2 + Jx * gp(qx);
            span_x = findspan(n_1d - 1, p, xi, Xi);
            Nx = basisfun(span_x, xi, p, Xi);
            for qy = 1:nqp
                eta = (eta_a + eta_b) / 2 + Jy * gp(qy);
                span_y = findspan(n_1d - 1, p, eta, Xi);
                Ny = basisfun(span_y, eta, p, Xi);

                ux_h = 0; uy_h = 0;
                for ii = 0:p
                    I = span_x - p + ii + 1;
                    for jj = 0:p
                        J = span_y - p + jj + 1;
                        glob = (J - 1) * n_1d + I;
                        ux_h = ux_h + Nx(ii + 1) * Ny(jj + 1) * ux(glob);
                        uy_h = uy_h + Nx(ii + 1) * Ny(jj + 1) * uy(glob);
                    end
                end

                u_exact = sin(pi * xi) * sin(pi * eta);
                wt = gw(qx) * Jx * gw(qy) * Jy;
                err_L2_x = err_L2_x + (ux_h - u_exact)^2 * wt;
                err_L2_y = err_L2_y + (uy_h - u_exact)^2 * wt;
            end
        end
    end
end
err_L2_x = sqrt(err_L2_x);
err_L2_y = sqrt(err_L2_y);
fprintf('\n  L2 error (u_x) = %.6e\n', err_L2_x);
fprintf('  L2 error (u_y) = %.6e\n', err_L2_y);

%% Plot displacement magnitude
n_plot = 30;
x_plot = linspace(0, 1 - 1e-10, n_plot);
y_plot = linspace(0, 1 - 1e-10, n_plot);
Ux = zeros(n_plot, n_plot);
Uy = zeros(n_plot, n_plot);

for ix = 1:n_plot
    xi = x_plot(ix);
    span_x = findspan(n_1d - 1, p, xi, Xi);
    Nx = basisfun(span_x, xi, p, Xi);
    for iy = 1:n_plot
        eta = y_plot(iy);
        span_y = findspan(n_1d - 1, p, eta, Xi);
        Ny = basisfun(span_y, eta, p, Xi);
        ux_h = 0; uy_h = 0;
        for ii = 0:p
            I = span_x - p + ii + 1;
            for jj = 0:p
                J = span_y - p + jj + 1;
                glob = (J - 1) * n_1d + I;
                ux_h = ux_h + Nx(ii + 1) * Ny(jj + 1) * ux(glob);
                uy_h = uy_h + Nx(ii + 1) * Ny(jj + 1) * uy(glob);
            end
        end
        Ux(iy, ix) = ux_h;
        Uy(iy, ix) = uy_h;
    end
end

U_mag = sqrt(Ux.^2 + Uy.^2);

figure(1);
subplot(1, 2, 1);
surf(x_plot, y_plot, U_mag);
title('Displacement Magnitude');
xlabel('x'); ylabel('y'); zlabel('|u|');
colorbar;

subplot(1, 2, 2);
surf(x_plot, y_plot, abs(U_mag - sin(pi * x_plot)' * sin(pi * y_plot) * sqrt(2)));
title('Error in |u|');
xlabel('x'); ylabel('y');
colorbar;
drawnow;

fprintf('\n=== 2D Elasticity Complete ===\n');
