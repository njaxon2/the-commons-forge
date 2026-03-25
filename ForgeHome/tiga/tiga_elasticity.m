%% TIGA 2D Plane Stress Elasticity (v2)
%  Full 2D element assembly with B-matrix approach
%  Manufactured solution: u_x = u_y = sin(pi*x)*sin(pi*y)
%  Tests: strain-displacement B matrix, plane stress, vector PDE, multi-DOF

clear;
fprintf('=== TIGA 2D Plane Stress Elasticity ===\n\n');

% Material
E = 1.0;
nu = 0.3;

% Plane stress constitutive matrix [3x3]
C11 = E / (1 - nu^2);
C12 = nu * E / (1 - nu^2);
C33 = E / (2 * (1 + nu));
Cmat = [C11, C12, 0; C12, C11, 0; 0, 0, C33];

fprintf('  E=%.1f, nu=%.1f\n', E, nu);
fprintf('  C11=%.4f, C12=%.4f, C33=%.4f\n', C11, C12, C33);

p = 2;
nel_1d = 8;

% Build 1D knot vector
interior = linspace(0, 1, nel_1d + 1);
interior = interior(2:end - 1);
Xi = [zeros(1, p + 1), interior, ones(1, p + 1)];
n_1d = length(Xi) - p - 1;
n_2d = n_1d * n_1d;
n_dof = 2 * n_2d;

fprintf('  p=%d, nel_1d=%d, n_1d=%d\n', p, nel_1d, n_1d);
fprintf('  2D nodes: %d, total DOFs: %d\n', n_2d, n_dof);

% Quadrature
nqp = p + 2;
[gp, gw] = gaussQuad(nqp);
knots_x = unique(Xi);
knots_y = unique(Xi);
nel_x = length(knots_x) - 1;
nel_y = length(knots_y) - 1;

%% Full 2D element-by-element assembly
K_glob = zeros(n_dof, n_dof);
f_glob = zeros(n_dof, 1);

for ex = 1:nel_x
    xi_a = knots_x(ex);
    xi_b = knots_x(ex + 1);
    if xi_b - xi_a < 1e-14; continue; end
    Jx = (xi_b - xi_a) / 2;

    for ey = 1:nel_y
        eta_a = knots_y(ey);
        eta_b = knots_y(ey + 1);
        if eta_b - eta_a < 1e-14; continue; end
        Jy = (eta_b - eta_a) / 2;

        for qx = 1:nqp
            xi = (xi_a + xi_b) / 2 + Jx * gp(qx);
            span_x = findspan(n_1d - 1, p, xi, Xi);
            ders_x = derbasisfun(span_x, xi, p, 1, Xi);
            Nx = ders_x(1, :);
            dNx = ders_x(2, :);

            for qy = 1:nqp
                eta = (eta_a + eta_b) / 2 + Jy * gp(qy);
                span_y = findspan(n_1d - 1, p, eta, Xi);
                ders_y = derbasisfun(span_y, eta, p, 1, Xi);
                Ny = ders_y(1, :);
                dNy = ders_y(2, :);

                wt = gw(qx) * Jx * gw(qy) * Jy;

                % Body force (from manufactured solution)
                % f_x = pi^2*((C11+C33)*sin*sin - (C12+C33)*cos*cos)
                % f_y = same (by symmetry since C11=C22 and u_x=u_y)
                S = sin(pi * xi) * sin(pi * eta);
                CC = cos(pi * xi) * cos(pi * eta);
                fx_val = pi^2 * ((C11 + C33) * S - (C12 + C33) * CC);
                fy_val = pi^2 * ((C33 + C11) * S - (C33 + C12) * CC);

                % Loop over test and trial functions
                for ii = 0:p
                    I_x = span_x - p + ii + 1;
                    for jj = 0:p
                        I_y = span_y - p + jj + 1;
                        glob_A = (I_y - 1) * n_1d + I_x;

                        % Test function A derivatives
                        dNA_dx = dNx(ii + 1) * Ny(jj + 1);
                        dNA_dy = Nx(ii + 1) * dNy(jj + 1);
                        NA = Nx(ii + 1) * Ny(jj + 1);

                        % Load vector
                        f_glob(glob_A) = f_glob(glob_A) + NA * fx_val * wt;
                        f_glob(n_2d + glob_A) = f_glob(n_2d + glob_A) + NA * fy_val * wt;

                        % Stiffness: loop over trial functions
                        for kk = 0:p
                            J_x = span_x - p + kk + 1;
                            for ll = 0:p
                                J_y = span_y - p + ll + 1;
                                glob_B = (J_y - 1) * n_1d + J_x;

                                % Trial function B derivatives
                                dNB_dx = dNx(kk + 1) * Ny(ll + 1);
                                dNB_dy = Nx(kk + 1) * dNy(ll + 1);

                                % K_xx block: C11*dNA/dx*dNB/dx + C33*dNA/dy*dNB/dy
                                K_glob(glob_A, glob_B) = K_glob(glob_A, glob_B) + (C11 * dNA_dx * dNB_dx + C33 * dNA_dy * dNB_dy) * wt;

                                % K_yy block: C33*dNA/dx*dNB/dx + C11*dNA/dy*dNB/dy
                                K_glob(n_2d + glob_A, n_2d + glob_B) = K_glob(n_2d + glob_A, n_2d + glob_B) + (C33 * dNA_dx * dNB_dx + C11 * dNA_dy * dNB_dy) * wt;

                                % K_xy block: C12*dNA/dx*dNB/dy + C33*dNA/dy*dNB/dx
                                K_glob(glob_A, n_2d + glob_B) = K_glob(glob_A, n_2d + glob_B) + (C12 * dNA_dx * dNB_dy + C33 * dNA_dy * dNB_dx) * wt;

                                % K_yx block: C12*dNA/dy*dNB/dx + C33*dNA/dx*dNB/dy
                                K_glob(n_2d + glob_A, glob_B) = K_glob(n_2d + glob_A, glob_B) + (C12 * dNA_dy * dNB_dx + C33 * dNA_dx * dNB_dy) * wt;
                            end
                        end
                    end
                end
            end
        end
    end
end

fprintf('  Assembled %d x %d stiffness\n', n_dof, n_dof);

%% Boundary conditions: u = 0 on all boundaries
bc_nodes = [];
for i = 1:n_1d
    bc_nodes = [bc_nodes, i, (n_1d - 1) * n_1d + i, (i - 1) * n_1d + 1, (i - 1) * n_1d + n_1d];
end
bc_nodes = unique(bc_nodes);

% BC dofs for both components
bc_dofs = [bc_nodes, bc_nodes + n_2d];
free_dofs = setdiff(1:n_dof, bc_dofs);

fprintf('  BC DOFs: %d, Free DOFs: %d\n', length(bc_dofs), length(free_dofs));

%% Solve
tic;
K_f = K_glob(free_dofs, free_dofs);
f_f = f_glob(free_dofs);
u_f = K_f \ f_f;
t_solve = toc;
u_sol = zeros(n_dof, 1);
u_sol(free_dofs) = u_f;

% Extract components
ux = u_sol(1:n_2d);
uy = u_sol(n_2d + 1:n_dof);

fprintf('  Solve time: %.4f s\n', t_solve);
fprintf('  max|u_x| = %.6f, max|u_y| = %.6f\n', max(abs(ux)), max(abs(uy)));

%% Compute L2 error
err_L2_x = 0;
err_L2_y = 0;

for ex = 1:nel_x
    xi_a = knots_x(ex);
    xi_b = knots_x(ex + 1);
    if xi_b - xi_a < 1e-14; continue; end
    Jx = (xi_b - xi_a) / 2;
    for ey = 1:nel_y
        eta_a = knots_y(ey);
        eta_b = knots_y(ey + 1);
        if eta_b - eta_a < 1e-14; continue; end
        Jy = (eta_b - eta_a) / 2;
        for qx = 1:nqp
            xi = (xi_a + xi_b) / 2 + Jx * gp(qx);
            span_x = findspan(n_1d - 1, p, xi, Xi);
            Nx_v = basisfun(span_x, xi, p, Xi);
            for qy = 1:nqp
                eta = (eta_a + eta_b) / 2 + Jy * gp(qy);
                span_y = findspan(n_1d - 1, p, eta, Xi);
                Ny_v = basisfun(span_y, eta, p, Xi);

                ux_h = 0; uy_h = 0;
                for ii = 0:p
                    I_x = span_x - p + ii + 1;
                    for jj = 0:p
                        I_y = span_y - p + jj + 1;
                        glob = (I_y - 1) * n_1d + I_x;
                        ux_h = ux_h + Nx_v(ii + 1) * Ny_v(jj + 1) * ux(glob);
                        uy_h = uy_h + Nx_v(ii + 1) * Ny_v(jj + 1) * uy(glob);
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

if err_L2_x < 1e-3
    fprintf('  CONVERGED: Error below 1e-3 threshold\n');
end

%% Plot
n_plot = 30;
x_plot = linspace(0, 1 - 1e-10, n_plot);
y_plot = linspace(0, 1 - 1e-10, n_plot);
Ux_plot = zeros(n_plot, n_plot);
Uy_plot = zeros(n_plot, n_plot);

for ix = 1:n_plot
    xi = x_plot(ix);
    span_x = findspan(n_1d - 1, p, xi, Xi);
    Nx_v = basisfun(span_x, xi, p, Xi);
    for iy = 1:n_plot
        eta = y_plot(iy);
        span_y = findspan(n_1d - 1, p, eta, Xi);
        Ny_v = basisfun(span_y, eta, p, Xi);
        ux_h = 0; uy_h = 0;
        for ii = 0:p
            I_x = span_x - p + ii + 1;
            for jj = 0:p
                I_y = span_y - p + jj + 1;
                glob = (I_y - 1) * n_1d + I_x;
                ux_h = ux_h + Nx_v(ii + 1) * Ny_v(jj + 1) * ux(glob);
                uy_h = uy_h + Nx_v(ii + 1) * Ny_v(jj + 1) * uy(glob);
            end
        end
        Ux_plot(iy, ix) = ux_h;
        Uy_plot(iy, ix) = uy_h;
    end
end

U_mag = sqrt(Ux_plot.^2 + Uy_plot.^2);

figure(1);
subplot(1, 3, 1);
surf(x_plot, y_plot, Ux_plot);
title('u_x (IGA)');
xlabel('x'); ylabel('y');
colorbar;

subplot(1, 3, 2);
surf(x_plot, y_plot, Uy_plot);
title('u_y (IGA)');
xlabel('x'); ylabel('y');
colorbar;

subplot(1, 3, 3);
U_exact_plot = sin(pi * x_plot)' * sin(pi * y_plot);
surf(x_plot, y_plot, abs(Ux_plot - U_exact_plot));
title('|u_x - exact|');
xlabel('x'); ylabel('y');
colorbar;
drawnow;

fprintf('\n=== 2D Elasticity Complete ===\n');
