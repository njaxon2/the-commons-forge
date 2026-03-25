%% TIGA Stress Recovery and Error Analysis
%  Compute stresses from IGA displacement solution
%  Verify strain-displacement relationship, constitutive law
%  Tests: derivative evaluation, stress computation, H1 error norm

clear;
fprintf('=== TIGA Stress Recovery ===\n\n');

% Material
E = 1.0;
nu = 0.3;
C11 = E / (1 - nu^2);
C12 = nu * E / (1 - nu^2);
C33 = E / (2 * (1 + nu));

p = 2;
nel_1d = 8;

% Build knot vector
interior = linspace(0, 1, nel_1d + 1);
interior = interior(2:end - 1);
Xi = [zeros(1, p + 1), interior, ones(1, p + 1)];
n_1d = length(Xi) - p - 1;
n_2d = n_1d * n_1d;
n_dof = 2 * n_2d;

fprintf('  p=%d, nel=%d, n_1d=%d, DOFs=%d\n', p, nel_1d, n_1d, n_dof);

% Quadrature
nqp = p + 2;
[gp, gw] = gaussQuad(nqp);
knots = unique(Xi);
nel = length(knots) - 1;

%% Assemble and solve (same as elasticity)
K_glob = zeros(n_dof, n_dof);
f_glob = zeros(n_dof, 1);

for ex = 1:nel
    xi_a = knots(ex); xi_b = knots(ex + 1);
    if xi_b - xi_a < 1e-14; continue; end
    Jx = (xi_b - xi_a) / 2;
    for ey = 1:nel
        eta_a = knots(ey); eta_b = knots(ey + 1);
        if eta_b - eta_a < 1e-14; continue; end
        Jy = (eta_b - eta_a) / 2;
        for qx = 1:nqp
            xi = (xi_a + xi_b) / 2 + Jx * gp(qx);
            span_x = findspan(n_1d - 1, p, xi, Xi);
            ders_x = derbasisfun(span_x, xi, p, 1, Xi);
            Nx = ders_x(1, :); dNx = ders_x(2, :);
            for qy = 1:nqp
                eta = (eta_a + eta_b) / 2 + Jy * gp(qy);
                span_y = findspan(n_1d - 1, p, eta, Xi);
                ders_y = derbasisfun(span_y, eta, p, 1, Xi);
                Ny = ders_y(1, :); dNy = ders_y(2, :);
                wt = gw(qx) * Jx * gw(qy) * Jy;

                S = sin(pi * xi) * sin(pi * eta);
                CC = cos(pi * xi) * cos(pi * eta);
                fx = pi^2 * ((C11 + C33) * S - (C12 + C33) * CC);
                fy = pi^2 * ((C33 + C11) * S - (C33 + C12) * CC);

                for ii = 0:p
                    Ix = span_x - p + ii + 1;
                    for jj = 0:p
                        Iy = span_y - p + jj + 1;
                        gA = (Iy - 1) * n_1d + Ix;
                        dA_dx = dNx(ii + 1) * Ny(jj + 1);
                        dA_dy = Nx(ii + 1) * dNy(jj + 1);
                        NA = Nx(ii + 1) * Ny(jj + 1);
                        f_glob(gA) = f_glob(gA) + NA * fx * wt;
                        f_glob(n_2d + gA) = f_glob(n_2d + gA) + NA * fy * wt;
                        for kk = 0:p
                            Jx_t = span_x - p + kk + 1;
                            for ll = 0:p
                                Jy_t = span_y - p + ll + 1;
                                gB = (Jy_t - 1) * n_1d + Jx_t;
                                dB_dx = dNx(kk + 1) * Ny(ll + 1);
                                dB_dy = Nx(kk + 1) * dNy(ll + 1);
                                K_glob(gA, gB) = K_glob(gA, gB) + (C11 * dA_dx * dB_dx + C33 * dA_dy * dB_dy) * wt;
                                K_glob(n_2d + gA, n_2d + gB) = K_glob(n_2d + gA, n_2d + gB) + (C33 * dA_dx * dB_dx + C11 * dA_dy * dB_dy) * wt;
                                K_glob(gA, n_2d + gB) = K_glob(gA, n_2d + gB) + (C12 * dA_dx * dB_dy + C33 * dA_dy * dB_dx) * wt;
                                K_glob(n_2d + gA, gB) = K_glob(n_2d + gA, gB) + (C12 * dA_dy * dB_dx + C33 * dA_dx * dB_dy) * wt;
                            end
                        end
                    end
                end
            end
        end
    end
end

% BCs
bc_nodes = [];
for i = 1:n_1d
    bc_nodes = [bc_nodes, i, (n_1d - 1) * n_1d + i, (i - 1) * n_1d + 1, (i - 1) * n_1d + n_1d];
end
bc_nodes = unique(bc_nodes);
bc_dofs = [bc_nodes, bc_nodes + n_2d];
free_dofs = setdiff(1:n_dof, bc_dofs);

% Solve
u_sol = zeros(n_dof, 1);
u_sol(free_dofs) = K_glob(free_dofs, free_dofs) \ f_glob(free_dofs);
ux = u_sol(1:n_2d);
uy = u_sol(n_2d + 1:n_dof);

fprintf('  Solution computed: max|u_x|=%.6f\n', max(abs(ux)));

%% Stress recovery: compute L2 error of stress
% Exact strains from u = sin(pi*x)*sin(pi*y):
%   eps_xx = pi*cos(pi*x)*sin(pi*y)
%   eps_yy = pi*sin(pi*x)*cos(pi*y)
%   gamma_xy = pi*sin(pi*x)*cos(pi*y) + pi*cos(pi*x)*sin(pi*y)
% Exact stresses:
%   sigma_xx = C11*eps_xx + C12*eps_yy
%   sigma_yy = C12*eps_xx + C11*eps_yy
%   sigma_xy = C33*gamma_xy

err_sxx = 0; err_syy = 0; err_sxy = 0;
err_H1 = 0;

for ex = 1:nel
    xi_a = knots(ex); xi_b = knots(ex + 1);
    if xi_b - xi_a < 1e-14; continue; end
    Jx = (xi_b - xi_a) / 2;
    for ey = 1:nel
        eta_a = knots(ey); eta_b = knots(ey + 1);
        if eta_b - eta_a < 1e-14; continue; end
        Jy = (eta_b - eta_a) / 2;
        for qx = 1:nqp
            xi = (xi_a + xi_b) / 2 + Jx * gp(qx);
            span_x = findspan(n_1d - 1, p, xi, Xi);
            ders_x = derbasisfun(span_x, xi, p, 1, Xi);
            Nx_v = ders_x(1, :); dNx_v = ders_x(2, :);
            for qy = 1:nqp
                eta = (eta_a + eta_b) / 2 + Jy * gp(qy);
                span_y = findspan(n_1d - 1, p, eta, Xi);
                ders_y = derbasisfun(span_y, eta, p, 1, Xi);
                Ny_v = ders_y(1, :); dNy_v = ders_y(2, :);
                wt = gw(qx) * Jx * gw(qy) * Jy;

                % Compute IGA strain at quadrature point
                eps_xx_h = 0; eps_yy_h = 0; gamma_xy_h = 0;
                dux_dx = 0; dux_dy = 0; duy_dx = 0; duy_dy = 0;
                for ii = 0:p
                    Ix = span_x - p + ii + 1;
                    for jj = 0:p
                        Iy = span_y - p + jj + 1;
                        glob = (Iy - 1) * n_1d + Ix;
                        dN_dx = dNx_v(ii + 1) * Ny_v(jj + 1);
                        dN_dy = Nx_v(ii + 1) * dNy_v(jj + 1);
                        dux_dx = dux_dx + dN_dx * ux(glob);
                        dux_dy = dux_dy + dN_dy * ux(glob);
                        duy_dx = duy_dx + dN_dx * uy(glob);
                        duy_dy = duy_dy + dN_dy * uy(glob);
                    end
                end
                eps_xx_h = dux_dx;
                eps_yy_h = duy_dy;
                gamma_xy_h = dux_dy + duy_dx;

                % IGA stress
                sxx_h = C11 * eps_xx_h + C12 * eps_yy_h;
                syy_h = C12 * eps_xx_h + C11 * eps_yy_h;
                sxy_h = C33 * gamma_xy_h;

                % Exact strain
                eps_xx_ex = pi * cos(pi * xi) * sin(pi * eta);
                eps_yy_ex = pi * sin(pi * xi) * cos(pi * eta);
                gamma_xy_ex = pi * sin(pi * xi) * cos(pi * eta) + pi * cos(pi * xi) * sin(pi * eta);

                % Exact stress
                sxx_ex = C11 * eps_xx_ex + C12 * eps_yy_ex;
                syy_ex = C12 * eps_xx_ex + C11 * eps_yy_ex;
                sxy_ex = C33 * gamma_xy_ex;

                % Accumulate L2 errors
                err_sxx = err_sxx + (sxx_h - sxx_ex)^2 * wt;
                err_syy = err_syy + (syy_h - syy_ex)^2 * wt;
                err_sxy = err_sxy + (sxy_h - sxy_ex)^2 * wt;

                % H1 seminorm error (gradient of displacement)
                dux_dx_ex = pi * cos(pi * xi) * sin(pi * eta);
                dux_dy_ex = pi * sin(pi * xi) * cos(pi * eta);
                duy_dx_ex = pi * cos(pi * xi) * sin(pi * eta);
                duy_dy_ex = pi * sin(pi * xi) * cos(pi * eta);
                err_H1 = err_H1 + ((dux_dx - dux_dx_ex)^2 + (dux_dy - dux_dy_ex)^2 + (duy_dx - duy_dx_ex)^2 + (duy_dy - duy_dy_ex)^2) * wt;
            end
        end
    end
end

err_sxx = sqrt(err_sxx);
err_syy = sqrt(err_syy);
err_sxy = sqrt(err_sxy);
err_H1 = sqrt(err_H1);

fprintf('\n  Stress L2 errors:\n');
fprintf('    sigma_xx: %.6e\n', err_sxx);
fprintf('    sigma_yy: %.6e\n', err_syy);
fprintf('    sigma_xy: %.6e\n', err_sxy);
fprintf('  H1 seminorm error: %.6e\n', err_H1);

%% Plot stress field
n_plot = 25;
x_plot = linspace(0.01, 0.99, n_plot);
y_plot = linspace(0.01, 0.99, n_plot);
Sxx = zeros(n_plot, n_plot);
Syy = zeros(n_plot, n_plot);
Sxy = zeros(n_plot, n_plot);
VM = zeros(n_plot, n_plot);

for ix = 1:n_plot
    xi = x_plot(ix);
    span_x = findspan(n_1d - 1, p, xi, Xi);
    ders_x = derbasisfun(span_x, xi, p, 1, Xi);
    Nx_v = ders_x(1, :); dNx_v = ders_x(2, :);
    for iy = 1:n_plot
        eta = y_plot(iy);
        span_y = findspan(n_1d - 1, p, eta, Xi);
        ders_y = derbasisfun(span_y, eta, p, 1, Xi);
        Ny_v = ders_y(1, :); dNy_v = ders_y(2, :);

        dux_dx = 0; dux_dy = 0; duy_dx = 0; duy_dy = 0;
        for ii = 0:p
            Ix = span_x - p + ii + 1;
            for jj = 0:p
                Iy = span_y - p + jj + 1;
                glob = (Iy - 1) * n_1d + Ix;
                dN_dx = dNx_v(ii + 1) * Ny_v(jj + 1);
                dN_dy = Nx_v(ii + 1) * dNy_v(jj + 1);
                dux_dx = dux_dx + dN_dx * ux(glob);
                dux_dy = dux_dy + dN_dy * ux(glob);
                duy_dx = duy_dx + dN_dx * uy(glob);
                duy_dy = duy_dy + dN_dy * uy(glob);
            end
        end

        sxx = C11 * dux_dx + C12 * duy_dy;
        syy = C12 * dux_dx + C11 * duy_dy;
        sxy = C33 * (dux_dy + duy_dx);

        Sxx(iy, ix) = sxx;
        Syy(iy, ix) = syy;
        Sxy(iy, ix) = sxy;
        % Von Mises stress
        VM(iy, ix) = sqrt(sxx^2 - sxx * syy + syy^2 + 3 * sxy^2);
    end
end

figure(1);
subplot(2, 2, 1);
surf(x_plot, y_plot, Sxx);
title('sigma_{xx}');
xlabel('x'); ylabel('y');
colorbar;

subplot(2, 2, 2);
surf(x_plot, y_plot, Syy);
title('sigma_{yy}');
xlabel('x'); ylabel('y');
colorbar;

subplot(2, 2, 3);
surf(x_plot, y_plot, Sxy);
title('sigma_{xy}');
xlabel('x'); ylabel('y');
colorbar;

subplot(2, 2, 4);
surf(x_plot, y_plot, VM);
title('Von Mises');
xlabel('x'); ylabel('y');
colorbar;
drawnow;

fprintf('\n=== Stress Recovery Complete ===\n');
