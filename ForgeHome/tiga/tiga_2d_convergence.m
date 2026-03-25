%% TIGA 2D h-Convergence Study
%  Verify optimal convergence rate for 2D IGA Poisson solver
%  -nabla^2 u = 2*pi^2*sin(pi*x)*sin(pi*y), u=0 on boundary

clear;
fprintf('=== TIGA 2D h-Convergence Study ===\n\n');

p = 2;
nel_values = [2 4 6 8];
n_mesh = length(nel_values);

dof_hist = zeros(1, n_mesh);
err_hist = zeros(1, n_mesh);
h_hist = zeros(1, n_mesh);

for im = 1:n_mesh
    nel_1d = nel_values(im);
    h = 1.0 / nel_1d;

    % Build 1D knot vector
    interior = linspace(0, 1, nel_1d + 1);
    interior = interior(2:end - 1);
    Xi = [zeros(1, p + 1), interior, ones(1, p + 1)];
    n_1d = length(Xi) - p - 1;
    n_2d = n_1d * n_1d;

    % Quadrature
    nqp = p + 2;
    [gp, gw] = gaussQuad(nqp);
    knots_unique = unique(Xi);
    nel = length(knots_unique) - 1;

    % 1D stiffness and mass
    K1 = zeros(n_1d, n_1d);
    M1 = zeros(n_1d, n_1d);

    for e = 1:nel
        xi_a = knots_unique(e);
        xi_b = knots_unique(e + 1);
        if xi_b - xi_a < 1e-14
            continue;
        end
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

    % 2D stiffness
    K2 = kron(K1, M1) + kron(M1, K1);

    % 2D load vector
    f2 = zeros(n_2d, 1);
    for ex = 1:nel
        xi_a = knots_unique(ex);
        xi_b = knots_unique(ex + 1);
        if xi_b - xi_a < 1e-14
            continue;
        end
        Jx = (xi_b - xi_a) / 2;
        for ey = 1:nel
            eta_a = knots_unique(ey);
            eta_b = knots_unique(ey + 1);
            if eta_b - eta_a < 1e-14
                continue;
            end
            Jy = (eta_b - eta_a) / 2;
            for qx = 1:nqp
                xi = (xi_a + xi_b) / 2 + Jx * gp(qx);
                span_x = findspan(n_1d - 1, p, xi, Xi);
                Nx = basisfun(span_x, xi, p, Xi);
                for qy = 1:nqp
                    eta = (eta_a + eta_b) / 2 + Jy * gp(qy);
                    span_y = findspan(n_1d - 1, p, eta, Xi);
                    Ny = basisfun(span_y, eta, p, Xi);
                    f_val = 2 * pi^2 * sin(pi * xi) * sin(pi * eta);
                    wt = gw(qx) * Jx * gw(qy) * Jy;
                    for ii = 0:p
                        I = span_x - p + ii + 1;
                        for jj = 0:p
                            J = span_y - p + jj + 1;
                            glob = (J - 1) * n_1d + I;
                            f2(glob) = f2(glob) + Nx(ii + 1) * Ny(jj + 1) * f_val * wt;
                        end
                    end
                end
            end
        end
    end

    % Boundary DOFs
    bc_dofs = [];
    for i = 1:n_1d
        bc_dofs = [bc_dofs, i, (n_1d - 1) * n_1d + i, (i - 1) * n_1d + 1, (i - 1) * n_1d + n_1d];
    end
    bc_dofs = unique(bc_dofs);
    free_dofs = setdiff(1:n_2d, bc_dofs);

    % Solve
    K_f = K2(free_dofs, free_dofs);
    f_f = f2(free_dofs);
    u_f = K_f \ f_f;
    u2 = zeros(n_2d, 1);
    u2(free_dofs) = u_f;

    % L2 error
    err_L2 = 0;
    for ex = 1:nel
        xi_a = knots_unique(ex);
        xi_b = knots_unique(ex + 1);
        if xi_b - xi_a < 1e-14
            continue;
        end
        Jx = (xi_b - xi_a) / 2;
        for ey = 1:nel
            eta_a = knots_unique(ey);
            eta_b = knots_unique(ey + 1);
            if eta_b - eta_a < 1e-14
                continue;
            end
            Jy = (eta_b - eta_a) / 2;
            for qx = 1:nqp
                xi = (xi_a + xi_b) / 2 + Jx * gp(qx);
                span_x = findspan(n_1d - 1, p, xi, Xi);
                Nx = basisfun(span_x, xi, p, Xi);
                for qy = 1:nqp
                    eta = (eta_a + eta_b) / 2 + Jy * gp(qy);
                    span_y = findspan(n_1d - 1, p, eta, Xi);
                    Ny = basisfun(span_y, eta, p, Xi);
                    u_h = 0;
                    for ii = 0:p
                        I = span_x - p + ii + 1;
                        for jj = 0:p
                            J = span_y - p + jj + 1;
                            glob = (J - 1) * n_1d + I;
                            u_h = u_h + Nx(ii + 1) * Ny(jj + 1) * u2(glob);
                        end
                    end
                    u_exact = sin(pi * xi) * sin(pi * eta);
                    err_L2 = err_L2 + (u_h - u_exact)^2 * gw(qx) * Jx * gw(qy) * Jy;
                end
            end
        end
    end
    err_L2 = sqrt(err_L2);

    dof_hist(im) = n_2d;
    err_hist(im) = err_L2;
    h_hist(im) = h;

    fprintf('  nel=%2d: n_2d=%4d, h=%.4f, L2 err = %.4e\n', nel_1d, n_2d, h, err_L2);
end

% Convergence rate
rate = log(err_hist(n_mesh - 1) / err_hist(n_mesh)) / log(h_hist(n_mesh - 1) / h_hist(n_mesh));
fprintf('\n  Convergence rate: %.2f (expected ~%d for p=%d)\n', rate, p + 1, p);

%% Plot
figure(1);
loglog(h_hist, err_hist, 'bo-', 'LineWidth', 1.5);
hold on;
% Reference slope
h_ref = [h_hist(1), h_hist(n_mesh)];
e_ref = err_hist(1) * (h_ref / h_ref(1)).^(p + 1);
loglog(h_ref, e_ref, 'r--', 'LineWidth', 1);
hold off;
xlabel('h');
ylabel('L2 Error');
title('2D IGA h-Convergence (p=2)');
legend('IGA', 'O(h^3) reference');
grid on;
drawnow;

fprintf('\n=== 2D Convergence Study Complete ===\n');
