%% TIGA Annulus Solution Gallery
%  Showcases IGA solutions and eigenmodes on NURBS quarter annulus
%  Multiple surface plots with different views and colormaps
%  Tests: surf, view, colormap, subplot, title, colorbar on mapped geometry

clear;
fprintf('=== TIGA Annulus Solution Gallery ===\n\n');

% Geometry
R1 = 0.5; R2 = 1.5; p = 2;
nel_r = 8;
interior_r = linspace(0, 1, nel_r + 1);
interior_r = interior_r(2:end - 1);
Xi_r = [zeros(1, p + 1), interior_r, ones(1, p + 1)];
n_r = length(Xi_r) - p - 1;
Xi_t = [0 0 0 1 1 1];
n_t = 3;
n_2d = n_r * n_t;

r_cp = linspace(R1, R2, n_r);
CPx = zeros(n_r, n_t);
CPy = zeros(n_r, n_t);
Cw = ones(n_r, n_t);
for i = 1:n_r
    r = r_cp(i);
    CPx(i, 1) = r;     CPy(i, 1) = 0;     Cw(i, 1) = 1;
    CPx(i, 2) = r;     CPy(i, 2) = r;     Cw(i, 2) = 1 / sqrt(2);
    CPx(i, 3) = 0;     CPy(i, 3) = r;     Cw(i, 3) = 1;
end

%% Assembly helper (shared code)
nqp = p + 2;
[gp, gw] = gaussQuad(nqp);
knots_r = unique(Xi_r);
knots_t = unique(Xi_t);
nel_rad = length(knots_r) - 1;
nel_cir = length(knots_t) - 1;

K = zeros(n_2d, n_2d);
M = zeros(n_2d, n_2d);
f = zeros(n_2d, 1);

for er = 1:nel_rad
    xi_a = knots_r(er); xi_b = knots_r(er + 1);
    if xi_b - xi_a < 1e-14; continue; end
    Jr = (xi_b - xi_a) / 2;
    for et = 1:nel_cir
        eta_a = knots_t(et); eta_b = knots_t(et + 1);
        if eta_b - eta_a < 1e-14; continue; end
        Jt = (eta_b - eta_a) / 2;
        for qr = 1:nqp
            xi = (xi_a + xi_b) / 2 + Jr * gp(qr);
            span_r = findspan(n_r - 1, p, xi, Xi_r);
            ders_r = derbasisfun(span_r, xi, p, 1, Xi_r);
            Nr = ders_r(1, :); dNr = ders_r(2, :);
            for qt = 1:nqp
                eta = (eta_a + eta_b) / 2 + Jt * gp(qt);
                span_t = findspan(n_t - 1, p, eta, Xi_t);
                ders_t = derbasisfun(span_t, eta, p, 1, Xi_t);
                Nt = ders_t(1, :); dNt = ders_t(2, :);
                wt_q = gw(qr) * Jr * gw(qt) * Jt;

                W = 0; dW_dxi = 0; dW_deta = 0;
                for a = 0:p
                    ir = span_r - p + a + 1;
                    for b = 0:p
                        it = span_t - p + b + 1;
                        ww = Cw(ir, it);
                        W = W + Nr(a+1) * Nt(b+1) * ww;
                        dW_dxi = dW_dxi + dNr(a+1) * Nt(b+1) * ww;
                        dW_deta = dW_deta + Nr(a+1) * dNt(b+1) * ww;
                    end
                end

                x_phys = 0; y_phys = 0;
                dx_dxi = 0; dx_deta = 0; dy_dxi = 0; dy_deta = 0;
                for a = 0:p
                    ir = span_r - p + a + 1;
                    for b = 0:p
                        it = span_t - p + b + 1;
                        ww = Cw(ir, it);
                        R_val = Nr(a+1) * Nt(b+1) * ww / W;
                        dR_dxi = (dNr(a+1) * Nt(b+1) * ww * W - Nr(a+1) * Nt(b+1) * ww * dW_dxi) / W^2;
                        dR_deta = (Nr(a+1) * dNt(b+1) * ww * W - Nr(a+1) * Nt(b+1) * ww * dW_deta) / W^2;
                        x_phys = x_phys + R_val * CPx(ir, it);
                        y_phys = y_phys + R_val * CPy(ir, it);
                        dx_dxi = dx_dxi + dR_dxi * CPx(ir, it);
                        dx_deta = dx_deta + dR_deta * CPx(ir, it);
                        dy_dxi = dy_dxi + dR_dxi * CPy(ir, it);
                        dy_deta = dy_deta + dR_deta * CPy(ir, it);
                    end
                end

                detJ = dx_dxi * dy_deta - dx_deta * dy_dxi;
                if abs(detJ) < 1e-15; continue; end

                inv_J11 = dy_deta / detJ;
                inv_J12 = -dy_dxi / detJ;
                inv_J21 = -dx_deta / detJ;
                inv_J22 = dx_dxi / detJ;

                r_phys = sqrt(x_phys^2 + y_phys^2);
                f_val = 16 * r_phys^2 - 4 * (R1^2 + R2^2);

                for a = 0:p
                    ir = span_r - p + a + 1;
                    for b = 0:p
                        it = span_t - p + b + 1;
                        ww_A = Cw(ir, it);
                        R_A = Nr(a+1) * Nt(b+1) * ww_A / W;
                        dR_A_dxi = (dNr(a+1) * Nt(b+1) * ww_A * W - Nr(a+1) * Nt(b+1) * ww_A * dW_dxi) / W^2;
                        dR_A_deta = (Nr(a+1) * dNt(b+1) * ww_A * W - Nr(a+1) * Nt(b+1) * ww_A * dW_deta) / W^2;
                        dR_A_dx = inv_J11 * dR_A_dxi + inv_J12 * dR_A_deta;
                        dR_A_dy = inv_J21 * dR_A_dxi + inv_J22 * dR_A_deta;
                        glob_A = (it - 1) * n_r + ir;
                        f(glob_A) = f(glob_A) + R_A * f_val * abs(detJ) * wt_q;
                        for c = 0:p
                            jr = span_r - p + c + 1;
                            for d = 0:p
                                jt = span_t - p + d + 1;
                                ww_B = Cw(jr, jt);
                                R_B = Nr(c+1) * Nt(d+1) * ww_B / W;
                                dR_B_dxi = (dNr(c+1) * Nt(d+1) * ww_B * W - Nr(c+1) * Nt(d+1) * ww_B * dW_dxi) / W^2;
                                dR_B_deta = (Nr(c+1) * dNt(d+1) * ww_B * W - Nr(c+1) * Nt(d+1) * ww_B * dW_deta) / W^2;
                                dR_B_dx = inv_J11 * dR_B_dxi + inv_J12 * dR_B_deta;
                                dR_B_dy = inv_J21 * dR_B_dxi + inv_J22 * dR_B_deta;
                                glob_B = (jt - 1) * n_r + jr;
                                K(glob_A, glob_B) = K(glob_A, glob_B) + (dR_A_dx * dR_B_dx + dR_A_dy * dR_B_dy) * abs(detJ) * wt_q;
                                M(glob_A, glob_B) = M(glob_A, glob_B) + R_A * R_B * abs(detJ) * wt_q;
                            end
                        end
                    end
                end
            end
        end
    end
end

fprintf('  Assembled K, M, f (%d x %d)\n', n_2d, n_2d);

%% BCs and solve
bc_dofs = [];
for j = 1:n_t
    bc_dofs = [bc_dofs, (j - 1) * n_r + 1, (j - 1) * n_r + n_r];
end
bc_dofs = unique(bc_dofs);
free_dofs = setdiff(1:n_2d, bc_dofs);

% Poisson solution
u_poisson = zeros(n_2d, 1);
u_poisson(free_dofs) = K(free_dofs, free_dofs) \ f(free_dofs);

% Eigenvalue solve
Kf = K(free_dofs, free_dofs);
Mf = M(free_dofs, free_dofs);
[V, D] = eig(Kf, Mf);
lambdas = real(diag(D));
[lambdas, idx] = sort(lambdas);
V = real(V(:, idx));
pos_idx = find(lambdas > 0);
lambdas = lambdas(pos_idx);
V = V(:, pos_idx);

% Build eigenvectors
n_modes = min(4, length(lambdas));
u_modes = zeros(n_2d, n_modes);
for k = 1:n_modes
    u_modes(free_dofs, k) = V(:, k);
    u_modes(:, k) = u_modes(:, k) / max(abs(u_modes(:, k)));
end

fprintf('  Poisson max|u| = %.6f\n', max(abs(u_poisson)));
fprintf('  First eigenvalue = %.4f\n', lambdas(1));

%% Evaluate all fields on plotting grid
n_plot = 50;
xi_v = linspace(0, 1 - 1e-10, n_plot);
eta_v = linspace(0, 1 - 1e-10, n_plot);
Xp = zeros(n_plot, n_plot);
Yp = zeros(n_plot, n_plot);
Up_poisson = zeros(n_plot, n_plot);
Up_exact = zeros(n_plot, n_plot);
Up_error = zeros(n_plot, n_plot);
Up_mode1 = zeros(n_plot, n_plot);
Up_mode2 = zeros(n_plot, n_plot);
Up_mode3 = zeros(n_plot, n_plot);

for i = 1:n_plot
    xi = xi_v(i);
    span_r = findspan(n_r - 1, p, xi, Xi_r);
    Nr = basisfun(span_r, xi, p, Xi_r);
    for j = 1:n_plot
        eta = eta_v(j);
        span_t = findspan(n_t - 1, p, eta, Xi_t);
        Nt = basisfun(span_t, eta, p, Xi_t);

        W = 0; x = 0; y = 0;
        u_h = 0;
        u_m = zeros(n_modes, 1);
        for a = 0:p
            ir = span_r - p + a + 1;
            for b = 0:p
                it = span_t - p + b + 1;
                ww = Cw(ir, it);
                W = W + Nr(a+1) * Nt(b+1) * ww;
            end
        end
        for a = 0:p
            ir = span_r - p + a + 1;
            for b = 0:p
                it = span_t - p + b + 1;
                Rv = Nr(a+1) * Nt(b+1) * Cw(ir, it) / W;
                x = x + Rv * CPx(ir, it);
                y = y + Rv * CPy(ir, it);
                glob = (it - 1) * n_r + ir;
                u_h = u_h + Rv * u_poisson(glob);
                for k = 1:n_modes
                    u_m(k) = u_m(k) + Rv * u_modes(glob, k);
                end
            end
        end
        Xp(j, i) = x;
        Yp(j, i) = y;
        Up_poisson(j, i) = u_h;
        r = sqrt(x^2 + y^2);
        Up_exact(j, i) = (r^2 - R1^2) * (R2^2 - r^2);
        Up_error(j, i) = abs(u_h - Up_exact(j, i));
        if n_modes >= 1; Up_mode1(j, i) = u_m(1); end
        if n_modes >= 2; Up_mode2(j, i) = u_m(2); end
        if n_modes >= 3; Up_mode3(j, i) = u_m(3); end
    end
end

err_max = max(max(Up_error));
fprintf('  Max pointwise error: %.4e\n', err_max);

%% Gallery figure: 2x3 subplot layout
figure(1);

% Panel 1: Poisson solution
subplot(2, 3, 1);
surf(Xp, Yp, Up_poisson);
title('IGA Solution');
xlabel('x'); ylabel('y');
colorbar;

% Panel 2: Exact solution
subplot(2, 3, 2);
surf(Xp, Yp, Up_exact);
title('Exact Solution');
xlabel('x'); ylabel('y');
colorbar;

% Panel 3: Error field
subplot(2, 3, 3);
surf(Xp, Yp, Up_error);
title('Pointwise Error');
xlabel('x'); ylabel('y');
colorbar;

% Panel 4: Mode 1
subplot(2, 3, 4);
surf(Xp, Yp, Up_mode1);
title(sprintf('Mode 1 (L=%.1f)', lambdas(1)));
xlabel('x'); ylabel('y');
colorbar;

% Panel 5: Mode 2
if n_modes >= 2
    subplot(2, 3, 5);
    surf(Xp, Yp, Up_mode2);
    title(sprintf('Mode 2 (L=%.1f)', lambdas(2)));
    xlabel('x'); ylabel('y');
    colorbar;
end

% Panel 6: Mode 3
if n_modes >= 3
    subplot(2, 3, 6);
    surf(Xp, Yp, Up_mode3);
    title(sprintf('Mode 3 (L=%.1f)', lambdas(3)));
    xlabel('x'); ylabel('y');
    colorbar;
end

drawnow;

%% Top-down view figure
figure(2);
subplot(1, 2, 1);
surf(Xp, Yp, Up_poisson);
view(2);
title('Poisson (top view)');
xlabel('x'); ylabel('y');
colorbar;

subplot(1, 2, 2);
surf(Xp, Yp, Up_mode1);
view(2);
title('Mode 1 (top view)');
xlabel('x'); ylabel('y');
colorbar;

drawnow;

fprintf('\n=== Gallery Complete ===\n');
