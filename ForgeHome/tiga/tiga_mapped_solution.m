%% TIGA Solution on Mapped Geometry
%  Solve Poisson equation on a quarter annulus using NURBS mapping
%  Visualize solution field on the physical geometry
%  Tests: geometry mapping, Jacobian computation, solution visualization

clear;
fprintf('=== TIGA Solution on Mapped Geometry ===\n\n');

% Geometry: quarter annulus [R1, R2] x [0, pi/2]
R1 = 0.5; R2 = 1.5;
p = 2;

% Radial knot vector with refinement
nel_r = 6;
interior_r = linspace(0, 1, nel_r + 1);
interior_r = interior_r(2:end - 1);
Xi_r = [zeros(1, p + 1), interior_r, ones(1, p + 1)];
n_r = length(Xi_r) - p - 1;

% Circumferential: single element NURBS (exact quarter circle)
Xi_t = [0 0 0 1 1 1];
n_t = 3;

% Total 2D DOFs
n_2d = n_r * n_t;
fprintf('  Annulus: R1=%.1f, R2=%.1f\n', R1, R2);
fprintf('  Radial: p=%d, nel=%d, n_r=%d\n', p, nel_r, n_r);
fprintf('  Circum: n_t=%d (NURBS quarter circle)\n', n_t);
fprintf('  Total nodes: %d\n', n_2d);

% Control point positions for quarter annulus
% Radial control points are linearly spaced between R1 and R2
% Each radial station has 3 circumferential control points (quarter circle)
r_cp = linspace(R1, R2, n_r);

% For each radial station, quarter circle control points:
% P1 = (r, 0), P2 = (r, r) with w=1/sqrt(2), P3 = (0, r)
CPx = zeros(n_r, n_t);
CPy = zeros(n_r, n_t);
Cw = ones(n_r, n_t);

for i = 1:n_r
    r = r_cp(i);
    CPx(i, 1) = r;     CPy(i, 1) = 0;     Cw(i, 1) = 1;
    CPx(i, 2) = r;     CPy(i, 2) = r;     Cw(i, 2) = 1 / sqrt(2);
    CPx(i, 3) = 0;     CPy(i, 3) = r;     Cw(i, 3) = 1;
end

% Quadrature
nqp = p + 2;
[gp, gw] = gaussQuad(nqp);
knots_r = unique(Xi_r);
knots_t = unique(Xi_t);
nel_rad = length(knots_r) - 1;
nel_cir = length(knots_t) - 1;

%% Assemble stiffness and load on mapped geometry
% Manufactured solution in physical coords: u(x,y) = sin(pi*r/R2) where r=sqrt(x^2+y^2)
% This gives u=0 at r=0 (not applicable) and u=0 at r=R2

K = zeros(n_2d, n_2d);
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
            Nr = ders_r(1, :);
            dNr = ders_r(2, :);

            for qt = 1:nqp
                eta = (eta_a + eta_b) / 2 + Jt * gp(qt);
                span_t = findspan(n_t - 1, p, eta, Xi_t);
                ders_t = derbasisfun(span_t, eta, p, 1, Xi_t);
                Nt = ders_t(1, :);
                dNt = ders_t(2, :);

                wt_q = gw(qr) * Jr * gw(qt) * Jt;

                % Compute NURBS geometry mapping: (xi,eta) -> (x,y)
                W = 0; dW_dxi = 0; dW_deta = 0;
                x_phys = 0; y_phys = 0;
                dx_dxi = 0; dx_deta = 0;
                dy_dxi = 0; dy_deta = 0;

                for a = 0:p
                    ir = span_r - p + a + 1;
                    for b = 0:p
                        it = span_t - p + b + 1;
                        ww = Cw(ir, it);
                        Nab = Nr(a + 1) * Nt(b + 1) * ww;
                        dNab_dxi = dNr(a + 1) * Nt(b + 1) * ww;
                        dNab_deta = Nr(a + 1) * dNt(b + 1) * ww;
                        W = W + Nr(a + 1) * Nt(b + 1) * ww;
                        dW_dxi = dW_dxi + dNr(a + 1) * Nt(b + 1) * ww;
                        dW_deta = dW_deta + Nr(a + 1) * dNt(b + 1) * ww;
                    end
                end

                % Physical coordinates and Jacobian
                for a = 0:p
                    ir = span_r - p + a + 1;
                    for b = 0:p
                        it = span_t - p + b + 1;
                        ww = Cw(ir, it);
                        R_val = Nr(a + 1) * Nt(b + 1) * ww / W;
                        dR_dxi = (dNr(a + 1) * Nt(b + 1) * ww * W - Nr(a + 1) * Nt(b + 1) * ww * dW_dxi) / W^2;
                        dR_deta = (Nr(a + 1) * dNt(b + 1) * ww * W - Nr(a + 1) * Nt(b + 1) * ww * dW_deta) / W^2;

                        x_phys = x_phys + R_val * CPx(ir, it);
                        y_phys = y_phys + R_val * CPy(ir, it);
                        dx_dxi = dx_dxi + dR_dxi * CPx(ir, it);
                        dx_deta = dx_deta + dR_deta * CPx(ir, it);
                        dy_dxi = dy_dxi + dR_dxi * CPy(ir, it);
                        dy_deta = dy_deta + dR_deta * CPy(ir, it);
                    end
                end

                % Jacobian and its determinant
                detJ = dx_dxi * dy_deta - dx_deta * dy_dxi;
                if abs(detJ) < 1e-15; continue; end

                % Inverse Jacobian for gradient transformation
                inv_J11 = dy_deta / detJ;
                inv_J12 = -dy_dxi / detJ;
                inv_J21 = -dx_deta / detJ;
                inv_J22 = dx_dxi / detJ;

                % Source term: f = -laplacian(u) for u = (r^2 - R1^2)*(R2^2 - r^2)
                r_phys = sqrt(x_phys^2 + y_phys^2);
                f_val = 16 * r_phys^2 - 4 * (R1^2 + R2^2);

                % Assembly
                for a = 0:p
                    ir = span_r - p + a + 1;
                    for b = 0:p
                        it = span_t - p + b + 1;
                        ww_A = Cw(ir, it);
                        R_A = Nr(a + 1) * Nt(b + 1) * ww_A / W;
                        dR_A_dxi = (dNr(a + 1) * Nt(b + 1) * ww_A * W - Nr(a + 1) * Nt(b + 1) * ww_A * dW_dxi) / W^2;
                        dR_A_deta = (Nr(a + 1) * dNt(b + 1) * ww_A * W - Nr(a + 1) * Nt(b + 1) * ww_A * dW_deta) / W^2;

                        % Physical gradients
                        dR_A_dx = inv_J11 * dR_A_dxi + inv_J12 * dR_A_deta;
                        dR_A_dy = inv_J21 * dR_A_dxi + inv_J22 * dR_A_deta;

                        glob_A = (it - 1) * n_r + ir;
                        f(glob_A) = f(glob_A) + R_A * f_val * abs(detJ) * wt_q;

                        for c = 0:p
                            jr = span_r - p + c + 1;
                            for d = 0:p
                                jt = span_t - p + d + 1;
                                ww_B = Cw(jr, jt);
                                R_B = Nr(c + 1) * Nt(d + 1) * ww_B / W;
                                dR_B_dxi = (dNr(c + 1) * Nt(d + 1) * ww_B * W - Nr(c + 1) * Nt(d + 1) * ww_B * dW_dxi) / W^2;
                                dR_B_deta = (Nr(c + 1) * dNt(d + 1) * ww_B * W - Nr(c + 1) * Nt(d + 1) * ww_B * dW_deta) / W^2;

                                dR_B_dx = inv_J11 * dR_B_dxi + inv_J12 * dR_B_deta;
                                dR_B_dy = inv_J21 * dR_B_dxi + inv_J22 * dR_B_deta;

                                glob_B = (jt - 1) * n_r + jr;
                                K(glob_A, glob_B) = K(glob_A, glob_B) + (dR_A_dx * dR_B_dx + dR_A_dy * dR_B_dy) * abs(detJ) * wt_q;
                            end
                        end
                    end
                end
            end
        end
    end
end

fprintf('  Assembled %d x %d stiffness\n', n_2d, n_2d);

%% Boundary conditions
% Inner boundary (radial index 1): u = 0
% Outer boundary (radial index n_r): u = 0
bc_dofs = [];
for j = 1:n_t
    bc_dofs = [bc_dofs, (j - 1) * n_r + 1, (j - 1) * n_r + n_r];
end
bc_dofs = unique(bc_dofs);
free_dofs = setdiff(1:n_2d, bc_dofs);

fprintf('  BC DOFs: %d, Free DOFs: %d\n', length(bc_dofs), length(free_dofs));

%% Solve
u_sol = zeros(n_2d, 1);
u_sol(free_dofs) = K(free_dofs, free_dofs) \ f(free_dofs);
fprintf('  max|u| = %.6f\n', max(abs(u_sol)));

%% Visualize solution on physical geometry
n_plot = 40;
xi_v = linspace(0, 1 - 1e-10, n_plot);
eta_v = linspace(0, 1 - 1e-10, n_plot);
Xp = zeros(n_plot, n_plot);
Yp = zeros(n_plot, n_plot);
Up = zeros(n_plot, n_plot);
Ue = zeros(n_plot, n_plot);

for i = 1:n_plot
    xi = xi_v(i);
    span_r = findspan(n_r - 1, p, xi, Xi_r);
    Nr = basisfun(span_r, xi, p, Xi_r);
    for j = 1:n_plot
        eta = eta_v(j);
        span_t = findspan(n_t - 1, p, eta, Xi_t);
        Nt = basisfun(span_t, eta, p, Xi_t);

        % Evaluate geometry and solution
        W = 0; x = 0; y = 0; u_h = 0;
        for a = 0:p
            ir = span_r - p + a + 1;
            for b = 0:p
                it = span_t - p + b + 1;
                ww = Cw(ir, it);
                W = W + Nr(a + 1) * Nt(b + 1) * ww;
            end
        end
        for a = 0:p
            ir = span_r - p + a + 1;
            for b = 0:p
                it = span_t - p + b + 1;
                Rv = Nr(a + 1) * Nt(b + 1) * Cw(ir, it) / W;
                x = x + Rv * CPx(ir, it);
                y = y + Rv * CPy(ir, it);
                glob = (it - 1) * n_r + ir;
                u_h = u_h + Rv * u_sol(glob);
            end
        end
        Xp(j, i) = x;
        Yp(j, i) = y;
        Up(j, i) = u_h;
        r = sqrt(x^2 + y^2);
        Ue(j, i) = (r^2 - R1^2) * (R2^2 - r^2);
    end
end

%% Plot
figure(1);
surf(Xp, Yp, Up);
title('IGA Solution on Quarter Annulus');
xlabel('x'); ylabel('y'); zlabel('u');
colorbar;
drawnow;

figure(2);
surf(Xp, Yp, abs(Up - Ue));
title('Error |u_h - u_{exact}|');
xlabel('x'); ylabel('y'); zlabel('error');
colorbar;
drawnow;

err_max = max(max(abs(Up - Ue)));
fprintf('\n  Max pointwise error: %.6e\n', err_max);
fprintf('\n=== Mapped Solution Complete ===\n');
