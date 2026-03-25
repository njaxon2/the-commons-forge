%% TIGA Eigenvalue Problem on Mapped Geometry
%  Solve -laplacian(u) = lambda*u on a quarter annulus
%  using NURBS geometry mapping
%  Exact eigenvalues: Bessel function zeros (known for annular domains)
%  Tests: mass matrix assembly, eigenvalue solve on curved geometry

clear;
fprintf('=== TIGA Eigenvalues on Quarter Annulus ===\n\n');

% Geometry: quarter annulus [R1, R2] x [0, pi/2]
R1 = 0.5; R2 = 1.5;
p = 2;

% Radial knot vector
nel_r = 10;
interior_r = linspace(0, 1, nel_r + 1);
interior_r = interior_r(2:end - 1);
Xi_r = [zeros(1, p + 1), interior_r, ones(1, p + 1)];
n_r = length(Xi_r) - p - 1;

% Circumferential: NURBS quarter circle
Xi_t = [0 0 0 1 1 1];
n_t = 3;

n_2d = n_r * n_t;
fprintf('  Annulus: R1=%.1f, R2=%.1f\n', R1, R2);
fprintf('  Radial: p=%d, nel=%d, n_r=%d\n', p, nel_r, n_r);
fprintf('  Circum: n_t=%d\n', n_t);
fprintf('  Total DOFs: %d\n', n_2d);

% Control points for quarter annulus
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

% Quadrature
nqp = p + 2;
[gp, gw] = gaussQuad(nqp);
knots_r = unique(Xi_r);
knots_t = unique(Xi_t);
nel_rad = length(knots_r) - 1;
nel_cir = length(knots_t) - 1;

%% Assemble stiffness K and mass M on mapped geometry
K = zeros(n_2d, n_2d);
M = zeros(n_2d, n_2d);

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

                % NURBS weights
                W = 0; dW_dxi = 0; dW_deta = 0;
                for a = 0:p
                    ir = span_r - p + a + 1;
                    for b = 0:p
                        it = span_t - p + b + 1;
                        ww = Cw(ir, it);
                        W = W + Nr(a + 1) * Nt(b + 1) * ww;
                        dW_dxi = dW_dxi + dNr(a + 1) * Nt(b + 1) * ww;
                        dW_deta = dW_deta + Nr(a + 1) * dNt(b + 1) * ww;
                    end
                end

                % Physical coordinates and Jacobian
                x_phys = 0; y_phys = 0;
                dx_dxi = 0; dx_deta = 0; dy_dxi = 0; dy_deta = 0;
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

                detJ = dx_dxi * dy_deta - dx_deta * dy_dxi;
                if abs(detJ) < 1e-15; continue; end

                % J^{-T} for gradient transformation
                inv_J11 = dy_deta / detJ;
                inv_J12 = -dy_dxi / detJ;
                inv_J21 = -dx_deta / detJ;
                inv_J22 = dx_dxi / detJ;

                % Assembly
                for a = 0:p
                    ir = span_r - p + a + 1;
                    for b = 0:p
                        it = span_t - p + b + 1;
                        ww_A = Cw(ir, it);
                        R_A = Nr(a + 1) * Nt(b + 1) * ww_A / W;
                        dR_A_dxi = (dNr(a + 1) * Nt(b + 1) * ww_A * W - Nr(a + 1) * Nt(b + 1) * ww_A * dW_dxi) / W^2;
                        dR_A_deta = (Nr(a + 1) * dNt(b + 1) * ww_A * W - Nr(a + 1) * Nt(b + 1) * ww_A * dW_deta) / W^2;
                        dR_A_dx = inv_J11 * dR_A_dxi + inv_J12 * dR_A_deta;
                        dR_A_dy = inv_J21 * dR_A_dxi + inv_J22 * dR_A_deta;

                        glob_A = (it - 1) * n_r + ir;

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
                                M(glob_A, glob_B) = M(glob_A, glob_B) + R_A * R_B * abs(detJ) * wt_q;
                            end
                        end
                    end
                end
            end
        end
    end
end

fprintf('  Assembled K and M (%d x %d)\n', n_2d, n_2d);

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

%% Solve generalized eigenvalue problem
Kf = K(free_dofs, free_dofs);
Mf = M(free_dofs, free_dofs);
[V, D] = eig(Kf, Mf);
lambdas = real(diag(D));

% Sort eigenvalues (take only positive real ones)
[lambdas, idx] = sort(lambdas);
V = real(V(:, idx));

% Remove non-physical (negative) eigenvalues
pos_idx = find(lambdas > 0);
lambdas = lambdas(pos_idx);
V = V(:, pos_idx);

n_eig = min(8, length(lambdas));
fprintf('\n  First %d eigenvalues:\n', n_eig);
for k = 1:n_eig
    fprintf('    lambda_%d = %.6f\n', k, lambdas(k));
end

%% Visualize first 4 eigenmodes on physical geometry
n_plot = 40;
xi_v = linspace(0, 1 - 1e-10, n_plot);
eta_v = linspace(0, 1 - 1e-10, n_plot);

for mode = 1:min(4, n_eig)
    % Build full eigenvector (including BC DOFs = 0)
    u_mode = zeros(n_2d, 1);
    u_mode(free_dofs) = V(:, mode);

    % Normalize
    u_mode = u_mode / max(abs(u_mode));

    Xp = zeros(n_plot, n_plot);
    Yp = zeros(n_plot, n_plot);
    Up = zeros(n_plot, n_plot);

    for i = 1:n_plot
        xi = xi_v(i);
        span_r = findspan(n_r - 1, p, xi, Xi_r);
        Nr = basisfun(span_r, xi, p, Xi_r);
        for j = 1:n_plot
            eta = eta_v(j);
            span_t = findspan(n_t - 1, p, eta, Xi_t);
            Nt = basisfun(span_t, eta, p, Xi_t);

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
                    u_h = u_h + Rv * u_mode(glob);
                end
            end
            Xp(j, i) = x;
            Yp(j, i) = y;
            Up(j, i) = u_h;
        end
    end

    figure(mode);
    surf(Xp, Yp, Up);
    title(sprintf('Mode %d, lambda=%.2f', mode, lambdas(mode)));
    xlabel('x'); ylabel('y'); zlabel('u');
    colorbar;
    drawnow;
end

fprintf('\n=== Eigenvalue Solve Complete ===\n');
