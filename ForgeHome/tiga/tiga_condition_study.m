%% TIGA Condition Number Study on Quarter Annulus
%  Investigate how stiffness matrix condition number scales with h and
%  how the annulus aspect ratio (R2/R1) affects conditioning
%  This is relevant because IGA on mapped geometries can have
%  dramatically different conditioning than standard FEA

clear;
fprintf('=== TIGA Condition Number Study ===\n\n');

p = 2;

%% Part 1: Condition number vs mesh size (fixed geometry)
fprintf('--- Part 1: cond(K) vs h ---\n');
R1 = 0.5; R2 = 1.5;
nel_list = [2, 4, 8, 16, 32];
cond_K = zeros(1, length(nel_list));
cond_M = zeros(1, length(nel_list));
h_vals = zeros(1, length(nel_list));

for trial = 1:length(nel_list)
    nel_r = nel_list(trial);
    h_vals(trial) = 1 / nel_r;

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
        CPx(i, 2) = r;     CPy(i, 2) = r;     Cw(i, 2) = 1/sqrt(2);
        CPx(i, 3) = 0;     CPy(i, 3) = r;     Cw(i, 3) = 1;
    end

    nqp = p + 2;
    [gp, gw] = gaussQuad(nqp);
    knots_r = unique(Xi_r);
    knots_t = unique(Xi_t);
    nel_rad = length(knots_r) - 1;
    nel_cir = length(knots_t) - 1;

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

    % BCs
    bc_dofs = [];
    for j = 1:n_t
        bc_dofs = [bc_dofs, (j - 1) * n_r + 1, (j - 1) * n_r + n_r];
    end
    bc_dofs = unique(bc_dofs);
    free_dofs = setdiff(1:n_2d, bc_dofs);

    Kf = K(free_dofs, free_dofs);
    Mf = M(free_dofs, free_dofs);

    cond_K(trial) = cond(Kf);
    cond_M(trial) = cond(Mf);
    fprintf('  nel=%2d  DOF=%3d  cond(K)=%.4e  cond(M)=%.4e\n', nel_r, n_2d, cond_K(trial), cond_M(trial));
end

% Condition number scaling rate
fprintf('\n  cond(K) scaling rates:\n');
for i = 2:length(nel_list)
    rate = log(cond_K(i) / cond_K(i-1)) / log(h_vals(i-1) / h_vals(i));
    fprintf('    h -> h/2: rate = %.2f\n', rate);
end

%% Part 2: Condition number vs aspect ratio R2/R1
fprintf('\n--- Part 2: cond(K) vs aspect ratio ---\n');
nel_r = 8;
R1_fixed = 0.5;
ratio_list = [1.5, 2.0, 3.0, 5.0, 10.0, 20.0];
cond_ratio = zeros(1, length(ratio_list));

for trial = 1:length(ratio_list)
    R2_var = R1_fixed * ratio_list(trial);

    interior_r = linspace(0, 1, nel_r + 1);
    interior_r = interior_r(2:end - 1);
    Xi_r = [zeros(1, p + 1), interior_r, ones(1, p + 1)];
    n_r = length(Xi_r) - p - 1;
    Xi_t = [0 0 0 1 1 1];
    n_t = 3;
    n_2d = n_r * n_t;

    r_cp = linspace(R1_fixed, R2_var, n_r);
    CPx = zeros(n_r, n_t);
    CPy = zeros(n_r, n_t);
    Cw = ones(n_r, n_t);
    for i = 1:n_r
        r = r_cp(i);
        CPx(i, 1) = r;     CPy(i, 1) = 0;     Cw(i, 1) = 1;
        CPx(i, 2) = r;     CPy(i, 2) = r;     Cw(i, 2) = 1/sqrt(2);
        CPx(i, 3) = 0;     CPy(i, 3) = r;     Cw(i, 3) = 1;
    end

    nqp = p + 2;
    [gp, gw] = gaussQuad(nqp);
    knots_r = unique(Xi_r);
    knots_t = unique(Xi_t);
    nel_rad = length(knots_r) - 1;
    nel_cir = length(knots_t) - 1;

    K = zeros(n_2d, n_2d);

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

                    for a = 0:p
                        ir = span_r - p + a + 1;
                        for b = 0:p
                            it = span_t - p + b + 1;
                            ww_A = Cw(ir, it);
                            dR_A_dxi = (dNr(a+1) * Nt(b+1) * ww_A * W - Nr(a+1) * Nt(b+1) * ww_A * dW_dxi) / W^2;
                            dR_A_deta = (Nr(a+1) * dNt(b+1) * ww_A * W - Nr(a+1) * Nt(b+1) * ww_A * dW_deta) / W^2;
                            dR_A_dx = inv_J11 * dR_A_dxi + inv_J12 * dR_A_deta;
                            dR_A_dy = inv_J21 * dR_A_dxi + inv_J22 * dR_A_deta;
                            glob_A = (it - 1) * n_r + ir;

                            for c = 0:p
                                jr = span_r - p + c + 1;
                                for d = 0:p
                                    jt = span_t - p + d + 1;
                                    ww_B = Cw(jr, jt);
                                    dR_B_dxi = (dNr(c+1) * Nt(d+1) * ww_B * W - Nr(c+1) * Nt(d+1) * ww_B * dW_dxi) / W^2;
                                    dR_B_deta = (Nr(c+1) * dNt(d+1) * ww_B * W - Nr(c+1) * Nt(d+1) * ww_B * dW_deta) / W^2;
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

    bc_dofs = [];
    for j = 1:n_t
        bc_dofs = [bc_dofs, (j - 1) * n_r + 1, (j - 1) * n_r + n_r];
    end
    bc_dofs = unique(bc_dofs);
    free_dofs = setdiff(1:n_2d, bc_dofs);

    Kf = K(free_dofs, free_dofs);
    cond_ratio(trial) = cond(Kf);
    fprintf('  R2/R1=%4.1f  R2=%5.1f  cond(K)=%.4e\n', ratio_list(trial), R2_var, cond_ratio(trial));
end

%% Plots
figure(1);
subplot(1, 2, 1);
loglog(1 ./ nel_list, cond_K, 'bo-');
hold on;
% Reference: cond ~ h^{-2} for Poisson
h_ref = [h_vals(1), h_vals(end)];
c0 = cond_K(1) / h_vals(1)^(-2);
loglog(h_ref, c0 * h_ref.^(-2), 'b--');
hold off;
xlabel('h');
ylabel('cond(K)');
title('Condition Number vs h');
legend('cond(K)', 'O(h^{-2})');

subplot(1, 2, 2);
semilogy(ratio_list, cond_ratio, 'rs-');
xlabel('R_2 / R_1');
ylabel('cond(K)');
title('Condition Number vs Aspect Ratio');

drawnow;

fprintf('\n=== Condition Number Study Complete ===\n');
