%% TIGA: NURBS Weight Effect on Conditioning
%  Compare condition numbers for:
%  1. Standard B-spline (all weights = 1) on parametric domain
%  2. NURBS on annulus (weights from circular arc parametrization)
%  3. NURBS with artificially perturbed weights
%  This isolates the effect of NURBS weights on system conditioning

clear;
fprintf('=== TIGA Weight Effect on Conditioning ===\n\n');

p = 2; R1 = 0.5; R2 = 1.5;

nel_list = [4, 8, 16, 32];
cond_bspline = zeros(1, length(nel_list));
cond_nurbs = zeros(1, length(nel_list));
cond_extreme = zeros(1, length(nel_list));
lambda_min_bs = zeros(1, length(nel_list));
lambda_max_bs = zeros(1, length(nel_list));
lambda_min_nr = zeros(1, length(nel_list));
lambda_max_nr = zeros(1, length(nel_list));

for trial = 1:length(nel_list)
    nel_r = nel_list(trial);

    interior_r = linspace(0, 1, nel_r + 1);
    interior_r = interior_r(2:end - 1);
    Xi_r = [zeros(1, p + 1), interior_r, ones(1, p + 1)];
    n_r = length(Xi_r) - p - 1;
    Xi_t = [0 0 0 1 1 1];
    n_t = 3;
    n_2d = n_r * n_t;

    r_cp = linspace(R1, R2, n_r);
    nqp = p + 2;
    [gp, gw] = gaussQuad(nqp);
    knots_r = unique(Xi_r);
    knots_t = unique(Xi_t);
    nel_rad = length(knots_r) - 1;
    nel_cir = length(knots_t) - 1;

    % Test three weight configurations
    for config = 1:3
        CPx = zeros(n_r, n_t);
        CPy = zeros(n_r, n_t);
        Cw = ones(n_r, n_t);

        for i = 1:n_r
            r = r_cp(i);
            CPx(i, 1) = r;     CPy(i, 1) = 0;
            CPx(i, 2) = r;     CPy(i, 2) = r;
            CPx(i, 3) = 0;     CPy(i, 3) = r;

            if config == 1
                % B-spline: all weights = 1
                Cw(i, 1) = 1; Cw(i, 2) = 1; Cw(i, 3) = 1;
            elseif config == 2
                % Standard NURBS quarter circle
                Cw(i, 1) = 1; Cw(i, 2) = 1/sqrt(2); Cw(i, 3) = 1;
            else
                % Extreme weight: middle control point weight = 0.1
                Cw(i, 1) = 1; Cw(i, 2) = 0.1; Cw(i, 3) = 1;
            end
        end

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

        ev = eig(Kf);
        ev = sort(real(ev));
        ev_pos = ev(ev > 1e-10);

        if config == 1
            cond_bspline(trial) = max(ev_pos) / min(ev_pos);
            lambda_min_bs(trial) = min(ev_pos);
            lambda_max_bs(trial) = max(ev_pos);
        elseif config == 2
            cond_nurbs(trial) = max(ev_pos) / min(ev_pos);
            lambda_min_nr(trial) = min(ev_pos);
            lambda_max_nr(trial) = max(ev_pos);
        else
            cond_extreme(trial) = max(ev_pos) / min(ev_pos);
        end
    end

    fprintf('  nel=%2d: B-spline cond=%.2e  NURBS cond=%.2e  Extreme cond=%.2e  ratio=%.2f\n', ...
        nel_r, cond_bspline(trial), cond_nurbs(trial), cond_extreme(trial), ...
        cond_nurbs(trial) / cond_bspline(trial));
end

fprintf('\n  Eigenvalue analysis (nel=8):\n');
fprintf('    B-spline: lambda_min=%.4e  lambda_max=%.4e\n', lambda_min_bs(2), lambda_max_bs(2));
fprintf('    NURBS:    lambda_min=%.4e  lambda_max=%.4e\n', lambda_min_nr(2), lambda_max_nr(2));

% Plot comparison
figure(1);
h_v = 1 ./ nel_list;
loglog(h_v, cond_bspline, 'bo-');
hold on;
loglog(h_v, cond_nurbs, 'rs-');
loglog(h_v, cond_extreme, 'gd-');
hold off;
xlabel('h');
ylabel('cond(K)');
title('NURBS Weight Effect on Conditioning');
legend('w=1 (B-spline)', 'w=1/sqrt(2) (exact circle)', 'w=0.1 (extreme)');
drawnow;

fprintf('\n=== Weight Effect Study Complete ===\n');
