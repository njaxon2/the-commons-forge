%% TIGA h-p Convergence on Quarter Annulus
%  Compare convergence rates for p=2,3,4 on NURBS-mapped annulus
%  Manufactured solution: u = (r^2-R1^2)*(R2^2-r^2)
%  Expected: L2 rate ~ p+1 for smooth solution

clear;
fprintf('=== TIGA h-p Convergence Study ===\n\n');

R1 = 0.5; R2 = 1.5;
p_list = [2, 3, 4];
nel_list = [2, 4, 8, 16];
n_trials = length(nel_list);

% Store results for all degrees
all_h = zeros(length(p_list), n_trials);
all_linf = zeros(length(p_list), n_trials);
all_l2 = zeros(length(p_list), n_trials);

for ip = 1:length(p_list)
    p = p_list(ip);
    fprintf('--- Degree p = %d ---\n', p);

    for trial = 1:n_trials
        nel_r = nel_list(trial);
        all_h(ip, trial) = 1 / nel_r;

        % Knot vector
        interior_r = linspace(0, 1, nel_r + 1);
        interior_r = interior_r(2:end - 1);
        Xi_r = [zeros(1, p + 1), interior_r, ones(1, p + 1)];
        n_r = length(Xi_r) - p - 1;

        % Circumferential NURBS (needs p+1 repeated knots at each end)
        Xi_t = [zeros(1, p + 1), ones(1, p + 1)];
        n_t = length(Xi_t) - p - 1;
        n_2d = n_r * n_t;

        % Control points for quarter annulus
        r_cp = linspace(R1, R2, n_r);
        CPx = zeros(n_r, n_t);
        CPy = zeros(n_r, n_t);
        Cw = ones(n_r, n_t);

        % For p=2: 3 control points per radial station
        % For p=3: 4 control points, For p=4: 5 control points
        % Use standard NURBS quarter circle parametrization
        if p == 2
            for i = 1:n_r
                r = r_cp(i);
                CPx(i, 1) = r;     CPy(i, 1) = 0;     Cw(i, 1) = 1;
                CPx(i, 2) = r;     CPy(i, 2) = r;     Cw(i, 2) = 1/sqrt(2);
                CPx(i, 3) = 0;     CPy(i, 3) = r;     Cw(i, 3) = 1;
            end
        elseif p == 3
            % 4-point NURBS quarter circle
            w1 = cos(pi/8);
            for i = 1:n_r
                r = r_cp(i);
                CPx(i, 1) = r;                    CPy(i, 1) = 0;                    Cw(i, 1) = 1;
                CPx(i, 2) = r;                    CPy(i, 2) = r*tan(pi/8);          Cw(i, 2) = w1;
                CPx(i, 3) = r*tan(pi/8);          CPy(i, 3) = r;                    Cw(i, 3) = w1;
                CPx(i, 4) = 0;                    CPy(i, 4) = r;                    Cw(i, 4) = 1;
            end
        elseif p == 4
            % 5-point NURBS quarter circle
            a1 = pi/16; a2 = pi/8; a3 = 3*pi/16;
            w1 = cos(a1); w2 = cos(a2);
            for i = 1:n_r
                r = r_cp(i);
                CPx(i, 1) = r;                    CPy(i, 1) = 0;                    Cw(i, 1) = 1;
                CPx(i, 2) = r*cos(a1);            CPy(i, 2) = r*sin(a1)*sec(a1);    Cw(i, 2) = w1;
                CPx(i, 3) = r*cos(a2);            CPy(i, 3) = r*sin(a2);            Cw(i, 3) = w2;
                CPx(i, 4) = r*sin(a1)*sec(a1);    CPy(i, 4) = r*cos(a1);            Cw(i, 4) = w1;
                CPx(i, 5) = 0;                    CPy(i, 5) = r;                    Cw(i, 5) = 1;
            end
        end

        nqp = p + 2;
        [gp, gw] = gaussQuad(nqp);
        knots_r = unique(Xi_r);
        knots_t = unique(Xi_t);
        nel_rad = length(knots_r) - 1;
        nel_cir = length(knots_t) - 1;

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

        % BCs
        bc_dofs = [];
        for j = 1:n_t
            bc_dofs = [bc_dofs, (j - 1) * n_r + 1, (j - 1) * n_r + n_r];
        end
        bc_dofs = unique(bc_dofs);
        free_dofs = setdiff(1:n_2d, bc_dofs);

        u_sol = zeros(n_2d, 1);
        u_sol(free_dofs) = K(free_dofs, free_dofs) \ f(free_dofs);

        % Error evaluation
        n_plot = 30;
        xi_v = linspace(0, 1 - 1e-10, n_plot);
        eta_v = linspace(0, 1 - 1e-10, n_plot);
        e_max = 0;
        e_l2_sum = 0;

        for i = 1:n_plot
            xi = xi_v(i);
            span_r = findspan(n_r - 1, p, xi, Xi_r);
            Nr_e = basisfun(span_r, xi, p, Xi_r);
            for j = 1:n_plot
                eta = eta_v(j);
                span_t = findspan(n_t - 1, p, eta, Xi_t);
                Nt_e = basisfun(span_t, eta, p, Xi_t);
                W = 0; x = 0; y = 0; u_h = 0;
                for a = 0:p
                    ir = span_r - p + a + 1;
                    for b = 0:p
                        it = span_t - p + b + 1;
                        W = W + Nr_e(a+1) * Nt_e(b+1) * Cw(ir, it);
                    end
                end
                for a = 0:p
                    ir = span_r - p + a + 1;
                    for b = 0:p
                        it = span_t - p + b + 1;
                        Rv = Nr_e(a+1) * Nt_e(b+1) * Cw(ir, it) / W;
                        x = x + Rv * CPx(ir, it);
                        y = y + Rv * CPy(ir, it);
                        glob = (it - 1) * n_r + ir;
                        u_h = u_h + Rv * u_sol(glob);
                    end
                end
                r = sqrt(x^2 + y^2);
                u_exact = (r^2 - R1^2) * (R2^2 - r^2);
                err = abs(u_h - u_exact);
                if err > e_max; e_max = err; end
                e_l2_sum = e_l2_sum + err^2;
            end
        end

        all_linf(ip, trial) = e_max;
        all_l2(ip, trial) = sqrt(e_l2_sum / (n_plot * n_plot));
        fprintf('  nel=%2d  DOF=%3d  Linf=%.4e  L2=%.4e\n', nel_r, n_2d, e_max, all_l2(ip, trial));
    end

    % Rates
    fprintf('  Rates (L2): ');
    for i = 2:n_trials
        rate = log(all_l2(ip, i-1) / all_l2(ip, i)) / log(all_h(ip, i-1) / all_h(ip, i));
        fprintf('%.2f ', rate);
    end
    fprintf('\n\n');
end

% Log-log plot comparing all degrees
figure(1);
colors = ['b', 'r', 'g'];
markers = ['o', 's', 'd'];
for ip = 1:length(p_list)
    p = p_list(ip);
    h_v = all_h(ip, :);
    loglog(h_v, all_l2(ip, :), [colors(ip), markers(ip), '-']);
    hold on;
    % Reference slope
    h_ref = [h_v(1), h_v(end)];
    c = all_l2(ip, 1) / h_v(1)^(p+1);
    loglog(h_ref, c * h_ref.^(p+1), [colors(ip), '--']);
end
hold off;

xlabel('h (element size)');
ylabel('L_2 Error');
title('h-p Convergence on Quarter Annulus');
legend('p=2', 'O(h^3)', 'p=3', 'O(h^4)', 'p=4', 'O(h^5)');
drawnow;

fprintf('=== h-p Convergence Complete ===\n');
