function [cond_val, lmin, lmax, err_val] = compute_annulus_cond(p, nel_r, nqp, R1, R2, w_mid)
%COMPUTE_ANNULUS_COND  Assemble and analyze stiffness matrix for NURBS quarter annulus
%  Returns condition number, extreme eigenvalues, and solution error

    interior_r = linspace(0, 1, nel_r + 1);
    interior_r = interior_r(2:end - 1);
    Xi_r = [zeros(1, p + 1), interior_r, ones(1, p + 1)];
    n_r = length(Xi_r) - p - 1;
    Xi_t = [0 0 0 1 1 1];
    n_t = 3;
    n_2d = n_r * n_t;

    r_cp = linspace(R1, R2, n_r);
    [gp, gw] = gaussQuad(nqp);
    knots_r = unique(Xi_r);
    knots_t = unique(Xi_t);
    nel_rad = length(knots_r) - 1;
    nel_cir = length(knots_t) - 1;

    CPx = zeros(n_r, n_t);
    CPy = zeros(n_r, n_t);
    Cw = ones(n_r, n_t);
    for i = 1:n_r
        r = r_cp(i);
        CPx(i, 1) = r;     CPy(i, 1) = 0;     Cw(i, 1) = 1;
        CPx(i, 2) = r;     CPy(i, 2) = r;     Cw(i, 2) = w_mid;
        CPx(i, 3) = 0;     CPy(i, 3) = r;     Cw(i, 3) = 1;
    end

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

    bc_dofs = [];
    for j = 1:n_t
        bc_dofs = [bc_dofs, (j - 1) * n_r + 1, (j - 1) * n_r + n_r];
    end
    bc_dofs = unique(bc_dofs);
    free_dofs = setdiff(1:n_2d, bc_dofs);

    Kf = K(free_dofs, free_dofs);
    ev = sort(real(eig(Kf)));
    ev_pos = ev(ev > 1e-10);

    cond_val = max(ev_pos) / min(ev_pos);
    lmin = min(ev_pos);
    lmax = max(ev_pos);

    % Solve and compute error
    u_sol = zeros(n_2d, 1);
    u_sol(free_dofs) = Kf \ f(free_dofs);
    e_max = 0;
    n_plot = 15;
    xi_v = linspace(0, 1 - 1e-10, n_plot);
    eta_v = linspace(0, 1 - 1e-10, n_plot);
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
        end
    end
    err_val = e_max;
end
