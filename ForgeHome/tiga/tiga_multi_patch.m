%% TIGA Multi-Patch Geometry Demo
%  Constructs a full annulus from 4 NURBS quarter-patches
%  Solves Poisson equation across all patches
%  Tests: multi-patch geometry, patch stitching, full annular domain

clear;
fprintf('=== TIGA Multi-Patch Full Annulus ===\n\n');

R1 = 0.5; R2 = 1.5; p = 2;

% Each patch covers a quarter of the annulus (90 degrees)
% Patch 1: theta = [0, pi/2]       Patch 2: theta = [pi/2, pi]
% Patch 3: theta = [pi, 3pi/2]     Patch 4: theta = [3pi/2, 2pi]

% Radial mesh (shared across patches)
nel_r = 8;
interior_r = linspace(0, 1, nel_r + 1);
interior_r = interior_r(2:end - 1);
Xi_r = [zeros(1, p + 1), interior_r, ones(1, p + 1)];
n_r = length(Xi_r) - p - 1;

Xi_t = [0 0 0 1 1 1];
n_t = 3;

n_patch = n_r * n_t;  % DOFs per patch
fprintf('  R1=%.1f, R2=%.1f, p=%d\n', R1, R2, p);
fprintf('  Per patch: %d radial CPs, %d circ CPs = %d DOFs\n', n_r, n_t, n_patch);

% Control points for each quarter
% Quarter circle at angle range [theta_start, theta_start + pi/2]:
%   CP1 = (r*cos(ts), r*sin(ts))           w=1
%   CP2 = (r*cos(ts), r*sin(ts)+r)  ... -> adjusted for each quadrant
%   CP3 = (r*cos(ts+pi/2), r*sin(ts+pi/2)) w=1
r_cp = linspace(R1, R2, n_r);

% Build 4 patches of control points
% Each stored as CPx{k}(n_r, n_t), CPy{k}(n_r, n_t), Cw{k}(n_r, n_t)
patch_angles = [0, pi/2, pi, 3*pi/2];
sq2 = 1/sqrt(2);

% Store as cell-like arrays (using separate variables since Forge cell is limited)
CPx1 = zeros(n_r, n_t); CPy1 = zeros(n_r, n_t); Cw1 = ones(n_r, n_t);
CPx2 = zeros(n_r, n_t); CPy2 = zeros(n_r, n_t); Cw2 = ones(n_r, n_t);
CPx3 = zeros(n_r, n_t); CPy3 = zeros(n_r, n_t); Cw3 = ones(n_r, n_t);
CPx4 = zeros(n_r, n_t); CPy4 = zeros(n_r, n_t); Cw4 = ones(n_r, n_t);

for i = 1:n_r
    r = r_cp(i);
    % Patch 1: 0 to pi/2
    CPx1(i,1) = r;  CPy1(i,1) = 0;  Cw1(i,1) = 1;
    CPx1(i,2) = r;  CPy1(i,2) = r;  Cw1(i,2) = sq2;
    CPx1(i,3) = 0;  CPy1(i,3) = r;  Cw1(i,3) = 1;

    % Patch 2: pi/2 to pi
    CPx2(i,1) = 0;   CPy2(i,1) = r;   Cw2(i,1) = 1;
    CPx2(i,2) = -r;  CPy2(i,2) = r;   Cw2(i,2) = sq2;
    CPx2(i,3) = -r;  CPy2(i,3) = 0;   Cw2(i,3) = 1;

    % Patch 3: pi to 3pi/2
    CPx3(i,1) = -r;  CPy3(i,1) = 0;   Cw3(i,1) = 1;
    CPx3(i,2) = -r;  CPy3(i,2) = -r;  Cw3(i,2) = sq2;
    CPx3(i,3) = 0;   CPy3(i,3) = -r;  Cw3(i,3) = 1;

    % Patch 4: 3pi/2 to 2pi
    CPx4(i,1) = 0;   CPy4(i,1) = -r;  Cw4(i,1) = 1;
    CPx4(i,2) = r;   CPy4(i,2) = -r;  Cw4(i,2) = sq2;
    CPx4(i,3) = r;   CPy4(i,3) = 0;   Cw4(i,3) = 1;
end

%% Global DOF mapping
% Interface DOFs are shared between adjacent patches
% Patch 1 theta=pi/2 edge = Patch 2 theta=0 edge (radial CPs, circ index 3 <-> 1)
% Patch 2 theta=pi edge = Patch 3 theta=0 edge
% etc.
% Interior DOFs: circ index 2 for each patch (unique)
% Shared DOFs: circ index 1 of patch k+1 = circ index 3 of patch k

% Total unique DOFs:
% Each patch has n_r * 3 = 3*n_r DOFs
% But edges are shared: 4 shared edges x n_r DOFs = 4*n_r shared
% n_unique_per_patch = n_r * 1 (interior, circ index 2)
% n_shared_edges = 4 * n_r
% Also inner/outer BCs subtract: 4*2*1 per edge + 4*1 per interior = lots
% Total: 4 * n_r (edge DOFs) + 4 * n_r (interior DOFs) = 8*n_r
% But boundary removes: inner (radial 1) and outer (radial n_r) per column
% Actually, let's just enumerate

% Global DOF numbering:
% Edge 1 (theta=0): DOFs 1..n_r  (shared between patch 4 end and patch 1 start)
% Interior 1 (patch 1, circ=2): DOFs n_r+1..2*n_r
% Edge 2 (theta=pi/2): DOFs 2*n_r+1..3*n_r  (shared between patch 1 end and patch 2 start)
% Interior 2 (patch 2, circ=2): DOFs 3*n_r+1..4*n_r
% Edge 3 (theta=pi): DOFs 4*n_r+1..5*n_r
% Interior 3 (patch 3, circ=2): DOFs 5*n_r+1..6*n_r
% Edge 4 (theta=3pi/2): DOFs 6*n_r+1..7*n_r
% Interior 4 (patch 4, circ=2): DOFs 7*n_r+1..8*n_r

n_total = 8 * n_r;
fprintf('  4 patches, total unique DOFs: %d\n', n_total);

% Build local-to-global maps for each patch
% patch_dof(patch, local_circ, local_rad) -> global
% For patch k: circ=1 -> edge k, circ=2 -> interior k, circ=3 -> edge k+1
% patch_glob(patch_idx, circ_idx) = base offset for that column of radial CPs

% Edge offsets:  edge k at (2*(k-1))*n_r + 1 .. (2*(k-1))*n_r + n_r
% Interior offsets: interior k at (2*(k-1)+1)*n_r + 1 .. (2*(k-1)+1)*n_r + n_r

%% Assemble global K, M, f
nqp = p + 2;
[gp, gw] = gaussQuad(nqp);
knots_r = unique(Xi_r);
knots_t = unique(Xi_t);
nel_rad = length(knots_r) - 1;
nel_cir = length(knots_t) - 1;

K_g = zeros(n_total, n_total);
M_g = zeros(n_total, n_total);
f_g = zeros(n_total, 1);

for patch = 1:4
    % Select control points for this patch
    if patch == 1
        CPx = CPx1; CPy = CPy1; Cw = Cw1;
    elseif patch == 2
        CPx = CPx2; CPy = CPy2; Cw = Cw2;
    elseif patch == 3
        CPx = CPx3; CPy = CPy3; Cw = Cw3;
    else
        CPx = CPx4; CPy = CPy4; Cw = Cw4;
    end

    % Local-to-global map for this patch
    % circ 1 -> edge (patch), circ 2 -> interior (patch), circ 3 -> edge (patch mod 4 + 1)
    edge_off_start = (2 * (patch - 1)) * n_r;
    int_off = (2 * (patch - 1) + 1) * n_r;
    next_patch = mod(patch, 4) + 1;
    edge_off_end = (2 * (next_patch - 1)) * n_r;

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

                            % Local to global
                            if it == 1
                                glob_A = edge_off_start + ir;
                            elseif it == 2
                                glob_A = int_off + ir;
                            else
                                glob_A = edge_off_end + ir;
                            end

                            f_g(glob_A) = f_g(glob_A) + R_A * f_val * abs(detJ) * wt_q;

                            for c = 0:p
                                jr = span_r - p + c + 1;
                                for d = 0:p
                                    jt = span_t - p + d + 1;
                                    ww_B = Cw(jr, jt);
                                    dR_B_dxi = (dNr(c+1) * Nt(d+1) * ww_B * W - Nr(c+1) * Nt(d+1) * ww_B * dW_dxi) / W^2;
                                    dR_B_deta = (Nr(c+1) * dNt(d+1) * ww_B * W - Nr(c+1) * Nt(d+1) * ww_B * dW_deta) / W^2;
                                    dR_B_dx = inv_J11 * dR_B_dxi + inv_J12 * dR_B_deta;
                                    dR_B_dy = inv_J21 * dR_B_dxi + inv_J22 * dR_B_deta;

                                    if jt == 1
                                        glob_B = edge_off_start + jr;
                                    elseif jt == 2
                                        glob_B = int_off + jr;
                                    else
                                        glob_B = edge_off_end + jr;
                                    end

                                    K_g(glob_A, glob_B) = K_g(glob_A, glob_B) + (dR_A_dx * dR_B_dx + dR_A_dy * dR_B_dy) * abs(detJ) * wt_q;
                                    M_g(glob_A, glob_B) = M_g(glob_A, glob_B) + R_A * (Nr(c+1) * Nt(d+1) * ww_B / W) * abs(detJ) * wt_q;
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    fprintf('  Patch %d assembled\n', patch);
end

%% Boundary conditions (inner and outer radius)
bc_dofs = [];
for col = 1:8
    base = (col - 1) * n_r;
    bc_dofs = [bc_dofs, base + 1, base + n_r];
end
bc_dofs = unique(bc_dofs);
free_dofs = setdiff(1:n_total, bc_dofs);
fprintf('  BCs: %d, free: %d\n', length(bc_dofs), length(free_dofs));

%% Solve
u_sol = zeros(n_total, 1);
u_sol(free_dofs) = K_g(free_dofs, free_dofs) \ f_g(free_dofs);
fprintf('  max|u| = %.6f (exact max = 1.0)\n', max(abs(u_sol)));

%% Visualize: evaluate solution on each patch
n_plot = 30;
xi_v = linspace(0, 1 - 1e-10, n_plot);
eta_v = linspace(0, 1 - 1e-10, n_plot);

figure(1);
err_max = 0;

for patch = 1:4
    if patch == 1
        CPx = CPx1; CPy = CPy1; Cw = Cw1;
    elseif patch == 2
        CPx = CPx2; CPy = CPy2; Cw = Cw2;
    elseif patch == 3
        CPx = CPx3; CPy = CPy3; Cw = Cw3;
    else
        CPx = CPx4; CPy = CPy4; Cw = Cw4;
    end

    edge_off_start = (2 * (patch - 1)) * n_r;
    int_off = (2 * (patch - 1) + 1) * n_r;
    next_patch = mod(patch, 4) + 1;
    edge_off_end = (2 * (next_patch - 1)) * n_r;

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
                    W = W + Nr(a+1) * Nt(b+1) * Cw(ir, it);
                end
            end
            for a = 0:p
                ir = span_r - p + a + 1;
                for b = 0:p
                    it = span_t - p + b + 1;
                    Rv = Nr(a+1) * Nt(b+1) * Cw(ir, it) / W;
                    x = x + Rv * CPx(ir, it);
                    y = y + Rv * CPy(ir, it);
                    if it == 1
                        glob = edge_off_start + ir;
                    elseif it == 2
                        glob = int_off + ir;
                    else
                        glob = edge_off_end + ir;
                    end
                    u_h = u_h + Rv * u_sol(glob);
                end
            end
            Xp(j, i) = x;
            Yp(j, i) = y;
            Up(j, i) = u_h;
            r = sqrt(x^2 + y^2);
            err = abs(u_h - (r^2 - R1^2) * (R2^2 - r^2));
            if err > err_max; err_max = err; end
        end
    end

    % Plot this patch
    subplot(2, 2, patch);
    surf(Xp, Yp, Up);
    title(sprintf('Patch %d', patch));
    xlabel('x'); ylabel('y');
    colorbar;
end
drawnow;

% Combined top-down view
figure(2);
hold on;
for patch = 1:4
    if patch == 1
        CPx = CPx1; CPy = CPy1; Cw = Cw1;
    elseif patch == 2
        CPx = CPx2; CPy = CPy2; Cw = Cw2;
    elseif patch == 3
        CPx = CPx3; CPy = CPy3; Cw = Cw3;
    else
        CPx = CPx4; CPy = CPy4; Cw = Cw4;
    end

    edge_off_start = (2 * (patch - 1)) * n_r;
    int_off = (2 * (patch - 1) + 1) * n_r;
    next_patch = mod(patch, 4) + 1;
    edge_off_end = (2 * (next_patch - 1)) * n_r;

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
                    W = W + Nr(a+1) * Nt(b+1) * Cw(ir, it);
                end
            end
            for a = 0:p
                ir = span_r - p + a + 1;
                for b = 0:p
                    it = span_t - p + b + 1;
                    Rv = Nr(a+1) * Nt(b+1) * Cw(ir, it) / W;
                    x = x + Rv * CPx(ir, it);
                    y = y + Rv * CPy(ir, it);
                    if it == 1
                        glob = edge_off_start + ir;
                    elseif it == 2
                        glob = int_off + ir;
                    else
                        glob = edge_off_end + ir;
                    end
                    u_h = u_h + Rv * u_sol(glob);
                end
            end
            Xp(j, i) = x;
            Yp(j, i) = y;
            Up(j, i) = u_h;
        end
    end
    surf(Xp, Yp, Up);
end
view(2);
title('Full Annulus (4 patches)');
xlabel('x'); ylabel('y');
colorbar;
drawnow;

fprintf('\n  Max pointwise error: %.4e\n', err_max);
fprintf('\n=== Multi-Patch Complete ===\n');
