%% TIGA: Anatomy of the Weight-Conditioning Tradeoff
%  Fine sweep + analytical investigation
%  Uses compute_annulus_cond.m as external function

clear;
fprintf('=== TIGA Weight-Conditioning Anatomy ===\n\n');

p = 2;
nel_r = 4;
nqp = p + 2;
R1 = 0.5; R2 = 1.5;

%% Part 1: Fine weight sweep
w_fine = logspace(-1, 1, 30);
n_w = length(w_fine);
cond_fine = zeros(1, n_w);
lmin_fine = zeros(1, n_w);
lmax_fine = zeros(1, n_w);
err_fine = zeros(1, n_w);

fprintf('Part 1: Fine weight sweep (nel=%d, %d points)\n', nel_r, n_w);
for iw = 1:n_w
    [cond_fine(iw), lmin_fine(iw), lmax_fine(iw), err_fine(iw)] = ...
        compute_annulus_cond(p, nel_r, nqp, R1, R2, w_fine(iw));
    fprintf('  w=%.4f  cond=%.4e  lmin=%.4e  lmax=%.4e  err=%.4e\n', ...
        w_fine(iw), cond_fine(iw), lmin_fine(iw), lmax_fine(iw), err_fine(iw));
end

% Find minimum by scanning
[cond_min, idx_min] = min(cond_fine);
w_approx_opt = w_fine(idx_min);
fprintf('\n  Approximate optimum: w*=%.4f, cond=%.4e\n', w_approx_opt, cond_min);

% Golden section refinement around minimum
if idx_min > 1 && idx_min < n_w
    a_gs = w_fine(idx_min - 1);
    b_gs = w_fine(idx_min + 1);
else
    a_gs = w_approx_opt * 0.5;
    b_gs = w_approx_opt * 2.0;
end

gr = (sqrt(5) - 1) / 2;
c_gs = b_gs - gr * (b_gs - a_gs);
d_gs = a_gs + gr * (b_gs - a_gs);
fc = compute_annulus_cond(p, nel_r, nqp, R1, R2, c_gs);
fd = compute_annulus_cond(p, nel_r, nqp, R1, R2, d_gs);

for iter = 1:30
    if (b_gs - a_gs) < 1e-6
        break;
    end
    if fc < fd
        b_gs = d_gs;
        d_gs = c_gs;
        fd = fc;
        c_gs = b_gs - gr * (b_gs - a_gs);
        fc = compute_annulus_cond(p, nel_r, nqp, R1, R2, c_gs);
    else
        a_gs = c_gs;
        c_gs = d_gs;
        fc = fd;
        d_gs = a_gs + gr * (b_gs - a_gs);
        fd = compute_annulus_cond(p, nel_r, nqp, R1, R2, d_gs);
    end
end
w_opt = (a_gs + b_gs) / 2;
cond_opt = compute_annulus_cond(p, nel_r, nqp, R1, R2, w_opt);

fprintf('  Golden section result: w*=%.6f, cond=%.4e\n', w_opt, cond_opt);

% Reference values
[c_geo, ~, ~, e_geo] = compute_annulus_cond(p, nel_r, nqp, R1, R2, 1/sqrt(2));
[c_bsp, ~, ~, e_bsp] = compute_annulus_cond(p, nel_r, nqp, R1, R2, 1.0);
fprintf('  w=1/sqrt(2): cond=%.4e, err=%.4e\n', c_geo, e_geo);
fprintf('  w=1.0:       cond=%.4e, err=%.4e\n', c_bsp, e_bsp);
fprintf('  Conditioning ratio: cond(1/sqrt2)/cond(1) = %.4f\n', c_geo/c_bsp);
fprintf('  Optimal ratio:      cond*/cond(1)         = %.4f\n', cond_opt/c_bsp);

%% Part 2: Geometry dependence of w*
fprintf('\nPart 2: How w* depends on R2/R1\n');
ratios = [1.2, 1.5, 2.0, 3.0, 5.0, 10.0];
n_rat = length(ratios);
w_opt_geom = zeros(1, n_rat);
cond_opt_geom = zeros(1, n_rat);

for ig = 1:n_rat
    R1g = 1.0;
    R2g = ratios(ig);
    % Coarse scan
    w_scan = linspace(0.5, 6.0, 20);
    c_scan = zeros(1, 20);
    for k = 1:20
        c_scan(k) = compute_annulus_cond(p, nel_r, nqp, R1g, R2g, w_scan(k));
    end
    [~, idx] = min(c_scan);
    % Golden section
    if idx > 1 && idx < 20
        ag = w_scan(idx-1); bg = w_scan(idx+1);
    else
        ag = 0.5; bg = 6.0;
    end
    cg = bg - gr*(bg-ag); dg = ag + gr*(bg-ag);
    fcg = compute_annulus_cond(p, nel_r, nqp, R1g, R2g, cg);
    fdg = compute_annulus_cond(p, nel_r, nqp, R1g, R2g, dg);
    for iter = 1:25
        if (bg - ag) < 1e-5; break; end
        if fcg < fdg
            bg = dg; dg = cg; fdg = fcg;
            cg = bg - gr*(bg-ag);
            fcg = compute_annulus_cond(p, nel_r, nqp, R1g, R2g, cg);
        else
            ag = cg; cg = dg; fcg = fdg;
            dg = ag + gr*(bg-ag);
            fdg = compute_annulus_cond(p, nel_r, nqp, R1g, R2g, dg);
        end
    end
    w_opt_geom(ig) = (ag + bg) / 2;
    cond_opt_geom(ig) = compute_annulus_cond(p, nel_r, nqp, R1g, R2g, w_opt_geom(ig));
    c_g = compute_annulus_cond(p, nel_r, nqp, R1g, R2g, 1/sqrt(2));
    c_b = compute_annulus_cond(p, nel_r, nqp, R1g, R2g, 1.0);
    fprintf('  R2/R1=%4.1f  w*=%.4f  cond*=%.2e  cond(geo)=%.2e  cond(bsp)=%.2e\n', ...
        ratios(ig), w_opt_geom(ig), cond_opt_geom(ig), c_g, c_b);
end

%% Part 3: Mesh independence
fprintf('\nPart 3: Mesh independence of w* and conditioning ratio\n');
nel_test = [2, 4, 8, 16];
for in = 1:length(nel_test)
    nel = nel_test(in);
    % Golden section at this mesh
    ag = 1.0; bg = 5.0;
    cg = bg-gr*(bg-ag); dg = ag+gr*(bg-ag);
    fcg = compute_annulus_cond(p, nel, nqp, R1, R2, cg);
    fdg = compute_annulus_cond(p, nel, nqp, R1, R2, dg);
    for iter = 1:25
        if (bg-ag) < 1e-5; break; end
        if fcg < fdg
            bg=dg; dg=cg; fdg=fcg;
            cg=bg-gr*(bg-ag);
            fcg=compute_annulus_cond(p,nel,nqp,R1,R2,cg);
        else
            ag=cg; cg=dg; fcg=fdg;
            dg=ag+gr*(bg-ag);
            fdg=compute_annulus_cond(p,nel,nqp,R1,R2,dg);
        end
    end
    w_m = (ag+bg)/2;
    c_m = compute_annulus_cond(p,nel,nqp,R1,R2,w_m);
    c_b = compute_annulus_cond(p,nel,nqp,R1,R2,1.0);
    c_g = compute_annulus_cond(p,nel,nqp,R1,R2,1/sqrt(2));
    fprintf('  nel=%2d  w*=%.4f  cond*=%.2e  ratio*/bsp=%.4f  ratio_geo/bsp=%.4f\n', ...
        nel, w_m, c_m, c_m/c_b, c_g/c_b);
end

%% Plots
figure(1);

subplot(2, 2, 1);
loglog(w_fine, cond_fine, 'b-', 'LineWidth', 1.5);
hold on;
loglog(1/sqrt(2), c_geo, 'rv', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
loglog(w_opt, cond_opt, 'g^', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
loglog(1.0, c_bsp, 'ks', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
hold off;
xlabel('Weight w');
ylabel('cond(K)');
title('Condition Number vs Weight');
legend('cond(K)', 'w=1/sqrt2', 'w* optimal', 'w=1 B-spline');

subplot(2, 2, 2);
loglog(w_fine, lmin_fine, 'r-', 'LineWidth', 1.5);
hold on;
loglog(w_fine, lmax_fine, 'b-', 'LineWidth', 1.5);
hold off;
xlabel('Weight w');
ylabel('Eigenvalue');
title('Extreme Eigenvalues');
legend('lambda_{min}', 'lambda_{max}');

subplot(2, 2, 3);
semilogx(ratios, w_opt_geom, 'bo-', 'MarkerFaceColor', 'b');
hold on;
semilogx([1, 20], [1/sqrt(2), 1/sqrt(2)], 'r--');
semilogx([1, 20], [1, 1], 'k--');
hold off;
xlabel('R_2/R_1');
ylabel('w*');
title('Optimal Weight vs Geometry');
legend('w*', '1/sqrt(2)', '1.0');

subplot(2, 2, 4);
loglog(w_fine, err_fine, 'g-', 'LineWidth', 1.5);
hold on;
loglog(1/sqrt(2), e_geo, 'rv', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
hold off;
xlabel('Weight w');
ylabel('Max Error');
title('Solution Error vs Weight');

saveas(1, '/tmp/weight_anatomy.png');
fprintf('\nPlot saved to /tmp/weight_anatomy.png\n');
fprintf('=== Anatomy Study Complete ===\n');
