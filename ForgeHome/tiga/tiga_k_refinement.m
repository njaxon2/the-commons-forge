%% TIGA k-Refinement Study
%  Compare C^0 (FEA-like) vs C^(p-1) (IGA maximal) continuity
%  Key advantage of IGA: higher inter-element continuity = better accuracy per DOF
%  Tests: knot insertion, continuity control, accuracy per DOF comparison

clear;
fprintf('=== TIGA k-Refinement Study ===\n');
fprintf('  Comparing C0 (FEA) vs Cp-1 (IGA) continuity\n\n');

% Problem: -u'' = pi^2*sin(pi*x), u(0)=u(1)=0, exact: u=sin(pi*x)

p_values = [2, 3, 4];
nel_values = [4, 8, 16, 32];

fprintf('  %3s | %4s | %6s | %12s | %12s | %8s\n', 'p', 'nel', 'DOFs', 'L2 err (IGA)', 'L2 err (C0)', 'Ratio');
fprintf('  %3s-|-%4s-|-%6s-|-%12s-|-%12s-|-%8s\n', '---', '----', '------', '------------', '-----------', '--------');

for ip = 1:length(p_values)
    p = p_values(ip);
    nqp = p + 2;
    [gp, gw] = gaussQuad(nqp);

    for im = 1:length(nel_values)
        nel = nel_values(im);

        %% IGA: maximal continuity C^(p-1)
        interior_iga = linspace(0, 1, nel + 1);
        interior_iga = interior_iga(2:end - 1);
        Xi_iga = [zeros(1, p + 1), interior_iga, ones(1, p + 1)];
        n_iga = length(Xi_iga) - p - 1;

        % Assemble IGA
        knots_iga = unique(Xi_iga);
        nel_iga = length(knots_iga) - 1;
        K_iga = zeros(n_iga, n_iga);
        f_iga = zeros(n_iga, 1);

        for e = 1:nel_iga
            xi_a = knots_iga(e); xi_b = knots_iga(e + 1);
            if xi_b - xi_a < 1e-14; continue; end
            J = (xi_b - xi_a) / 2;
            for q = 1:nqp
                xi = (xi_a + xi_b) / 2 + J * gp(q);
                span = findspan(n_iga - 1, p, xi, Xi_iga);
                ders = derbasisfun(span, xi, p, 1, Xi_iga);
                wt = gw(q) * J;
                for ii = 0:p
                    I = span - p + ii + 1;
                    f_iga(I) = f_iga(I) + ders(1, ii + 1) * pi^2 * sin(pi * xi) * wt;
                    for jj = 0:p
                        JJ = span - p + jj + 1;
                        K_iga(I, JJ) = K_iga(I, JJ) + ders(2, ii + 1) * ders(2, jj + 1) * wt;
                    end
                end
            end
        end

        free_iga = 2:n_iga - 1;
        u_iga = zeros(n_iga, 1);
        u_iga(free_iga) = K_iga(free_iga, free_iga) \ f_iga(free_iga);

        % L2 error IGA
        err_iga = 0;
        for e = 1:nel_iga
            xi_a = knots_iga(e); xi_b = knots_iga(e + 1);
            if xi_b - xi_a < 1e-14; continue; end
            J = (xi_b - xi_a) / 2;
            for q = 1:nqp
                xi = (xi_a + xi_b) / 2 + J * gp(q);
                span = findspan(n_iga - 1, p, xi, Xi_iga);
                N = basisfun(span, xi, p, Xi_iga);
                u_h = 0;
                for k = 0:p
                    u_h = u_h + N(k + 1) * u_iga(span - p + k + 1);
                end
                err_iga = err_iga + (u_h - sin(pi * xi))^2 * gw(q) * J;
            end
        end
        err_iga = sqrt(err_iga);

        %% C0: repeated interior knots (FEA-like)
        % Each interior knot repeated p times -> C^0 at element boundaries
        interior_c0 = [];
        for k = 1:nel - 1
            xi_k = k / nel;
            for r = 1:p
                interior_c0 = [interior_c0, xi_k];
            end
        end
        Xi_c0 = [zeros(1, p + 1), interior_c0, ones(1, p + 1)];
        n_c0 = length(Xi_c0) - p - 1;

        % Assemble C0
        knots_c0 = unique(Xi_c0);
        nel_c0 = length(knots_c0) - 1;
        K_c0 = zeros(n_c0, n_c0);
        f_c0 = zeros(n_c0, 1);

        for e = 1:nel_c0
            xi_a = knots_c0(e); xi_b = knots_c0(e + 1);
            if xi_b - xi_a < 1e-14; continue; end
            J = (xi_b - xi_a) / 2;
            for q = 1:nqp
                xi = (xi_a + xi_b) / 2 + J * gp(q);
                span = findspan(n_c0 - 1, p, xi, Xi_c0);
                ders = derbasisfun(span, xi, p, 1, Xi_c0);
                wt = gw(q) * J;
                for ii = 0:p
                    I = span - p + ii + 1;
                    f_c0(I) = f_c0(I) + ders(1, ii + 1) * pi^2 * sin(pi * xi) * wt;
                    for jj = 0:p
                        JJ = span - p + jj + 1;
                        K_c0(I, JJ) = K_c0(I, JJ) + ders(2, ii + 1) * ders(2, jj + 1) * wt;
                    end
                end
            end
        end

        free_c0 = 2:n_c0 - 1;
        u_c0 = zeros(n_c0, 1);
        u_c0(free_c0) = K_c0(free_c0, free_c0) \ f_c0(free_c0);

        % L2 error C0
        err_c0 = 0;
        for e = 1:nel_c0
            xi_a = knots_c0(e); xi_b = knots_c0(e + 1);
            if xi_b - xi_a < 1e-14; continue; end
            J = (xi_b - xi_a) / 2;
            for q = 1:nqp
                xi = (xi_a + xi_b) / 2 + J * gp(q);
                span = findspan(n_c0 - 1, p, xi, Xi_c0);
                N = basisfun(span, xi, p, Xi_c0);
                u_h = 0;
                for k = 0:p
                    u_h = u_h + N(k + 1) * u_c0(span - p + k + 1);
                end
                err_c0 = err_c0 + (u_h - sin(pi * xi))^2 * gw(q) * J;
            end
        end
        err_c0 = sqrt(err_c0);

        ratio = err_c0 / err_iga;
        fprintf('  %3d | %4d | %3d/%3d | %12.4e | %12.4e | %8.2f\n', p, nel, n_iga, n_c0, err_iga, err_c0, ratio);
    end
    fprintf('\n');
end

%% Plot comparison for p=3
p = 3;
nel = 8;
nqp = p + 2;
[gp, gw] = gaussQuad(nqp);

% IGA solution
interior = linspace(0, 1, nel + 1);
interior = interior(2:end - 1);
Xi_iga = [zeros(1, p + 1), interior, ones(1, p + 1)];
n_iga = length(Xi_iga) - p - 1;
knots_iga = unique(Xi_iga);
nel_iga = length(knots_iga) - 1;
K = zeros(n_iga); f = zeros(n_iga, 1);
for e = 1:nel_iga
    xi_a = knots_iga(e); xi_b = knots_iga(e + 1);
    if xi_b - xi_a < 1e-14; continue; end
    J = (xi_b - xi_a) / 2;
    for q = 1:nqp
        xi = (xi_a + xi_b) / 2 + J * gp(q);
        span = findspan(n_iga - 1, p, xi, Xi_iga);
        ders = derbasisfun(span, xi, p, 1, Xi_iga);
        wt = gw(q) * J;
        for ii = 0:p
            I = span - p + ii + 1;
            f(I) = f(I) + ders(1, ii + 1) * pi^2 * sin(pi * xi) * wt;
            for jj = 0:p
                JJ = span - p + jj + 1;
                K(I, JJ) = K(I, JJ) + ders(2, ii + 1) * ders(2, jj + 1) * wt;
            end
        end
    end
end
free = 2:n_iga - 1;
u_iga_plot = zeros(n_iga, 1);
u_iga_plot(free) = K(free, free) \ f(free);

% Evaluate and plot
n_plot = 200;
x_plot = linspace(0, 1 - 1e-10, n_plot);
y_iga = zeros(1, n_plot);
for ix = 1:n_plot
    xi = x_plot(ix);
    span = findspan(n_iga - 1, p, xi, Xi_iga);
    N = basisfun(span, xi, p, Xi_iga);
    for k = 0:p
        y_iga(ix) = y_iga(ix) + N(k + 1) * u_iga_plot(span - p + k + 1);
    end
end

figure(1);
subplot(1, 2, 1);
plot(x_plot, abs(y_iga - sin(pi * x_plot)), 'b-', 'LineWidth', 1.5);
title('Error: IGA (C2) vs exact, p=3, 8 elem');
xlabel('x'); ylabel('|u_h - u_{exact}|');
grid on;

subplot(1, 2, 2);
% Derivative comparison (shows smoothness advantage)
dy_iga = zeros(1, n_plot);
for ix = 1:n_plot
    xi = x_plot(ix);
    span = findspan(n_iga - 1, p, xi, Xi_iga);
    ders = derbasisfun(span, xi, p, 1, Xi_iga);
    for k = 0:p
        dy_iga(ix) = dy_iga(ix) + ders(2, k + 1) * u_iga_plot(span - p + k + 1);
    end
end
plot(x_plot, dy_iga, 'b-', 'LineWidth', 1.5);
hold on;
plot(x_plot, pi * cos(pi * x_plot), 'r--', 'LineWidth', 1);
hold off;
title('Derivative: IGA vs exact');
xlabel('x'); ylabel('du/dx');
legend('IGA', 'Exact');
grid on;
drawnow;

fprintf('=== k-Refinement Study Complete ===\n');
