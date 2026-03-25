%% TIGA Mode Shape Visualization
%  Compute and plot first 4 vibration mode shapes of a bar
%  Tests: subplot, multiple plots, eigenvectors, normalized plots

clear;
fprintf('=== TIGA Mode Shape Visualization ===\n\n');

%% Setup: p=3 cubic B-spline, 16 elements
p = 3;
num_el = 16;
h = 1.0 / num_el;

% Build knot vector
interior = linspace(0, 1, num_el + 1);
interior = interior(2:end-1);
Xi = [zeros(1, p+1), interior, ones(1, p+1)];
n = length(Xi) - p - 1;

fprintf('  p=%d, %d elements, n=%d DOFs\n', p, num_el, n);

% Gauss quadrature
nqp = p + 2;
[gp, gw] = gaussQuad(nqp);
knots_unique = unique(Xi);
nel = length(knots_unique) - 1;

% Assemble K and M
K = zeros(n, n);
M = zeros(n, n);

for e = 1:nel
    xi_a = knots_unique(e);
    xi_b = knots_unique(e + 1);
    if xi_b - xi_a < 1e-14
        continue;
    end
    J_xi = (xi_b - xi_a) / 2;

    for q = 1:nqp
        xi = (xi_a + xi_b) / 2 + J_xi * gp(q);
        span = findspan(n-1, p, xi, Xi);
        ders = derbasisfun(span, xi, p, 1, Xi);
        N_val = ders(1, :);
        dN_dx = ders(2, :);
        wt = gw(q) * J_xi;

        for ii = 0:p
            I = span - p + ii + 1;
            for jj = 0:p
                J = span - p + jj + 1;
                K(I, J) = K(I, J) + dN_dx(ii+1) * dN_dx(jj+1) * wt;
                M(I, J) = M(I, J) + N_val(ii+1) * N_val(jj+1) * wt;
            end
        end
    end
end

% Apply BCs: fixed-fixed
free = 2:n-1;
K_f = K(free, free);
M_f = M(free, free);

% Solve generalized eigenvalue problem [V,D] = eig(K,M)
[V, D] = eig(K_f, M_f);

% Extract eigenvalues from diagonal
n_free = length(free);
lambda_all = zeros(n_free, 1);
for k = 1:n_free
    lambda_all(k) = D(k, k);
end

% Sort by eigenvalue
[lambda_sorted, idx] = sort(lambda_all);

% Print first 6 frequencies
n_show = min(6, n_free);
fprintf('\n  Natural frequencies (first %d modes):\n', n_show);
for m = 1:n_show
    exact = (m * pi)^2;
    omega = sqrt(lambda_sorted(m));
    omega_exact = m * pi;
    fprintf('    Mode %d: omega = %8.4f (exact: %8.4f), error = %.2e\n', ...
        m, omega, omega_exact, abs(omega - omega_exact) / omega_exact);
end

%% Evaluate mode shapes at fine grid points
n_pts = 200;
x_plot = linspace(0, 1, n_pts);
modes = zeros(n_pts, 4);

for mode_num = 1:4
    % Get eigenvector (reorder by sorted index)
    phi = V(:, idx(mode_num));
    % Normalize
    phi = phi / max(abs(phi));
    
    % Full DOF vector (with BCs = 0)
    phi_full = zeros(n, 1);
    phi_full(free) = phi;
    
    % Evaluate B-spline at each plot point
    for pt = 1:n_pts
        xi = x_plot(pt);
        if xi > 1 - 1e-14
            xi = 1 - 1e-14;
        end
        span = findspan(n-1, p, xi, Xi);
        N = basisfun(span, xi, p, Xi);
        
        val = 0;
        for k = 0:p
            val = val + N(k+1) * phi_full(span - p + k + 1);
        end
        modes(pt, mode_num) = val;
    end
end

%% Plot mode shapes
figure(1);
for m = 1:4
    subplot(2, 2, m);
    plot(x_plot, modes(:, m), 'b-', 'LineWidth', 1.5);
    hold on;
    % Plot exact: sin(m*pi*x)
    exact_mode = sin(m * pi * x_plot);
    exact_mode = exact_mode / max(abs(exact_mode));
    plot(x_plot, exact_mode, 'r--', 'LineWidth', 1);
    hold off;
    title(sprintf('Mode %d', m));
    xlabel('x');
    ylabel('Displacement');
    grid on;
    legend('IGA', 'Exact');
end
drawnow;

fprintf('\n=== Mode Shape Visualization Complete ===\n');
