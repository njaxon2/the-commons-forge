%% TIGA Free Vibration Analysis
%  Solve generalized eigenvalue problem: K*phi = omega^2 * M * phi
%  For a simply-supported beam (Euler-Bernoulli)
%  Exact: omega_n = (n*pi)^2 * sqrt(EI/(rho*A*L^4))
%  Tests: generalized eigenvalue, mass matrix, mode shapes

clear;
fprintf('=== TIGA Free Vibration Analysis ===\n\n');

% Beam properties
L = 1.0;     % length
EI = 1.0;    % bending stiffness
rhoA = 1.0;  % mass per unit length

p = 3;       % cubic B-splines (C2 continuous, good for 4th order problem)
nel = 12;

% Build knot vector
interior = linspace(0, L, nel + 1);
interior = interior(2:end - 1);
Xi = [zeros(1, p + 1), interior, L * ones(1, p + 1)];
n = length(Xi) - p - 1;

fprintf('  Beam: L=%.1f, EI=%.1f, rhoA=%.1f\n', L, EI, rhoA);
fprintf('  IGA: p=%d, nel=%d, n=%d\n', p, nel, n);

% Quadrature
nqp = p + 2;
[gp, gw] = gaussQuad(nqp);
knots_unique = unique(Xi);
nel_actual = length(knots_unique) - 1;

% Assemble stiffness and mass
% For Euler-Bernoulli beam: K uses 2nd derivatives, M uses 0th
K = zeros(n, n);
M = zeros(n, n);

for e = 1:nel_actual
    xi_a = knots_unique(e);
    xi_b = knots_unique(e + 1);
    if xi_b - xi_a < 1e-14
        continue;
    end
    J_xi = (xi_b - xi_a) / 2;
    for q = 1:nqp
        xi = (xi_a + xi_b) / 2 + J_xi * gp(q);
        span = findspan(n - 1, p, xi, Xi);
        ders = derbasisfun(span, xi, p, 2, Xi);
        N_val = ders(1, :);
        d2N_dx2 = ders(3, :);
        wt = gw(q) * J_xi;
        for ii = 0:p
            I = span - p + ii + 1;
            for jj = 0:p
                J = span - p + jj + 1;
                K(I, J) = K(I, J) + EI * d2N_dx2(ii + 1) * d2N_dx2(jj + 1) * wt;
                M(I, J) = M(I, J) + rhoA * N_val(ii + 1) * N_val(jj + 1) * wt;
            end
        end
    end
end

% BCs: simply-supported = u(0) = u(L) = 0 (no moment BCs needed for IGA)
free = 2:n - 1;
K_f = K(free, free);
M_f = M(free, free);

% Solve generalized eigenvalue problem
[V, D] = eig(K_f, M_f);

% Extract eigenvalues and sort
eigenvalues = diag(D);
[eigenvalues, idx] = sort(eigenvalues);
V = V(:, idx);

% Compute frequencies
n_modes = min(8, length(eigenvalues));
fprintf('\n  Natural frequencies (first %d modes):\n', n_modes);
fprintf('  %5s | %14s | %14s | %10s\n', 'Mode', 'omega^2 (IGA)', 'omega^2 (exact)', 'Rel Error');
fprintf('  %5s-|-%14s-|-%14s-|-%10s\n', '-----', '--------------', '---------------', '----------');

for i = 1:n_modes
    omega2_iga = eigenvalues(i);
    omega2_exact = (i * pi / L)^4 * EI / rhoA;
    rel_err = abs(omega2_iga - omega2_exact) / omega2_exact;
    fprintf('  %5d | %14.4f | %14.4f | %10.2e\n', i, omega2_iga, omega2_exact, rel_err);
end

%% Plot mode shapes
figure(1);
n_plot = 200;
x_plot = linspace(0, L - 1e-10, n_plot);

for mode = 1:min(4, n_modes)
    subplot(2, 2, mode);

    % IGA mode shape
    phi = zeros(n, 1);
    phi(free) = V(:, mode);

    y_iga = zeros(1, n_plot);
    for ix = 1:n_plot
        xi = x_plot(ix);
        span = findspan(n - 1, p, xi, Xi);
        N_val = basisfun(span, xi, p, Xi);
        for k = 0:p
            y_iga(ix) = y_iga(ix) + N_val(k + 1) * phi(span - p + k + 1);
        end
    end

    % Normalize
    max_amp = max(abs(y_iga));
    if max_amp > 0
        y_iga = y_iga / max_amp;
    end

    % Exact mode shape
    y_exact = sin(mode * pi * x_plot / L);

    % Fix sign (eigenvectors can be +/-)
    if sum(y_iga .* y_exact) < 0
        y_iga = -y_iga;
    end

    plot(x_plot, y_iga, 'b-', 'LineWidth', 1.5);
    hold on;
    plot(x_plot, y_exact, 'r--', 'LineWidth', 1);
    hold off;

    omega2_iga = eigenvalues(mode);
    omega2_exact = (mode * pi / L)^4;
    title(sprintf('Mode %d (err=%.1e)', mode, abs(omega2_iga - omega2_exact) / omega2_exact));
    xlabel('x'); ylabel('phi');
    if mode == 1
        legend('IGA', 'Exact');
    end
    grid on;
end

drawnow;

fprintf('\n=== Vibration Analysis Complete ===\n');
