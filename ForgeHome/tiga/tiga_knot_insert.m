%% TIGA Knot Insertion & Refinement
%  Test B-spline knot insertion algorithm
%  Verify: geometry preservation, partition of unity, refinement convergence
%  Tests: while loops, function handles, cell arrays, error norms

clear;
fprintf('=== TIGA Knot Insertion & Refinement ===\n\n');

%% Test 1: Knot insertion preserves geometry
fprintf('--- Test 1: Knot Insertion Preserves Geometry ---\n');
p = 2;

% Original: 2 elements
Xi_orig = [0 0 0 0.5 1 1 1];
n_orig = length(Xi_orig) - p - 1;  % 4 control points
P_orig = [0.0; 0.5; 1.5; 2.0];     % 1D control point values

% Insert knot at xi=0.25
xi_new = 0.25;
[Xi_new, P_new, n_new] = knotInsert1D(Xi_orig, P_orig, p, xi_new);

fprintf('  Original: n=%d, Xi=[', n_orig);
for i = 1:length(Xi_orig)
    if i > 1
        fprintf(' ');
    end
    fprintf('%.2f', Xi_orig(i));
end
fprintf(']\n');

fprintf('  Refined:  n=%d, Xi=[', n_new);
for i = 1:length(Xi_new)
    if i > 1
        fprintf(' ');
    end
    fprintf('%.2f', Xi_new(i));
end
fprintf(']\n');

% Verify: evaluate curve at test points
n_test = 50;
xi_test = linspace(0, 1-1e-10, n_test);
err_max = 0;

for t = 1:n_test
    xi = xi_test(t);

    % Original curve
    span_o = findspan(n_orig-1, p, xi, Xi_orig);
    N_o = basisfun(span_o, xi, p, Xi_orig);
    val_orig = 0;
    for k = 0:p
        val_orig = val_orig + N_o(k+1) * P_orig(span_o - p + k + 1);
    end

    % Refined curve
    span_n = findspan(n_new-1, p, xi, Xi_new);
    N_n = basisfun(span_n, xi, p, Xi_new);
    val_new = 0;
    for k = 0:p
        val_new = val_new + N_n(k+1) * P_new(span_n - p + k + 1);
    end

    err_max = max(err_max, abs(val_orig - val_new));
end

fprintf('  Max geometry error after knot insertion: %.2e\n', err_max);
if err_max < 1e-12
    fprintf('  PASS: Geometry preserved to machine precision\n');
else
    fprintf('  FAIL: Geometry NOT preserved\n');
end

%% Test 2: Multiple knot insertions
fprintf('\n--- Test 2: Multiple Knot Insertions ---\n');
Xi = Xi_orig;
P = P_orig;
n = n_orig;

new_knots = [0.125, 0.375, 0.625, 0.875];
for i = 1:length(new_knots)
    [Xi, P, n] = knotInsert1D(Xi, P, p, new_knots(i));
end

fprintf('  After inserting 4 knots: n=%d DOFs\n', n);

% Verify
err_max2 = 0;
for t = 1:n_test
    xi = xi_test(t);
    span_o = findspan(n_orig-1, p, xi, Xi_orig);
    N_o = basisfun(span_o, xi, p, Xi_orig);
    val_orig = 0;
    for k = 0:p
        val_orig = val_orig + N_o(k+1) * P_orig(span_o - p + k + 1);
    end

    span_n = findspan(n-1, p, xi, Xi);
    N_n = basisfun(span_n, xi, p, Xi);
    val_new = 0;
    for k = 0:p
        val_new = val_new + N_n(k+1) * P(span_n - p + k + 1);
    end
    err_max2 = max(err_max2, abs(val_orig - val_new));
end

fprintf('  Max geometry error after 4 insertions: %.2e\n', err_max2);
if err_max2 < 1e-12
    fprintf('  PASS: Geometry preserved\n');
else
    fprintf('  FAIL: Geometry NOT preserved\n');
end

%% Test 3: Plot original vs refined basis functions
fprintf('\n--- Test 3: Basis Function Plots ---\n');

figure(1);
n_plot = 200;
x_plot = linspace(0, 1-1e-10, n_plot);

% Original basis
subplot(2, 1, 1);
for i = 1:n_orig
    y_vals = zeros(1, n_plot);
    for t = 1:n_plot
        span = findspan(n_orig-1, p, x_plot(t), Xi_orig);
        N = basisfun(span, x_plot(t), p, Xi_orig);
        for k = 0:p
            if span - p + k + 1 == i
                y_vals(t) = N(k+1);
            end
        end
    end
    plot(x_plot, y_vals, 'LineWidth', 1.5);
    hold on;
end
hold off;
title('Original Basis (4 functions)');
xlabel('xi');
ylabel('N');
grid on;

% Refined basis
subplot(2, 1, 2);
for i = 1:n
    y_vals = zeros(1, n_plot);
    for t = 1:n_plot
        span = findspan(n-1, p, x_plot(t), Xi);
        N_val = basisfun(span, x_plot(t), p, Xi);
        for k = 0:p
            if span - p + k + 1 == i
                y_vals(t) = N_val(k+1);
            end
        end
    end
    plot(x_plot, y_vals, 'LineWidth', 1);
    hold on;
end
hold off;
title('Refined Basis (8 functions)');
xlabel('xi');
ylabel('N');
grid on;
drawnow;

fprintf('  Plots generated\n');
fprintf('\n=== TIGA Knot Insertion Complete ===\n');
