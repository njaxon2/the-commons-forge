%% TIGA Knot Insertion and p-Refinement
%  Test knot insertion (h-refinement) and demonstrate
%  the effect on basis function quality
%  Following Section 3 of tigaSPM13.pdf

clear;
fprintf('=== TIGA Knot Insertion (h-Refinement) ===\n');

%% Initial knot vector
p = 2;
Xi0 = [0 0 0 1 1 1];  % Initial: single element
n0 = length(Xi0) - p - 1;
fprintf('Initial: p=%d, n=%d, elements=%d\n', p, n0, length(unique(Xi0))-1);

%% First refinement: insert midpoint
Xi1 = [0 0 0 0.5 1 1 1];
n1 = length(Xi1) - p - 1;
fprintf('After 1 insertion: n=%d, elements=%d\n', n1, length(unique(Xi1))-1);

%% Second refinement: insert quarter points
Xi2 = [0 0 0 0.25 0.5 0.75 1 1 1];
n2 = length(Xi2) - p - 1;
fprintf('After 3 insertions: n=%d, elements=%d\n', n2, length(unique(Xi2))-1);

%% Third refinement: insert eighth points
Xi3 = [0 0 0 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1 1 1];
n3 = length(Xi3) - p - 1;
fprintf('After 7 insertions: n=%d, elements=%d\n', n3, length(unique(Xi3))-1);

%% Plot basis functions at each level
num_plot = 201;
u_plot = linspace(0, 1, num_plot);

% Helper: compute all basis function values
knot_vectors = {Xi0, Xi1, Xi2, Xi3};
labels = {'1 element', '2 elements', '4 elements', '8 elements'};

for level = 1:4
    Xi_cur = knot_vectors{level};
    n_cur = length(Xi_cur) - p - 1;
    B_cur = zeros(n_cur, num_plot);

    for k = 1:num_plot
        u = u_plot(k);
        if u >= 1
            u = 1 - 1e-10;
        end
        span = findspan(n_cur-1, p, u, Xi_cur);
        N = basisfun(span, u, p, Xi_cur);
        for j = 0:p
            idx = span - p + j + 1;
            B_cur(idx, k) = N(j+1);
        end
    end

    figure(level);
    hold on;
    for i = 1:n_cur
        plot(u_plot, B_cur(i,:));
    end
    hold off;
    title_str = sprintf('B-spline basis (p=%d, %s)', p, labels{level});
    title(title_str);
    xlabel('xi');
    ylabel('N_{i,p}(xi)');
    grid on;
    drawnow;
end

%% Verify partition of unity at finest level
fprintf('\n--- Partition of Unity Check (finest mesh) ---\n');
max_pou_err = 0;
for k = 1:num_plot
    u = u_plot(k);
    if u >= 1
        u = 1 - 1e-10;
    end
    span = findspan(n3-1, p, u, Xi3);
    N = basisfun(span, u, p, Xi3);
    pou = sum(N);
    err = abs(pou - 1.0);
    if err > max_pou_err
        max_pou_err = err;
    end
end
fprintf('Max partition of unity error: %e\n', max_pou_err);

%% Greville abscissae (control point parameters)
fprintf('\n--- Greville Abscissae ---\n');
for level = 1:4
    Xi_cur = knot_vectors{level};
    n_cur = length(Xi_cur) - p - 1;
    grev = zeros(1, n_cur);
    for i = 1:n_cur
        s = 0;
        for j = 1:p
            s = s + Xi_cur(i + j);
        end
        grev(i) = s / p;
    end
    fprintf('Level %d Greville: ', level);
    for i = 1:n_cur
        fprintf('%.4f ', grev(i));
    end
    fprintf('\n');
end

%% Test with cubic (p=3) for comparison
fprintf('\n--- Cubic B-splines (p=3) ---\n');
p3 = 3;
Xi3_cubic = [0 0 0 0 0.25 0.5 0.75 1 1 1 1];
n3_cubic = length(Xi3_cubic) - p3 - 1;
fprintf('Cubic: n=%d, elements=%d\n', n3_cubic, length(unique(Xi3_cubic))-1);

B_cubic = zeros(n3_cubic, num_plot);
for k = 1:num_plot
    u = u_plot(k);
    if u >= 1
        u = 1 - 1e-10;
    end
    span = findspan(n3_cubic-1, p3, u, Xi3_cubic);
    N = basisfun(span, u, p3, Xi3_cubic);
    for j = 0:p3
        idx = span - p3 + j + 1;
        B_cubic(idx, k) = N(j+1);
    end
end

figure(5);
hold on;
for i = 1:n3_cubic
    plot(u_plot, B_cubic(i,:));
end
hold off;
title('Cubic B-spline Basis (p=3, 4 elements)');
xlabel('xi');
ylabel('N_{i,p}(xi)');
grid on;
drawnow;

% Verify partition of unity for cubic
max_pou_cubic = 0;
for k = 1:num_plot
    u = u_plot(k);
    if u >= 1
        u = 1 - 1e-10;
    end
    span = findspan(n3_cubic-1, p3, u, Xi3_cubic);
    N = basisfun(span, u, p3, Xi3_cubic);
    pou_val = sum(N);
    err_val = abs(pou_val - 1.0);
    if err_val > max_pou_cubic
        max_pou_cubic = err_val;
    end
end
fprintf('Cubic partition of unity error: %e\n', max_pou_cubic);

fprintf('\n=== TIGA Knot Insertion Complete ===\n');
