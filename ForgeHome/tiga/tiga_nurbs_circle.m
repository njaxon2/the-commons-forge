%% TIGA NURBS Circle - Exact Geometry Representation
%  Demonstrate NURBS quarter circle (exact to machine precision)
%  Following the TIGA framework from tigaSPM13.pdf

clear;
fprintf('=== TIGA NURBS Quarter Circle ===\n');

%% Define NURBS quarter circle
% Degree
p = 2;

% Knot vector (open)
U = [0 0 0 1 1 1];

% Control points for quarter circle (weighted)
% Format: [x*w, y*w, w]
w = cos(pi/4);  % weight for mid control point = 1/sqrt(2)
Pw = [1 0 1;
      w w w;
      0 1 1];

fprintf('Control points (weighted):\n');
disp(Pw);
fprintf('Mid-point weight: %f\n', w);

%% Evaluate NURBS curve
num_pts = 101;
u_vals = linspace(0, 1, num_pts);
C = nurbsCurveEval(p, U, Pw, u_vals);

fprintf('\nCurve evaluated at %d points\n', num_pts);
fprintf('First point: (%f, %f)\n', C(1,1), C(1,2));
fprintf('Last point:  (%f, %f)\n', C(num_pts,1), C(num_pts,2));

%% Check circle accuracy
% Each point should have x^2 + y^2 = 1
radius_err = zeros(1, num_pts);
for k = 1:num_pts
    r = sqrt(C(k,1)^2 + C(k,2)^2);
    radius_err(k) = abs(r - 1.0);
end

max_radius_err = max(radius_err);
fprintf('\nMax radius error: %e (should be near machine eps)\n', max_radius_err);

%% Compare with exact circle
theta = linspace(0, pi/2, num_pts);
x_exact = cos(theta);
y_exact = sin(theta);

geom_err = zeros(1, num_pts);
for k = 1:num_pts
    dx = C(k,1) - x_exact(k);
    dy = C(k,2) - y_exact(k);
    geom_err(k) = sqrt(dx^2 + dy^2);
end
max_geom_err = max(geom_err);
fprintf('Max geometric error vs exact: %e\n', max_geom_err);

%% Plot
figure(1);
plot(C(:,1), C(:,2), 'b-');
hold on;
plot(x_exact, y_exact, 'r--');
plot(Pw(:,1)./Pw(:,3), Pw(:,2)./Pw(:,3), 'ko-');
hold off;
title('NURBS Quarter Circle');
xlabel('x');
ylabel('y');
legend('NURBS', 'Exact', 'Control Points');
axis equal;
grid on;
drawnow;
fprintf('Figure 1: NURBS circle plotted\n');

figure(2);
semilogy(u_vals, radius_err + eps);
title('Radius Error |r - 1|');
xlabel('u');
ylabel('Error');
grid on;
drawnow;
fprintf('Figure 2: Radius error plotted\n');

%% Full circle (4 NURBS arcs)
fprintf('\n--- Full Circle from 4 Arcs ---\n');
n_arc = 51;
u_arc = linspace(0, 1, n_arc);

% Arc 1: (1,0) to (0,1) - quarter 1 (already computed above)
% Arc 2: (0,1) to (-1,0) - quarter 2
Pw2 = [0 1 1; -w w w; -1 0 1];
C2 = nurbsCurveEval(p, U, Pw2, u_arc);

% Arc 3: (-1,0) to (0,-1) - quarter 3
Pw3 = [-1 0 1; -w -w w; 0 -1 1];
C3 = nurbsCurveEval(p, U, Pw3, u_arc);

% Arc 4: (0,-1) to (1,0) - quarter 4
Pw4 = [0 -1 1; w -w w; 1 0 1];
C4 = nurbsCurveEval(p, U, Pw4, u_arc);

% Check closure
C1_arc = nurbsCurveEval(p, U, Pw, u_arc);
closure_err = sqrt((C4(end,1) - C1_arc(1,1))^2 + (C4(end,2) - C1_arc(1,2))^2);
fprintf('Closure error: %e\n', closure_err);

% Check max radius error across all arcs
all_x = [C1_arc(:,1); C2(:,1); C3(:,1); C4(:,1)];
all_y = [C1_arc(:,2); C2(:,2); C3(:,2); C4(:,2)];
all_r = sqrt(all_x.^2 + all_y.^2);
max_full_err = max(abs(all_r - 1));
fprintf('Max radius error (full circle): %e\n', max_full_err);

figure(3);
plot(C1_arc(:,1), C1_arc(:,2), 'b-');
hold on;
plot(C2(:,1), C2(:,2), 'r-');
plot(C3(:,1), C3(:,2), 'g-');
plot(C4(:,1), C4(:,2), 'm-');
hold off;
axis equal;
title('Full NURBS Circle (4 Arcs)');
grid on;
drawnow;
fprintf('Figure 3: Full circle plotted\n');

fprintf('\n=== TIGA NURBS Circle Complete ===\n');
