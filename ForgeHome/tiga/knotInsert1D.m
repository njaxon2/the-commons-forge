function [Xi_new, P_new, n_new] = knotInsert1D(Xi, P, p, xi_bar)
% KNOTINSERT1D Insert a single knot into a B-spline
%   Input:  Xi - knot vector, P - control points, p - degree, xi_bar - new knot
%   Output: Xi_new - new knot vector, P_new - new control points, n_new - new n

n = length(Xi) - p - 1;

% Find knot span
k = 0;
for i = 1:length(Xi)-1
    if xi_bar >= Xi(i) && xi_bar < Xi(i+1)
        k = i;
        break;
    end
end
if k == 0
    k = length(Xi) - p - 1;
end

% New knot vector
Xi_new = [Xi(1:k), xi_bar, Xi(k+1:end)];

% New control points
n_new = n + 1;
P_new = zeros(n_new, 1);

for i = 1:n_new
    if i <= k - p
        P_new(i) = P(i);
    elseif i >= k + 1
        P_new(i) = P(i - 1);
    else
        alpha = (xi_bar - Xi(i)) / (Xi(i + p) - Xi(i));
        P_new(i) = alpha * P(i) + (1 - alpha) * P(i - 1);
    end
end
end
