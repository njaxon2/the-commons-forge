function C = nurbsCurveEval(p, U, Pw, u_vals)
% NURBSCURVEEVAL Evaluate a NURBS curve at given parameter values
%   Input: p - degree
%          U - knot vector
%          Pw - weighted control points (n x dim+1), last col = weights
%          u_vals - parameter values to evaluate at
%   Output: C - evaluated curve points (len(u_vals) x dim)

n = size(Pw, 1) - 1;
dim = size(Pw, 2) - 1;
num_pts = length(u_vals);
C = zeros(num_pts, dim);

for k = 1:num_pts
    u = u_vals(k);
    span = findspan(n, p, u, U);
    N = basisfun(span, u, p, U);
    
    Cw = zeros(1, dim+1);
    for i = 0:p
        Cw = Cw + N(i+1) * Pw(span-p+i+1, :);
    end
    
    % Divide by weight
    for d = 1:dim
        C(k, d) = Cw(d) / Cw(dim+1);
    end
end
end
