function N = basisfun(i, u, p, U)
% BASISFUN Compute nonvanishing basis functions
%   Based on Algorithm A2.2 from "The NURBS Book"
%   Input: i - knot span index (from findspan)
%          u - parametric point
%          p - degree
%          U - knot vector
%   Output: N - vector of basis function values (length p+1)

N = zeros(1, p+1);
left = zeros(1, p+1);
right = zeros(1, p+1);

N(1) = 1.0;

for j = 1:p
    left(j+1) = u - U(i+1-j+1);
    right(j+1) = U(i+j+1) - u;
    saved = 0.0;
    for r = 0:j-1
        temp = N(r+1) / (right(r+2) + left(j-r+1));
        N(r+1) = saved + right(r+2) * temp;
        saved = left(j-r+1) * temp;
    end
    N(j+1) = saved;
end
end
