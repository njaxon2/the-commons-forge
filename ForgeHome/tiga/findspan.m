function mid = findspan(n, p, u, U)
% FINDSPAN Find the knot span index
%   Based on Algorithm A2.1 from "The NURBS Book" (Piegl & Tiller)
%   Input: n - number of basis functions minus 1
%          p - degree
%          u - parametric point
%          U - knot vector
%   Output: mid - knot span index

if u == U(n+2)
    mid = n;
    return;
end

low = p;
high = n + 1;
mid = floor((low + high) / 2);

while u < U(mid+1) || u >= U(mid+2)
    if u < U(mid+1)
        high = mid;
    else
        low = mid;
    end
    mid = floor((low + high) / 2);
end
end
