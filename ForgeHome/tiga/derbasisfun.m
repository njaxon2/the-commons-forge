function ders = derbasisfun(i, u, p, n_deriv, U)
% DERBASISFUN Compute derivatives of basis functions
%   Based on Algorithm A2.3 from "The NURBS Book"
%   Input: i - knot span
%          u - parametric point
%          p - degree
%          n_deriv - number of derivatives to compute
%          U - knot vector
%   Output: ders - (n_deriv+1) x (p+1) matrix of derivatives

ders = zeros(n_deriv+1, p+1);
ndu = zeros(p+1, p+1);
left = zeros(1, p+1);
right = zeros(1, p+1);
a = zeros(2, p+1);

ndu(1,1) = 1.0;

for j = 1:p
    left(j+1) = u - U(i+1-j+1);
    right(j+1) = U(i+j+1) - u;
    saved = 0.0;
    for r = 0:j-1
        ndu(j+1, r+1) = right(r+2) + left(j-r+1);
        temp = ndu(r+1, j) / ndu(j+1, r+1);
        ndu(r+1, j+1) = saved + right(r+2) * temp;
        saved = left(j-r+1) * temp;
    end
    ndu(j+1, j+1) = saved;
end

% Load basis functions
for j = 0:p
    ders(1, j+1) = ndu(j+1, p+1);
end

% Compute derivatives
for r = 0:p
    s1 = 0;
    s2 = 1;
    a(1, 1) = 1.0;
    for k = 1:n_deriv
        d = 0.0;
        rk = r - k;
        pk = p - k;
        if r >= k
            a(s2+1, 1) = a(s1+1, 1) / ndu(pk+2, rk+1);
            d = a(s2+1, 1) * ndu(rk+1, pk+1);
        end
        if rk >= -1
            j1 = 1;
        else
            j1 = -rk;
        end
        if r-1 <= pk
            j2 = k - 1;
        else
            j2 = p - r;
        end
        for j = j1:j2
            a(s2+1, j+1) = (a(s1+1, j+1) - a(s1+1, j)) / ndu(pk+2, rk+j+1);
            d = d + a(s2+1, j+1) * ndu(rk+j+1, pk+1);
        end
        if r <= pk
            a(s2+1, k+1) = -a(s1+1, k) / ndu(pk+2, r+1);
            d = d + a(s2+1, k+1) * ndu(r+1, pk+1);
        end
        ders(k+1, r+1) = d;
        j_temp = s1;
        s1 = s2;
        s2 = j_temp;
    end
end

% Multiply by correct factors
r_val = p;
for k = 1:n_deriv
    for j = 0:p
        ders(k+1, j+1) = ders(k+1, j+1) * r_val;
    end
    r_val = r_val * (p - k);
end
end
