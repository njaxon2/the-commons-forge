function dN = basisfunder(i, u, p, U, n_deriv)
% BASISFUNDER Wrapper for derbasisfun - returns derivative row
%   Returns the n_deriv-th derivative of basis functions
%   Output: 1 x (p+1) row vector of derivatives
ders = derbasisfun(i, u, p, n_deriv, U);
dN = ders(n_deriv+1, :);
end
