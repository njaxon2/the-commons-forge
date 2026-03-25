function [gp, gw] = gaussQuad(npts)
% GAUSSQUAD Return Gauss-Legendre quadrature points and weights on [-1,1]
%   Input: npts - number of quadrature points (1 to 5)
%   Output: gp - quadrature points
%           gw - quadrature weights

switch npts
    case 1
        gp = 0;
        gw = 2;
    case 2
        gp = [-1/sqrt(3), 1/sqrt(3)];
        gw = [1, 1];
    case 3
        gp = [-sqrt(3/5), 0, sqrt(3/5)];
        gw = [5/9, 8/9, 5/9];
    case 4
        gp = [-0.861136311594953, -0.339981043584856, ...
               0.339981043584856,  0.861136311594953];
        gw = [0.347854845137454, 0.652145154862546, ...
              0.652145154862546, 0.347854845137454];
    case 5
        gp = [-0.906179845938664, -0.538469310105683, 0, ...
               0.538469310105683,  0.906179845938664];
        gw = [0.236926885056189, 0.478628670499366, 0.568888888888889, ...
              0.478628670499366, 0.236926885056189];
    otherwise
        error('gaussQuad: npts must be 1-5');
end
end
