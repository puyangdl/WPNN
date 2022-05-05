% RWFDE Initialisation
% Use input to compute the translation parameter k->trans and dialation parameter r
% Other parameters are calculated for future usage% 

% @Input:
%   x: data
%   wavelet: linear/quad/cubic wavelet
%   j0: dialation parameter
%               
%@Output
%   dimension: number of feature for input x
%   m: m-th order b-spline
%   constant: 2^j0
%   r: dialation parameter
%   R: length of r
%   trans: translation parameter
% 
% Author: E. S. Garcia-Trevino, Pu Yang, and J. A. Barria
% Paper: Wavelet Probabilistic Neural Networks, 
%            IEEE Transactions on Neural Networks and Learning Systems, 
%            2022
% Institution: Imperial College London
% Date: Jan-2022

function [dimension, m, constant, r, R, trans]=fun_wpnn_initialisation(x, wavelet, j0)

switch wavelet
    case 'bsplines_linear'
        m=2;
        u=1;
    case 'bsplines_quad'
        m=3;
        u=1;
    case 'bsplines_cubic'
        m=4;
        u=2;
end

[dimension]=size(x,2);
constant=2^(j0);

r = -u:(constant+u);
R = length(r);

trans = r;
for j=1:dimension-1
    trans=combvec(trans,r);%all the combination of the possible phi
end
trans = trans.';

end
