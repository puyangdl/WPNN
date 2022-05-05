% Algorithm 3: Update network parameters in the online stationary case
%
% @Input:
%   x: data input
%   r: dialation parameter
%   m: m-th order b-spline
%   constant: 2^j0
%   k: translation parameter
%   R: length of r
%   n: number of feature for input x
%   wavelet: order of wavelet
%   M: size of the translation vector K
%   b: index of the relevant frame functions
%   w: network parameter for specific class

%
%@Output
%   w: updated network parameter for specific class
%
% Author: E. S. Garcia-Trevino, Pu Yang, and J. A. Barria
% Paper: Wavelet Probabilistic Neural Networks, 
%            IEEE Transactions on Neural Networks and Learning Systems, 
%            2022
% Institution: Imperial College London
% Date: Jan-2022

function [w] = fun_wpnn_online_updating_stationary(x,r,m,constant,k,R,n,wavelet,M,w,t,b)

    for j =1:numel(b)
        idx = b(j);
        z = sqrt(sum( (constant.*x- k(idx,:)).^2) ) + m/2;
        w(idx) = w(idx) + (fun_radial_bspline(z, constant^(n/2), wavelet) - w(idx) )/t;
    end
    f = 1:M;
    f(b)=[];
    w(f) = w(f) - w(f)/t;
end