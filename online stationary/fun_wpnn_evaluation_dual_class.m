% Algorithm 5: WPNN testing
% calculate the corresponding pdf for given input
% dual class due to labels={0,1}
%
% @Input:
%   q: data input
%   w1: network parameter for class 1
%   w2: network parameter for class 2
%   r: dialation parameter
%   m: m-th order b-spline
%   constant: 2^j0
%   k: translation parameter
%   R: length of r
%   n: number of feature for input x
%   wavelet: order of wavelet
%   b: index of the relevant frame functions

%
%@Output
%   p1: pdf for the first class
%   p2: pdf for the second class
%
% Author: E. S. Garcia-Trevino, Pu Yang, and J. A. Barria
% Paper: Wavelet Probabilistic Neural Networks, 
%            IEEE Transactions on Neural Networks and Learning Systems, 
%            2022
% Institution: Imperial College London
% Date: Jan-2022

function [p1,p2] = fun_wpnn_evaluation_dual_class(q,w1,w2,r,m,constant,k,R,n,wavelet,b)

Neval = size(q,1);
p1 = zeros(1,Neval);
p2 = zeros(1,Neval);

phi = [];
if(b~=0)
    for j =1:length(b)
        z = sqrt(sum( (constant.*q - k(b(j),:)).^2) ) + m/2;
        phi = [phi fun_radial_bspline(z, constant^(n/2), wavelet)];
    end
    
    tmp_1= w1(b) * phi';
    tmp_2= w2(b) * phi';
    if(isempty(tmp_1))
        p1 = 0;
    else
        p1 = w1(b) * phi';
    end
    
    if(isempty(tmp_2))
        p2 = 0;
    else
        p2 = w2(b) * phi';
    end
    
end

end