% Algorithm 1: find relevant frame for given datapoit
% find the relevant frame function for given input x
%
% @Input:
%   x: data
%   r: dialation parameter
%   m: m-th order b-spline
%   constant: 2^j0
%   R: length of r
%   dimension: number of feature for input x
%               
%@Output
%   b: the index of the relevant frame function
% 
% Author: E. S. Garcia-Trevino, Pu Yang, and J. A. Barria
% Paper: Wavelet Probabilistic Neural Networks, 
%            IEEE Transactions on Neural Networks and Learning Systems, 
%            2022
% Institution: Imperial College London
% Date: Jan-2022

function [b] = fun_relevant_frame_for_given_datapoint(x,r,m,constant,R,dimension)

h = zeros(1,dimension);

A = [];
for j = 1:R%number of frame functions
    mmin = (r(j)-m/2)/constant;
    mmax =  (r(j)+m/2)/constant;
    for i = 1:dimension %number of dimension of input data
        if (x(i) >= mmin) && (x(i) <mmax)
            h(i)=h(i)+1;
            if(i==1)
                A(i,h(i)) = j;
            else
                A(i,h(i)) = (j-1)*R^(i-1);
            end
        end
    end
end

if (size(A,1) == dimension && all(all(isnan(A))) == false)
    b = A(1,:);
    for j = 1:(dimension-1)
        g = [];
        for i = 1:length(b)
            g = [g; b(i)+A(j+1,:)];
        end
        b = g;
    end
    b = [b(~isnan(b))];
    b = nonzeros(b);
else
    b = [];
end

end