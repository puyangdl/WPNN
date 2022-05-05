% Gaussian kernel KDE
%
% @Input:
%   x: x test
%   xi: x train
%   bandwidth: hyperparameter
%               
%@Output
%   fx: pdf
% 
% Author: E. S. Garcia-Trevino, Pu Yang, and J. A. Barria
% Paper: Wavelet Probabilistic Neural Networks, 
%            IEEE Transactions on Neural Networks and Learning Systems, 
%            2022
% Institution: Imperial College London
% Date: Jan-2022

function fx=funmyKDE(x,xi,bandwidth)

n1=size(x,1); n2=size(xi,1); ps=zeros(n1,n2);
 
 for i=1:n2
  ps(:,i) = sum( abs((x - ones(n1,1)*xi(i,:))).^2 ,2)./bandwidth.^2/2;
 end;
 
K = exp(-ps);

fx=sum(K,2);
fx=fx/sum(fx);
end


