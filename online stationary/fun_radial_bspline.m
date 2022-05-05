% Table 1 & Eq 7: Explicit expression for B-spline functions
%
% @Input:
%   z: input from Nm(z)
%   nconstant: 2^{j0*n/2}
%   wavelet: wavelet with specific order
%               
%@Output
%   phi: \Phi from equation 7
% 
% Author: E. S. Garcia-Trevino, Pu Yang, and J. A. Barria
% Paper: Wavelet Probabilistic Neural Networks, 
%            IEEE Transactions on Neural Networks and Learning Systems, 
%            2022
% Institution: Imperial College London
% Date: Jan-2022

function [phi] = fun_radial_bspline(z,nconstant,wavelet)
switch wavelet
    
    case 'bsplines_linear'
        
        if (0<=z && z<1)
            phi=z;
        elseif (1<=z && z<2)
            phi=(2-z);
        else
            phi=0;
        end
        
    case 'bsplines_quad'
        if (0<=z && z <1)
            phi =(1/2*z ^2);
        elseif (1<=z  && z <2)
            phi =(3/4-(z -3/2)^2);
        elseif (2<=z  && z <3)
            phi =(1/2*(z -3)^2);
        else
            phi =0;
        end
        
    case 'bsplines_cubic'
        if (0<=z  && z <1)
            phi =(1/6*z ^3);
        elseif (1<=z  && z <2)
            phi =(1/6) * (- 3*z ^3 +12*z ^2 -12*z  +4 );
        elseif (2<=z  && z <3)
            phi =(1/6)* ( 3*z ^3 -24*z ^2 +60*z  -44 ) ;
        elseif (3<=z  && z <4)
            phi =(1/6)* (4 - z )^3;
        else
            phi =0;
        end
        
        
        
end

phi=nconstant*phi;

end