function theta = subspace_gpca(A,B)
%  SUBSPACE_GPCA
%     Angle between subspaces.
%     SUBSPACE_GPCA(A,B) finds the angle between two
%     subspaces specified by the columns of A and B.
%
%  Same as MATLAB's built in SUBSPACE command, but assumes that A and B are
%  already orthonormal, saving a couple extra SVD's and speeding up the
%  code.

if size(A,2) < size(B,2),
    % Swap A and B.
    tmp = A;
    A = B;
    B = tmp;
end 

% Compute orthonormal bases, using SVD in "orth" to avoid problems  
% when A and/or B is nearly rank deficient.  
%A = orth(A);   
%B = orth(B);                     

% Compute the projection the most accurate way, according to [1].
for k=1:size(A,2)  
    B = B - A(:,k)*(A(:,k)'*B);  
end                     
% Make sure it's magnitude is less than 1. 
theta = asin(min(1,(norm(B)))); 