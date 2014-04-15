% Given a three dimensional vector v, construct the
% skew symmetric 3x3 "cross product" matrix.
function mat=hat(v)
mat=[0 -v(3) v(2)
   v(3) 0  -v(1)
   -v(2) v(1) 0];

% Check out the eigenvalues
% eig(hat(rand(3,1)*10))