% rand_special_orthogonal(order)
%
% Compute special orthogonal matrices that are in some sense uniform.
% If you take any point and transform it by the matrix from this function,
% it will lie on the hypersphere with the same magnitude, and it will be
% equally likely to be anywhere on the surface of this hypersphere.
function X = rand_special_orthogonal(n);

X = zeros(n,n);
x = randn(n,1);
x = x./norm(x);
X(:,1) = x;  % Generate a random first column.
for m=2:n, % For every other column of the matrix,
    x = randn(n,1); % Generate a unit vector in a random direction.
    x = x./norm(x);
    for r = 1:m-1,
        old = X(:,r);
        x = x - (x'*old)*old/norm(old);  % Subtract off the projection onto the other basis vectors.
        x = x./norm(x);
    end
    X(:,m) = x;
end
% If the determinant of x is negative, flip the sign of a uniformly
% randomly chosen basis vector.
if det(X)<0,
    l = ceil(rand*n);
    X(:,l) = -1*X(:,l);
end


        
    