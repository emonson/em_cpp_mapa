% Tests the random special orthogonal matrix function.

n = 10;

X = rand_special_orthogonal(n);

% Test that all rows and columns have unity magnitude.
for col = 1:10,
    if abs(norm(X(:,col))-1) > .000001,
        error('Found column of non-unity norm')
    end
    if abs(norm(X(col,:))-1) > .000001,
        error('Found row of non-unity norm')
    end
end

% Check that the determinant of the matrix is unity
if abs(det(X)-1) > .0001,
    error('Matrix does not have unity determinant')
end

% Check distribution by taking a random vector and seeing where a bunch of
% X's take it.
n = 2;
x = randn(n,1);
numpoints = 500;
XX = zeros(n,numpoints);
for k = 1:numpoints,
    X=rand_special_orthogonal(n);
    xx = X*x;
    XX(:,k) = xx;
end
if n==2,
    plot(XX(1,:),XX(2,:),'ro')
end
if n==3,
plot3(XX(1,:),XX(2,:),XX(3,:),'ro')
end