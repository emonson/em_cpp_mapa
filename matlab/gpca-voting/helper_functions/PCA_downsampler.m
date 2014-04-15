function reducedX=PCA_downsampler(X,dimension, threshold)

[U,S,V]=svds(X,dimension);
if nargin==3
    % Thresholding
    singluarDimension=sum(diag(S)>(S(1,1)*threshold));
    dimension = min(dimension,singluarDimension);
end

reducedX=V(:,1:dimension)';
reducedX = S(1:dimension,1:dimension)*reducedX;