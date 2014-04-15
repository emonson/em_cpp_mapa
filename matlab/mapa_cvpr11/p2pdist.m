function dists = p2pdist(X,centers,bases)

% This code computes (squared) points to planes distances.

N = size(X,1);
K = length(centers);

dists = Inf(N, K);
for k = 1:K

    if ~isempty(centers{k}) && ~isempty(bases{k})
        Y = X - repmat(centers{k},N,1);
        dists(:,k) = sum(Y.^2,2) - sum((Y*bases{k}').^2,2);
    end
    
end