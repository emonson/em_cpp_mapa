function mse = L2error(data, dim, idx, ctr, dir)

D = size(data,2);

K = max(idx);

if length(dim) == 1
    dim = dim*ones(K,1);
end

if nargin<4
    [ctr,dir] = computing_bases(data,idx,dim);
end

mse = zeros(1,K);
for k = 1:K
    cls_k = data((idx==k),:);
    n_k = size(cls_k,1);
    if n_k > dim(k)
        cls_k = cls_k - repmat(ctr{k,1},n_k,1);
        disp(k);
        disp(sum(cls_k.^2,2)');
        disp(sum((cls_k*dir{k,1}').^2,2)');
        mse(k) = sum(sum(cls_k.^2,2) - sum((cls_k*dir{k,1}').^2,2))/(D-dim(k));
    end
end

mse = sqrt(sum(mse)/sum(idx>0));