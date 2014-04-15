function [centers,bases] = computing_bases(data,labels,dims)

%D = size(data,2);

K = max(labels);

if length(dims) == 1 && K>1
    dims = dims*ones(K,1);
end

% intialization
centers = cell(K,1);
bases = cell(K,1);

for k = 1:K
    
    cls_k = data((labels==k),:);
    n_k = size(cls_k,1);
    
    if n_k >= dims(k)+1
        
        centers{k} = mean(cls_k,1);
        cls_k = cls_k - repmat(centers{k},n_k,1);
        [~,~,vk] = svds(cls_k,dims(k));
        
        bases{k} = vk';
        
    end
    
end