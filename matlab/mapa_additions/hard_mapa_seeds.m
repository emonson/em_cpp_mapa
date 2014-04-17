function [seeds, k] = hard_mapa_seeds(U,K)
%find initial centers

    [N,D] = size(U);

    tol = 1e-8;
    seeds = zeros(K,D);

    u0 = mean(U,1);
    [~,ind_m] = max(sum((U-repmat(u0,N,1)).^2,2));
    seeds(1,:) = U(ind_m(1),:);

    k = 1;
    U1 = U(sum((U - repmat(seeds(1,:),N,1)).^2,2)>tol,:);
    while k < K && size(U1,1)>0
        [~,ind_m] = max(sum((repmat(U1,1,k)-repmat(reshape(seeds(1:k,:)',[],1)',size(U1,1),1)).^2,2));
        %[um,ind_m] = min(max(U1*seeds(1:k,:)',[],2));
        k = k+1;
        seeds(k,:) = U1(ind_m(1),:);
        U1 = U1(sum((U1 - repmat(seeds(k,:),size(U1,1),1)).^2,2)>tol,:);
    end
end