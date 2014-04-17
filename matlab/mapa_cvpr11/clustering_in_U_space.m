function indicesKmeans = clustering_in_U_space(U,K,opts)

REX = 1;

[N,D] = size(U);

if isempty(K); K = size(U,2); end;

if opts.normalizeU,
    rowNorms = sqrt(sum(U.^2,2));
    rowNorms(rowNorms==0) = 1;
    U = U./repmat(rowNorms,1,D);
end

if ~REX,
    switch opts.seedType

        case 'hard'

            [seeds, k] = expanded_seed_code(U,K);

        case 'soft'

            u0 = mean(U,1);
            w = sum((U-repmat(u0,N,1)).^2,2);
            w = w/sum(w);
            seeds(1,:) = U(randsample(N,1,true,w),:);

            k = 1;
            U1 = U(sum((U - repmat(seeds(1,:),N,1)).^2,2)>tol,:);
            while k < K && size(U1,1)>0
                w = sum((repmat(U1,1,k)-repmat(reshape(seeds(1:k,:)',[],1)',size(U1,1),1)).^2,2);
                w = w/sum(w);
                %[um,ind_m] = min(max(U1*seeds(1:k,:)',[],2));
                k = k+1;
                seeds(k,:) = U1(randsample(size(U1,1),1,true,w),:);
                U1 = U1(sum((U1 - repmat(seeds(k,:),size(U1,1),1)).^2,2)>tol,:);
            end
    end

    if k<K
        indicesKmeans = ceil(K*rand(N,1));
    else
        indicesKmeans = kmeans(U,K,'start',seeds,'EmptyAction','drop');
    end
end

if REX,
    [~, indicesKmeans] = KMeansRex(U, K, 25, 'mapa');
end

end



%figure; do_plot_data(U, indicesKmeans);
