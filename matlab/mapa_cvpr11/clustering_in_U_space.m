function indicesKmeans = clustering_in_U_space(U,K,opts)

[N,D] = size(U);

if isempty(K); K = size(U,2); end;

if opts.normalizeU,
    rowNorms = sqrt(sum(U.^2,2));
    rowNorms(rowNorms==0) = 1;
    U = U./repmat(rowNorms,1,D);
end

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


function [seeds, k] = expanded_seed_code(U,K)
%find initial centers

    [N,D] = size(U);

    % some minimal distance don't want to consider points within when
    % picking a new center
    tol = 1e-8;
    
    seeds = zeros(K,D);

    % Mean position within U points
    u0 = mean(U,1);
    sq_dist_from_mean = (U-repmat(u0,N,1)).^2;
    sum_sq_dist_from_mean = sum(sq_dist_from_mean,2);
    % find index of point furthest from mean
    [~,ind_m] = max(sum_sq_dist_from_mean);
    seeds(1,:) = U(ind_m(1),:);

    k = 1;
    sq_dist_from_first_seed = (U - repmat(seeds(1,:),N,1)).^2;
    is_far_enough_away = sum(sq_dist_from_first_seed, 2) > tol;
    U1 = U(is_far_enough_away,:);
    % while we have fewer points than we wanted, and are not out of
    % potential points
    while k < K && size(U1,1)>0
        seeds_row_arranged = reshape(seeds(1:k,:)',[],1)';
        seeds_row_arranged_duped = repmat(seeds_row_arranged, size(U1,1), 1);
        copies_of_U_row_arranged = repmat(U1,1,k);
        % find the index of the point the furthest away from the 
        sum_sq_dists_from_all_seeds = sum((copies_of_U_row_arranged - seeds_row_arranged_duped).^2, 2);
        [~,ind_m] = max( sum_sq_dists_from_all_seeds );
        
        k = k+1;
        seeds(k,:) = U1(ind_m(1),:);
        sq_dist_from_curr_seed = (U1 - repmat(seeds(k,:),size(U1,1),1)).^2;
        is_far_enough_away = sum(sq_dist_from_curr_seed, 2) > tol;
        % only keep points far enough away from seed for next round
        U1 = U1(is_far_enough_away,:);
    end
end

function [seeds, k] = orig_seed_code(U,K) %#ok<DEFNU>
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

%figure; do_plot_data(U, indicesKmeans);
