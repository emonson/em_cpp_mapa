function [seeds, k] = hard_mapa_seeds_expanded(U,K)
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