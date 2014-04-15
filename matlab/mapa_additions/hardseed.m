function seeds = hardseed(U,K,opts)
if nargin == 0
    % test code with randn data
    U = randn(200,4);
    K = 7;
    opts.normalizeU = 1;
end

[N,D] = size(U);

if isempty(K); K = size(U,2); end;

if opts.normalizeU,
    rowNorms = sqrt(sum(U.^2,2));
    rowNorms(rowNorms==0) = 1;
    U = U./repmat(rowNorms,1,D);
end

%find initial centers
tol = 1e-8;
seeds = zeros(K,D);

% pick first seed farthest from mean
u0 = mean(U,1);
[~,ind_m] = max(sqdist(U',u0'));
seeds(1,:) = U(ind_m(1),:);

k = 1;

% keep points farther than tol from 1st seed
idxs_to_keep = find(sqdist(U',seeds(1,:)') > tol);
U1 = U(idxs_to_keep,:);

% loop until all seeds have been chosen, or we run out of points in U

while k < K && size(U1,1)>0
    
    dists = sqdist(U1',seeds(1:k,:)'); % calc dists to seeds
    dists = sum(dists,2); % find sum of dists to seeds
    
    k = k + 1;
    
    [~,ind_m] = max(dists);
    
    seeds(k,:) = U1(ind_m,:);
    
    % keep points farther than tol from kth seed
    idxs_to_keep = find(sqdist(U1',seeds(k,:)') > tol);
    U1 = U1(idxs_to_keep,:);
    
end


%% test against old code, see if there are differences;

oldseeds = old_seed_code(U,K,opts);

% test if error
sum(sum(abs(seeds - oldseeds)))


function seeds = old_seed_code(U,K,opts)
[N,D] = size(U);

if isempty(K); K = size(U,2); end;

if opts.normalizeU,
    rowNorms = sqrt(sum(U.^2,2));
    rowNorms(rowNorms==0) = 1;
    U = U./repmat(rowNorms,1,D);
end

%find initial centers
tol = 1e-8;
seeds = zeros(K,D);

u0 = mean(U,1);
[um,ind_m] = max(sum((U-repmat(u0,N,1)).^2,2));
seeds(1,:) = U(ind_m(1),:);

k = 1;
U1 = U(sum((U - repmat(seeds(1,:),N,1)).^2,2)>tol,:);
while k < K && size(U1,1)>0
    
    [um,ind_m] = max(sum((repmat(U1,1,k)-repmat(reshape(seeds(1:k,:)',[],1)',size(U1,1),1)).^2,2));
    %[um,ind_m] = min(max(U1*seeds(1:k,:)',[],2));
    k = k+1;
    seeds(k,:) = U1(ind_m(1),:);
    U1 = U1(sum((U1 - repmat(seeds(k,:),size(U1,1),1)).^2,2)>tol,:);
end


