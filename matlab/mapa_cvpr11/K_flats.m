function [idx,m2] = K_flats(data, dim, idx, max_loops)

% function [idx,m2] = K_flats(data, dim, idx)
%   applies the iterative K-flats algorithm to cluster data into
%   K (=length(dim)) flats of dimensions (specified by dim),
%   possibly based on an initial guess of the labels (idx).
%
% Input:
%   data: N-by-D matrix
%   dim: dimensions of the planes;
%       can be a single number if all dimensions are the same AND idx is given 
%   idx: initial labels of the points;
%       if not given, will randomly assign points to subspaces        
%
% Output:
%   idx: labels of the data points associated with the clusters
%   m2: averaged L2 error of the final model

if nargin<2
    error('Dimensions need to be specified!')
else
    K = length(dim);
end

N = size(data,1);

if nargin<3
    idx = ceil(K*rand(N,1));
elseif K == 1
    K = max(idx); 
    dim = dim*ones(1,K);
end

if nargin<4
   max_loops = 1000;
end

[ctr,dir] = computing_centers_and_bases(data,idx,dim);

m1 = 0;
m2 = L2error(data, dim, idx, ctr, dir);

tol = 1e-6;
cnt = 0;
while cnt<max_loops && abs(m1-m2)>tol

    cnt = cnt+1;

    inds_in = (idx>0);
    dist = p2pdist(data(inds_in,:),ctr,dir);
    [~, idx(inds_in)] = min(dist,[],2);
    
    [ctr,dir] = computing_centers_and_bases(data,idx,dim);
    
    m1 = m2;
    m2 = L2error(data, dim, idx, ctr, dir);

end
