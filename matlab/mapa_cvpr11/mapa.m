function [labels, planeDims, MoreOutput] = mapa(X,opts)

%
% Multiscale Analysis of Plane Arrangments (MAPA)
%
% This algorithm estimates an arrangement of affine subspaces given 
% only coordinates of data points sampled from a hybrid linear model.
%
% USAGE
%   [labels, planeDims, MoreOutput] = mapa(X,opts)
%
% INPUT
%   X: N-by-D data matrix (rows are points, columns are dimensions)
%   opts: a structure of the following optional parameters:
%       .dmax: upper bound on plane dimensions (default = D-1)
%       .K: number of planes in the model; 
%            if unknown, then provide an upper bound .Kmax (see below)
%       .Kmax: upper bound on the number of planes (default = 10)
%            This field is not needed when .K is provided.
%       .alpha0: cutoff slope for distinguishing tangential singular values 
%            from noise ones. Default = 0.3/sqrt(.dmax).
%       .n0: sampling parameter. Default = 20*.Kmax or 20*.K 
%            (depending on which is provided). Multiscale SVD analysis
%            will be performed at n0 randomly selected locations.  
%       .seeds: sampled points around which MSVD analysis is performed. 
%            If provided, its length should equal to .n0;
%            If not provided, it equals randsample(N, .n0) 
%       .MinNetPts: first scale (in terms of number of points).
%            Default=.dmax+2
%       .nScales: number of scales used in MSVD (default = 50)
%       .nPtsPerScale: number of points per scale.
%            Default = min(N/5,50*.dmax*log(.dmax)) / .nScales
%       .isLinear: 1 (all linear subspaces), 0 (otherwise). Default = 0
%       .discardRows: percentage of bad rows of the matrix A to be discarded 
%            (default = 0)
%       .discardCols: percentage of bad columns of A to be discarded 
%            (default = 0)
%       .nOutliers: number of outliers (if >=1), or percentage (if <1)
%            (default=0)
%       .averaging: 'L1' or 'L2'(default) mean of the local errors, which is
%            referred to as the tolerance (tau) in the CVPR paper
%       .plotFigs: whether or not to show the figures such as pointwise
%            dimension estimates, their optimal scales, the affinity
%            matrix A, and the model selection curve (for K)
%            Default = false.
%       .showSpectrum: An integer representing the number of randomly 
%            sampled locations for which multiscale singular values as well as 
%            good scales are shown.  Default = 0.
%       .postOptimization: whether to apply the K-planes algorithm to further
%            improve the clustering using the estimated model
%
% OUTPUT
%   labels: a vector of clustering labels corresponding to the best model 
%        determined by the algorithm (Outliers have label zero).
%   planeDims: vector of the plane dimensions inferred by the algorithm;
%        (its length is the number of planes determined by the algorithm)
%   MoreOutput: structure of all other relevant quantities:
%       .opts: the optional parameter values used by the algorithm
%       .seeds: sampled points at which MSVD analysis is performed
%       .localDims: estimated local dimensions at the seed points
%       .optLocRegions: cell array of optimal local regions at the seed points
%       .matA: the affinity matrix A
%       .matU: the matrix U consisting in columns of the top few left 
%            singular values of a normalized version of A
%       .eps: estimated model error (per dimension), i.e., the tolerance
%       .planeDims: cell array of plane dimensions corresponding to all
%            different K examined by the algorithm 
%            (in fact only those not exceeding the truth are checked)
%       .labels: cell array of labels corresponding to the different K
%       .L2Errors: vector of approximation errors for all tested K
%
% EXAMPLE
%   % Generate data using the function generate_samples.m, borrowed
%   % from the GPCA-voting package at the following url:
%   % http://perception.csl.uiuc.edu/software/GPCA/gpca-voting.tar.gz
%   [Xt, aprioriSampleLabels, aprioriGroupBases] = generate_samples(...
%       'ambientSpaceDimension', 3,...
%       'groupSizes', [200 200 200],...
%       'basisDimensions', [1 1 2],...
%       'noiseLevel', 0.04/sqrt(3),...
%       'noiseStatistic', 'gaussian', ...
%       'isAffine', 0,...
%       'outlierPercentage', 0, ...
%       'minimumSubspaceAngle', pi/6);
%   X = Xt'; % Xt is D-by-N, X is N-by-D
%         
%   % set mapa parameters
%   opts = struct('n0',20*3, 'dmax',2, 'Kmax',6, 'plotFigs',true);
%                
%   % apply mapa
%   tic; [labels, planeDims, MoreOutput] = mapa(X,opts); TimeUsed = toc
%   MisclassificationRate = clustering_error(labels,aprioriSampleLabels)
%
% PUBLICATION
%   Multiscale Geometric and Spectral Analysis of Plane Arrangements
%   G. Chen and M. Maggioni, Proc. CVPR 2011, Colorado Springs, CO
%
% (c)2011 Mauro Maggioni and Guangliang Chen, Duke University
%   {mauro, glchen}@math.duke.edu. 

%% set parameter values and check compatibility
if nargin<1, 
    error('Data set needs to be provided!');
end

[N,D] = size(X);

if nargin<2; 
    opts = struct(); 
end;

if ~isfield(opts, 'alpha0')
    if ~isfield(opts, 'dmax') || opts.dmax>=D
        opts.dmax = D-1; 
        opts.alpha0 = 0.2;
    else
        opts.alpha0 = 0.3/sqrt(opts.dmax);
    end
end

if ~isfield(opts, 'K') && ~isfield(opts, 'Kmax'),
    opts.Kmax = 10;
end

if ~isfield(opts, 'seeds')
    if ~isfield(opts, 'n0')
        if ~isfield(opts, 'K')
            opts.n0 = 20*opts.Kmax;
        else
            opts.n0 = 20*opts.K;
        end
    end
    if opts.n0<N
        opts.seeds = sort(randsample(N,opts.n0));
    else
        opts.seeds = 1:N;
        if opts.n0 > N
            opts.n0 = N;
            warning('The sampling parameter n0 has been modified to N!'); %#ok<WNTAG>
        end
    end  
else % seeds provided
    if isfield(opts, 'n0') && opts.n0 ~= length(opts.seeds)
        warning('The parameter values of n0 and seeds are incompatible. n0 has been changed to the length of seeds.') %#ok<WNTAG>
    end
    opts.n0 = length(opts.seeds);
end

maxKNN = round(min(N/5, 50*opts.dmax*log(max(3,opts.dmax))));

if ~isfield(opts, 'MinNetPts')
    opts.MinNetPts = opts.dmax+2;
end

if ~isfield(opts, 'nScales')
    opts.nScales = min(50, maxKNN);
end

if ~isfield(opts, 'nPtsPerScale')
    opts.nPtsPerScale = round( maxKNN / opts.nScales );
end

if ~isfield(opts, 'isLinear')
    opts.isLinear = false;
end

if ~isfield(opts, 'discardRows')
    opts.discardRows = 0;
end

if ~isfield(opts, 'discardCols')
    opts.discardCols =  0;
end

if ~isfield(opts,'averaging')
    opts.averaging = 'L2';
end

if ~isfield(opts, 'nOutliers')
    opts.nOutliers = 0;
elseif opts.nOutliers<1
    opts.nOutliers = round(N*opts.nOutliers);
end
    
if ~isfield(opts, 'plotFigs')
    opts.plotFigs = false;
end

if ~isfield(opts, 'showSpectrum')
    opts.showSpectrum = 0;
end

if ~isfield(opts, 'postOptimization')
    opts.postOptimization = false;
end

%%
if nargout>2
    MoreOutput = struct();
    MoreOutput.opts = opts;
end

%% linear multiscale svd analysis
[optLocRegions, seeds, localDims] = lmsvd(X, opts);
n0 = numel(seeds); % Returned seed points could be fewer, because bad ones are thrown away in lmsvd.

allPtsInOptRegions = 1:N;
flags = zeros(1,N); flags([optLocRegions{:}]) = 1;
allPtsInOptRegions = allPtsInOptRegions(flags>0);

n = numel(allPtsInOptRegions);

invRowMap = zeros(1,N);
invRowMap(allPtsInOptRegions) = 1:n;

%% spectral analysis
heights = zeros(n, n0); % distances from the n points in allPtsInOptRegions to the n0 local planes
eps = zeros(1,n0); % estimated local errors

if nargout>2,
    MoreOutput.localCtrs = cell(1,n0);
    MoreOutput.localBases = cell(1, n0);
    MoreOutput.localSigs = cell(1,n0);
end

for i = 1:n0
    
    R_i = optLocRegions{i}; % indices of points in the current local region
    n_i = numel(R_i);
    
    if opts.isLinear
        X_c = X(allPtsInOptRegions,:);
    else
        ctr = mean(X(R_i,:), 1);
        X_c = X(allPtsInOptRegions,:)- repmat(ctr, n, 1);
    end
    
    [~,s,v] = svd(X_c(invRowMap(R_i),:),0);
    s = diag(s); % local singular values
    
    eps(i) = sum(s(localDims(i)+1:end).^2) / (n_i-1); % local approximation error
    
    heights(:,i) = (sum(X_c.^2,2)-sum((X_c*v(:,1:localDims(i))).^2,2)) / (2*eps(i));
    
    if nargout>2,
        if ~opts.isLinear,
            MoreOutput.localCtrs{i} = ctr;
        end
        MoreOutput.localBases{i} = v(:,1:localDims(i));
        MoreOutput.localSigs{i} = s;
    end
    
end

A = exp(-abs(heights)); 
A(isnan(A)) = 1;

if opts.plotFigs
    figure; imagesc(A); title('Elements of the Matrix A', 'fontSize',  14); colorbar
end

%figure; imagesc(A*A');

%% discarding bad rows and columns
if opts.discardCols > 0
    
    colStdA = std(A,0,1);
    goodCols = find(colStdA > quantile(colStdA, opts.discardCols));
    
    eps = eps(goodCols);
    optLocRegions = optLocRegions(goodCols);
    seeds = seeds(goodCols);
    localDims = localDims(goodCols);
    
    if nargout>2,
        MoreOutput.localCtrs = MoreOutput.localCtrs(goodCols);
        MoreOutput.localBases = MoreOutput.localBases(goodCols);
        MoreOutput.localSigs = MoreOutput.localSigs(goodCols);
    end
    
    allPtsInOptRegions = 1:N;
    flags = zeros(1,N); flags([optLocRegions{:}]) = 1;
    allPtsInOptRegions = allPtsInOptRegions(flags>0);
    
    A = A(invRowMap(allPtsInOptRegions), goodCols);
    if opts.plotFigs
        figure; imagesc(A); title('Elements of the Matrix A (column reduced)', 'fontSize',  14)
    end
    
    [n,n0] = size(A);
    
end

invColMap = zeros(1,N);
invColMap(seeds) = 1:n0;

switch opts.averaging
    case {'l2','L2'}
        eps = sqrt(mean(eps./(D-localDims)));
    case {'l1','L1'}
        eps = mean(sqrt(eps./(D-localDims)));
end

if opts.discardRows>0
    
    rowStdA = std(A,0,2);
    goodRows = (rowStdA>quantile(rowStdA, opts.discardRows));
    
    A = A(goodRows,:);
    if opts.plotFigs
        figure; imagesc(A); title('Elements of the Matrix A (row reduced)', 'fontSize',  14)
    end
    
    allPtsInOptRegions = allPtsInOptRegions(goodRows);
    n = numel(allPtsInOptRegions);
    
end

invRowMap = zeros(1,N);
invRowMap(allPtsInOptRegions) = 1:n;

%% Record Intermediate results
if nargout>2
    MoreOutput.seeds = seeds;
    MoreOutput.localDims = localDims;
    MoreOutput.optLocRegions = optLocRegions;
    MoreOutput.matA = A;
    MoreOutput.eps = eps;
    MoreOutput.invRowMap = invRowMap;
    MoreOutput.invColMap = invColMap;
end

%% normalize the spectral matrix A
degrees = A*sum(A,1).';
degrees((degrees == 0)) = 1;
A = repmat(1./sqrt(degrees),1,n0).*A;

% if opts.plotFigs
%     figure; imagesc(A); title('Elements of the Matrix A (normalized)', 'fontSize',  14)
% end

%% Directly cluster data (when K is provided)
if isfield(opts, 'K'),
    
    K = opts.K;
    
    [U,S] = svds(A, K+1);
    if opts.plotFigs
        figure; do_plot_data(diag(S)); title('Top Singular Values of L', 'fontSize',  14); box on
    end
    
    [planeDims, labels, err] =  spectral_analysis(X, U(:,1:K), allPtsInOptRegions, invColMap, localDims, opts.nOutliers);
    
    if nargout>2
        MoreOutput.matU = U(:,1:K);
        MoreOutput.L2Error = err;
    end
    
%% also select a model when only upper bound is given
elseif isfield(opts, 'Kmax'),
    
    [U,S] = svds(A, opts.Kmax+1);
    if opts.plotFigs
        figure; do_plot_data(diag(S)); title('Top Singular Values of L', 'fontSize', 14); box on
    end
    
    MoreOutput.planeDims = cell(1, opts.Kmax);
    MoreOutput.labels = cell(1, opts.Kmax);
    MoreOutput.L2Errors = zeros(1, opts.Kmax);
    
    MoreOutput.planeDims{1} = mode(localDims);
    MoreOutput.labels{1} = ones(1,N);
    MoreOutput.L2Errors(1) = L2error(X, MoreOutput.planeDims{1}, MoreOutput.labels{1});
    
    K = 1;
    while K<opts.Kmax && MoreOutput.L2Errors(K) > 1.05*eps
        K = K+1;
        [MoreOutput.planeDims{K}, MoreOutput.labels{K}, MoreOutput.L2Errors(K)] = ...
            spectral_analysis(X, U(:,1:K), allPtsInOptRegions, invColMap, localDims, opts.nOutliers);
        %figure; do_plot_data(X, MoreOutput.labels{k}); title(['K = ' num2str(k) ', d_k = ' num2str(MoreOutput.planeDims{k})], 'fontSize', 14);
    end
    
    planeDims = MoreOutput.planeDims{K};
    labels = MoreOutput.labels{K};
    
    %% use the effective dimension idea as in the GPCA paper
    %     effectiveDim = zeros(1,opts.Kmax);
    %     for k = 1:opts.Kmax,
    %         effectiveDim(k) = ( sum(MoreOutput.planeDims{k}.*(D-MoreOutput.planeDims{k})) + sum(MoreOutput.planeDims{k}(MoreOutput.labels{k})) ) / N;
    %     end
    %
    %     objectiveFun = (MoreOutput.L2Errors).^2 + 2*eps^2*effectiveDim;
    %     figure; plot(objectiveFun,'-v')
    %     [minObjFun, K] = min(objectiveFun);
    %     hold on
    %     plot(K, objectiveFun(K), 'ro')
    
    %%
    if opts.plotFigs
        figure; plot(MoreOutput.L2Errors*sqrt(D), '-*')
        hold on
        plot([1 opts.Kmax],  [eps eps]*sqrt(D), 'k-.')
        plot(K, MoreOutput.L2Errors(K)*sqrt(D), 'ro', 'MarkerSize' , 12)
        title(['Final Model: K = ' num2str(K) ', d_k = ' num2str(planeDims)], 'fontSize', 14);
        xlabel('K', 'fontSize', 14)
        ylabel('e(K)', 'fontSize', 14)
        set(gca, 'xTick', 1:opts.Kmax)
    end
    
end

%% use K-planes to optimize clustering
if opts.postOptimization    
    [labels, L2Error] = K_flats(X, planeDims, labels);
    MoreOutput.labels{K} = labels;
    MoreOutput.L2Errors(K) = L2Error;
end

%% show final results
if opts.plotFigs
    figure; do_plot_data(X, labels); title(['Final Model: K = ' num2str(K) ', d_k = ' num2str(planeDims)], 'fontSize', 14);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [planeDims, labels, err] =  spectral_analysis(X, U, allPtsInOptRegions, invColMap, localDims, nOutliers)

K = size(U,2);

SCCopts = struct();
SCCopts.normalizeU = 1;
SCCopts.seedType = 'hard';
indicesKmeans = clustering_in_U_space(U,K,SCCopts);

planeDims = zeros(1,K);
for k = 1:K
    class_k = allPtsInOptRegions(indicesKmeans == k);
    temp = invColMap(class_k);  
    temp = temp(temp>0);
    planeDims(k) = mode(localDims(temp));
end

[planeCenters, planeBases] = computing_bases(X(allPtsInOptRegions,:), indicesKmeans, planeDims);
dists = p2pdist(X,planeCenters,planeBases);

%[N,D] = size(X);
%dists = dists./repmat(D-planeDims, N, 1);

[dists,labels] = min(dists,[],2);

if nOutliers>0
    % new labels
    labels1 = labels;
    objectiveFun1 = sum(sqrt(dists));
    outliers1 = [];
    % old labels
    objectiveFun = Inf;
    labels = [];
    outliers = [];
    while objectiveFun1<objectiveFun
        labels = labels1;
        objectiveFun = objectiveFun1;
        outliers = outliers1;       
        [~, I] = sort(dists, 'descend');
        outliers1=I(1:nOutliers);
        labels1(outliers1)=0;
        [planeCenters, planeBases] = computing_bases(X, labels1, planeDims);
        dists = p2pdist(X,planeCenters,planeBases);
        [dists,labels1] = min(dists,[],2);
        objectiveFun1 = sum(sqrt(dists));
    end
    labels(outliers)=0;
end

err = L2error(X, planeDims, labels);
