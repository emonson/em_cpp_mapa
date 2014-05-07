function [goodLocalRegions, goodSeedPoints, estDims] = lmsvd(X, opts)

%
% Multiscale SVD analysis for Linear manifolds (LMSVD)
%
% function [goodLocalRegions, goodSeedPoints, estDims] = lmsvd(X, opts)
%
% INPUT
%   X: N-by-D data matrix
%   opts: structure of the following optional parameters:
%       .seeds: seed points at which multiscale SVD analysis is performed
%       .dmax: upper bound on the plane dimensions 
%       .MinNetPts: minimum scale
%       .nScales: number of scales
%       .nPtsPerScale: number of points per scale
%       .showSpectrum: number of randomly selected seed points for which 
%           the spectrum will be shown
%       .alpha0: cutoff slope for separating noise s.v. from tangential s.v.
%
% OUTPUT
%   goodLocalRegions: a cell array of good local scales
%   goodSeedPoints: the sampled points at which the msvd analysis is performed
%   estDims: estimated intrinsic dimensions of the above local regions
%
% (c)2011 Mauro Maggioni and Guangliang Chen
% Contact: {mauro, glchen}@math.duke.edu. 
%

%% Set parameters
if nargin<1,
    error('Data set needs to be provided!')
end

[N,D] = size(X);

if nargin<2,
    opts = struct();
end;

if ~isfield(opts, 'seeds')
    opts.seeds = 1:N;
end

if ~isfield(opts, 'alpha0')
    if ~isfield(opts, 'dmax') || opts.dmax>=D
        opts.dmax = D-1; 
        opts.alpha0 = 0.2;
    else
        opts.alpha0 = 0.3/sqrt(opts.dmax);
    end
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

if ~isfield(opts, 'plotFigs')
    opts.plotFigs = false;
end

if ~isfield(opts, 'showSpectrum')
    opts.showSpectrum = 0;
end

%%
n_seeds = numel(opts.seeds);
% "rounded" version of maxKNN so get integer indices between MinNetPts and
% maxKNN with nPtsPerScale stride
maxKNN = opts.MinNetPts + opts.nPtsPerScale*(opts.nScales-1);

% Compute the distance between the seed point and the maxKNN nearest points in X. 
% opts.seeds are indices of X to compute distances from
% e.g. if X is [600x3], and if there are 60 seed points, and 102 maxKNN
%      idxs is [60x102] integers containing the indices of the nearest neighbors to the 60 seed points
%      statdists is [60x102] doubles containing the distances from the 60 seed points to the 102 NNs
[~, nn_idxs, statdists] = nrsearch(X, uint32(opts.seeds), maxKNN, [], [], struct('XIsTransposed',true,'ReturnAsArrays',true));

% e.g. statdists(:, 4:2:102) which is [60x50], which is [n_seed_pts x opts.nScales]
% Miles: "Delta is the distance to the farthest neighbor used in that scale. 
%   so instead of thinking of the scale as a number of nearest neighbors in the data, 
%   we can think of it as a distance needed to collect that many neighbors."
Delta = statdists(:, opts.MinNetPts:opts.nPtsPerScale:maxKNN );

%%
estDims = zeros(1, n_seeds);
GoodScales = zeros(n_seeds,2);
goodLocalRegions = cell(1,n_seeds);

for i_seed = 1:n_seeds,
    
    Nets_S = zeros(opts.nScales, D);
    
    for i_scale = 1:opts.nScales,
        % We have a minimum number of points to go out from each seed, and then
        % are increasing the number of points with each scale
        Nets_count = opts.MinNetPts + (i_scale-1)*opts.nPtsPerScale;

        % Grab NNidxs over all seed points up to a certain number of NN for
        % this scale
        % actual point coords for the NNs for this seed point and scale
        net = X( nn_idxs(i_seed, 1:Nets_count), :);
        % center this set of net points and do an SVD to get the singular
        % values
        sigs = svd(net - repmat(mean(net,1), Nets_count, 1));
        % make into a row vector and normalize the singular values by the
        % sqrt of the number of net points
        sigs = sigs'/sqrt(Nets_count);

        Nets_S(i_scale,:) = sigs;
    end
        
%     disp([i_seed i_scale]);
%     disp(Nets_S);
    
    lStats = EstimateDimFromSpectra(Delta(i_seed,:)', Nets_S, opts.alpha0, i_seed);
    % DEBUG
    % disp(Nets_S);
    % writeDMAT_binary(['artdat_rev1_lmsvd_mid_spectra' num2str(i_seed) '.dmat'], Nets_S);
	% fprintf(1,'%.70f\n', opts.alpha0);
    % if (i_seed == 2 || i_seed == 1 || i_seed == 60),
    %     disp(lStats.DimEst);
    %     disp(lStats.GoodScales);
    % end
    estDims(i_seed) = lStats.DimEst;
    GoodScales(i_seed,:) = lStats.GoodScales;
    maxScale = GoodScales(i_seed,2);
    goodLocalRegions{i_seed} = nn_idxs(i_seed, 1:(opts.MinNetPts + (maxScale-1)*opts.nPtsPerScale));
    
end

disp(estDims);

goodSeedPoints = (cellfun(@length, goodLocalRegions)>2*estDims & estDims<D);

goodLocalRegions = goodLocalRegions(goodSeedPoints);
estDims = estDims(goodSeedPoints);
goodSeedPoints = opts.seeds(goodSeedPoints);

%%

lMinScale = GoodScales(:,1)';
lMaxScale = GoodScales(:,2)';

if opts.plotFigs
    figure;scatter3(X(goodSeedPoints,1),X(goodSeedPoints,2),X(goodSeedPoints,3),20,estDims,'filled');colorbar;
    title('Pointwise Dimension Estimates', 'fontSize', 14); grid off; axis tight

    figure;scatter3(X(opts.seeds,1),X(opts.seeds,2),X(opts.seeds,3),30,lMaxScale,'filled');colorbar;
    title('Optimal Local Scales', 'fontSize', 14); grid off; axis tight
end

if opts.showSpectrum > 0
    
    R = zeros(1,N); 
    R([goodLocalRegions{:}]) = 1; 
    R = find(R>0);
    n_seeds = numel(goodSeedPoints);
    
    tempDelta = Delta';
    lMaxScale  = tempDelta((0:size(tempDelta,1):size(tempDelta,1)*(size(tempDelta,2)-1)) + lMaxScale)';
    lMinScale  = tempDelta((0:size(tempDelta,1):size(tempDelta,1)*(size(tempDelta,2)-1)) + lMinScale)';
    lMaxScale = tempDelta((0:size(tempDelta,1):size(tempDelta,1)*(size(tempDelta,2)-1)) + lMaxScale)';

    seeds = randsample(1:n_seeds, opts.showSpectrum);
    for seed_idx = seeds
        figure
        plot3(X(R,1), X(R,2), X(R,3), '.');
        hold on
        plot3(X(goodLocalRegions{seed_idx},1),X(goodLocalRegions{seed_idx},2),X(goodLocalRegions{seed_idx},3),'rx')
        plot3(X(opts.seeds(seed_idx),1), X(opts.seeds(seed_idx),2), X(opts.seeds(seed_idx),3), 'k+', 'MarkerSize', 20)
        title(['Est. Dim. = ' num2str(estDims(seed_idx))], 'fontSize', 14)
        
        figure;
        hold on
        sigs = (squeeze(MSVD_Stats(seed_idx,:,:)))';
        for dim = 1:D
            plot(Delta(seed_idx,:),sigs(:,dim), 'v-')
        end
        plot(repmat(lMinScale(seed_idx),1,2), [0, sigs((Delta(seed_idx,:)==lMinScale(seed_idx)),1)], 'r-')
        plot(repmat(lMaxScale(seed_idx),1,2), [0, sigs((Delta(seed_idx,:)==lMaxScale(seed_idx)),1)], 'g-')
        plot(repmat(lMaxScale(seed_idx),1,2), [0, sigs((Delta(seed_idx,:)==lMaxScale(seed_idx)),1)], 'r-')
        title(['Est. Dim. = ' num2str(estDims(seed_idx))], 'fontSize', 14)
        xlabel('Scale', 'fontSize', 14)
        ylabel('Sing.Vals.', 'fontSize', 14)
        hold off
        axis equal
                
    end
    
end

