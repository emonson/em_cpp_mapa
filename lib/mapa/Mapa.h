#ifndef LMSVD_H
#define LMSVD_H

/* Mapa.h

Based on the Matlab code of
Guangliang Chen & Mauro Maggioni

Eric E Monson – 2014
Duke University


// % Multiscale Analysis of Plane Arrangments (MAPA)
// %
// % This algorithm estimates an arrangement of affine subspaces given 
// % only coordinates of data points sampled from a hybrid linear model.
// %
// % USAGE
// %   [labels, planeDims] = mapa(X,opts)
// %
// % INPUT
// %   X: N-by-D data matrix (rows are points, columns are dimensions)
// %   opts: a structure of the following optional parameters:
// %       .dmax: upper bound on plane dimensions (default = D-1)
// %       .K: number of planes in the model; 
// %            if unknown, then provide an upper bound .Kmax (see below)
// %       .Kmax: upper bound on the number of planes (default = 10)
// %            This field is not needed when .K is provided.
// %       .alpha0: cutoff slope for distinguishing tangential singular values 
// %            from noise ones. Default = 0.3/sqrt(.dmax).
// %       .n0: sampling parameter. Default = 20*.Kmax or 20*.K 
// %            (depending on which is provided). Multiscale SVD analysis
// %            will be performed at n0 randomly selected locations.  
// %       .seeds: sampled points around which MSVD analysis is performed. 
// %            If provided, its length should equal to .n0;
// %            If not provided, it equals randsample(N, .n0) 
// %       .MinNetPts: first scale (in terms of number of points).
// %            Default=.dmax+2
// %       .nScales: number of scales used in MSVD (default = 50)
// %       .nPtsPerScale: number of points per scale.
// %            Default = min(N/5,50*.dmax*log(.dmax)) / .nScales
// %       .isLinear: 1 (all linear subspaces), 0 (otherwise). Default = 0
// %       .discardRows: fraction of bad rows of the matrix A to be discarded 
// %            (default = 0)
// %       .discardCols: fraction of bad columns of A to be discarded 
// %            (default = 0)
// %       .nOutliers: number of outliers (if >=1), or percentage (if <1)
// %            (default=0)
// %       .averaging: 'L1' or 'L2'(default) mean of the local errors, which is
// %            referred to as the tolerance (tau) in the CVPR paper
// %       .plotFigs: whether or not to show the figures such as pointwise
// %            dimension estimates, their optimal scales, the affinity
// %            matrix A, and the model selection curve (for K)
// %            Default = false.
// %       .showSpectrum: An integer representing the number of randomly 
// %            sampled locations for which multiscale singular values as well as 
// %            good scales are shown.  Default = 0.
// %       .postOptimization: whether to apply the K-planes algorithm to further
// %            improve the clustering using the estimated model
// %
// % OUTPUT
// %   labels: a vector of clustering labels corresponding to the best model 
// %        determined by the algorithm (Outliers have label zero).
// %   planeDims: vector of the plane dimensions inferred by the algorithm;
// %        (its length is the number of planes determined by the algorithm)
// %
// % EXAMPLE
// %   % Generate data using the function generate_samples.m, borrowed
// %   % from the GPCA-voting package at the following url:
// %   % http://perception.csl.uiuc.edu/software/GPCA/gpca-voting.tar.gz
// %   [Xt, aprioriSampleLabels, aprioriGroupBases] = generate_samples(...
// %       'ambientSpaceDimension', 3,...
// %       'groupSizes', [200 200 200],...
// %       'basisDimensions', [1 1 2],...
// %       'noiseLevel', 0.04/sqrt(3),...
// %       'noiseStatistic', 'gaussian', ...
// %       'isAffine', 0,...
// %       'outlierPercentage', 0, ...
// %       'minimumSubspaceAngle', pi/6);
// %   X = Xt'; % Xt is D-by-N, X is N-by-D
// %         
// %   % set mapa parameters
// %   opts = struct('n0',20*3, 'dmax',2, 'Kmax',6, 'plotFigs',true);
// %                
// %   % apply mapa
// %   tic; [labels, planeDims] = mapa(X,opts); TimeUsed = toc
// %   MisclassificationRate = clustering_error(labels,aprioriSampleLabels)
// %
// % PUBLICATION
// %   Multiscale Geometric and Spectral Analysis of Plane Arrangements
// %   G. Chen and M. Maggioni, Proc. CVPR 2011, Colorado Springs, CO
// %
// % (c)2011 Mauro Maggioni and Guangliang Chen, Duke University
// %   {mauro, glchen}@math.duke.edu. 
*/

#include <Eigen/Core>
#include <Eigen/SVD>

#include <vector>
#include <iostream>
#include <cmath>

#include <igl/slice.h>
#include <igl/slice_into.h>
#include "Options.h"
#include "NRsearch.h"
#include "LMsvd.h"

using namespace Eigen;


namespace MAPA {

class Mapa {

public:

    Mapa(const ArrayXXd &X, MAPA::Opts opts)
    {

// %% linear multiscale svd analysis
// [optLocRegions, seeds, localDims] = lmsvd(X, opts);
// % Returned seed points could be fewer, because bad ones are thrown away in lmsvd.
// % localdims is the same length as seeds
// n0 = numel(seeds); 
// 
// % allPtsInOptRegions are point indices
// allPtsInOptRegions = unique([optLocRegions{:}]);
// 
// n = numel(allPtsInOptRegions);
// 
// % Maps indices of original data points to indices of allPtsInOptRegions,
// % which will also be indices of the rows of A in a bit.
// invRowMap = zeros(1,N);
// invRowMap(allPtsInOptRegions) = 1:n;
// 
// %% spectral analysis
// heights = zeros(n, n0); % distances from the n points in allPtsInOptRegions to the n0 local planes
// eps = zeros(1,n0); % estimated local errors
// 
// for i = 1:n0
//     
//     R_i = optLocRegions{i}; % indices of points in the current local region
//     n_i = numel(R_i);
//     
//     if opts.isLinear
//         X_c = X;
//     else
//         ctr = mean(X(R_i,:), 1);
//         X_c = X - repmat(ctr, N, 1);
//     end
//     
//     [~,s,v] = svd(X_c(R_i,:),0);
//     s = diag(s); % local singular values
//     
//     eps(i) = sum(s(localDims(i)+1:end).^2) / (n_i-1); % local approximation error
//     
//     heights(:,i) = (sum(X_c(allPtsInOptRegions,:).^2,2)-sum((X_c(allPtsInOptRegions,:)*v(:,1:localDims(i))).^2,2)) / (2*eps(i));
//         
// end
// 
// A = exp(-abs(heights)); 
// A(isnan(A)) = 1;
// 
// 
// %% discarding bad rows and columns
// if opts.discardCols > 0
//     
//     colStdA = std(A,0,1);
//     goodCols = find(colStdA > quantile(colStdA, opts.discardCols));
//     
//     eps = eps(goodCols);
//     optLocRegions = optLocRegions(goodCols);
//     seeds = seeds(goodCols);
//     localDims = localDims(goodCols);
//     
//     % When we throw away columns, we are throwing away seed points, and
//     % when we throw away seed points, we throw away data points that were
//     % in the optimal local regions of those seed points, so that gets rid
//     % of rows of A as well.
//     allPtsInOptRegions = unique([optLocRegions{:}]);
//     
//     A = A(invRowMap(allPtsInOptRegions), goodCols);
//     
//     [n,n0] = size(A);
//     
// end
// 
// % Maps the indices of the original data set to indices of the seed points,
// % which are also the indices of the columns of A
// invColMap = zeros(1,N);
// invColMap(seeds) = 1:n0;
// 
// switch opts.averaging
//     case {'l2','L2'}
//         eps = sqrt(mean(eps./(D-localDims)));
//     case {'l1','L1'}
//         eps = mean(sqrt(eps./(D-localDims)));
// end
// 
// if opts.discardRows>0
//     
//     rowStdA = std(A,0,2);
//     goodRows = (rowStdA>quantile(rowStdA, opts.discardRows));
//     
//     A = A(goodRows,:);
//     
//     allPtsInOptRegions = allPtsInOptRegions(goodRows);
//     n = numel(allPtsInOptRegions);
//     
// end
// 
// invRowMap = zeros(1,N);
// invRowMap(allPtsInOptRegions) = 1:n;
// 
// %% normalize the spectral matrix A
// degrees = A*sum(A,1).';
// degrees((degrees == 0)) = 1;
// A = repmat(1./sqrt(degrees),1,n0).*A;
// 
// 
// %% Directly cluster data (when K is provided)
// if isfield(opts, 'K'),
//     
//     K = opts.K;
//     
//     [U,S] = svds(A, K+1);
//     if opts.plotFigs
//         figure; do_plot_data(diag(S)); title('Top Singular Values of L', 'fontSize',  14); box on
//     end
//     
//     [planeDims, labels, err] =  spectral_analysis(X, U(:,1:K), allPtsInOptRegions, invColMap, localDims, opts.nOutliers);
//     
// %% also select a model when only upper bound is given
// elseif isfield(opts, 'Kmax'),
//     
//     [U,S] = svds(A, opts.Kmax+1);
//     if opts.plotFigs
//         figure; do_plot_data(diag(S)); title('Top Singular Values of L', 'fontSize', 14); box on
//     end
//     
//     planeDims = mode(localDims);
//     labels = ones(1,N);
//     L2Errors = L2error(X, planeDims, labels);
//     
//     K = 1;
//     while K<opts.Kmax && L2Errors > 1.05*eps
//         K = K+1;
//         [planeDims, labels, L2Errors] = ...
//             spectral_analysis(X, U(:,1:K), allPtsInOptRegions, invColMap, localDims, opts.nOutliers);
//     end
//     
// end
// 
// %% use K-planes to optimize clustering
// if opts.postOptimization    
//     [labels, L2Error] = K_flats(X, planeDims, labels);
// end


    };

    ArrayXi GetLabels()
    {
        return labels;
    };
    
    ArrayXi GetPlaneDims()
    {
        return planeDims;
    };


private:

    ArrayXi labels;
    ArrayXi planeDims;
    

}; // class def

} // namespace MAPA

#endif
