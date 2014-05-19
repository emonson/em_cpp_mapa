#ifndef MAPA_H
#define MAPA_H

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
// %            (default = 0, value [0 1]) throwing out low standard deviation
// %       .discardCols: fraction of bad columns of A to be discarded 
// %            (default = 0, value [0 1]) throwing out low standard deviation
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
#include "UtilityCalcs.h"
#include "ComputingBases.h"
#include "SpectralAnalysis.h"

using namespace Eigen;


namespace MAPA {

class Mapa {

public:

    Mapa(const ArrayXXd &X, MAPA::Opts opts)
    {

        // %% linear multiscale svd analysis
        // [optLocRegions, seeds, localDims] = lmsvd(X, opts);
        
        std::cout << "LMSVD IN" << std::endl;
        MAPA::LMsvd lmsvd(X, opts);
        std::vector<ArrayXi> optLocRegions = lmsvd.GetGoodLocalRegions();
        ArrayXi seeds = lmsvd.GetGoodSeedPoints();
        ArrayXi localDims = lmsvd.GetGoodEstimatedDims();

        // % Returned seed points could be fewer, because bad ones are thrown away in lmsvd.
        // % localdims is the same length as seeds
        // n0 = numel(seeds); 
        
        int n0 = seeds.size();

        // % allPtsInOptRegions are point indices
        // allPtsInOptRegions = unique([optLocRegions{:}]);
        // n = numel(allPtsInOptRegions);
        
        ArrayXi allPtsInOptRegions = MAPA::UtilityCalcs::UniqueMembers( optLocRegions );
        int n = allPtsInOptRegions.size();
        
        // % Maps indices of original data points to indices of allPtsInOptRegions,
        // % which will also be indices of the rows of A in a bit.
        // invRowMap = zeros(1,N);
        // invRowMap(allPtsInOptRegions) = 1:n;
        
        // note: indices 0-based
        // == 0:(n-1)
        ArrayXi optPtsIdxs = ArrayXi::LinSpaced(n, 0, n-1);
        // initialize all unused values to -1 so they will give an error if we use them as indices
        ArrayXi invRowMap = ArrayXi::Constant(opts.N, -1);
        // invRowMap(allPtsInOptRegions) = 0:(n-1)
        igl::slice_into(optPtsIdxs, allPtsInOptRegions, invRowMap);

        // %% spectral analysis
        // heights = zeros(n, n0); % distances from the n points in allPtsInOptRegions to the n0 local planes
        // eps = zeros(1,n0); % estimated local errors
        
        ArrayXXd heights = ArrayXXd::Zero(n, n0);
        ArrayXd eps = ArrayXd::Zero(n0);

        // for i = 1:n0
        int ii = 0;
        for (std::vector<ArrayXi>::iterator it = optLocRegions.begin(); it != optLocRegions.end(); ++it)
        {
            // R_i = optLocRegions{i}; % indices of points in the current local region
            // n_i = numel(R_i);
            ArrayXi R_i = *it;
            int n_i = R_i.size();
            
            // if opts.isLinear
            //     X_c = X;
            // else
            //     ctr = mean(X(R_i,:), 1);
            //     X_c = X - repmat(ctr, N, 1);
            // end
            ArrayXXd X_c;
            ArrayXXd X_loc;
            if (opts.isLinear)
            {
                X_c = X;
            }
            else
            {
                // Default
                // NOTE: more extra copies than we need...
                // X_loc = X(R_i,:)
                igl::slice(X, R_i, 1, X_loc);
                ArrayXd ctr = X_loc.colwise().mean();
                X_c = X.rowwise() - ctr.transpose();
            }
                        
            // [~,s,v] = svd(X_c(R_i,:),0);
            // s = diag(s); % local singular values
            
            // X_loc = X_c(R_i,:)
            igl::slice(X_c, R_i, 1, X_loc);
            // Eigen std SVD
            JacobiSVD<MatrixXd> svd(X_loc, Eigen::ComputeThinU | Eigen::ComputeThinV);
            ArrayXd s = svd.singularValues();

            int loc_dim = localDims(ii);
            
            // eps(i) = sum(s(localDims(i)+1:end).^2) / (n_i-1); % local approximation error
            int num_noisy_dims = s.size() - (loc_dim + 1) + 1;
            eps(ii) = (s.tail(num_noisy_dims).square() / (n_i - 1)).sum();
            
            ArrayXXd X_c_allOptPts;
            // X_c_allOptPts = X_c(allPtsInOptRegions,:)
            igl::slice(X_c, allPtsInOptRegions, 1, X_c_allOptPts);
            MatrixXd v = svd.matrixV();
            // v_good = v0(:,1:localDims(i))
            MatrixXd v_good = v.leftCols(loc_dim);
            // heights(:,i) = (sum(X_c(allPtsInOptRegions,:).^2,2)-sum((X_c(allPtsInOptRegions,:)*v(:,1:localDims(i))).^2,2)) / (2*eps(i));
            heights.col(ii) = ( X_c_allOptPts.array().square().rowwise().sum() - (X_c_allOptPts.matrix() * v_good).array().square().rowwise().sum() ) / (2 * eps(ii));
            
            ii++;
                 
        // end
        }
                
        // A = exp(-abs(heights)); 
        // A(isnan(A)) = 1;
        
        ArrayXXd A = (-1.0 * heights.abs()).exp();
        // TODO: Can probably find a slicker way of doing this NaN check...
        for (int ii = 0; ii < A.size(); ii++)
        {
            if (isnan(A(ii)))
            {
                A(ii) = 1.0;
            }
        }

        // %% discarding bad rows and columns
        // if opts.discardCols > 0
        if (opts.discardCols > 0)
        {

            //     colStdA = std(A,0,1);
            //     goodCols = find(colStdA > quantile(colStdA, opts.discardCols));
        
            // Calculate standard deviation of the columns
            ArrayXd colStdA = ((A.rowwise() - A.colwise().mean()).square().colwise().sum() / (double)(A.rows()-1)).sqrt();
            ArrayXi goodCols = MAPA::UtilityCalcs::IdxsAboveQuantile( colStdA, opts.discardCols );

            //     eps = eps(goodCols);
            //     optLocRegions = optLocRegions(goodCols);
            //     seeds = seeds(goodCols);
            //     localDims = localDims(goodCols);
        
            ArrayXd tmp = eps;
            igl::slice(tmp, goodCols, eps);
            ArrayXi tmpi = seeds;
            igl::slice(tmpi, goodCols, seeds);
            tmpi = localDims;
            igl::slice(tmpi, goodCols, localDims);
        
            std::vector<ArrayXi> tmpv;
            for (int ii = 0; ii < goodCols.size(); ii++)
            {
                tmpv.push_back(optLocRegions[goodCols(ii)]);
            }
            optLocRegions = tmpv;

            //     % When we throw away columns, we are throwing away seed points, and
            //     % when we throw away seed points, we throw away data points that were
            //     % in the optimal local regions of those seed points, so that gets rid
            //     % of rows of A as well.
            //     allPtsInOptRegions = unique([optLocRegions{:}]);
            //     A = A(invRowMap(allPtsInOptRegions), goodCols);
            //     [n,n0] = size(A);
            allPtsInOptRegions = MAPA::UtilityCalcs::UniqueMembers( optLocRegions );
            
            ArrayXi goodRowIdxs;
            igl::slice(invRowMap, allPtsInOptRegions, goodRowIdxs);
            ArrayXXd tmpdd = A;
            igl::slice(tmpdd, goodRowIdxs, goodCols, A);
            n = A.rows();
            n0 = A.cols();

        // end
        }
        
        // 
        // % Maps the indices of the original data set to indices of the seed points,
        // % which are also the indices of the columns of A
        // invColMap = zeros(1,N);
        // invColMap(seeds) = 1:n0;
        
        // note: indices 0-based
        // seedsIdxs = 0:(n0-1)
        ArrayXi seedsIdxs = ArrayXi::LinSpaced(n0, 0, n0-1);
        // initialize all unused values to -1 so they will give an error if we use them as indices
        ArrayXi invColMap = ArrayXi::Constant(opts.N, -1);
        // invColMap(seeds) = 0:(n-1) == seedsIdxs
        igl::slice_into(seedsIdxs, seeds, invColMap);
        
        // switch opts.averaging
        //     case {'l2','L2'}
        //         eps = sqrt(mean(eps./(D-localDims)));
        //     case {'l1','L1'}
        //         eps = mean(sqrt(eps./(D-localDims)));
        // end
        
        double eps_mean;
        if ((opts.averaging == "L2") || (opts.averaging == "l2"))
        {
            eps_mean = std::sqrt((eps / (opts.D - localDims).cast<double>()).mean());
        }
        else if ((opts.averaging == "L1") || (opts.averaging == "l1"))
        {
            eps_mean = (eps / (opts.D - localDims).cast<double>()).sqrt().mean();
        }
        
        // if opts.discardRows>0
        if (opts.discardRows > 0)
        {
            //     rowStdA = std(A,0,2);
            //     goodRows = (rowStdA>quantile(rowStdA, opts.discardRows));

            // Calculate standard deviation of the rows
            ArrayXd rowStdA = ((A.colwise() - A.rowwise().mean()).square().rowwise().sum() / (double)(A.cols()-1)).sqrt();
            ArrayXi goodRows = MAPA::UtilityCalcs::IdxsAboveQuantile( rowStdA, opts.discardRows );
            
            //     A = A(goodRows,:);
            ArrayXXd tmpd2 = A;
            igl::slice(tmpd2, goodRows, 1, A);

            //     allPtsInOptRegions = allPtsInOptRegions(goodRows);
            //     n = numel(allPtsInOptRegions);
            ArrayXi tmp_allPts = allPtsInOptRegions;
            igl::slice(tmp_allPts, goodRows, allPtsInOptRegions);
            n = allPtsInOptRegions.size();

        // end
        }
        
        // invRowMap = zeros(1,N);
        // invRowMap(allPtsInOptRegions) = 1:n;
        
        // note: indices 0-based
        // initialize all unused values to -1 so they will give an error if we use them as indices
        invRowMap = ArrayXi::Constant(opts.N, -1);
        // optPtsIdxs = 0:(n-1)
        optPtsIdxs = ArrayXi::LinSpaced(n, 0, n-1);
        // invRowMap(allPtsInOptRegions) = 0:(n-1)
        igl::slice_into(optPtsIdxs, allPtsInOptRegions, invRowMap);

        
        // %% normalize the spectral matrix A
        // degrees = A*sum(A,1).';
        // degrees((degrees == 0)) = 1;
        // A = repmat(1./sqrt(degrees),1,n0).*A;
        
        ArrayXXd degrees = A.matrix() * A.colwise().sum().transpose().matrix();
        degrees = (degrees == 0).select(1, degrees);
        ArrayXd invDegrees = 1.0 / degrees.sqrt();
        A = invDegrees.replicate(1,n0) * A;

        JacobiSVD<MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        ArrayXXd U = svd.matrixU();

            // * * * * OKAY TO HERE * * * *

        // Directly cluster data (when K is provided)
        // if isfield(opts, 'K'),
        if (opts.K > 0)
        {
            //     K = opts.K;
            int K = opts.K;
            
            //     [U,S] = svds(A, K+1);
            
            //     [planeDims, labels, err] =  spectral_analysis(X, U(:,1:K), allPtsInOptRegions, invColMap, localDims, opts.nOutliers);
            MAPA::SpectralAnalysis spectral_analysis(X, U.leftCols(K), allPtsInOptRegions, invColMap, localDims, opts.nOutliers);
            
            planeDims = spectral_analysis.GetPlaneDims();
            labels = spectral_analysis.GetLabels();
            distance_error = spectral_analysis.GetError();
       }

        // Also select a model when only upper bound is given
        // elseif isfield(opts, 'Kmax'),
        else if (opts.Kmax > 0)
        {
            //     [U,S] = svds(A, opts.Kmax+1);

            //     planeDims = mode(localDims);
            //     labels = ones(1,N);
            int dim_winner = MAPA::UtilityCalcs::Mode(localDims);
            planeDims = ArrayXi::Constant(opts.N, dim_winner);
            labels = ArrayXi::Ones(opts.N);
            
            // NOTE: Why would we recompute the bases for all points for this error calculation??
            // PROBLEM: computing_bases_all with faces_clustering
            MAPA::ComputingBases computing_bases_all(X, labels, planeDims);
            std::vector<ArrayXd> planeCenters_all = computing_bases_all.GetCenters();
            std::vector<ArrayXXd> planeBases_all = computing_bases_all.GetBases();
        
            //     L2Errors = L2error(X, labels, planeDims);
            distance_error = MAPA::UtilityCalcs::L2error( X, labels, planeDims, planeCenters_all, planeBases_all );

            //     K = 1;
            int K = 1;
            
            //     while K<opts.Kmax && L2Errors > 1.05*eps
            while ((K < opts.Kmax) && (distance_error > (1.05 * eps_mean)))
            {
                // K = K+1;
                K += 1;
                // [planeDims, labels, L2Errors] = ...
                //    spectral_analysis(X, U(:,1:K), allPtsInOptRegions, invColMap, localDims, opts.nOutliers);
                MAPA::SpectralAnalysis spectral_analysis(X, U.leftCols(K), allPtsInOptRegions, invColMap, localDims, opts.nOutliers);
                planeDims = spectral_analysis.GetPlaneDims();
                labels = spectral_analysis.GetLabels();
                distance_error = spectral_analysis.GetError();
            // end
            }
            //     
        // end
        }
        
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
    
    double GetDistanceError()
    {
        return distance_error;
    };


private:

    ArrayXi labels;
    ArrayXi planeDims;
    double distance_error;

}; // class def

} // namespace MAPA

#endif
