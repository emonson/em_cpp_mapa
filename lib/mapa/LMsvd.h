#ifndef LMSVD_H
#define LMSVD_H

/* LMsvd.h

Based on the Matlab code of
Guangliang Chen & Mauro Maggioni

Eric E Monson – 2014
Duke University

*/

#include <Eigen/Core>
#include <Eigen/SVD>

#include <vector>
#include <iostream>
#include <cmath>

#include <igl/slice.h>
#include "Options.h"
#include "NRsearch.h"

using namespace Eigen;


namespace MAPA {

class LMsvd {

public:

    LMsvd(const ArrayXXd &X, MAPA::Opts opts)
    {
        int n_seeds = opts.seeds.size();
        
        // "rounded" version of maxKNN so get integer indices between MinNetPts and
        // maxKNN with nPtsPerScale stride
        int maxKNN = opts.MinNetPts + opts.nPtsPerScale * (opts.nScales - 1);

        // Compute the distance between the seed point and the maxKNN nearest points in X. 
        // opts.seeds are indices of X to compute distances from
        // e.g. if X is [600x3], and if there are 60 seed points, and 102 maxKNN
        // idxs is [60x102] integers containing the indices of the nearest neighbors to the 60 seed points
        // statdists is [60x102] doubles containing the distances from the 60 seed points to the 102 NNs

        MAPA::NRsearch ann(X);
        ann.computeANN(opts.seeds, maxKNN, 0);
    
        ArrayXXi nn_idxs = ann.GetIdxs();
        ArrayXXd statdists = ann.GetDistances();
        
        // e.g. statdists(:, 4:2:102) which is [60x50], which is [n_seed_pts x opts.nScales]
        // Miles: "Delta is the distance to the farthest neighbor used in that scale. 
        // so instead of thinking of the scale as a number of nearest neighbors in the data, 
        // we can think of it as a distance needed to collect that many neighbors."
        
        ArrayXi row_idxs, col_idxs;
        ArrayXXd Delta;
        
        row_idxs.setLinSpaced(statdists.rows(), 0, statdists.rows()-1); 
        col_idxs.setLinSpaced(opts.nScales, opts.MinNetPts-1, maxKNN-1);
        
        igl::slice( statdists, row_idxs, col_idxs, Delta );
        
        allEstDims = ArrayXi::Zero(n_seeds);
        allGoodScales = ArrayXXi::Zero(n_seeds,2);

        ArrayXXd Nets_S(opts.nScales, opts.D);
        int Nets_count, maxScale, seed_est_dim;
        double Nets_count_sqrt;
        ArrayXi allXcols = ArrayXi::LinSpaced(X.cols(), 0, X.cols()-1);
        ArrayXXd net, net_centered;
        ArrayXd sigs;
        ArrayXi seed_nn_idxs, seed_local_region;
        ArrayXi isSeedPointGood(n_seeds);
        
        MAPA::EstimateDimFromSpectra estdim;
        
        for (int i_seed = 0; i_seed < n_seeds; i_seed++)
        {
            Nets_S.setZero();
        
            for (int i_scale = 0; i_scale < opts.nScales; i_scale++)
            {
                // We have a minimum number of points to go out from each seed, and then
                // are increasing the number of points with each scale                
                // NOTE: i_scale already 0-based here, so no -1 !
                Nets_count = opts.MinNetPts + (i_scale) * opts.nPtsPerScale;
                Nets_count_sqrt = std::sqrt((double)Nets_count);
        
                // Grab NNidxs over all seed points up to a certain number of NN for
                // this scale actual point coords for the NNs for this seed point and scale
                seed_nn_idxs = nn_idxs.row(i_seed);
                igl::slice(X, seed_nn_idxs.head(Nets_count), allXcols, net);
                                
                // center this set of net points and do an SVD to get the singular values
                net_centered = net.rowwise() - net.colwise().mean();
                
                // Eigen std SVD
                JacobiSVD<MatrixXd> svd(net_centered, Eigen::ComputeThinU | Eigen::ComputeThinV);
                sigs = svd.singularValues();

                // make into a row vector and normalize the singular values by the
                // sqrt of the number of net points
                sigs /= Nets_count_sqrt;
                
                Nets_S.row(i_scale) = sigs.transpose();
            }
        
            // lStats = EstimateDimFromSpectra(Delta(i_seed,:)', Nets_S, opts.alpha0, i_seed);
        	estdim.EstimateDimensionality( Delta.row(i_seed), Nets_S, opts.alpha0);
        		
            // estDims(i_seed) = lStats.DimEst;
            // GoodScales(i_seed,:) = lStats.GoodScales;
            // maxScale = GoodScales(i_seed,2);
            // goodLocalRegions{i_seed} = nn_idxs(i_seed, 1:(opts.MinNetPts + (maxScale-1)*opts.nPtsPerScale));
            seed_est_dim = estdim.GetDimension();
            allEstDims(i_seed) = seed_est_dim;
            allGoodScales(i_seed,0) = estdim.GetLowerScaleIdx(); // NOTE: Matlab 1s-based now!!!             
            allGoodScales(i_seed,1) = estdim.GetUpperScaleIdx(); // NOTE: Matlab 1s-based now!!!  

                // *** FINE UP TO THIS POINT ** 

            maxScale = allGoodScales(i_seed,1);
            seed_local_region = nn_idxs.row(i_seed).head(opts.MinNetPts + (maxScale-1)*opts.nPtsPerScale);
            allLocalRegions.push_back(seed_local_region);
            
            isSeedPointGood(i_seed) = (int)(seed_local_region.size() > (2 * seed_est_dim)) && (seed_est_dim < opts.D);
            if (isSeedPointGood(i_seed))
            {
            	goodLocalRegions.push_back(seed_local_region);
            } 
        }
        // estDims = estDims(goodSeedPoints);
        // goodSeedPoints = opts.seeds(goodSeedPoints);
        
        // TODO: Maybe can do this with some boolean indexing?
        int n_good_seeds = isSeedPointGood.sum();
        estDims = ArrayXi::Zero(n_good_seeds);
        goodSeedPoints = ArrayXi::Zero(n_good_seeds);
        int jj = 0;
        for (int ii = 0; ii < n_seeds; ii++)
        {
        	if (isSeedPointGood(ii))
        	{
        		estDims(jj) = allEstDims(ii);
        		goodSeedPoints(jj) = opts.seeds(ii);
        		jj++;
        	}
        }
    };

    std::vector<ArrayXi> GetGoodLocalRegions()
    {
        return goodLocalRegions;
    };
    
    ArrayXi GetGoodSeedPoints()
    {
        return goodSeedPoints;
    };
    
    ArrayXi GetEstimatedDims()
    {
        return estDims;
    };
    
    ArrayXi GetAllEstimatedDims()
    {
        return allEstDims;
    };
    
    ArrayXXi GetAllGoodScales()
    {
        return allGoodScales;
    };
    

private:

    std::vector<ArrayXi> goodLocalRegions;
    ArrayXi goodSeedPoints;
    ArrayXi estDims;
    
    // Results before filtering out seeds with high-D or not enough neighbors
    std::vector<ArrayXi> allLocalRegions;
    ArrayXXi allGoodScales;
    ArrayXi allEstDims;

}; // class def

} // namespace MAPA

#endif
