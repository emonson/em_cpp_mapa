#ifndef LMSVD_H
#define LMSVD_H

/* LMsvd.h

Based on the Matlab code of
Guangliang Chen & Mauro Maggioni

Eric E Monson – 2014
Duke University

*/

#include <Eigen/Core>
#include <vector>
#include <iostream>
#include "Options.h"

using namespace Eigen;


namespace MAPA {

class LMsvd {

public:

    LMsvd(const ArrayXXd &Xin, MAPA::Options opts)
    {
        // n_seeds = numel(opts.seeds);
        int n_seeds = opts.seeds.size();
        
        // % "rounded" version of maxKNN so get integer indices between MinNetPts and
        // % maxKNN with nPtsPerScale stride
        // maxKNN = opts.MinNetPts + opts.nPtsPerScale*(opts.nScales-1);
        int maxKNN = opts.MinNetPts + opts.nPtsPerScale * (opts.nScales - 1);

        // % Compute the distance between the seed point and the maxKNN nearest points in X. 
        // % opts.seeds are indices of X to compute distances from
        // % e.g. if X is [600x3], and if there are 60 seed points, and 102 maxKNN
        // %      idxs is [60x102] integers containing the indices of the nearest neighbors to the 60 seed points
        // %      statdists is [60x102] doubles containing the distances from the 60 seed points to the 102 NNs
        // [~, nn_idxs, statdists] = nrsearch(X, uint32(opts.seeds), maxKNN, [], [], struct('XIsTransposed',true,'ReturnAsArrays',true));

        // ANN calc object
        MAPA::NRsearch ann(X);
    
        ann.computeANN(seed_subset, maxKNN, eps);
    
        // std::cout << ann.GetIdxs() << std::endl;
        // std::cout << ann.GetDistances() << std::endl;

        
        
        // 
        // % e.g. statdists(:, 4:2:102) which is [60x50], which is [n_seed_pts x opts.nScales]
        // % Miles: "Delta is the distance to the farthest neighbor used in that scale. 
        // %   so instead of thinking of the scale as a number of nearest neighbors in the data, 
        // %   we can think of it as a distance needed to collect that many neighbors."
        // Delta = statdists(:, opts.MinNetPts:opts.nPtsPerScale:maxKNN );
        // 
        // %%
        // MSVD_Stats = zeros(n_seeds, D, opts.nScales);
        // estDims = zeros(1, n_seeds);
        // GoodScales = zeros(n_seeds,2);
        // goodLocalRegions = cell(1,n_seeds);
        // 
        // for i_seed = 1:n_seeds,
        //     
        //     Nets_S = zeros(opts.nScales, D);
        //     
        //     for i_scale = 1:opts.nScales,
        //         % We have a minimum number of points to go out from each seed, and then
        //         % are increasing the number of points with each scale
        //         Nets_count = opts.MinNetPts + (i_scale-1)*opts.nPtsPerScale;
        // 
        //         % Grab NNidxs over all seed points up to a certain number of NN for
        //         % this scale
        //         % actual point coords for the NNs for this seed point and scale
        //         net = X( nn_idxs(i_seed, 1:Nets_count), :);
        //         % center this set of net points and do an SVD to get the singular
        //         % values
        //         sigs = svd(net - repmat(mean(net,1), Nets_count, 1));
        //         % make into a row vector and normalize the singular values by the
        //         % sqrt of the number of net points
        //         sigs = sigs'/sqrt(Nets_count);
        // 
        //         Nets_S(i_scale,:) = sigs;
        //     end
        //         
        //     lStats = EstimateDimFromSpectra(Delta(i_seed,:)', Nets_S, opts.alpha0, i_seed);
        //     fprintf(1,'%.70f\n', opts.alpha0);
        //     estDims(i_seed) = lStats.DimEst;
        //     GoodScales(i_seed,:) = lStats.GoodScales;
        //                 if (i_seed == 2 || i_seed == 1 || i_seed == 60),
        //                     disp(lStats.DimEst);
        //                     disp(lStats.GoodScales);
        //                 end
        //     maxScale = GoodScales(i_seed,2);
        //     goodLocalRegions{i_seed} = nn_idxs(i_seed, 1:(opts.MinNetPts + (maxScale-1)*opts.nPtsPerScale));
        //     
        // end
        // 
        // goodSeedPoints = (cellfun(@length, goodLocalRegions)>2*estDims & estDims<D);
        // 
        // goodLocalRegions = goodLocalRegions(goodSeedPoints);
        // estDims = estDims(goodSeedPoints);
        // goodSeedPoints = opts.seeds(goodSeedPoints);
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
    

private:

    std::vector<ArrayXi> goodLocalRegions;
    ArrayXi goodSeedPoints;
    ArrayXi estDims;

}; // class def

} // namespace MAPA

#endif
