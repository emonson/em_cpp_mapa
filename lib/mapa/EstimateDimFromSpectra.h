#ifndef ESTIMATEDIMFROMSPECTRA_H
#define ESTIMATEDIMFROMSPECTRA_H


#include <Eigen/Core>
#include <iostream>     // std::cout
#include <algorithm>    // std::min

using namespace Eigen;


namespace MAPA {

class EstimateDimFromSpectra {
// function vStats = EstimateDimFromSpectra( cDeltas, S_MSVD, alpha0 )
// %
// % Estimates intrinsic dimensionality and good scales given the multiscale singular values
// %
// % IN:
// %   cDeltas         : (#scales) vector of scales
// %   S_MSVD          : (#scales)*(#dimensions) matrix of singular values: the (i,j) entry is the j-th singular value of a cell at scale cDeltas(i) around a point
// %
// % OUT:
// %   vStats          : structure containing the following fields:
// %                       DimEst      : estimate of intrinsic dimensionality
// %                       GoodScales  : vector of indices (into the vector cDeltas) of good scales used for dimension estimate
// %
// 

public:

    EstimateDimFromSpectra()
    {
        DimEst = -1;
        GoodScalesLow = -1;
        GoodScalesHigh = -1;
        width = -1;
    };
    
    void EstimateDimensionality(const ArrayXd &cDelta, const ArrayXXd &S_MSVD, double alpha0 = 0.2)
    {
        double slope;
        ArrayXd current_to_prev_spectra_diff;
        ArrayXd largest_to_prev_spectra_diff;
        double mean_fractional_spectrum_rise;
        
        int lowerScaleIdx;
        int upperScaleIdx;
        
        // [nScales,nDims] = size(S_MSVD);
        // 
        // scale_distances = cDeltas;
        // 
        // if nargin<3; alpha0 = 0.2; end
        // 
        // width = 5;
        // lowerScaleIdxMax = min(12,nScales);
        // upperScaleIdxMax = nScales;
        // 
        // vStats.GoodScales = [1,nScales];
        // vStats.DimEst = nDims;
        
        int nScales = S_MSVD.rows();
        int nDims = S_MSVD.cols();
        
        scale_distances = cDelta;
        spectra = S_MSVD;
        
        width = 5;
        
        // NOTE: remember using zero-based indices unlike Matlab 1-based !!
        int lowerScaleIdxMax = std::min(12-1, nScales-1);
        int upperScaleIdxMax = nScales-1;
        GoodScalesLow = 1-1;
        GoodScalesHigh = nScales-1;
        // NOTE: Watch this one – Don't know if should keep it more intuitive, or switch to zero-based for use as index...?
        // Keeping it as intuitive dimensionality value, so need to subtract one when using it as an index...!!
        DimEst = nDims;

        // Start with the smallest singular values, which should be the most noisy
        // and loop through decreasing dim index down to 1, and break out early if
        // we hit a dim at which the slope never flattens out below 0.1 at any
        // r/delta
        
        // for dim = nDims:-1:1,
        
        for (int dim = nDims-1; dim >= 0; dim--)
        {
             
        //     vStats.DimEst = dim;
        //     spectrum = S_MSVD(:,dim);
            
            // Note: difference between dim as index and DimEst as dimensionality!
            DimEst = dim + 1;
            
            // find a lower bound for the optimal scale
            // try to find an r/delta index at which the slope drops below 0.1
            // (flattens out)
        
        //     for ii = width:lowerScaleIdxMax+1,
        //         lowerScaleIdx = ii;
        //         slope = compute_slope(scale_distances, spectrum, width, lowerScaleIdx);
        //         if (slope < 0.1),
        //             break;
        //         end
        //     end
            
            for (int ii = width - 1; ii <= lowerScaleIdxMax+1; ii++)
            {
                lowerScaleIdx = ii;
                slope = compute_slope(dim, ii);
                if (slope < 0.1)
                {
                    break;
                }
            }

            // Special case for the first (highest dim, noisiest singular values)
            // Testing whether slope never flattened out (or only flattened out on the
            // last iteration. If it didn't flatten out (go < 0.1), soon enough, 
            // then it's "not a noisy singular value" and don't
            // need to check when the flat started to rise again.
            // This would also mean we don't have to test any lower dim. All scales are
            // "good", and dimensionality of the manifold is the full dimensionality of
            // the system.
        
        //     if (dim == nDims) && (lowerScaleIdx == lowerScaleIdxMax+1),
        //         return;
        //     end
        
            if ((dim == nDims-1) && (lowerScaleIdx == lowerScaleIdxMax+1))
            {
                return;
            }
        
            // find an upper bound for the optimal scale
            // Special case for first iteration, no test for slope <= alpha0 for
            // some reason.
            // If the slope never even dropped below alpha0
            // then don't bother trying to find another rise from
            // the nonexistent flat portion, and break out of this loop because it's
            // assumed once the slope always stays above alpha0, none of the rest of
            // the dims are "noisy", i.e. those larger singular value dims are all
            // part of the manifold around this net point, so we can set the
            // estimation of the dimension, and we won't get any more help from
            // these dims judging the window of good scales (r/delta indices)
        
        //     if (dim == nDims) || (slope <= alpha0)
            
            if ((dim == nDims-1) || (slope <= alpha0))
            {

                // Set j either at iMax+1 or at the r/delta index at which the slope
                // flattened out below 0.1

        //         for jj = lowerScaleIdx:upperScaleIdxMax,
        //             upperScaleIdx = jj;
        //             slope = compute_slope(scale_distances, spectrum, width, upperScaleIdx);
        //             if (slope > alpha0),
        //                 break;
        //             end
        //         end
        //         upperScaleIdx = upperScaleIdx - 1;

                for (int jj = lowerScaleIdx; jj <= upperScaleIdxMax; jj++)
                {
                    upperScaleIdx = jj;
                    slope = compute_slope(dim, jj);
                    if (slope > alpha0)
                    {
                        break;
                    }
                }
                
                // Only compute gap after first iteration
        
        //         if(dim < nDims),
        
                if (dim < nDims-1)
                {
                
                    // If the curve flattens out for a while, but the gap between it and
                    // the previous dim spectrum is wide enough, going to count it as a
                    // "non-noisy" real dim that's part of the manifold
        
        //             idx_window = upperScaleIdx-width+1:upperScaleIdx;
        //             current_to_prev_spectra_diff = (spectrum(idx_window)-S_MSVD(idx_window,dim+1));
        //             largest_to_prev_spectra_diff = (S_MSVD(idx_window,1)-S_MSVD(idx_window,dim+1));
        //             mean_fractional_spectrum_rise = mean(current_to_prev_spectra_diff./largest_to_prev_spectra_diff);
                    
                    int idx = upperScaleIdx-width+1;
                    current_to_prev_spectra_diff = spectra.col(dim-1).segment(idx,width) - spectra.col(dim+1-1).segment(idx,width);
                    largest_to_prev_spectra_diff = spectra.col(1-1).segment(idx,width) - spectra.col(dim+1-1).segment(idx,width);
                    mean_fractional_spectrum_rise = (current_to_prev_spectra_diff / largest_to_prev_spectra_diff).mean();
                    
                    
        //             if mean_fractional_spectrum_rise > 0.2
        //                 % real manifold dim, so leave everything the same and return
        //                 return;
        //             end
                    
                    if (mean_fractional_spectrum_rise > 0.2)
                    {
                        // real manifold dim, so leave everything the same and return
                        return;
                    }

        //         end
                
                }

                // still a noisy scale with a low-enough flat portion that is
                // helping us find the "good scales", so update those. 
                // We want to use the values from the last dim found to be
                // "noisy", i.e. flat in some portion and not risen enough from
                // the previous, higher, dim, so keep updating GoodScales each
                // time, so when we break out of the loop with a noisy == false
                // will have the proper r/delta indices recoreded
        
        //         vStats.GoodScales = [lowerScaleIdx-1 upperScaleIdx];
                
                GoodScalesLow = lowerScaleIdx-1;
                GoodScalesHigh = upperScaleIdx;

        //     else
            }
            else
            {
        
        // slope never dropped below alpha0, leave everything the same and return
        
        //         return;

        //     end
            }

        // end
        }
        
        return;
    };  

    int GetDimension()
    {
        return DimEst;
    };
    
    int GetLowerScaleIdx()
    {
        return GoodScalesLow;
    };
    
    int GetUpperScaleIdx()
    {
        return GoodScalesHigh;
    };
    
    
private:

    int DimEst;
    int GoodScalesLow;
    int GoodScalesHigh;
    ArrayXd scale_distances;
    ArrayXXd spectra;
    int width;
    
	double compute_slope(int dim, int idx)
	{
		ArrayXd spectrum = spectra.col(dim);
		double slope, numerator, denominator;
		
        // function slope = compute_slope(distances, spectrum, width, idx)
        //     s1 = distances(idx-width+1:idx);
        //     sp = spectrum(idx-width+1:idx);
        //     slope = (sum(s1.*sp) - sum(s1)*sum(sp)/numel(s1)) / (sum(s1.^2)-sum(s1)^2/numel(s1));
        // end
        
		ArrayXd s1 = scale_distances.segment(idx-width+1, width);
		ArrayXd sp = spectrum.segment(idx-width+1, width);
		// slope = (sum(s1.*sp) - sum(s1)*sum(sp)/numel(s1)) / (sum(s1.^2)-sum(s1)^2/numel(s1));
		
		numerator = (s1 * sp).sum() - (s1.sum()*sp.sum()/(double)s1.size());
		denominator = s1.square().sum() - (s1.sum()*s1.sum())/(double)s1.size();
		
		slope = numerator / denominator;
		
		return slope;
	};

}; // class def

} // namespace MAPA

#endif
