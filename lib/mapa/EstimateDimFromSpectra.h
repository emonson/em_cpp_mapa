#ifndef ESTIMATEDIMFROMSPECTRA_H
#define ESTIMATEDIMFROMSPECTRA_H


#include <Eigen/Core>
#include <iostream>     // std::cout

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
        GoodScalesLow = -1;
        GoodScalesHigh = -1;
        DimEst = -1;
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
        
        int nScales = S_MSVD.rows();
        int nDims = S_MSVD.cols();
        distances = cDelta;
        spectra = S_MSVD;
        width = 5;
        
        // NOTE: Trying all of these same as Matlab but just subtracting one when using as an index
        int lowerScaleIdxMax = nScales < 12 ? nScales : 12;
        int upperScaleIdxMax = nScales;
        GoodScalesLow = 1;
        GoodScalesHigh = nScales;
        DimEst = nDims;

        // Start with the smallest singular values, which should be the most noisy
        // and loop through decreasing dim index down to 1, and break out early if
        // we hit a dim at which the slope never flattens out below 0.1 at any
        // r/delta
        for (int dim = nDims; dim >= 1; dim--)
        {
            // Note: difference between dim as index and DimEst as dimensionality!
            DimEst = dim;
            
            // find a lower bound for the optimal scale
            // try to find an r/delta index at which the slope drops below 0.1
            // (flattens out)
            for (int ii = width; ii <= lowerScaleIdxMax+1; ii++)
            {
                lowerScaleIdx = ii;
                slope = compute_slope(dim, lowerScaleIdx);
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
            if ((dim == nDims) && (lowerScaleIdx == lowerScaleIdxMax+1))
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
            if ((dim == nDims) || (slope <= alpha0))
            {
                // Set j either at iMax+1 or at the r/delta index at which the slope
                // flattened out below 0.1
                for (int jj = lowerScaleIdx; jj <= upperScaleIdxMax; jj++)
                {
                    upperScaleIdx = jj;
                    slope = compute_slope(dim, upperScaleIdx);
                    if (slope > alpha0)
                    {
                        break;
                    }
                }
                
                // Only compute gap after first iteration
                if (dim < nDims)
                {
                    // If the curve flattens out for a while, but the gap between it and
                    // the previous dim spectrum is wide enough, going to count it as a
                    // "non-noisy" real dim that's part of the manifold
                    int idx = upperScaleIdx-width+1;
                    current_to_prev_spectra_diff = spectra.col(dim-1).segment(idx-1,width) - spectra.col(dim+1-1).segment(idx-1,width);
                    largest_to_prev_spectra_diff = spectra.col(1-1).segment(idx-1,width) - spectra.col(dim+1-1).segment(idx-1,width);
                    mean_fractional_spectrum_rise = (current_to_prev_spectra_diff / largest_to_prev_spectra_diff).mean();
                    
                    if (mean_fractional_spectrum_rise > 0.2)
                    {
                        // real manifold dim, so leave everything the same and return
                        return;
                    }
                }

                // still a noisy scale with a low-enough flat portion that is
                // helping us find the "good scales", so update those. 
                // We want to use the values from the last dim found to be
                // "noisy", i.e. flat in some portion and not risen enough from
                // the previous, higher, dim, so keep updating GoodScales each
                // time, so when we break out of the loop with a noisy == false
                // will have the proper r/delta indices recoreded
                GoodScalesLow = lowerScaleIdx-1;
                GoodScalesHigh = upperScaleIdx;
            }
            else
            {       
                // slope never dropped below alpha0, leave everything the same and return
                return;
            }
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
    ArrayXd distances;
    ArrayXXd spectra;
    int width;
    
	double compute_slope(int dim, int idx)
	{
		// NOTE: idx getting passed as Matlab 1s-based, so need to subtract 1 for real indexing
		double slope, numerator, denominator;
		
		ArrayXd s1 = distances.segment(idx-1-width+1, width);
		ArrayXd sp = spectra.col(dim-1).segment(idx-1-width+1, width);
		
		numerator = (s1 * sp).sum() - (s1.sum()*sp.sum()/(double)s1.size());
		denominator = s1.square().sum() - (s1.sum()*s1.sum())/(double)s1.size();
		slope = numerator / denominator;
		
		return slope;
	};

}; // class def

} // namespace MAPA

#endif
