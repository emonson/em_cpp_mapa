#ifndef OPTIONS_H
#define OPTIONS_H

/* lmsvd.h

Based on the Matlab code of
Guangliang Chen & Mauro Maggioni

Eric E Monson – 2014
Duke University

*/

// %   opts: a structure of the following optional parameters:
// %
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
// %       .discardRows: percentage of bad rows of the matrix A to be discarded 
// %            (default = 0)
// %       .discardCols: percentage of bad columns of A to be discarded 
// %            (default = 0)
// %       .nOutliers: number of outliers (if >=1), or percentage (if <1) **NOTE: not supporting percentage for now...
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

/* Usage:
    The constructor initializes all the values, but doesn't set them to useful
    defaults yet. Once the Opts object is created, desired option (member variable)
    values should be set. After that, run .SetDefaults(X) to set the rest of the
    values to natural defaults depending on the data. X is [N, D] */
    
#include "Eigen/Dense"
#include <math.h>
#include <string>
#include <iostream>
#include "UtilityCalcs.h"
#include "igl/sort.h"

using namespace Eigen;


namespace MAPA {

class Opts {

public:

    Opts()
    {
        N = 0;			// number of points
        D = 0;			// dimensionality of points
        dmax = 0;
        K = 0;
        Kmax = 0;
        alpha0 = 0.0;
        n0 = 0;
        // seeds;
        MinNetPts = 0;
        nScales = 0;
        nPtsPerScale = 0;
        isLinear = false;
        discardRows = false;
        discardCols = false;
        nOutliers = 0;
        averaging = "L2";
        plotFigs = false;
        showSpectrum = false;
        postOptimization = false;
        maxKNN = 0;
    };
    
    void SetDefaults(ArrayXXd &X)
    {
        N = X.rows();
        D = X.cols();
        
        // alpha0 -----------
        if (!(alpha0 > 0.0))
        {
            if ((dmax == 0) || (dmax >= D))
            {
                dmax = D - 1;
                alpha0 = 0.2;
            }
            else
            {
                alpha0 = 0.3 / sqrt(dmax);
            }
        }
        
        // Kmax -----------
        if ((K == 0) && (Kmax == 0))
        {
            Kmax = 10;
        }
        
        // n0 -----------
        if (seeds.size() == 0)
        {
        	if (n0 == 0)
        	{
        		if (K == 0)
        		{
        			n0 = 20 * Kmax;
        		}
        		else
        		{
        			n0 = 20 * K;
        		}
        	}
        	if (n0 < N)
        	{
        		bool sorted = true;
        		seeds = UtilityCalcs::RandSample(N, n0, sorted);
        	}
        	else
        	{
						seeds.setLinSpaced(0,N-1);
						if (n0 > N)
						{
							std::cout << "Warning: The sampling parameter n0 has been modified to N!" << std::endl;
							n0 = N;
						}
        	}
        }
        else
        {
        	// Seeds provided
        	if ((n0 > 0) && n0 != seeds.size())
        	{
        		std::cout << "Warning: The parameter values of n0 and seeds are incompatible. n0 has been changed to the length of seeds." << std::endl;
        		n0 = seeds.size();
        	}
        }
        
        // maxKNN -----------
        maxKNN = (unsigned int)round(fmin((double)N/5.0, 50.0*dmax*log(fmax(3.0,(double)dmax))));
        
        // MinNetPts -----------
        if (MinNetPts == 0)
        {
        	MinNetPts = dmax + 2;
        }
        
        // nScales -----------
        if (nScales == 0)
        {
        	nScales = (maxKNN < 50) ? maxKNN : 50;
        }
        
        // nPtsPerScale -----------
        if (nPtsPerScale == 0)
        {
        	nPtsPerScale = round( (double)maxKNN / (double)nScales );
        }
        
        // NOTE: isLinear, discardRows, discardCols, averaging,
        // plotFigs, showSpectrum, postOptimization already at default
        
        // if ~isfield(opts, 'nOutliers')
        //     opts.nOutliers = 0;
        // elseif opts.nOutliers<1
        //     opts.nOutliers = round(N*opts.nOutliers);
        // end
        // ** NOTE ** Not supporting nOutliers fraction as percentage for now...
        
    }
    
    // Member varialbes
		unsigned int N;
		unsigned int D;
		unsigned int dmax;
		unsigned int K;
		unsigned int Kmax;
		unsigned int maxKNN;
		double alpha0;
		unsigned int n0;
		ArrayXi seeds;
		unsigned int MinNetPts;
		unsigned int nScales;
		unsigned int nPtsPerScale;
		bool isLinear;
		bool discardRows;
		bool discardCols;
		unsigned int nOutliers; // reall should be able to take float according to original...
		std::string averaging;
		bool plotFigs;
		bool showSpectrum;
		bool postOptimization;

		// --------------------------
		friend std::ostream& operator<<(std::ostream& os, const Opts& op);

private:

}; // class def


std::ostream& operator<<(std::ostream& os, const Opts& op)
{
    os << "N: " << op.N << std::endl;
    os << "D: " << op.D << std::endl;
    os << "dmax: " << op.dmax << std::endl;
    os << "K: " << op.K << std::endl;
    os << "Kmax: " << op.Kmax << std::endl;
    os << "maxKNN: " << op.maxKNN << std::endl;
    os << "alpha0: " << op.alpha0 << std::endl;
    os << "n0: " << op.n0 << std::endl;
    os << "seeds: " << op.seeds.transpose() << std::endl;
    os << "MinNetPts: " << op.MinNetPts << std::endl;
    os << "nScales: " << op.nScales << std::endl;
    os << "nPtsPerScale: " << op.nPtsPerScale << std::endl;
    return os;
}


} // namespace MAPA

#endif
