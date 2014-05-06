#ifndef ESTIMATEDIMFROMSPECTRA_H
#define ESTIMATEDIMFROMSPECTRA_H


#include <Eigen/Core>
#include <stdlib.h>

using namespace Eigen;


namespace MAPA {

class EstimateDimFromSpectra {

public:

    EstimateDimFromSpectra()
    {
			DimEst = -1;
			GoodScalesLow = -1;
			GoodScalesHigh = -1;
    };
    
    void EstimateDimensionality(const ArrayXd &cDelta, ArrayXXd &S_MSVD, double alpha0)
    {
    	
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
    
	double compute_slope(int dim, int idx, int width)
	{
		ArrayXd spectrum = spectra.col(dim);
		double slope;
		
		ArrayXd s1 = scale_distances.segment(idx-width+1, width);
		ArrayXd sp = spectrum.segment(idx-width+1, width);
		// slope = (sum(s1.*sp) - sum(s1)*sum(sp)/numel(s1)) / (sum(s1.^2)-sum(s1)^2/numel(s1));
		
		return slope;
	};

}; // class def

} // namespace MAPA

#endif
