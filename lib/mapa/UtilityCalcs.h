#ifndef UTILITYCALCS_H
#define UTILITYCALCS_H

/* UtilityCalcs.h

Based on the Matlab code of
Guangliang Chen & Mauro Maggioni

Eric E Monson – 2014
Duke University

*/

#include "Eigen/Dense"
#include "mersenneTwister2002.h"
#include <igl/slice.h>
#include <igl/sort.h>
#include <stdlib.h>
#include "ANN.h"

using namespace Eigen;


namespace MAPA {

class UtilityCalcs {

	public:

    static ArrayXXd P2Pdist(const ArrayXXd &X, const std::vector<ArrayXd> &centers, const std::vector<ArrayXXd> &bases)
    {
      std::cerr << "MAPA::UtilityCalcs::P2Pdist – Not implemented yet!!" << std::endl;
			return Array<double,1,1>::Zero();
    };

    static ArrayXd L2error(const ArrayXXd &data, const ArrayXi &dim, const ArrayXi &idx)
    {
      std::cerr << "MAPA::UtilityCalcs::L2error – Not implemented yet!!" << std::endl;
			return Array<double,1,1>::Zero();
    };

    static double ClusteringError(const ArrayXi &indices, const ArrayXi &trueLabels)
    {
      std::cerr << "MAPA::UtilityCalcs::ClusteringError – Not implemented yet!!" << std::endl;
			return 0;
    };
    
    static ArrayXi RandSample(unsigned int N, unsigned int K, bool sorted=false)
    {
			/* Y = randsample(N,K) returns Y as a column vector of K values sampled
			uniformly at random, without replacement, from the integers 0:N-1 */

			if (K > N)
			{
      	std::cerr << "MAPA::UtilityCalcs::RandSample Error – Number of samples K can't be larger than N" << std::endl;
      	return ArrayXi::Zero(K);
			}
			
			ArrayXd randX(N);
			randX.setRandom();
			
			// sort columns independently (only one here)
			int dim = 1;
			// sort ascending order
			int ascending = true;
			// Sorted output matrix
			ArrayXd Y;
			// sorted indices for sort dimension
			ArrayXi IX;
			
			igl::sort(randX,1,ascending,Y,IX);

			if (!sorted)
			{
				return IX.head(K);			
			}
			else
			{
				ArrayXi Yidxs;
				ArrayXi Xidxs = IX.head(K);
				
				igl::sort(Xidxs, 1, ascending, Yidxs, IX);
				return Yidxs;
			}
    };

   
}; // class def

} // namespace MAPA

#endif
