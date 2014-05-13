#ifndef UTILITYCALCS_H
#define UTILITYCALCS_H

/* UtilityCalcs.h

Based on the Matlab code of
Guangliang Chen & Mauro Maggioni

Eric E Monson – 2014
Duke University

*/

#include <Eigen/Core>
#include "mersenneTwister2002.h"
#include "ANN.h"
#include <igl/sort.h>
#include <igl/slice.h>

#include <stdlib.h>
#include <iostream>
#include <vector>
#include <set>
#include <algorithm>

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

    static ArrayXi UniqueMembers(const std::vector<ArrayXi> neighborhoods)
    {
        // Actually find unique members by adding them to a set
        std::set<int> unique_members;
        for (std::vector<int>::size_type ii = 0; ii != neighborhoods.size(); ii++)
        {
            int *nb = (int*)neighborhoods[ii].data();
            int nb_size = neighborhoods[ii].size();
            unique_members.insert(nb, nb+nb_size);
        }
        
        // Copy values over into an Eigen array
        ArrayXi unique_array(unique_members.size());
        std::set<int>::iterator it;
        int ii = 0;
        for (it = unique_members.begin(); it != unique_members.end(); ++it)
        {
            unique_array(ii) = *it;
            ii++;
        }

        return unique_array;
    };

    static ArrayXi IdxsAboveQuantile(const ArrayXd invals, double q_cutoff)
    {
        int N = 10;
        double q_cutoff = 0.2;

        ArrayXd randvec = ArrayXd::Random(N);
    
        // std::cout.precision(3);
        std::cout << randvec.transpose() << std::endl;
    
        ArrayXd Yd;
        ArrayXi IX;
        bool ascending = true;
        igl::sort(randvec, 1, ascending, Yd, IX);
    
        // Create an array cumulative probabilities
        ArrayXd quants = ArrayXd::LinSpaced(Eigen::Sequential, N, 0.5, N-0.5) / (double)N;
        std::cout << quants << std::endl;

        // Replace all indices with -1 that don't pass the test
        ArrayXi IX_found = (quants.array() > q_cutoff).select(IX, -1);
    
        // Use stable_partition and the gtezero ( >= 0 ) to place all good indices
        // still in their original order, before bound
        int *bound;
        bound = std::stable_partition( IX_found.data(), IX_found.data()+IX_found.size(), gtezero);
        std::cout << IX_found.transpose() << std::endl;
    
        // Resize indices array to exclude all of the -1s
        IX_found.conservativeResize(bound-IX_found.data());
        std::cout << IX_found.transpose() << std::endl;
    
        // Resort indices back to original order
        ArrayXi Yi; 
        igl::sort(IX_found, 1, ascending, Yi, IX);
        std::cout << Yi.transpose() << std::endl;
        
        return Yi;
    };

  private:
    
    bool gtezero(int val) { return val >= 0; }
   
}; // class def

} // namespace MAPA

#endif
