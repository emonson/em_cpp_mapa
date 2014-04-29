#include <Eigen/Core>
#include <igl/readDMAT.h>
#include <igl/slice.h>
#include <cstdio>
#include "UtilityCalcs.h"

int main(int argc, char * argv[])
{
    // Read in test data
    Eigen::ArrayXXd X;
    std::cout << "Reading in Artifical 3D test data (rev1)" << std::endl;
    igl::readDMAT( "/Users/emonson/Programming/em_cpp_mapa/data/artificial_data_rev1.dmat", X );
    
    // Read in seed points
    Eigen::ArrayXi seeds;
    std::cout << "Reading in seed points for test data (rev1)" << std::endl;
    igl::readDMAT( "/Users/emonson/Programming/em_cpp_mapa/data/artdat_rev1_nrsearch_seeds.dmat", seeds);
		
		// NOTE: seeds is matlab-style 1s index!!
		seeds -= 1;
		
		ArrayXi seed_subset = seeds.head(5);
		
		std::cout << seed_subset << std::endl;
		
		int maxKNN = 10;
		// NOTE: computeANN needs point vectors as columns, so need to create column major ordering
		// and transposed from normal arrangement!
		Eigen::ArrayXXi idxs(maxKNN, seed_subset.size());
		Eigen::ArrayXXd statDists(maxKNN, seed_subset.size());
		
		MAPA::UtilityCalcs::computeANN(X.transpose(), seed_subset, idxs, statDists, 0.0);
		
		// Get data back in usual orientation and convert squared to actual distances
		idxs.transposeInPlace();
		statDists.transposeInPlace();
		statDists = statDists.sqrt();
		
		std::cout << idxs << std::endl;
		std::cout << statDists << std::endl;

    return 0;
}
