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
		
		seeds = seeds.head(5);
		
		std::cout << seeds << std::endl;
		
		// Testing seeds slice into X
// 		Eigen::ArrayXXd B;
// 		igl::slice(X, seeds, Eigen::ArrayXi::LinSpaced(X.cols(), 0, X.cols()-1), B);
// 		
// 		std::cout << B << std::endl;
		
		int maxKNN = 10;
		Eigen::ArrayXXi idxs(seeds.size(), maxKNN);
		Eigen::ArrayXXd statDists(seeds.size(), maxKNN);
		
		MAPA::UtilityCalcs::nrsearch(X, seeds, maxKNN, idxs, statDists);
		
		std::cout << idxs << std::endl;
		std::cout << statDists << std::endl;

    return 0;
}
