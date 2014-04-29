#include <Eigen/Core>
#include <igl/readDMAT.h>
#include <cstdio>
#include "NRsearch.h"
// #include <time.h>

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
    
    // Compute ANNs of seed points
    int maxKNN = 10;
    double eps = 0.0;
    ArrayXi seed_subset = seeds.head(5);
    
    // clock_t t1 = clock();

    // ANN calc object
    MAPA::NRsearch ann(X);
    
    ann.computeANN(seed_subset, maxKNN, eps);
    
    // clock_t t2 = clock();
    // std::cout << (t2-t1)/(double)CLOCKS_PER_SEC << std::endl;

    std::cout << seed_subset.transpose() << std::endl;
    std::cout << ann.GetIdxs() << std::endl;
    std::cout << ann.GetDistances() << std::endl;

    // Compute ANNs of a different set of seed points
    maxKNN = 4;
    seed_subset = seeds.tail(10);
    std::cout << seed_subset.transpose() << std::endl;
    
    ann.computeANN(seed_subset, maxKNN, eps);
    
    std::cout << ann.GetIdxs() << std::endl;
    std::cout << ann.GetDistances() << std::endl;

    return 0;
}
