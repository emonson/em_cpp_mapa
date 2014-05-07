#include <Eigen/Core>
#include <igl/readDMAT.h>
#include <iostream>
#include "Options.h"
#include "EstimateDimFromSpectra.h"
#include "LMsvd.h"

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
    
    Array3i aSubDims;
    aSubDims << 1, 2, 1;

    int K = aSubDims.size();

    MAPA::Opts opts;
    opts.n0 = 20*K;
    opts.dmax = aSubDims.maxCoeff();
    opts.Kmax = 2*K;
    opts.seeds = seeds;
    
    opts.SetDefaults(X);
    
    // std::cout << opts << std::endl;
    
    MAPA::LMsvd lmsvd(X, opts);
    
    std::cout << "lmsvd" << std::endl;
    // std::cout << lmsvd.GetGoodLocalRegions().at(0) << std::endl;
    std::cout << lmsvd.GetGoodSeedPoints() << std::endl;
    std::cout << lmsvd.GetEstimatedDims() << std::endl;

    return 0;
}
