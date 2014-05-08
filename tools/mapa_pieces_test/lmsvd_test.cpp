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
		
    // Read in real lmsvd all good seeds
    Eigen::ArrayXXi AllGoodScales;
    std::cout << "Reading in Artifical test data (rev1) matlab-generated GoodScales" << std::endl;
    igl::readDMAT( "/Users/emonson/Programming/em_cpp_mapa/data/artdat_rev1_lmsvd_allgoodscales.dmat", AllGoodScales );

    // Read in real lmsvd all est dims
    // since it's a row vector, reading in as XXi
    Eigen::ArrayXXi AllEstDims;
    std::cout << "Reading in Artifical test data (rev1) matlab-generated EstDims" << std::endl;
    igl::readDMAT( "/Users/emonson/Programming/em_cpp_mapa/data/artdat_rev1_lmsvd_allestdims.dmat", AllEstDims );

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
        
    MAPA::LMsvd lmsvd(X, opts);
    
    MatrixXi scales_comparison;
    int largest_dim = AllGoodScales.rows() > lmsvd.GetAllGoodScales().rows() ? AllGoodScales.rows() : lmsvd.GetAllGoodScales().rows();
    scales_comparison.resize(largest_dim, 4);
    scales_comparison << AllGoodScales.matrix(), lmsvd.GetAllGoodScales().matrix();
    
    std::cout << "lmsvd" << std::endl;
    std::cout << scales_comparison << std::endl << std::endl;
    
    std::cout << AllEstDims << std::endl << std::endl;
    std::cout << lmsvd.GetAllEstimatedDims().transpose() << std::endl << std::endl;

    return 0;
}
