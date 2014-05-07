#include <Eigen/Core>
#include <igl/readDMAT.h>
#include <cstdio>
#include <string>
#include "EstimateDimFromSpectra.h"
// #include <time.h>

int main(int argc, char * argv[])
{
    double alpha0 = 0.212132034355964227412272293804562650620937347412109375;
    MAPA::EstimateDimFromSpectra estdim;
    
    // -------------------------------------
    // Read in distances
    Eigen::ArrayXd Delta_1;
    std::cout << "Reading in Artifical test data (rev1) seed 1 scale distances" << std::endl;
    igl::readDMAT( "/Users/emonson/Programming/em_cpp_mapa/data/artdat_rev1_estdim_delta1.dmat", Delta_1 );
    
    // Read in real lmsvd all good seeds
    Eigen::ArrayXXd AllGoodScales;
    std::cout << "Reading in Artifical test data (rev1) seed 1 scale distances" << std::endl;
    igl::readDMAT( "/Users/emonson/Programming/em_cpp_mapa/data/artdat_rev1_lmsvd_allgoodscales.dmat", AllGoodScales );

    // Read in real lmsvd all est dims
    Eigen::ArrayXi AllEstDims;
    std::cout << "Reading in Artifical test data (rev1) seed 1 scale distances" << std::endl;
    igl::readDMAT( "/Users/emonson/Programming/em_cpp_mapa/data/artdat_rev1_lmsvd_allestdims.dmat", AllEstDims );

    // Spectra
    Eigen::ArrayXXd NetS;
    
    for (int i_seed = 1; i_seed <= 60; i_seed++)
    {
        std::cout << "Reading in seed " << i_seed << " spectra for test data (rev1)" << std::endl;
        igl::readDMAT( "/Users/emonson/Programming/em_cpp_mapa/data/artdat_rev1_estdim_s1.dmat", NetS);
        
        estdim.EstimateDimensionality(Delta_1, NetS_1, alpha0);
    
        std::cout << "dim = 2, " << estdim.GetDimension() << std::endl;
        std::cout << "lo = 9, " << estdim.GetLowerScaleIdx() << std::endl;
        std::cout << "hi = 15, " << estdim.GetUpperScaleIdx() << std::endl;
    }

    return 0;
}
