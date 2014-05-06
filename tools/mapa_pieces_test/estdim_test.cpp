#include <Eigen/Core>
#include <igl/readDMAT.h>
#include <cstdio>
#include "EstimateDimFromSpectra.h"
// #include <time.h>

int main(int argc, char * argv[])
{
    double alpha0 = 0.3;
    MAPA::EstimateDimFromSpectra estdim;
    
    // Read in distances
    Eigen::ArrayXd Delta_1;
    std::cout << "Reading in Artifical test data (rev1) seed 1 scale distances" << std::endl;
    igl::readDMAT( "/Users/emonson/Programming/em_cpp_mapa/data/artdat_rev1_estdim_delta1.dmat", Delta_1 );
    
    // Read in spectra
    Eigen::ArrayXXd NetS_1;
    std::cout << "Reading in seed 1 spectra for test data (rev1)" << std::endl;
    igl::readDMAT( "/Users/emonson/Programming/em_cpp_mapa/data/artdat_rev1_estdim_s1.dmat", NetS_1);
		
    // NOTE: i_seed is matlab-style 1s index!!
    
		// i_seed == 1 (matlab idx)
		//    GoodScales: [9 15]
		//         DimEst: 2

    estdim.EstimateDimensionality(Delta_1, NetS_1, alpha0);
    
    std::cout << estdim.GetDimension() << std::endl;
    std::cout << estdim.GetLowerScaleIdx() << std::endl;
    std::cout << estdim.GetUpperScaleIdx() << std::endl;

    // Read in distances
    Eigen::ArrayXd Delta_60;
    std::cout << "Reading in Artifical test data (rev1) seed 60 scale distances" << std::endl;
    igl::readDMAT( "/Users/emonson/Programming/em_cpp_mapa/data/artdat_rev1_estdim_delta60.dmat", Delta_60 );
    
    // Read in spectra
    Eigen::ArrayXXd NetS_60;
    std::cout << "Reading in seed 60 spectra for test data (rev1)" << std::endl;
    igl::readDMAT( "/Users/emonson/Programming/em_cpp_mapa/data/artdat_rev1_estdim_s60.dmat", NetS_60);
		
		// i_seed == 60 (matlab idx)
		//     GoodScales: [1 50]
		//         DimEst: 3

    estdim.EstimateDimensionality(Delta_60, NetS_60, alpha0);
    
    std::cout << estdim.GetDimension() << std::endl;
    std::cout << estdim.GetLowerScaleIdx() << std::endl;
    std::cout << estdim.GetUpperScaleIdx() << std::endl;

    return 0;
}
