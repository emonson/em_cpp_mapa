#include <Eigen/Core>
#include <igl/readDMAT.h>
#include <iostream>
#include "Options.h"
#include "Mapa.h"

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
		
    // Read in seed points
    Eigen::ArrayXXi aprioriSampleLabels;
    std::cout << "Reading in apriori sample labels for test data (rev1)" << std::endl;
    igl::readDMAT( "/Users/emonson/Programming/em_cpp_mapa/data/artificial_aprioriLabels_rev1.dmat", aprioriSampleLabels);
	
    // NOTE: seeds is matlab-style 1s index!!
    seeds -= 1;
    
    Array3i aSubDims;
    aSubDims << 1, 2, 1;

    int K = aSubDims.size();

    MAPA::Opts opts;
    opts.n0 = 20*K;
    opts.dmax = aSubDims.maxCoeff();
    opts.Kmax = 2*K;
    // opts.K = 3;
    opts.seeds = seeds;
    // opts.discardCols = 0.2;
    // opts.discardRows = 0.2;
    
    opts.SetDefaults(X);
    std::cout << "options" << std::endl;
    std::cout << opts << std::endl;
        
    MAPA::Mapa mapa(X, opts);
    
    std::cout << std::endl << "Mapa labels:" << std::endl;
    std::cout << mapa.GetLabels().transpose() << std::endl;
    std::cout << std::endl << "Mapa plane dims:" << std::endl;
    std::cout << mapa.GetPlaneDims().transpose() << std::endl;
    std::cout << std::endl << "Mapa disance-based error:" << std::endl;
    std::cout << mapa.GetDistanceError() << std::endl;

    double MisclassificationRate = MAPA::UtilityCalcs::ClusteringError(mapa.GetLabels(), aprioriSampleLabels);

    std::cout << std::endl << "Mapa misclassification rate:" << std::endl;
    std::cout << MisclassificationRate << std::endl;

    return 0;
}
