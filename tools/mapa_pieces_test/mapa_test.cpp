#include <Eigen/Core>
#include <igl/readDMAT.h>
#include <iostream>
#include <stdio.h> // printf
#include "Options.h"
#include "Mapa.h"

#include <stdlib.h>             /* NULL */
#include <time.h>               /* time */
#include <stdio.h>              /* srand */


int main(int argc, char * argv[])
{
    // Read in test data
    Eigen::ArrayXXd X;
    std::cout << "Reading in Artifical 3D test data (rev1)" << std::endl;
    igl::readDMAT( "/home/emonson/Programming/em_cpp_mapa/data/artificial_data_rev1.dmat", X );
		
    // Read in seed points
    Eigen::ArrayXXi seeds_in;
    std::cout << "Reading in seed points for test data (rev1)" << std::endl;
    igl::readDMAT( "/home/emonson/Programming/em_cpp_mapa/data/artdat_rev1_nrsearch_seeds.dmat", seeds_in);
    Eigen::ArrayXi seeds = seeds_in.col(0);

    // Read in seed points
    Eigen::ArrayXXi aprioriSampleLabels_in;
    std::cout << "Reading in apriori sample labels for test data (rev1)" << std::endl;
    igl::readDMAT( "/home/emonson/Programming/em_cpp_mapa/data/artificial_aprioriLabels_rev1.dmat", aprioriSampleLabels_in);
    Eigen::ArrayXi aprioriSampleLabels = aprioriSampleLabels_in.row(0);

    // NOTE: seeds is matlab-style 1s index!!
    seeds -= 1;
    aprioriSampleLabels -= 1;
    
    Array3i aSubDims;
    aSubDims << 1, 2, 1;

    int K = aSubDims.size();

    // Reseed random number generator since Eigen Random.h doesn't do this itself
    srand( (unsigned int)time(NULL) );

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
        
    clock_t t = clock();
    MAPA::Mapa mapa(X, opts);
    ArrayXi inferred_labels = mapa.GetLabels();
    double MisclassificationRate = MAPA::UtilityCalcs::ClusteringError(inferred_labels, aprioriSampleLabels);
    t = clock() - t;
    
    std::cout << std::endl << "Mapa labels:" << std::endl;
    std::cout << mapa.GetLabels().transpose() << std::endl;
    std::cout << std::endl << "Mapa plane dims:" << std::endl;
    std::cout << mapa.GetPlaneDims().transpose() << std::endl;
    std::cout << std::endl << "Mapa disance-based error:" << std::endl;
    std::cout << mapa.GetDistanceError() << std::endl << std::endl;

    printf("Misclassification Rate: %.10f\n", MisclassificationRate );
    printf("Elapsed time: %.10f sec.\n", (double)t/CLOCKS_PER_SEC );

    return 0;
}
