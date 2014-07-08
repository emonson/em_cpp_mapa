#include <Eigen/Core>
#include <igl/readDMAT.h>
#include <iostream>
#include <stdio.h> // printf

#include "mapa_config.h"
#include "UtilityCalcs.h"
#include "Options.h"
#include "Mapa.h"

#include <stdlib.h>             /* NULL */
#include <time.h>               /* time */
#include <stdio.h>              /* srand */


int main(int argc, char * argv[])
{
    // Read in test data
	std::string data_dir = MAPA::UtilityCalcs::PathAppend(MAPA_SOURCE_DIR, "data");

    Eigen::ArrayXXd X;
    std::cout << "Reading in Face clustering data (svd to 30d)" << std::endl;
	std::string data_file = MAPA::UtilityCalcs::PathAppend(data_dir, "face_clustering_svd_in_data.dmat");
    igl::readDMAT( data_file, X );
		
    // Read in true labels
    Eigen::ArrayXXi true_labels_in;
    std::cout << "Reading in true labels for test data (rev1)" << std::endl;
	data_file = MAPA::UtilityCalcs::PathAppend(data_dir, "face_clustering_svd_in_labels.dmat");
    igl::readDMAT( data_file, true_labels_in);
    Eigen::ArrayXi true_labels = true_labels_in.row(0);
	std::cout << true_labels.transpose() << std::endl;

    // NOTE: seeds is matlab-style 1s index!!
    true_labels -= 1;
    
    // Reseed random number generator since Eigen Random.h doesn't do this itself
    srand( (unsigned int)time(NULL) );

    // opts = struct('dmax',3, 'Kmax',15, 'n0',640, 'plotFigs',true);
    MAPA::Opts opts;
    opts.dmax = 3;
    opts.Kmax = 15;
    opts.n0 = 10;
    
    opts.SetDefaults(X);
    std::cout << "options" << std::endl;
    std::cout << opts << std::endl;
        
    clock_t t = clock();
    MAPA::Mapa mapa(X, opts);
    t = clock() - t;
    std::cout << std::endl << "Mapa labels:" << std::endl;
    std::cout << mapa.GetLabels().transpose() << std::endl;
    std::cout << std::endl << "Mapa plane dims:" << std::endl;
    std::cout << mapa.GetPlaneDims().transpose() << std::endl;
    std::cout << std::endl << "Mapa disance-based error:" << std::endl;
    std::cout << mapa.GetDistanceError() << std::endl << std::endl;

    printf("Elapsed time: %.10f sec.\n", (double)t/CLOCKS_PER_SEC );

    double MisclassificationRate = MAPA::UtilityCalcs::ClusteringError(mapa.GetLabels(), true_labels);
    
    t = clock() - t;
    printf("Elapsed time: %.10f sec.for %ld d result\n", (double)t/CLOCKS_PER_SEC, mapa.GetPlaneDims().size() );
    printf("Misclassification Rate: %.10f\n", MisclassificationRate );

    return 0;
}
