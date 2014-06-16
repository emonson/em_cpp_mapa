#include <Eigen/Core>
#include <igl/readDMAT.h>
#include <iostream>
#include <string>
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
    std::stringstream data_file;
	std::string data_dir = MAPA::UtilityCalcs::PathAppend(MAPA_SOURCE_DIR, "data");
    std::string data_start_str = MAPA::UtilityCalcs::PathAppend(data_dir, "MotionSegmentation_set");
    
    for (int ss = 1; ss <= 3; ss++)
    {
        // Read in test data
        Eigen::ArrayXXd X;
        std::cout << std::endl << "# -------------------------" << std::endl;
        std::cout << "Reading in Motion segmentation set " << ss << " data" << std::endl;
        data_file.str("");
        data_file << data_start_str << ss << "_data_10d.dmat";
        igl::readDMAT( data_file.str().c_str(), X);
    
        // Read in true labels
        Eigen::ArrayXXi true_labels_in;
        std::cout << "Reading in Motion segmentation set " << ss << " labels" << std::endl;
        data_file.str("");
        data_file << data_start_str << ss << "_labels.dmat";
        igl::readDMAT( data_file.str().c_str(), true_labels_in);
        Eigen::ArrayXi true_labels = true_labels_in.col(0);

        // NOTE: seeds is matlab-style 1s index!!
        true_labels -= 1;

        // Reseed random number generator since Eigen Random.h doesn't do this itself
        srand( (unsigned int)time(NULL) );

        // opts = struct('dmax',3, 'Kmax',5, 'n0',N, 'plotFigs',true);
        MAPA::Opts opts;
        opts.dmax = 3;
        opts.Kmax = 5;
        opts.n0 = X.rows();

        opts.SetDefaults(X);
        // std::cout << "options" << std::endl;
        // std::cout << opts << std::endl;
    
        clock_t t = clock();
        MAPA::Mapa mapa(X, opts);
        t = clock() - t;
        
        // kanatani1
        // L2Error 0.0040595
        // TimeUsed: 2.5056
        // PlaneDims: 2  2
        // MisclassificationRate: 0
        // 
        // kanatani2
        // L2Error 0.0067473
        // TimeUsed: 2.024
        // PlaneDims: 2  2
        // MisclassificationRate: 0
        // 
        // kanatani3
        // L2Error 0.05947
        // TimeUsed: 2.1978
        // PlaneDims: 2  2
        // MisclassificationRate: 0.10959
        
        ArrayXi inferred_labels = mapa.GetLabels();
        
        // std::cout << std::endl << "Mapa labels:" << std::endl;
        // std::cout << inferred_labels.transpose() << std::endl;
        std::cout << std::endl << "Mapa plane dims:" << std::endl;
        std::cout << mapa.GetPlaneDims().transpose() << std::endl;
        std::cout << std::endl << "Mapa disance-based error:" << std::endl;
        std::cout << mapa.GetDistanceError() << std::endl << std::endl;

        printf("Elapsed time: %.10f sec.\n", (double)t/CLOCKS_PER_SEC );

        double MisclassificationRate = MAPA::UtilityCalcs::ClusteringError(inferred_labels, true_labels);

        printf("Misclassification Rate: %.10f\n", MisclassificationRate );
    }

    return 0;
}
