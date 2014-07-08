#include <Eigen/Core>
#include <igl/readDMAT.h>
#include <cstdio>

#include "mapa_config.h"
#include "UtilityCalcs.h"
#include "Options.h"

int main(int argc, char * argv[])
{
	std::string data_dir = MAPA::UtilityCalcs::PathAppend(MAPA_SOURCE_DIR, "data");

    // Read in test data
    Eigen::ArrayXXd X;
    std::cout << "Reading in Artifical 3D test data (rev1)" << std::endl;
	std::string data_file = MAPA::UtilityCalcs::PathAppend(data_dir, "artificial_data_rev1.dmat");
    igl::readDMAT( data_file, X );
		
		Array3i aSubDims;
		aSubDims << 1, 2, 1;
		
		int K = aSubDims.size();
		
    MAPA::Opts opts1;
    opts1.n0 = 20*K;
    opts1.dmax = aSubDims.maxCoeff();
    opts1.Kmax = 2*K;
    
    opts1.SetDefaults(X);
    
    std::cout << opts1;

    return 0;
}
