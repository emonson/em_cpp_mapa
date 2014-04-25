#include <Eigen/Core>
#include <igl/readDMAT.h>
#include <cstdio>
#include "Options.h"

int main(int argc, char * argv[])
{
    // Read in test data
    Eigen::ArrayXXd X;
    std::cout << "Reading in Artifical 3D test data (rev1)" << std::endl;
    igl::readDMAT( "/Users/emonson/Programming/em_cpp_mapa/data/artificial_data_rev1.dmat", X );
		
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
