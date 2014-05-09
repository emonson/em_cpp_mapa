#include <Eigen/Core>
#include <igl/readDMAT.h>
#include <igl/cat.h>
#include <iostream>
#include <vector>
#include <string>
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

    // Read in good seed ponits
    Eigen::ArrayXi GoodSeedPoints;
    std::cout << "Reading in Artifical test data (rev1) matlab-generated EstDims" << std::endl;
    igl::readDMAT( "/Users/emonson/Programming/em_cpp_mapa/data/artdat_rev1_lmsvd_out_goodseeds.dmat", GoodSeedPoints );

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
    igl::cat(2, AllGoodScales.matrix(), lmsvd.GetAllGoodScales().matrix(), scales_comparison);
    
    std::cout << "lmsvd" << std::endl;
    std::cout << scales_comparison << std::endl << std::endl;
    
    std::cout << "estimated dimensionalities" << std::endl;
    std::cout << AllEstDims << std::endl << std::endl;
    std::cout << lmsvd.GetAllEstimatedDims().transpose() << std::endl << std::endl;
    
    std::cout << "good seed points (1st matlab, which are 1s-based)" << std::endl;
    std::cout << GoodSeedPoints.transpose() << std::endl;
    std::cout << lmsvd.GetGoodSeedPoints().transpose() << std::endl << std::endl;

    // http://stackoverflow.com/questions/409348/iteration-over-vector-in-c
    // 
    // Using std::vector
    // 
    // Using iterators
    // 
    // for(std::vector<T>::iterator it = v.begin(); it != v.end(); ++it) {
    //     /* std::cout << *it; ... */
    // }
    // 
    // Using indices
    // 
    // for(std::vector<int>::size_type i = 0; i != v.size(); i++) {
    //     /* std::cout << someVector[i]; ... */
    // }
    
    std::vector<ArrayXi> goodRegions = lmsvd.GetGoodLocalRegions();
    
    int file_idx = 1;
    std::stringstream locreg_file;
    ArrayXXi matGoodLocalRegion;
    std::cout << "good local regions (NN idxs -- 1st matlab, which are 1s-based)" << std::endl;
    for (std::vector<ArrayXi>::iterator it = goodRegions.begin(); it != goodRegions.end(); ++it)
    {
        // std::cout << "Reading in seed " << i_seed << " spectra for test data (rev1)" << std::endl;
        locreg_file.str("");
        locreg_file << "/Users/emonson/Programming/em_cpp_mapa/data/lmsvd_out/artdat_rev1_lmsvd_out_goodlocreg_" << file_idx << ".dmat";
        igl::readDMAT( locreg_file.str().c_str(), matGoodLocalRegion);
        std::cout << "* " << matGoodLocalRegion << std::endl;
        file_idx++;
        
        std::cout << "- " << (*it).transpose() << std::endl;
    }

    return 0;
}
