#include <Eigen/Core>
#include <igl/readDMAT.h>
#include <cstdio>
#include <string>

#include "mapa_config.h"
#include "UtilityCalcs.h"
#include "EstimateDimFromSpectra.h"
#include "Options.h"

// #include <time.h>

int main(int argc, char * argv[])
{
    MAPA::Opts opts;
    opts.alpha0 = 0.212132034355964227412272293804562650620937347412109375;
    std::stringstream spectra_file;
    std::string d_off, lo_off, hi_off;
    MAPA::EstimateDimFromSpectra estdim;
    
    // -------------------------------------
    // Read in distances
	std::string data_dir = MAPA::UtilityCalcs::PathAppend(MAPA_SOURCE_DIR, "data");

    Eigen::ArrayXd Delta_1;
    std::cout << "Reading in Artifical test data (rev1) scale distances" << std::endl;
	std::string data_file = MAPA::UtilityCalcs::PathAppend(data_dir, "artdat_rev1_estdim_delta1.dmat");
    igl::readDMAT( data_file, Delta_1 );
    
    // Read in real lmsvd all good seeds
    Eigen::ArrayXXd AllGoodScales;
    std::cout << "Reading in Artifical test data (rev1) matlab-generated GoodScales" << std::endl;
	data_file = MAPA::UtilityCalcs::PathAppend(data_dir, "artdat_rev1_lmsvd_allgoodscales.dmat");
    igl::readDMAT( data_file, AllGoodScales );

    // Read in real lmsvd all est dims
    Eigen::ArrayXXi AllEstDims;
    std::cout << "Reading in Artifical test data (rev1) matlab-generated EstDims" << std::endl;
	data_file = MAPA::UtilityCalcs::PathAppend(data_dir, "artdat_rev1_lmsvd_allestdims.dmat");
    igl::readDMAT( data_file, AllEstDims );

    // Spectra
    Eigen::ArrayXXd NetS;
    
    std::cout << "i \td    \t\tlo    \t\thi" << std::endl;
    std::cout << "——\t—————\t\t——————\t\t——————" << std::endl;
    
	std::string lmout_data_dir = MAPA::UtilityCalcs::PathAppend(MAPA_SOURCE_DIR, "data");
	lmout_data_dir = MAPA::UtilityCalcs::PathAppend(lmout_data_dir, "lmsvd_out");
	lmout_data_dir = MAPA::UtilityCalcs::PathAppend(lmout_data_dir, "artdat_rev1_lmsvd_mid_spectra");

    for (int i_seed = 1; i_seed <= 60; i_seed++)
    {
        // std::cout << "Reading in seed " << i_seed << " spectra for test data (rev1)" << std::endl;
        spectra_file.str("");
        spectra_file << lmout_data_dir << i_seed << ".dmat";
        igl::readDMAT( spectra_file.str().c_str(), NetS);
        
        estdim.EstimateDimensionality(Delta_1, NetS, opts);
        
        d_off = AllEstDims(i_seed-1) == estdim.GetDimension() ? "  ": " *";
        lo_off = AllGoodScales(i_seed-1,0) == estdim.GetLowerScaleIdx() ? "  ": " *";
        hi_off = AllGoodScales(i_seed-1,1) == estdim.GetUpperScaleIdx() ? "  ": " *";
    
        std::cout << i_seed;
        std::cout << "\t" << AllEstDims(i_seed-1) << " " << estdim.GetDimension() << d_off;
        std::cout << "\t\t" << AllGoodScales(i_seed-1,0) << " " << estdim.GetLowerScaleIdx() << lo_off;
        std::cout << "\t\t" << AllGoodScales(i_seed-1,1) << " " << estdim.GetUpperScaleIdx() << hi_off << std::endl;
    }

    return 0;
}
