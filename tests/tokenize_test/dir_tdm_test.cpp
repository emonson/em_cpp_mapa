#include <iostream>
#include <string>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "mapa_config.h"
#include "UtilityCalcs.h"
#include "DIRtokenizer.h"
#include "TDMgenerator.h"

int main( int argc, const char** argv )
{
	std::string data_dir = MAPA::UtilityCalcs::PathAppend(MAPA_SOURCE_DIR, "data");
    std::string dirname = MAPA::UtilityCalcs::PathAppend(data_dir, "SNData");

    int min_term_length = 3;
    int min_term_count = 5;
    MAPA::TDMgenerator tdm_gen(min_term_length, min_term_count);
    tdm_gen.addStopwords("science news year years work university");
    
	MAPA::DIRtokenizer dir_tok(dirname, &tdm_gen);
    
    Eigen::SparseMatrix<double,0,long> tdm = tdm_gen.getTDM();
    
    std::cout << "TDM: " << tdm.rows() << " x " << tdm.cols() << ", " << tdm.nonZeros() << std::endl << std::endl;

    return EXIT_SUCCESS;
}
