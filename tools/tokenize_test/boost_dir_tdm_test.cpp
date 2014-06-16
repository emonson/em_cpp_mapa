#include <iostream>
#include <string>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "mapa_config.h"
#include "UtilityCalcs.h"
#include "BoostDIRtokenizer.h"

int main( int argc, const char** argv )
{
	std::string data_dir = MAPA::UtilityCalcs::PathAppend(MAPA_SOURCE_DIR, "data");
    std::string dirname = MAPA::UtilityCalcs::PathAppend(data_dir, "SNData");

    int min_term_length = 2;
    int min_term_count = 2;
	MAPA::BoostDIRtokenizer dir_tok(dirname, min_term_length, min_term_count);
    
    Eigen::SparseMatrix<double,0,long> tdm = dir_tok.getTDM();
    
    std::cout << "TDM: " << tdm.rows() << " x " << tdm.cols() << ", " << tdm.nonZeros();
    std::cout << " = " << (double)tdm.nonZeros()/(double)(tdm.rows()*tdm.cols()) << std::endl << std::endl;

    return EXIT_SUCCESS;
}
