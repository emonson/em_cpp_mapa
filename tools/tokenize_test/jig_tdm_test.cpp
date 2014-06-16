#include <iostream>
#include <string>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "mapa_config.h"
#include "UtilityCalcs.h"
#include "TDMgenerator.h"
#include "JIGtokenizer.h"

int main( int argc, const char** argv )
{
	std::string data_dir = MAPA::UtilityCalcs::PathAppend(MAPA_SOURCE_DIR, "data");
    std::string filename = MAPA::UtilityCalcs::PathAppend(data_dir, "InfovisVAST-papers.jig");

    int min_term_length = 3;
    int min_term_count = 5;
    MAPA::TDMgenerator tdm_gen(min_term_length, min_term_count);
	MAPA::JIGtokenizer jig_tok(filename, &tdm_gen);
    
    Eigen::SparseMatrix<double,0,long> tdm = tdm_gen.getTDM();
    
    std::cout << "TDM: " << tdm.rows() << " x " << tdm.cols() << ", " << tdm.nonZeros() << std::endl << std::endl;

    return EXIT_SUCCESS;
}
