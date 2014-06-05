#include <iostream>
#include <string>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "DIRtokenizer.h"

int main( int argc, const char** argv )
{
    #if defined( _MSC_VER ) && defined( DEBUG )
        _CrtMemCheckpoint( &startMemState );
    #endif
    
    std::string dirname = "/Users/emonson/Data/Fodava/EMoDocDataSets/SNData";

    int min_term_length = 3;
    int min_term_count = 5;
	MAPA::DIRtokenizer dir_tok(dirname, min_term_length, min_term_count);
    
    Eigen::SparseMatrix<double,0,long> tdm = dir_tok.getTDM();
    
    std::cout << "TDM: " << tdm.rows() << " x " << tdm.cols() << ", " << tdm.nonZeros() << std::endl << std::endl;

    return EXIT_SUCCESS;
}
