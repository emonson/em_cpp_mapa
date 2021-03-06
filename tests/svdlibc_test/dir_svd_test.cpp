#include <iostream>
#include <string>
#include <time.h>

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/SparseCore>

#include "mapa_config.h"
#include "UtilityCalcs.h"
#include "TDMgenerator.h"
#include "DIRtokenizer.h"
#include "SvdlibcSVD.h"

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
    
    // SVDLIBC sparse SVD
    int rank = 5;
    clock_t t = clock();
    
    MAPA::SvdlibcSVD svds(tdm, rank);
    
    t = clock() - t;
    printf("SVD sparse Elapsed time: %.10f sec.\n", (double)t/CLOCKS_PER_SEC );
    
    std::cout << "U: " << svds.matrixU().rows() << " x " << svds.matrixU().cols() << std::endl;
    // std::cout << svds.matrixU() << std::endl << std::endl;
    std::cout << "V: " << svds.matrixV().rows() << " x " << svds.matrixV().cols() << std::endl;
    // std::cout << svds.matrixV() << std::endl << std::endl;
    std::cout << "S" << std::endl;
    std::cout << svds.singularValues().transpose() << std::endl << std::endl;

    Eigen::MatrixXd Xred = svds.matrixV() * svds.singularValues().asDiagonal();
    
    std::cout << "Xred: " << Xred.rows() << " x " << Xred.cols() << std::endl;

    // SVDLIBC dense SVD
    Eigen::MatrixXd tdm_dense = tdm;
    t = clock();

    MAPA::SvdlibcSVD svdsd(tdm_dense, rank);

    t = clock() - t;
    printf("SVD dense Elapsed time: %.10f sec.\n", (double)t/CLOCKS_PER_SEC );
    
    std::cout << "U: " << svdsd.matrixU().rows() << " x " << svdsd.matrixU().cols() << std::endl;
    // std::cout << svdsd.matrixU() << std::endl << std::endl;
    std::cout << "V: " << svdsd.matrixV().rows() << " x " << svdsd.matrixV().cols() << std::endl;
    // std::cout << svdsd.matrixV() << std::endl << std::endl;
    std::cout << "S" << std::endl;
    std::cout << svdsd.singularValues().head(rank).transpose() << std::endl << std::endl;

    Eigen::MatrixXd Xred_d = svdsd.matrixV() * svdsd.singularValues().asDiagonal();
    
    std::cout << "Xred_d: " << Xred_d.rows() << " x " << Xred_d.cols() << std::endl;
    
    // Eigen standard SVD
    t = clock();
    
    JacobiSVD<MatrixXd> svd_e(tdm_dense, Eigen::ComputeThinU | Eigen::ComputeThinV);
    
    t = clock() - t;
    printf("JacobiSVD Elapsed time: %.10f sec.\n", (double)t/CLOCKS_PER_SEC );
    
    std::cout << "U: " << svd_e.matrixU().rows() << " x " << svd_e.matrixU().cols() << std::endl;
    // std::cout << svd_e.matrixU() << std::endl << std::endl;
    std::cout << "V: " << svd_e.matrixV().rows() << " x " << svd_e.matrixV().cols() << std::endl;
    // std::cout << svd_e.matrixV() << std::endl << std::endl;
    std::cout << "S" << std::endl;
    std::cout << svd_e.singularValues().head(rank).transpose() << std::endl << std::endl;

    Eigen::MatrixXd Xred_e = svd_e.matrixV().leftCols(rank) * svd_e.singularValues().head(rank).asDiagonal();
    
    std::cout << "Xred_e: " << Xred_e.rows() << " x " << Xred_e.cols() << std::endl;
    
    
    return EXIT_SUCCESS;
}
