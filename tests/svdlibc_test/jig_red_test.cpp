#include <iostream>
#include <string>
#include <time.h>

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/SparseCore>

#include "mapa_config.h"
#include "UtilityCalcs.h"
#include "TDMgenerator.h"
#include "JIGtokenizer.h"
#include "SvdlibcSVD.h"

#include "redsvd.hpp"
#include "EigenRandomSVD.h"

int main( int argc, const char** argv )
{
	std::string data_dir = MAPA::UtilityCalcs::PathAppend(MAPA_SOURCE_DIR, "data");
	std::string filename = MAPA::UtilityCalcs::PathAppend(data_dir, "InfovisVAST-papers.jig");

    int min_term_length = 3;
    int min_term_count = 5;
    MAPA::TDMgenerator tdm_gen(min_term_length, min_term_count);
	MAPA::JIGtokenizer jig_tok(filename, &tdm_gen);
    
    Eigen::SparseMatrix<double,0,long> tdm = tdm_gen.getTDM();
    
    std::cout << std::endl << "TDM (sparse input matrix): " << tdm.rows() << " x " << tdm.cols() << ", " << tdm.nonZeros() << " nonzeros" << std::endl << std::endl;
    
    int rank = 5;

    // --------------------------------
    // SVDLIBC (sparse)
    
    clock_t t = clock();    
    MAPA::SvdlibcSVD svds(tdm, rank);
    
    t = clock() - t;
    printf("SVDLIBC Elapsed time: %.10f sec.\n", (double)t/CLOCKS_PER_SEC );
    
    std::cout << "U: " << svds.matrixU().rows() << " x " << svds.matrixU().cols() << std::endl;
    // std::cout << svds.matrixU() << std::endl << std::endl;
    std::cout << "V: " << svds.matrixV().rows() << " x " << svds.matrixV().cols() << std::endl;
    // std::cout << svds.matrixV() << std::endl << std::endl;
    std::cout << "S: ";
    std::cout << svds.singularValues().transpose() << std::endl;

    Eigen::MatrixXd Xred = svds.matrixV() * svds.singularValues().asDiagonal();
    std::cout << "X reduced: " << Xred.rows() << " x " << Xred.cols() << std::endl << std::endl;

    // --------------------------------
    // Eigen standard JacobiSVD (dense)
    
    Eigen::MatrixXd tdm_dense = tdm;
    t = clock();
    JacobiSVD<MatrixXd> svd_e(tdm_dense, Eigen::ComputeThinU | Eigen::ComputeThinV);
    t = clock() - t;
    printf("Eigen standard JacobiSVD Elapsed time: %.10f sec.\n", (double)t/CLOCKS_PER_SEC );
    
    std::cout << "U: " << svd_e.matrixU().rows() << " x " << svd_e.matrixU().cols() << std::endl;
    // std::cout << svd_e.matrixU() << std::endl << std::endl;
    std::cout << "V: " << svd_e.matrixV().rows() << " x " << svd_e.matrixV().cols() << std::endl;
    // std::cout << svd_e.matrixV() << std::endl << std::endl;
    std::cout << "S: ";
    std::cout << svd_e.singularValues().head(rank).transpose() << std::endl;

    Eigen::MatrixXd Xred_e = svd_e.matrixV().leftCols(rank) * svd_e.singularValues().head(rank).asDiagonal();
    std::cout << "X reduced Eigen: " << Xred_e.rows() << " x " << Xred_e.cols() << std::endl << std::endl;
    
    // --------------------------------
    // RedSVD test (row-major sparse)
    
    REDSVD::SMatrixXf tdm_r = tdm.cast<float>();

    t = clock();
    REDSVD::RedSVD svd_r(tdm_r, rank);
    t = clock() - t;
    printf("RedSVD Elapsed time: %.10f sec.\n", (double)t/CLOCKS_PER_SEC );

    std::cout << "U: " << svd_r.matrixU().rows() << " x " << svd_r.matrixU().cols() << std::endl;
    // std::cout << svd_r.matrixU() << std::endl << std::endl;
    std::cout << "V: " << svd_r.matrixV().rows() << " x " << svd_r.matrixV().cols() << std::endl;
    // std::cout << svd_r.matrixV() << std::endl << std::endl;
    std::cout << "S: ";
    std::cout << svd_r.singularValues().head(rank).transpose() << std::endl;

    Eigen::MatrixXf Xred_r = svd_r.matrixV().leftCols(rank) * svd_r.singularValues().head(rank).asDiagonal();
    std::cout << "X reduced redsvd: " << Xred_r.rows() << " x " << Xred_r.cols() << std::endl << std::endl;
    
    // --------------------------------
    // Eigen random SVD (dense – Sam)
    
    Eigen::MatrixXd tdm_dense_r = tdm;
    t = clock();
    EigenLinalg::RandomSVD svd_er(tdm_dense_r, rank, 3);
    t = clock() - t;
    printf("Eigen Random SVD Elapsed time: %.10f sec.\n", (double)t/CLOCKS_PER_SEC );
    
    std::cout << "U: " << svd_er.U.rows() << " x " << svd_er.U.cols() << std::endl;
    // std::cout << svd_er.matrixU() << std::endl << std::endl;
    std::cout << "S: ";
    std::cout << svd_er.S.head(rank).transpose() << std::endl << std::endl << std::endl;

    return EXIT_SUCCESS;
}
