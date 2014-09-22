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
#include "LinalgIO.h"
#include "Precision.h"
#include "RandomSVD.h"
#include "SVD.h"

int main( int argc, const char** argv )
{
	std::string data_dir = MAPA::UtilityCalcs::PathAppend(MAPA_SOURCE_DIR, "data");
	std::string filename = MAPA::UtilityCalcs::PathAppend(data_dir, "InfovisVAST-papers.jig");

    int min_term_length = 3;
    int min_term_count = 5;
    MAPA::TDMgenerator tdm_gen(min_term_length, min_term_count);
	MAPA::JIGtokenizer jig_tok(filename, &tdm_gen);
    
    // Eigen sparse matrix
    Eigen::SparseMatrix<double,0,long> tdm = tdm_gen.getTDM();
    // Eigen dense matrix
    Eigen::MatrixXd tdm_dense = tdm;
    // Sam dense matrix
    FortranLinalg::DenseMatrix<Precision> tdm_sam(tdm.rows(), tdm.cols());
    tdm_sam.setDataPointer(tdm_dense.data());

    std::cout << std::endl << "TDM (sparse input matrix): " << tdm.rows() << " x " << tdm.cols() << ", " << tdm.nonZeros() << " nonzeros" << std::endl << std::endl;
    
    int rank = 5;
    int power_iterations = 3;

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
    
    t = clock();
    EigenLinalg::RandomSVD svd_er(tdm_dense, rank, 3);
    t = clock() - t;
    printf("Eigen Random SVD Elapsed time: %.10f sec.\n", (double)t/CLOCKS_PER_SEC );
    
    std::cout << "U: " << svd_er.U.rows() << " x " << svd_er.U.cols() << std::endl;
    // std::cout << svd_er.matrixU() << std::endl << std::endl;
    std::cout << "S: ";
    std::cout << svd_er.S.head(rank).transpose() << std::endl << std::endl << std::endl;

    // --------------------------------
    // Sam dense standard Fortran version
    
    t = clock();
    FortranLinalg::SVD<Precision> svd_fs(tdm_sam, true);
    t = clock() - t;
    std::cout << "SVD" << std::endl;
    std::cout << (t2-t1)/(double)CLOCKS_PER_SEC << std::endl;

    printf("LAPACK standard Elapsed time: %.10f sec.\n", (double)t/CLOCKS_PER_SEC );

    std::cout << "U: " << svd_fs.U.M() << " x " << svd_fs.U.N() << std::endl;
    std::cout << "S: ";
    for (int ii = 0; ii < rank; ii++)
    {
        std::cout << svd_fs.S(ii) << " " << std::endl;
    }

    // --------------------------------
    // Sam dense random Fortran version
    
    t = clock();
    FortranLinalg::RandomSVD<Precision> svd_fr(tdm_sam, rank, power_iterations, true);
    t = clock() - t;

    printf("LAPACK random Elapsed time: %.10f sec.\n", (double)t/CLOCKS_PER_SEC );

    std::cout << "U: " << svd_fr.U.M() << " x " << svd_fr.U.N() << std::endl;
    std::cout << "S: ";
    for (int ii = 0; ii < rank; ii++)
    {
        std::cout << svd_fr.S(ii) << " " << std::endl;
    }

    return EXIT_SUCCESS;
}
