#include <stdlib.h>		/* NULL */
#include <time.h>			/* time */
#include <stdio.h>		/* srand */
#include <iostream>
#include "EigenRandomSVD.h"
#include "LinalgIO.h"
#include "Precision.h"
#include "RandomSVD.h"
#include "SVD.h"
#include <Eigen/SVD>

#include "CmdLine.h"

int main(int argc, char **argv) {

    //Command line parsing
    TCLAP::CmdLine cmd("SinkhornTransport", ' ', "1");

    TCLAP::ValueArg<std::string> xArg("x","X", "Matrix to decompose", true, "",
                                      "matrix header file");
    cmd.add(xArg);

    TCLAP::ValueArg<int> pArg("p","power", "Number of power iterations", false, 3, "integer");
    cmd.add(pArg);

    TCLAP::ValueArg<int> dArg("d","dimension", "Number of dimensions for SVD. To get k accurate dimension set d = k+5", false, 3, "integer");
    cmd.add(dArg);

    try
    {
        cmd.parse( argc, argv );
    }
    catch (TCLAP::ArgException &e)
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
        return -1;
    }

    int p = pArg.getValue();
    int d = dArg.getValue();

    FortranLinalg::DenseMatrix<Precision> Xf =
            FortranLinalg::LinalgIO<Precision>::readMatrix( xArg.getValue() );

    //Both eigen and my matrix format use column major order by defualt so
    //mapping should be correct
    Eigen::MatrixXd Xe = Eigen::MatrixXd::Map(Xf.data(),Xf.M(), Xf.N() );

    // Lapack std SVD
    FortranLinalg::SVD<Precision> svd_std_f(Xf, true);

    // Eigen std SVD
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_std_e(Xe, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd svg_std_e_S = svd_std_e.singularValues();

    // Reseed random number generator since Eigen Random.h doesn't do this itself
    // srand( (unsigned int)time(NULL) );

    std::cout.precision(10);

    // Eigen RandomSVD
    Eigen::MatrixXd rand_range_e = EigenLinalg::RandomRange::FindRange(Xe,d,p);
    std::cout << "RandomRange matrix (0,0)" << std::endl;
    std::cout << "Eigen: " << rand_range_e(0,0) << std::endl;
    EigenLinalg::RandomSVD svd_rand_e(Xe, d, p);

    // Lapack RandomSVD
    FortranLinalg::DenseMatrix<Precision> rand_range_f = FortranLinalg::RandomRange<Precision>::FindRange(Xf,d,p);
    std::cout << "Lapack: " << rand_range_f(0,0) << std::endl;
    std::cout << std::endl;
    FortranLinalg::RandomSVD<double> svd_rand_f(Xf, d, p, true);
    // FortranLinalg::LinalgIO<Precision>::writeVector("svd_orig.data", svd_orig.S);

    std::cout << "svd.S values" << std::endl;
    std::cout << "Lapack_std"<< "    " << "Lapack_rand"  << "   " << "Eigen_std " << "    " << "Eigen_rand" << std::endl;
    for(int i=0; i < d; i++)
    {
        std::cout << svd_std_f.S(i) << "   " << svd_rand_f.S(i) << "   " << svg_std_e_S(i) << "   " << svd_rand_e.S(i) << std::endl;
    }


    return 0;
}
