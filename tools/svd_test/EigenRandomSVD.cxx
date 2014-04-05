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

int main(int argc, char **argv){

  //Command line parsing
  TCLAP::CmdLine cmd("SinkhornTransport", ' ', "1");

  TCLAP::ValueArg<std::string> xArg("x","X", "Matrix to decompose", true, "",
      "matrix header file");
  cmd.add(xArg);
  
  TCLAP::ValueArg<int> pArg("p","power", "Number of power iterations", false, 3, "integer"); 
  cmd.add(pArg);
  
  TCLAP::ValueArg<int> dArg("d","dimension", "Number of dimensions for SVD. To get k accurate dimension set d = k+5", false, 3, "integer");
  cmd.add(dArg);

  try{ 
    cmd.parse( argc, argv );
  } 
  catch (TCLAP::ArgException &e){ 
    std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; 
    return -1;
  }

  int p = pArg.getValue();
  int d = dArg.getValue();

  FortranLinalg::DenseMatrix<Precision> Xf =
    FortranLinalg::LinalgIO<Precision>::readMatrix( xArg.getValue() );

  //Both eigen and my matrix format use column major order by defualt so
  //mapping should be correct 
  Eigen::MatrixXd X = Eigen::MatrixXd::Map(Xf.data(),Xf.M(), Xf.N() );
  
  // Lapack SVD
  FortranLinalg::SVD<Precision> svd_nonrand(Xf, true);
  std::cout << "SVD" << std::endl;

	// Eigen SVD
	Eigen::JacobiSVD<Eigen::MatrixXd> svd_eigen(X, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::VectorXd S = svd_eigen.singularValues();
	
	// Reseed random number generator since Eigen Random.h doesn't do this itself
	// srand( (unsigned int)time(NULL) );
	
	std::cout.precision(10);

	// Eigen RandomSVD
	Eigen::MatrixXd Q = EigenLinalg::RandomRange::FindRange(X,d,p);
	std::cout << "RandomRange matrix (0,0)" << std::endl;
	std::cout << "Eigen: " << Q(0,0) << std::endl;
	
  EigenLinalg::RandomSVD svd(X, d, p);
  
  // Lapack RandomSVD
  FortranLinalg::DenseMatrix<Precision> Q_orig = FortranLinalg::RandomRange<Precision>::FindRange(Xf,d,p);
	std::cout << "Lapack: " << Q_orig(0,0) << std::endl;
	std::cout << std::endl;

  FortranLinalg::RandomSVD<double> svd_orig(Xf, d, p, true);
	// FortranLinalg::LinalgIO<Precision>::writeVector("svd_orig.data", svd_orig.S);

	std::cout << "svd.S values" << std::endl;
	std::cout << "LPNonRand " << "\t" << "Eigen     " << "\t" << "ENonRand" << "\t" << "Lapack    " << std::endl;
  for(int i=0; i<d; i++){
    std::cout << svd_nonrand.S(i) << "\t" << svd.S(i) << "\t" << S(i) << "\t" << svd_orig.S(i) << std::endl;
  }
  
	
  return 0;
}
