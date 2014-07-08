#include "Precision.h"

#include <stdio.h>
#include <time.h>

#include "LinalgIO.h"
#include "SVD.h"
#include "RandomSVD.h"

#include "CmdLine.h"

int main(int argc, char **argv){
  
  //Command line parsing
  TCLAP::CmdLine cmd("SVD", ' ', "1");


  TCLAP::ValueArg<int> dArg("d","dimension", "Dimension for randomized SVD", true, 10, "integer");
  cmd.add(dArg);
  
  TCLAP::ValueArg<int> pArg("p","power", "Number of power iterations for randomized SVD", true, 1, "integer");
  cmd.add(pArg);
  
  TCLAP::ValueArg<std::string> dataArg("x","data", "Data file",  true, "", "matrix header file");
  cmd.add(dataArg);

  try{
	  cmd.parse( argc, argv );
	} 
  catch (TCLAP::ArgException &e){ 
    std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; 
    return -1;
  }


  FortranLinalg::DenseMatrix<Precision> X = FortranLinalg::LinalgIO<Precision>::readMatrix(dataArg.getValue());
  int d = dArg.getValue();
  int p = pArg.getValue();

  clock_t t1 = clock();
  FortranLinalg::SVD<Precision> svd(X, true);
  clock_t t2 = clock();
  std::cout << "SVD" << std::endl;
  std::cout << (t2-t1)/(double)CLOCKS_PER_SEC << std::endl;
  FortranLinalg::LinalgIO<Precision>::writeVector("S.data", svd.S);
  FortranLinalg::LinalgIO<Precision>::writeMatrix("U.data", svd.U);

  t1 = clock();
  FortranLinalg::RandomSVD<Precision> rsvd(X, d, p, true);
  t2 = clock();
  std::cout << "Random SVD" << std::endl;
  std::cout << (t2-t1)/(double)CLOCKS_PER_SEC << std::endl;
  FortranLinalg::LinalgIO<Precision>::writeVector("rS.data", rsvd.S);
  FortranLinalg::LinalgIO<Precision>::writeMatrix("rU.data", rsvd.U);


  for(int i=0; i<d; i++){
    Precision dot = FortranLinalg::Linalg<Precision>::DotColumnColumn( svd.U, i, rsvd.U, i);
    std::cout << "<rsvd, svd>: " << dot << std::endl;
    dot = FortranLinalg::Linalg<Precision>::DotColumnColumn( rsvd.U, i, rsvd.U, i);
    std::cout << "<rsvd, rsvd>: " << dot << std::endl;
  }
  
  return 0;

}
