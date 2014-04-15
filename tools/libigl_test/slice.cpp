#include <Eigen/Dense>
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#define EIGEN_IM_MAD_AS_HELL_AND_IM_NOT_GOING_TO_TAKE_IT_ANYMORE
#include <Eigen/Sparse>
using namespace Eigen;

#include <cstdio>
#include <iostream>
using namespace std;

#include <stdlib.h>		/* NULL */
#include <time.h>			/* time */
#include <stdio.h>		/* srand */

#include <igl/slice.h>
#include <igl/speye.h>
#include <igl/print_ijv.h>
using namespace igl;

int main(int argc, char * argv[])
{
  // Dense
  MatrixXd A = MatrixXd::Identity(5,5);
  cout<<"A=["<<endl<<A<<endl<<"];"<<endl;
  // Row and column indices
  MatrixXi R(6,1);
  R<<0,1,2,4,4,0;
  cout<<"R=["<<endl<<R<<endl<<"];"<<endl;
  MatrixXi C(4,1);
  C<<2,2,4,0;
  cout<<"C=["<<endl<<C<<endl<<"];"<<endl;
  MatrixXd B;
  cout<<"B=A(R,C);"<<endl;
  slice(A,R,C,B);
  cout<<"B=["<<endl<<B<<endl<<"];"<<endl;
  
  // Reseed random number generator since Eigen Random.h doesn't do this itself
  srand( (unsigned int)time(NULL) );

  // Trying A(R,:) type Eigen slice
  MatrixXd AA = MatrixXd::Random(5,5);
  cout << "AA" << endl;
  cout << AA << endl;
  VectorXd CC1 = AA.col(1);
  cout << "AA[:,1]" << endl;
  cout << CC1 << endl;
  
  cout<<endl;

  //SparseMatrix
  SparseMatrix<double> spA;
  speye(5,5,spA);
  cout<<"spA_IJV=["<<endl;print_ijv(spA,1);cout<<endl<<"];"<<endl;
  cout<<"spA=sparse(spA_IJV(:,1),spA_IJV(:,2),spA_IJV(:,3));"<<endl;
  SparseMatrix<double> spB;
  cout<<"spB=spA(R,C);"<<endl;
  slice(spA,R,C,spB);
  cout<<"spB_IJV=["<<endl;print_ijv(spB,1);cout<<endl<<"];"<<endl;
  cout<<"spB=sparse(spB_IJV(:,1),spB_IJV(:,2),spB_IJV(:,3));"<<endl;
  
  // Eigen select
  MatrixXi m(1,5);
  m << 1, 2, 3, 4, 5;
  // select equivalent of loop over members(i), (m(i) < 3) ? 0 : m(i)
  m = (m.array() < 3).select(0,m);
  cout << m << endl;
  cout << (m.array() < 3) << endl;

  return 0;
}
