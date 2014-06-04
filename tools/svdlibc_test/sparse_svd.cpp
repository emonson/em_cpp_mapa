#include <Eigen/Core>
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#define EIGEN_IM_MAD_AS_HELL_AND_IM_NOT_GOING_TO_TAKE_IT_ANYMORE
#include <Eigen/SparseCore>
using namespace Eigen;

#include <cmath>
#include <string>
#include <vector>
#include <iostream>

#include "SvdlibcSVD.h"

using namespace std;

int main(int argc, char * argv[])
{
    int rows = 4;
    int cols = 3;
    int rank = 3;
    
    // Filling sparse Eigen matrix with triplets
    typedef Triplet<double,long> T;
    std::vector<T> tripletList;
    tripletList.reserve(4);
    
    tripletList.push_back(T(0, 0, 5.3));
    tripletList.push_back(T(1, 0, 2));
    tripletList.push_back(T(2, 1, 8.91));
    tripletList.push_back(T(3, 2, 0.11));

    // NOTE: the "long" specification for the indices is necessary to match the 
    //   pointer type for the SVDLIBC matrices...
    SparseMatrix<double,0,long> m(rows,cols);
    m.setFromTriplets(tripletList.begin(), tripletList.end());
    // m is ready to go!
    cout << "Original (sparse) matrix" << endl;
    cout << m << endl << endl;
    
    MAPA::SvdlibcSVD svds(m, rank);
    
    cout << "U" << endl;
    cout << svds.matrixU() << endl << endl;
    cout << "V" << endl;
    cout << svds.matrixV() << endl << endl;
    cout << "S" << endl;
    cout << svds.singularValues() << endl << endl;

    // Test from Matlab svd()
    //
    // U =
    // 
    //          0   -0.9356         0   -0.3531
    //          0   -0.3531         0    0.9356
    //    -1.0000         0         0         0
    //          0         0   -1.0000         0
    // 
    // >> V
    // 
    // V =
    // 
    //      0    -1     0
    //     -1     0     0
    //      0     0    -1
    // 
    // >> diag(S)
    // 
    // ans =
    // 
    //     8.9100
    //     5.6648
    //     0.1100

    return 0;
}
