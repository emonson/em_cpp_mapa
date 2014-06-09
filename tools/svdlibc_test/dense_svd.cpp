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
    
    // Making dense matrix
    Eigen::MatrixXd md = Eigen::MatrixXd::Zero(4,3);
    
    md(0, 0) = 5.3;
    md(1, 0) = 2;
    md(2, 1) = 8.91;
    md(3, 2) = 0.11;

    cout << "Original (dense) matrix" << endl;
    cout << md << endl << endl;
    
    MAPA::SvdlibcSVD svds(md, rank);
    
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
