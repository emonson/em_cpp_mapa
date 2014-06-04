#include <Eigen/Core>
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#define EIGEN_IM_MAD_AS_HELL_AND_IM_NOT_GOING_TO_TAKE_IT_ANYMORE
#include <Eigen/SparseCore>
using namespace Eigen;

#include <cmath>
#include <string>
#include <vector>
#include <iostream>

#include "svdlib.h"

using namespace std;

int main(int argc, char * argv[])
{
    int rows = 4;
    int cols = 3;
    int rank = 2;
    
    // Filling sparse Eigen matrix with triplets
    typedef Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(4);
    
    tripletList.push_back(T(0, 0, 5.3));
    tripletList.push_back(T(1, 0, 2));
    tripletList.push_back(T(2, 1, 8.91));
    tripletList.push_back(T(3, 2, 0.11));

    SparseMatrix<double,0,long> m(rows,cols);
    m.setFromTriplets(tripletList.begin(), tripletList.end());
    // m is ready to go!
    cout << m << endl;
    
    // Allocate dynamic memory for a svdlibc sparse matrix
    SMat s_mat = svdNewSMat(m.rows(), m.cols(), m.nonZeros()); 

    // see if we can directly build with the same pointers...
    s_mat->value = m.valuePtr();
    s_mat->rowind = m.innerIndexPtr();
    s_mat->pointr = m.outerIndexPtr();
        
    SVDRec svd_result = svdNewSVDRec(); // allocate dynamic memory for a svdlibc svd record for storing the result of applying the lanczos method on the input matrix
    if(svd_result == NULL)
    {
        printf("memory allocation for svd_result variable in the sparse_svd() function failed\n");
        fflush(stdout);
        exit(3);
    }
    int iterations = 0; // number of lanczos iterations - 0 means until convergence
    double las2end[2] = {-1.0e-30, 1.0e-30}; // tolerance interval
    double kappa = 1e-6; // another tolerance parameter
    
    // computing the sparse svd of the input matrix using lanczos method
    // struct svdrec {
    //   int d;      /* Dimensionality (rank) */
    //   DMat Ut;    /* Transpose of left singular vectors. (d by m)
    //                  The vectors are the rows of Ut. */
    //   double *S;  /* Array of singular values. (length d) */
    //   DMat Vt;    /* Transpose of right singular vectors. (d by n)
    //                  The vectors are the rows of Vt. */
    // };
    svd_result = svdLAS2(s_mat, rank, iterations, las2end, kappa); 

    for (int ii = 0; ii < rank; ii++)
    {
        std::cout << svd_result->S[ii] << std::endl;
    }
    
    char U_filename[128];
    string out_file_U = "Ut";
    sprintf(U_filename, "%s.txt", out_file_U.c_str());
    svdWriteDenseMatrix(svd_result->Ut, U_filename, SVD_F_DT);
    
    char V_filename[128];
    string out_file_V = "Vt";
    sprintf(V_filename, "%s.txt", out_file_V.c_str());
    svdWriteDenseMatrix(svd_result->Vt, V_filename, SVD_F_DT);
    
    // svdFreeSMat(s_mat);
    return 0;
}
