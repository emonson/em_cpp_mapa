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
    
    // Allocate dynamic memory for a svdlibc sparse matrix
    // NOTE: SMat is a typedef for a pointer to a sparse matrix structure
    SMat s_mat = svdNewSMat(m.rows(), m.cols(), m.nonZeros()); 
    if(s_mat == NULL)
    {
        printf("memory allocation for svdlibc_sparse_matrix variable in the sparse_svd() function failed\n");
        fflush(stdout);
        exit(3);
    }

    // Do the mapping to the SVDLIBC matrix directly from the Eigen pointers
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

    /* Row-major dense matrix.  Rows are consecutive vectors. */
    // struct dmat {
    //   long rows;
    //   long cols;
    //   double **value; /* Accessed by [row][col]. Free value[0] and value to free.*/
    // };
    
    Map< Matrix<double, Dynamic, Dynamic, RowMajor> > e_Ut(*svd_result->Ut->value, svd_result->Ut->rows, svd_result->Ut->cols);
    cout << "Ut" << endl;
    cout << e_Ut << endl << endl;

    Map< Matrix<double, Dynamic, Dynamic, RowMajor> > e_Vt(*svd_result->Vt->value, svd_result->Vt->rows, svd_result->Vt->cols);
    cout << "Vt" << endl;
    cout << e_Vt << endl << endl;
    
    Map< VectorXd > e_S(svd_result->S, svd_result->d);
    cout << "S" << endl;
    // NOTE: use vector.asDiagonal() for optimized matrix multiplications with diagonal
    //   matrix formed directly from vector of diagonal elements
    // This copy is only for display, we don't need it for any computations
    MatrixXd e_S_copy = e_S.asDiagonal();
    cout << e_S_copy << endl << endl;


    // Test from Matlab svd()
    // >> U'
    // 
    // ans =
    // 
    //          0         0   -1.0000         0
    //    -0.9356   -0.3531         0         0
    //          0         0         0   -1.0000
    //    -0.3531    0.9356         0         0
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

    // I guess you don't need to do this...
    //   svdFreeSMat(s_mat);
    // or this...
    //   svdFreeSVDRec(svd_result);
    // or you get this:
    //   sparse_svd(3400,0x7fff7de6f310) malloc: *** error for object 0x7fee8a500080: pointer being freed was not allocated
    //   *** set a breakpoint in malloc_error_break to debug
    //   Abort trap: 6
    
    return 0;
}
