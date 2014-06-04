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
    
    // Filling sparse Eigen matrix by insertion
    SparseMatrix<double,0,long> mat(rows,cols);         // default is column major
    mat.reserve(VectorXi::Constant(cols,2));	// reserve enough space for 2 nonzeros per column
    mat.insert(0,0) = 5.3;                    // alternative: mat.coeffRef(i,j) += v_ij;
    mat.insert(1,0) = 2;
    mat.insert(2,1) = 8.91;
    mat.insert(3,2) = 0.11;
    mat.makeCompressed();                        // optional

    cout << mat << endl;
    
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
    SMat s_mat = svdNewSMat(mat.rows(), mat.cols(), mat.nonZeros()); 

    /* Harwell-Boeing sparse matrix. */
    // SVDLIBC sparse matrix
    // struct smat {
    //   long rows;
    //   long cols;
    //   long vals;     /* Total non-zero entries. */
    //   long *pointr;  /* For each col (plus 1), index of first non-zero entry. */
    //   long *rowind;  /* For each nz entry, the row index. */
    //   double *value; /* For each nz entry, the value. */
    // };
    
    // EIGEN sparse matrix
    // sm1.valuePtr(); // Pointer to the values
    // sm1.innerIndexPtr(); // Pointer to the indices.
    // sm1.outerIndexPtr(); //Pointer to the beginning of each inner vector
    
    // see if we can directly build with the same pointers...
    s_mat->value = m.valuePtr();
    s_mat->rowind = m.innerIndexPtr();
    s_mat->pointr = m.outerIndexPtr();
        
    char s_filename[128];
    string out_file = "s_mat";
    sprintf(s_filename, "%s.txt", out_file.c_str());
    svdWriteSparseMatrix(s_mat, s_filename, 0);
    
    // Try to make dense matrix and write out
    DMat d_mat;
    d_mat = svdConvertStoD(s_mat);
    
    char d_filename[128];
    string out_file2 = "d_mat";
    sprintf(d_filename, "%s.txt", out_file2.c_str());
    svdWriteDenseMatrix(d_mat, d_filename, SVD_F_DT);
    
    // svdFreeSMat(s_mat);
    return 0;
}
