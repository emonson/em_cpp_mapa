#include <Eigen/Core>
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#define EIGEN_IM_MAD_AS_HELL_AND_IM_NOT_GOING_TO_TAKE_IT_ANYMORE
#include <Eigen/SparseCore>
using namespace Eigen;

#include <cmath>
#include <string>
#include <iostream>

#include "svdlib.h"

using namespace std;

int main(int argc, char * argv[])
{
    int rows = 4;
    int cols = 3;
    SparseMatrix<double> mat(rows,cols);         // default is column major
    mat.reserve(VectorXi::Constant(cols,2));	// reserve enough space for 2 nonzeros per column
    mat.insert(0,0) = 5.3;                    // alternative: mat.coeffRef(i,j) += v_ij;
    mat.insert(1,0) = 2;
    mat.insert(2,1) = 8.91;
    mat.insert(3,2) = 0.11;
    mat.makeCompressed();                        // optional

    cout << mat << endl;
    
    // allocate dynamic memory for a svdlibc sparse matrix
    SMat s_mat = svdNewSMat(mat.rows(), mat.cols(), mat.nonZeros()); 

    /* Harwell-Boeing sparse matrix. */
    // struct smat {
    //   long rows;
    //   long cols;
    //   long vals;     /* Total non-zero entries. */
    //   long *pointr;  /* For each col (plus 1), index of first non-zero entry. */
    //   long *rowind;  /* For each nz entry, the row index. */
    //   double *value; /* For each nz entry, the value. */
    // };
    
    // sm1.valuePtr(); // Pointer to the values
    // sm1.innerIndexPtr(); // Pointer to the indices.
    // sm1.outerIndexPtr(); //Pointer to the beginning of each inner vector
    
    // see if we can directly build with the same pointers...
    s_mat->value = mat.valuePtr();
    s_mat->rowind = (long *)mat.innerIndexPtr();
    s_mat->pointr = (long *)mat.outerIndexPtr();
    
    char filename[128];
    string out_file = "s_mat";
    sprintf(filename, "%s.txt", out_file.c_str());
    svdWriteSparseMatrix(s_mat, filename, 0);
    
    // svdFreeSMat(s_mat);
    return 0;
}
