#ifndef SVDLIBCSVD_H
#define SVDLIBCSVD_H

/* SvdlibcSVD.h

Wrapper for SVDLIBC sparse svd routine using Eigen matrix as input.

Using very similar API to Eigen's JacobiSVD() so it's easy to try
swapping in and out. Leaving rank required to be explicitly supplied for now.

Eric E Monson – 2014
Duke University

*/

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "svdlib.h"

#include <vector>
#include <iostream>

using namespace Eigen;


namespace MAPA {

class SvdlibcSVD {

public:

    SvdlibcSVD( SparseMatrix<double,0,long> &mat, int rank)
    {
        run(mat, rank);
    };
    
    SvdlibcSVD( MatrixXd &dmat, int rank )
    {
        // Convert dense matrix to sparse
        SparseMatrix<double,0,long> mat = dmat.sparseView();

        run(mat, rank);

    };

    SvdlibcSVD( ArrayXXd &darray, int rank )
    {
        // Convert dense matrix to sparse
        SparseMatrix<double,0,long> mat = darray.matrix().sparseView();

        run(mat, rank);

    };

    VectorXd singularValues()
    {
        return S;
    };
    
    MatrixXd matrixU()
    {
        return Ut.transpose();
    };
    
    MatrixXd matrixV()
    {
        return Vt.transpose();
    };
    

private:

    VectorXd S;
    MatrixXd Ut;
    MatrixXd Vt;
    
    void run(SparseMatrix<double,0,long> &mat, int rank)
    {
        // make double-sure that matrix is in compressed format
        if (!mat.isCompressed())
        {
            mat.makeCompressed();
        }
        
        // Allocate dynamic memory for a svdlibc sparse matrix
        // NOTE: SMat is a typedef for a pointer to a sparse matrix structure
        SMat s_mat = svdNewSMat(mat.rows(), mat.cols(), mat.nonZeros()); 
        if(s_mat == NULL)
        {
            printf("memory allocation for svdlibc_sparse_matrix variable in the sparse_svd() function failed\n");
            fflush(stdout);
            exit(3);
        }

        // Do the mapping to the SVDLIBC matrix directly from the Eigen pointers
        s_mat->value = mat.valuePtr();
        s_mat->rowind = mat.innerIndexPtr();
        s_mat->pointr = mat.outerIndexPtr();
        
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
        Map< Matrix<double, Dynamic, Dynamic, RowMajor> > e_Vt(*svd_result->Vt->value, svd_result->Vt->rows, svd_result->Vt->cols);
        Map< VectorXd > e_S(svd_result->S, svd_result->d);

        // Copy over into actual matrices. Probably not necessary...
        Ut = e_Ut;
        Vt = e_Vt;
        S = e_S;
        
        // I guess you don't need to do this...
        //   svdFreeSMat(s_mat);
        // or this...
        //   svdFreeSVDRec(svd_result);
        // or you get this:
        //   sparse_svd(3400,0x7fff7de6f310) malloc: *** error for object 0x7fee8a500080: pointer being freed was not allocated
        //   *** set a breakpoint in malloc_error_break to debug
        //   Abort trap: 6    };
    };

}; // class def

} // namespace MAPA

#endif
