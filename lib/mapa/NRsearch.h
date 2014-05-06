#ifndef NRSEARCH_H
#define NRSEARCH_H


#include <Eigen/Core>
#include <stdlib.h>
#include "ANN.h"

using namespace Eigen;


namespace MAPA {

class NRsearch {

public:

    NRsearch(const ArrayXXd &Xin)
    {
        // Want to be able to pass in data [n_points x dim] even through ANN calc
        // routine wants the transpose of that.
        X = Xin.transpose();
        
        setupANNTree();
    };
    
    ~NRsearch()
    {
        delete[] dataPointers;
        annClose(); // done with ANN
        delete annTree;    
    };

    // Modified from Sam Gerber's Geometry.h computeANN() to use Eigen arrays
    /* ANN routines need pointers to vectors, so since default in Eigen is column-major
       arrangement, ** data needs to be sent here transposed from the original / desired! ** 
       So,
       data [dim x n_points]
       seeds [dim x n_seed_points]
       idxs [n_knn x n_seed_points]
       dists [n_knn x n_seed_points]
    */
    void computeANN(const ArrayXi &seedIdxs, int n_knn, double eps)
    {
    
        // Eigen data are not directly convertible to double**, so need to construct it
        // explicitly
        
        int n_seed_points = seedIdxs.size();
        
        // Create Eigen arrays with orientation needed for ANN calc, but will tranpose
        // afterwards before access
        idxs = ArrayXXi::Zero(n_knn, n_seed_points);
        statDists = ArrayXXd::Zero(n_knn, n_seed_points);
        
        // Create the arrays of pointers to the columns
        int **idxsPointers;
        double **distsPointers;

        idxsPointers = new int*[n_seed_points];
        distsPointers = new double*[n_seed_points];
        
        for (int ii = 0; ii < n_seed_points; ii++)
        {
            idxsPointers[ii] = idxs.col(ii).data();
            distsPointers[ii] = statDists.col(ii).data();
        }
    
        for(unsigned int i = 0; i < n_seed_points; i++){
            annTree->annkSearch( pts[seedIdxs(i)], n_knn, idxsPointers[i], distsPointers[i], eps);
        }
        
        // Put into desired arrangement and do distances rather than squared
        idxs.transposeInPlace();
        statDists.transposeInPlace();
        statDists = statDists.sqrt();

        delete[] idxsPointers;
        delete[] distsPointers;
    };  

    ArrayXXi GetIdxs()
    {
        return idxs;
    };
    
    ArrayXXd GetDistances()
    {
        return statDists;
    };
    
private:

    ArrayXXd X;
    ArrayXXi idxs;
    ArrayXXd statDists;
    
    double **dataPointers;

    ANNpointArray pts;
    ANNkd_tree *annTree;
    
    int dim;
    int n_points;

    void setupANNTree()
    {    
        // Eigen data are not directly convertible to double**, so need to construct it
        // explicitly        

        dim = X.rows();
        n_points = X.cols();

        // Create the arrays of pointers to the columns
        dataPointers = new double*[n_points];
        for (int ii = 0; ii < n_points; ii++)
        {
            dataPointers[ii] = (double*)X.col(ii).data();
        }

        pts = dataPointers;

        annTree = new ANNkd_tree( pts, n_points, dim); 
    };  

}; // class def

} // namespace MAPA

#endif
