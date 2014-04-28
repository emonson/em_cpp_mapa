#ifndef UTILITYCALCS_H
#define UTILITYCALCS_H

/* UtilityCalcs.h

Based on the Matlab code of
Guangliang Chen & Mauro Maggioni

Eric E Monson – 2014
Duke University

*/

#include "Eigen/Dense"
#include "mersenneTwister2002.h"
#include <igl/slice.h>
#include <igl/sort.h>
#include <stdlib.h>
#include "ANN.h"

using namespace Eigen;


namespace MAPA {

class UtilityCalcs {

	public:

    static ArrayXXd P2Pdist(const ArrayXXd &X, const std::vector<ArrayXd> &centers, const std::vector<ArrayXXd> &bases)
    {
      std::cerr << "MAPA::UtilityCalcs::P2Pdist – Not implemented yet!!" << std::endl;
			return Array<double,1,1>::Zero();
    };

    static ArrayXd L2error(const ArrayXXd &data, const ArrayXi &dim, const ArrayXi &idx)
    {
      std::cerr << "MAPA::UtilityCalcs::L2error – Not implemented yet!!" << std::endl;
			return Array<double,1,1>::Zero();
    };

    static double ClusteringError(const ArrayXi &indices, const ArrayXi &trueLabels)
    {
      std::cerr << "MAPA::UtilityCalcs::ClusteringError – Not implemented yet!!" << std::endl;
			return 0;
    };
    
    static ArrayXi RandSample(unsigned int N, unsigned int K, bool sorted=false)
    {
			/* Y = randsample(N,K) returns Y as a column vector of K values sampled
			uniformly at random, without replacement, from the integers 0:N-1 */

			if (K > N)
			{
      	std::cerr << "MAPA::UtilityCalcs::RandSample Error – Number of samples K can't be larger than N" << std::endl;
      	return ArrayXi::Zero(K);
			}
			
			ArrayXd randX(N);
			randX.setRandom();
			
			// sort columns independently (only one here)
			int dim = 1;
			// sort ascending order
			int ascending = true;
			// Sorted output matrix
			ArrayXd Y;
			// sorted indices for sort dimension
			ArrayXi IX;
			
			igl::sort(randX,1,ascending,Y,IX);

			if (!sorted)
			{
				return IX.head(K);			
			}
			else
			{
				ArrayXi Yidxs;
				ArrayXi Xidxs = IX.head(K);
				
				igl::sort(Xidxs, 1, ascending, Yidxs, IX);
				return Yidxs;
			}
    };

		static void nrsearch(const ArrayXXd &X, const ArrayXi &seedIdxs, int maxKNN,
		                      Eigen::Ref<ArrayXXi> idxs, Eigen::Ref<ArrayXXd> statDists)
		{
			// Create seed points out of seed indices
			ArrayXXd seedPoints;
			igl::slice(X, seedIdxs, ArrayXi::LinSpaced(X.cols(),0,X.cols()-1), seedPoints);
			
			// Make sure output arrays are allocated to the proper size before ANN call
			idxs = Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(seedIdxs.size(), maxKNN);
			statDists = Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(seedIdxs.size(), maxKNN);
			
			computeANN(seedPoints, idxs, statDists, 0.0);
		};
		
	private:
		
		// Modified from Sam Gerber's Geometry.h computeANN() to use Eigen arrays
		// data expected to be [n_points x dim]
		// knn should be pre-allocated to 
		// knn and dists should be RowMajor!!
		static void computeANN(const ArrayXXd &data, Eigen::Ref<ArrayXXi> knn, Eigen::Ref<ArrayXXd> dists, double eps){
		
			// Eigen data are not directly convertible to double**, so need to construct it
			// explicitly
		
			// First, make sure data is in RowMajor order, so point coordinates are stored contiguously
			Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dataRows = data;
			
		  // std::cout << data << std::endl;
		
			// TODO: clean this up and check dimensions match before running...
			
			// Now create the array of pointers to the rows
			int n = data.rows();
			int m = dataRows.cols();

			double **rowPointers;
			rowPointers = new double*[n];
			int **knnPointers;
			knnPointers = new int*[n];
			double **distsPointers;
			distsPointers = new double*[n];
			
			
			for (int ii = 0; ii < dataRows.rows(); ii++)
			{
				rowPointers[ii] = dataRows.data() + (ii*m)*sizeof(double);
				knnPointers[ii] = knn.data() + (ii*m)*sizeof(int);
				distsPointers[ii] = dists.data() + (ii*m)*sizeof(double);
			}
		
			ANNpointArray pts = rowPointers;
		
	// 	ANNkd_tree(							// build from point array
	// 		ANNpointArray	pa,				// point array
	// 		int				n,				// number of points
	// 		int				dd,				// dimension
	// 		int				bs = 1,			// bucket size
	// 		ANNsplitRule	split = ANN_KD_SUGGEST);	// splitting method

			// orig had (pts, data.N(), data.M())...
			ANNkd_tree *annTree = new ANNkd_tree( pts, dataRows.rows(), dataRows.cols()); 

	// 		int **knnData = knn.data();
	// 		double **distData = dists.data();

	// 	virtual void annkSearch(			// approx k near neighbor search
	// 		ANNpoint		q,				// query point
	// 		int				k,				// number of near neighbors to return
	// 		ANNidxArray		nn_idx,			// nearest neighbor array (modified)
	// 		ANNdistArray	dd,				// dist to near neighbors (modified)
	// 		double			eps=0.0			// error bound
	// 		) = 0;							// pure virtual (defined elsewhere)
			
			int maxKNN = knn.cols();
			int *knnPointer = knn.data();
			double *distsPointer = dists.data();
			
			
			for(unsigned int i = 0; i < dataRows.rows(); i++){
				annTree->annkSearch( pts[i], maxKNN, knnPointers[i], distsPointers[i], eps);
			}

			delete annTree;
			delete[] rowPointers;
			delete[] knnPointers;
			delete[] distsPointers;
			
			annClose(); // done with ANN

		};  

   
}; // class def

} // namespace MAPA

#endif
