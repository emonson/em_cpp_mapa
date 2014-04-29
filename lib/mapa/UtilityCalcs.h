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
			int n_points = X.rows();
			int n_seed_points = seedIdxs.size();
			
			// Create seed points out of seed indices
			ArrayXXd seedPoints;
			igl::slice(X, seedIdxs, ArrayXi::LinSpaced(X.cols(),0,X.cols()-1), seedPoints);
			
			// Make sure output arrays are allocated to the proper size before ANN call
			// idxs = Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(seedIdxs.size(), maxKNN);
			// statDists = Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(seedIdxs.size(), maxKNN);
			
			// Arrays need to be fed into computeANN as 
		  // data [dim x n_seed_points]
		  // knn [n_knn x n_seed_points]
		  // dists [n_knn x n_seed_points]
		  // so want to after
		  
		  seedPoints.transposeInPlace();
		  idxs = ArrayXXi::Zero(maxKNN, n_seed_points);
		  statDists = ArrayXXd::Zero(maxKNN, n_seed_points);
		  
			computeANN(seedPoints, idxs, statDists, 0.0);
			
			idxs.transposeInPlace();
			statDists.transposeInPlace();
		};
		
	private:
		
		// Modified from Sam Gerber's Geometry.h computeANN() to use Eigen arrays
		/* ANN routines need pointers to vectors, so since default in Eigen is column-major
		   arrangement, data needs to be sent here transposed from the original / desired. 
		   So,
		   data [dim x n_points]
		   knn [n_knn x n_points]
		   dists [n_knn x n_points]
		   Note: n_points is the number of seed points, as all pairs are calculated here, so
		   only pass in data for seed points.
		*/
		static void computeANN(const ArrayXXd &data, Eigen::Ref<ArrayXXi> knn, Eigen::Ref<ArrayXXd> dists, double eps){
		
			// Eigen data are not directly convertible to double**, so need to construct it
			// explicitly
			
			int dim = data.rows();
			int n_points = data.cols();
			int n_knn = knn.rows();
			// TODO: check for column major and any dim sanity checks...

			// Create the arrays of pointers to the columns
			double **dataPointers;
			int **knnPointers;
			double **distsPointers;

			dataPointers = new double*[n_points];
			knnPointers = new int*[n_points];
			distsPointers = new double*[n_points];
			
			for (int ii = 0; ii < n_points; ii++)
			{
				dataPointers[ii] = (double*)data.col(ii).data();
				knnPointers[ii] = knn.col(ii).data();
				distsPointers[ii] = dists.col(ii).data();
			}
		
			ANNpointArray pts = dataPointers;
		
	// 	ANNkd_tree(							// build from point array
	// 		ANNpointArray	pa,				// point array
	// 		int				n,				// number of points
	// 		int				dd,				// dimension
	// 		int				bs = 1,			// bucket size
	// 		ANNsplitRule	split = ANN_KD_SUGGEST);	// splitting method

			// orig had (pts, data.N(), data.M())...
			ANNkd_tree *annTree = new ANNkd_tree( pts, n_points, dim); 

	// 		int **knnData = knn.data();
	// 		double **distData = dists.data();

	// 	virtual void annkSearch(			// approx k near neighbor search
	// 		ANNpoint		q,				// query point
	// 		int				k,				// number of near neighbors to return
	// 		ANNidxArray		nn_idx,			// nearest neighbor array (modified)
	// 		ANNdistArray	dd,				// dist to near neighbors (modified)
	// 		double			eps=0.0			// error bound
	// 		) = 0;							// pure virtual (defined elsewhere)
			
			for(unsigned int i = 0; i < n_points; i++){
				annTree->annkSearch( pts[i], n_knn, knnPointers[i], distsPointers[i], eps);
			}

			delete annTree;
			delete[] dataPointers;
			delete[] knnPointers;
			delete[] distsPointers;
			
			annClose(); // done with ANN

		};  

   
}; // class def

} // namespace MAPA

#endif
