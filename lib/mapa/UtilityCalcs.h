#ifndef UTILITYCALCS_H
#define UTILITYCALCS_H

/* UtilityCalcs.h

Based on the Matlab code of
Guangliang Chen & Mauro Maggioni

Eric E Monson – 2014
Duke University

*/

#include <Eigen/Core>
#include "mersenneTwister2002.h"
#include "kMeansRex.h"
#include "ANN.h"
#include <igl/sort.h>
#include <igl/slice.h>

#include <stdlib.h>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>

using namespace Eigen;


namespace MAPA {

class UtilityCalcs {

  public:

    static ArrayXXd P2Pdists(const ArrayXXd &X, const std::vector<ArrayXd> &centers, const std::vector<ArrayXXd> &bases)
    {
        // function dists = p2pdist(X,centers,bases)
        // 
        // % This code computes (squared) points to planes distances.
        // 
        // N = size(X,1);
        // K = length(centers);
        int N = X.rows();
        int K = centers.size();

        // dists = Inf(N, K);
        ArrayXXd dists = ArrayXXd::Constant(N,K,INFINITY);

        // for k = 1:K
        for (int k = 0; k < K; k++)
        {
            // if ~isempty(centers{k}) && ~isempty(bases{k})
            if ((centers[k].size() > 0) && (bases[k].size() > 0))
            {
                // Y = X - repmat(centers{k},N,1);
                ArrayXXd Y = X.rowwise() - centers[k].transpose();
                
                // dists(:,k) = sum(Y.^2,2) - sum((Y*bases{k}').^2,2);
                dists.col(k) = Y.square().rowwise().sum() - (Y.matrix() * bases[k].matrix().transpose()).array().square().rowwise().sum();
            //     end
            }
        // end
        }
        
        return dists;
    };

    static double L2error(const ArrayXXd &data, 
                          const ArrayXi &labels, 
                          const ArrayXi &dims,
                          const std::vector<ArrayXd> &ctr,
                          const std::vector<ArrayXXd> &dir)
    {
        // function mse = L2error(data, idx, dim, ctr, dir)
        // 
        // D = size(data,2);
        // K = max(labels);
        
        // original point dimensionality
        int D = data.cols();
        // number of clusters
        int K = labels.maxCoeff()+1;

        // NOTE: ignoring this option for now...
        // if length(dims) == 1
        //     dims = dims*ones(K,1);
        // end

        // NOTE: forcing these to be passed explicitly for now...
        // if nargin<4
        //     [ctr,dir] = computing_bases(data,labels,dims);
        // end

        // mse = zeros(1,K);
        ArrayXd mse_array = ArrayXd::Zero(K);
        
        // for k = 1:K
        for (int k = 0; k < K; k++)
        {
            // cls_k = data((labels==k),:);
            // n_k = size(cls_k,1);
            
            ArrayXi cls_k_idxs = MAPA::UtilityCalcs::IdxsFromComparison(labels, "eq", k);
            ArrayXXd cls_k;
            igl::slice(data, cls_k_idxs, 1, cls_k);
            int n_k = cls_k.rows();

            // if n_k >= dims(k)+1
            if (n_k > dims(k))
            {
                // cls_k = cls_k - repmat(centers{k},n_k,1);
                cls_k.rowwise() -= ctr[k].transpose();
                
                // mse(k) = sum( sum(cls_k.^2,2) - sum((cls_k*dir{k,1}').^2,2) ) / (D-dims(k));
                ArrayXd sum_squares = cls_k.square().rowwise().sum();
                ArrayXd proj_sum_squares = (cls_k.matrix() * dir[k].matrix().transpose()).array().square().rowwise().sum();
                mse_array[k] = (sum_squares - proj_sum_squares).sum() / (D-dims[k]);
            // end
            }
            
        // end
        }
        
        // mse = sqrt(sum(mse)/sum(labels>0));
        int N = labels.size();
        ArrayXi ones = ArrayXi::Ones(N);
        ArrayXi zeros = ArrayXi::Zero(N);
        ArrayXi good_labels = (labels > -1).select(ones, zeros);
        int good_labels_count = good_labels.sum();
        double mse = std::sqrt( mse_array.sum() / good_labels_count );
        
        return mse;
    };

    static double ClusteringError(const ArrayXi &clustered_labels, const ArrayXi &true_labels)
    {
        // function p = clustering_error_improved(clustered_labels, true_labels)
        // 
        // % Right now this only works for equal numbers of clusters in both the true
        // %   and inferred labels. 
        // % For unequal numbers, the inefficient way would be to make K equal to the
        // %   larger of the two numbers of labels and do all the rest the same. That
        // %   is what we have implemented at this point. 
        // % The most efficient solution would be to make J the larger of the two and
        // %   K the smaller, and then instead of all of the permutations in the
        // %   subroutine, generate all of the nchoosek combinations, (j choose k), and
        // %   then generate all of the permutations of each of those and check them for
        // %   which is best. That would make k! * (n!/((n-k)!*k!)) possibilities over
        // %   which to find the max.
        // 
        // % Make sure both are row vectors (or columns will work)
        // clustered_labels = reshape(clustered_labels, 1, []);
        // true_labels = reshape(true_labels, 1, []);
        // 
        int N = clustered_labels.size();
        if (N != true_labels.size())
        {
            std::cerr << "MAPA::UtilityCalcs::ClusteringError -- number of true and clustered labels must be equal!" << std::endl;
            return 1.0;
        }
        
        // Js = unique(clustered_labels);
        // Ks = unique(true_labels);
        // K = max([length(Js) length(Ks)]);
        ArrayXi Js = MAPA::UtilityCalcs::UniqueMembers(clustered_labels);
        ArrayXi Ks = MAPA::UtilityCalcs::UniqueMembers(true_labels);
        int K = Js.size() > Ks.size() ? Js.size() : Ks.size();
        
        // planeSizes = zeros(1,K);
        ArrayXi planeSizes = ArrayXi::Zero(K);
        
        // for k = 1:length(Ks)
        //     planeSizes(k) = sum(true_labels == Ks(k));
        // end
        for (int k = 0; k < Ks.size(); k++)
        {
            ArrayXi cls_k_idxs = MAPA::UtilityCalcs::IdxsFromComparison(true_labels, "eq", Ks(k));
            planeSizes(k) = cls_k_idxs.size();
        }

        // counts_mtx = zeros(K,K);
        ArrayXXi counts_mtx = ArrayXXi::Zero(K,K);
        ArrayXi ones = ArrayXi::Ones(N);
        ArrayXi zeros = ArrayXi::Zero(N);
        
        // % k is the index of true labels
        // for k = 1:length(Ks)
        for (int k = 0; k < Ks.size(); k++)
        {
            //     % j is the index of the inferred, clustering results labels
            //     for j = 1:length(Js)
            for (int j = 0; j < Js.size(); j++)
            {
                //         % count up how many matches there are with this k and j combination
                //         counts_mtx(k,j) = sum( (clustered_labels == Js(j)) & (true_labels == Ks(k)) );
                ArrayXi clustered_matching = (clustered_labels == Js(j)).select(ones, zeros);
                ArrayXi true_matching = (true_labels == Ks(k)).select(ones, zeros);
                counts_mtx(k,j) = (clustered_matching * true_matching).sum();
            //     end
            }
        // end
        }
        

        // [n_correct, ~] = number_of_correctly_classified_points(counts_mtx);
        // p = 1 - n_correct/length(true_labels);
        
        int correct = MAPA::UtilityCalcs::NumberOfCorrectlyClassifiedPoints(counts_mtx);
        double p = 1.0 - (double)correct / (double)N;
        
        return p;
    };

    static int NumberOfCorrectlyClassifiedPoints(const ArrayXXi &counts_mtx)
    {

        // K = size(counts_mtx,1);
        int K = counts_mtx.rows();
        
        // % There are K! permutations of the integers 1:K
        // % We will consider k always to be the sequence 1:K
        // % and j will be the permuted forms
        // 
        // % Generate all permutations. Each row is one permutation.
        // permuted_js = perms(1:K);
        // 
        // % We'll keep track of the results for all of the permutations
        // % and then just use the permutation that gives us the best results
        // % (max number of correct answers)
        // n_permutations = size(permuted_js,1); % == K!
        // num_correct = zeros(1, n_permutations);
        
        // For the C++ version not going to store all results. Can generate next permutation
        // as we go, and just keep track of max and best permutation.
        
        // Sorted ascending is the lexographically lowest order, so start with that
        ArrayXi permuted_js = ArrayXi::LinSpaced(K, 0, K-1);
        int *pt = (int*)permuted_js.data();
        int num_correct = 0;
        int max_correct = -INFINITY;
        
        // Not returning these for now...
        ArrayXi optimal_js = permuted_js;
        
        // for pp = 1:n_permutations,
        //     for k = 1:K
        //         num_correct(pp) = num_correct(pp) + counts_mtx(k, permuted_js(pp,k));
        //     end
        // end
        
        do {
            num_correct = 0;
            for (int k = 0; k < K; k++)
            {
                num_correct += counts_mtx(k, permuted_js(k));
            }
            if (num_correct > max_correct)
            {
                max_correct = num_correct;
                optimal_js = permuted_js;
            }
        } while ( std::next_permutation(pt, pt+K) );

        // [n,I] = max(num_correct);
        // opt_perm = permuted_js(I);
        
        return max_correct;
    };

    // http://stackoverflow.com/questions/13290395/how-to-remove-a-certain-row-or-column-while-using-eigen-library-c
    template <typename Derived>
    static void removeRow(Array<Derived, Dynamic, Dynamic> &matrix, unsigned int rowToRemove)
    {
        unsigned int numRows = matrix.rows()-1;
        unsigned int numCols = matrix.cols();

        if( rowToRemove < numRows )
            matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);

        matrix.conservativeResize(numRows,numCols);
    };

    // http://stackoverflow.com/questions/13290395/how-to-remove-a-certain-row-or-column-while-using-eigen-library-c
    template <typename Derived>
    static void removeColumn(Array<Derived, Dynamic, Dynamic> &matrix, unsigned int colToRemove)
    {
        unsigned int numRows = matrix.rows();
        unsigned int numCols = matrix.cols()-1;

        if( colToRemove < numCols )
            matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.block(0,colToRemove+1,numRows,numCols-colToRemove);

        matrix.conservativeResize(numRows,numCols);
    }

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

    static ArrayXi UniqueMembers(const ArrayXi &labels)
    {
        // Find unique members by adding them to a std::set
        std::set<int> unique_members;
        for (int ii = 0; ii < labels.size(); ii++)
        {
            unique_members.insert(labels(ii));
        }
        
        // Copy values over into an Eigen array
        ArrayXi unique_array(unique_members.size());
        std::set<int>::iterator it;
        int ii = 0;
        for (it = unique_members.begin(); it != unique_members.end(); ++it)
        {
            unique_array(ii) = *it;
            ii++;
        }

        return unique_array;
    };
    
    static ArrayXi UniqueMembers(const std::vector<ArrayXi> &neighborhoods)
    {
        // Actually find unique members by adding them to a set
        std::set<int> unique_members;
        for (std::vector<int>::size_type ii = 0; ii != neighborhoods.size(); ii++)
        {
            int *nb = (int*)neighborhoods[ii].data();
            int nb_size = neighborhoods[ii].size();
            unique_members.insert(nb, nb+nb_size);
        }
        
        // Copy values over into an Eigen array
        ArrayXi unique_array(unique_members.size());
        std::set<int>::iterator it;
        int ii = 0;
        for (it = unique_members.begin(); it != unique_members.end(); ++it)
        {
            unique_array(ii) = *it;
            ii++;
        }

        return unique_array;
    };
    
    template <typename Derived>
    static ArrayXi IdxsFromComparison(const ArrayBase<Derived> &inarray, std::string comp, double value)
    {
        // Create an array of indices
        ArrayXi idxs = ArrayXi::LinSpaced(Eigen::Sequential, inarray.size(), 0, inarray.size()-1);
        ArrayXi found;
        
        // Replace all indices with -1 that don't pass the test
        if (comp == "gt")
        {
            found = (inarray > value).select(idxs, -1);
        }
        else if (comp == "gte")
        {
            found = (inarray >= value).select(idxs, -1);
        }
        else if (comp == "lt")
        {
            found = (inarray < value).select(idxs, -1);
        }
        else if (comp == "lte")
        {
            found = (inarray <= value).select(idxs, -1);
        }
        else if (comp == "eq")
        {
            found = (inarray == value).select(idxs, -1);
        }
        else
        {
            return found;
        }
       
        // Use stable_partition and the gtezero ( >= 0 ) to place all good indices
        // still in their original order, before bound
        int *bound;
        bound = std::stable_partition( found.data(), found.data()+found.size(), gtezero);

        // Resize indices array to exclude all of the -1s
        found.conservativeResize(bound-found.data());
        
        return found;
    }
    
    template <typename Derived>
    static ArrayXi IdxsAboveQuantile(const Array<Derived, Dynamic, 1> &inarray, double q_cutoff)
    {
        int N = inarray.size();

        Array<Derived, Dynamic, 1> Yd;
        ArrayXi IX;
        bool ascending = true;
        igl::sort(inarray, 1, ascending, Yd, IX);
    
        // Create an array cumulative probabilities
        ArrayXd quants = ArrayXd::LinSpaced(Eigen::Sequential, N, 0.5, N-0.5) / (double)N;

        // Replace all indices with -1 that don't pass the test
        ArrayXi IX_found = (quants > q_cutoff).select(IX, -1);
    
        // Use stable_partition and the gtezero ( >= 0 ) to place all good indices
        // still in their original order, before bound
        // http://www.cplusplus.com/reference/algorithm/stable_partition/
        int *bound;
        bound = std::stable_partition( IX_found.data(), IX_found.data()+IX_found.size(), gtezero);
    
        // Resize indices array to exclude all of the -1s
        IX_found.conservativeResize(bound-IX_found.data());
    
        // Resort indices back to original order
        ArrayXi Yi; 
        igl::sort(IX_found, 1, ascending, Yi, IX);
        
        return Yi;
    };

    static ArrayXi ClusteringInUSpace(const ArrayXXd &U, bool normalizeU)
    {
        // function indicesKmeans = clustering_in_U_space_min(U,opts)
        // % will cluster the rows of U (same number as the rows of A)
        // % minimal version using KMeansRex and only 'hard' seeds
        // 
        // K = size(U,2);
        
        ArrayXXd U_copy = U;
        int K = U.cols();
        
        // if opts.normalizeU,
        if (normalizeU)
        {
            //     rowNorms = sqrt(sum(U.^2,2));
            //     rowNorms(rowNorms==0) = 1;
            //     U = U./repmat(rowNorms,1,K);
            ArrayXd rowNorms = U_copy.square().rowwise().sum().sqrt();
            rowNorms = (rowNorms == 0).select(1.0, rowNorms);
            U_copy = U_copy.colwise() / rowNorms;
        // end
        }
        

        // [~, indicesKmeans] = KMeansRex(U, K, 100, 'mapa');
        KMeans::KMeansRex km(U_copy, K);
        ArrayXi indicesKmeans = km.GetClusterAssignments();

        // % This is a C++ routine that has 0-based indices, so add one here
        // indicesKmeans = indicesKmeans + 1;
        // 
        // 
        // end

        return indicesKmeans;
    };
    
    static int Mode(const ArrayXi &invec)
    {
        // Like Matlab implementation of mode(), this returns the lowest value
        // that ties the highest count
        
        int max_count = 0;
        int low_key = INFINITY;
    
        std::map<int, int> counts;
        std::map<int, int>::iterator it;
    
        for (int ii = 0; ii < invec.size(); ii++)
        {
            int vv = invec(ii);
            if (counts.find(vv) == counts.end())
            {
                counts[vv] = 1;
            }
            else
            {
                counts[vv] += 1;
            }
        
            if ((counts[vv] == max_count) && (vv < low_key))
            {
                low_key = vv;
            }
            else if (counts[vv] > max_count)
            {
                max_count = counts[vv];
                low_key = vv;
            }
        }
        return low_key;
    }

  private:
    
    static bool gtezero(int val) { return val >= 0; }
   
}; // class def

} // namespace MAPA

#endif
