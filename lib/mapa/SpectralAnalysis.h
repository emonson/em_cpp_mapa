#ifndef SPECTRALANALYSIS_H
#define SPECTRALANALYSIS_H

/* SpectralAnalysis.h

Based on the Matlab code of
Guangliang Chen & Mauro Maggioni

Eric E Monson – 2014
Duke University

*/

#include <Eigen/Core>
#include <Eigen/SVD>

#include <iostream>


using namespace Eigen;


namespace MAPA {

class SpectralAnalysis {

public:

    // function [planeDims, labels, err] =  spectral_analysis(X, U, allPtsInOptRegions, invColMap, localDims, nOutliers)
    SpectralAnalysis(const ArrayXXd &X, 
                     const ArrayXXd &U, 
                     ArrayXi allPtsInOptRegions,
                     ArrayXi invColMap,
                     ArrayXi localDims,
                     int nOutliers = 0)
    {
        // K = size(U,2);
        int K = U.cols();
        
        // SCCopts = struct();
        // SCCopts.normalizeU = 1;
        // SCCopts.seedType = 'hard';
        // indicesKmeans = clustering_in_U_space(U,K,SCCopts);
        
        // NOTE: Only allowing 'hard' seed type right now...
        bool normalizeU = true;
        ArrayXi indicesKmeans = MAPA::UtilityCalcs::ClusteringInUSpace(U, normalizeU);
        
        // planeDims = zeros(1,K);
        planeDims = ArrayXi::Zero(K);
        
        // for k = 1:K
        for (int k = 0; k < K; k++)
        {
            //     % Find the original point indices of the rows of A/U in this cluster
            //     class_k = allPtsInOptRegions(indicesKmeans == k);
            ArrayXi cluster_k_idxs = MAPA::UtilityCalcs::IdxsFromComparison(indicesKmeans, "eq", k);
            ArrayXi class_k;
            igl::slice(allPtsInOptRegions, cluster_k_idxs, class_k);
        
            //     % Figure out which of these points are seed points
            //     temp = invColMap(class_k);  
            //     temp = temp(temp>0);
            ArrayXi temp0;
            igl::slice(invColMap, class_k, temp0);
            ArrayXi temp1 = MAPA::UtilityCalcs::IdxsFromComparison(temp0, "gte", 0);
            ArrayXi temp_valid;
            igl::slice(temp0, temp1, temp_valid);
        
            //     % Then see what dimensionality most of these seed points in this
            //     % cluster have
            //     planeDims(k) = mode(localDims(temp));
            ArrayXi cluster_dims;
            igl::slice(localDims, temp_valid, cluster_dims);
            std::cout << cluster_dims.transpose() << std::endl;
            planeDims(k) = MAPA::UtilityCalcs::Mode(cluster_dims);
            
        // end
        }
        
            // * * * * OKAY TO HERE * * * *
        
        // 
        // [planeCenters, planeBases] = computing_bases(X(allPtsInOptRegions,:), indicesKmeans, planeDims);
        // dists = p2pdist(X,planeCenters,planeBases);
        // 
        // %[N,D] = size(X);
        // %dists = dists./repmat(D-planeDims, N, 1);
        // 
        // [dists,labels] = min(dists,[],2);
        // 
        // if nOutliers>0
        //     % new labels
        //     labels1 = labels;
        //     objectiveFun1 = sum(sqrt(dists));
        //     outliers1 = [];
        //     % old labels
        //     objectiveFun = Inf;
        //     labels = [];
        //     outliers = [];
        //     while objectiveFun1<objectiveFun
        //         labels = labels1;
        //         objectiveFun = objectiveFun1;
        //         outliers = outliers1;       
        //         [~, I] = sort(dists, 'descend');
        //         outliers1=I(1:nOutliers);
        //         labels1(outliers1)=0;
        //         [planeCenters, planeBases] = computing_bases(X, labels1, planeDims);
        //         dists = p2pdist(X,planeCenters,planeBases);
        //         [dists,labels1] = min(dists,[],2);
        //         objectiveFun1 = sum(sqrt(dists));
        //     end
        //     labels(outliers)=0;
        // end
        // 
        // err = L2error(X, planeDims, labels);
        // 
    };

    ArrayXi GetPlaneDims()
    {
        return planeDims;
    };
    
    ArrayXi GetLabels()
    {
        return labels;
    };
    
    double GetError()
    {
        return err;
    };
    

private:

    ArrayXi planeDims;
    ArrayXi labels;
    double err;
    
}; // class def

} // namespace MAPA

#endif
