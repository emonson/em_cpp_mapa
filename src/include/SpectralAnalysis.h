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

#include "ComputingBases.h"

#include <iostream>
#include <vector>
#include <cmath>

using namespace Eigen;


namespace MAPA {

class SpectralAnalysis {

public:

    SpectralAnalysis(const ArrayXXd &X, 
                     const ArrayXXd &U, 
                     ArrayXi allPtsInOptRegions,
                     ArrayXi invColMap,
                     ArrayXi localDims,
                     int nOutliers = 0)
    {
        err = INFINITY;
        int K = U.cols();
                
        // NOTE: Only allowing 'hard' seed type right now...
        bool normalizeU = true;
        ArrayXi indicesKmeans = MAPA::UtilityCalcs::ClusteringInUSpace(U, normalizeU);
        
        planeDims = ArrayXi::Zero(K);
        
        for (int k = 0; k < K; k++)
        {
            //     % Find the original point indices of the rows of A/U in this cluster
            ArrayXi cluster_k_idxs = MAPA::UtilityCalcs::IdxsFromComparison(indicesKmeans, "eq", k);
            ArrayXi class_k;
            igl::slice(allPtsInOptRegions, cluster_k_idxs, class_k);
        
            //     % Figure out which of these points are seed points
            ArrayXi temp0;
            igl::slice(invColMap, class_k, temp0);
            ArrayXi temp1 = MAPA::UtilityCalcs::IdxsFromComparison(temp0, "gte", 0);
            ArrayXi temp_valid;
            igl::slice(temp0, temp1, temp_valid);
        
            //     % Then see what dimensionality most of these seed points in this
            //     % cluster have
            ArrayXi cluster_dims;
            igl::slice(localDims, temp_valid, cluster_dims);

            if (cluster_dims.size() > 0)
            {
                planeDims(k) = MAPA::UtilityCalcs::Mode(cluster_dims);
            }
            else
            {
                // HACK!! -- Sometimes there are no seed points in a particular cluster...
                // TODO: This probably isn't right!!
                planeDims(k) = MAPA::UtilityCalcs::Mode(localDims);
            }
        }
        
        ArrayXXd X_allPtsOpt;
        igl::slice(X, allPtsInOptRegions, 1, X_allPtsOpt);
        MAPA::ComputingBases computing_bases(X_allPtsOpt, indicesKmeans, planeDims);
        std::vector<ArrayXd> planeCenters = computing_bases.GetCenters();
        std::vector<ArrayXXd> planeBases = computing_bases.GetBases();
        
        ArrayXXd all_dists = MAPA::UtilityCalcs::P2Pdists(X, planeCenters, planeBases);
        
        int N = all_dists.rows();
        ArrayXd dists = ArrayXd::Constant(N, INFINITY);
        labels = ArrayXi::Constant(N, -1);
        
        for (int ii = 0; ii < N; ii++)
        {
            dists[ii] = all_dists.row(ii).minCoeff(&labels[ii]);
        }
        
        // TODO: Not implementing for now since not default and confused...
        
        // if nOutliers>0
        //     % new labels
        //     labels1 = labels;
        //     objectiveFun1 = sum(sqrt(dists));
        //     outliers1 = [];
        //     % old labels
        //     objectiveFun = Inf;
        //     labels = [];
        //     outliers = [];
        //     % NOTE: I don't quite see how this is a progressive optimization...?
        //     while objectiveFun1<objectiveFun
        //         % if we're doing better, go ahead and grab the labels from the last round
        //         labels = labels1;
        //         % and keep track of sum of distances
        //         objectiveFun = objectiveFun1;
        //         % and lock in the outliers from last time, too
        //         outliers = outliers1;
        //         % sort descending so first points are furtheset pointss
        //         [~, I] = sort(dists, 'descend');
        //         % grab indices of points farthest from any planes
        //         outliers1=I(1:nOutliers);
        //         % setting labels as zero gets them ignored in computing_bases
        //         labels1(outliers1)=0;
        //         % compute new planes ignoring these outlying points
        //         [planeCenters, planeBases] = computing_bases(X, labels1, planeDims);
        //         % calculate distances from all points to these new bases
        //         dists = p2pdist(X,planeCenters,planeBases);
        //         % figure out which groups all points should belong to using these new bases
        //         [dists,labels1] = min(dists,[],2);
        //         objectiveFun1 = sum(sqrt(dists));
        //     end
        //     labels(outliers)=0;
        // end
        
        // Recompute the bases for all points for this error calculation
        MAPA::ComputingBases computing_bases_all(X, labels, planeDims);
        planeCenters_all = computing_bases_all.GetCenters();
        planeBases_all = computing_bases_all.GetBases();
        
        err = MAPA::UtilityCalcs::L2error( X, labels, planeDims, planeCenters_all, planeBases_all );
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
    
    std::vector<ArrayXd> GetPlaneCenters()
    {
        return planeCenters_all;
    };
    
    std::vector<ArrayXXd> GetPlaneBases()
    {
        return planeBases_all;
    };


private:

    ArrayXi planeDims;
    ArrayXi labels;
    double err;
    std::vector<ArrayXd> planeCenters_all;
    std::vector<ArrayXXd> planeBases_all;
    
}; // class def

} // namespace MAPA

#endif
