#ifndef COMPUTINGBASES_H
#define COMPUTINGBASES_H

/* ComputingBases.h

Based on the Matlab code of
Guangliang Chen & Mauro Maggioni

Eric E Monson – 2014
Duke University

*/

#include "Options.h"
#include "SvdlibcSVD.h"
#include "UtilityCalcs.h"
#include <vector>
#include "Eigen/Dense"

using namespace Eigen;


namespace MAPA {

class ComputingBases {

public:

    ComputingBases(const ArrayXXd &data, const ArrayXi &labels, const ArrayXi &dims)
    {
        int K = labels.maxCoeff();
        
        // NOTE: labels that should be ignored need to be set negative
        for (int k = 0; k <= K; k++)
        {            
            ArrayXi cls_k_idxs = MAPA::UtilityCalcs::IdxsFromComparison(labels, "eq", k);
            ArrayXXd cls_k;
            int n_k;
            if (cls_k_idxs.size() > 0)
            {
                igl::slice(data, cls_k_idxs, 1, cls_k);
                n_k = cls_k.rows();
            }
            else
            {
                n_k = 0;
            }

            if (n_k >= (dims(k) + 1))
            {
                centers.push_back( cls_k.colwise().mean() );
                
                cls_k.rowwise() -= centers[k].transpose();
                
                SvdlibcSVD svd(cls_k, dims(k));
                ArrayXXd vk = svd.matrixV();           
                
                bases.push_back(vk.leftCols(dims(k)).transpose());
            }
            else
            {
                ArrayXd empty1d;
                ArrayXXd empty2d;
                centers.push_back( empty1d );
                bases.push_back( empty2d );
            }
        }
    };

    std::vector<ArrayXd> GetCenters()
    {
        return centers;
    };
    
    std::vector<ArrayXXd> GetBases()
    {
        return bases;
    };
    
private:

    std::vector<ArrayXd> centers;
    std::vector<ArrayXXd> bases;

}; // class def

} // namespace MAPA

#endif
