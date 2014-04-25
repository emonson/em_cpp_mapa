#ifndef UTILITYCALCS_H
#define UTILITYCALCS_H

/* UtilityCalcs.h

Based on the Matlab code of
Guangliang Chen & Mauro Maggioni

Eric E Monson – 2014

*/

#include "Eigen/Dense"

using namespace Eigen;


namespace MAPA {

class UtilityCalcs {

public:

    static ArrayXXd P2Pdist(const ArrayXXd &X, const ArrayXXd &centers, const ArrayXXd &bases)
    {

    };

    static ArrayXd L2error(const ArrayXXd &data, const ArrayXi &dim, const ArrayXi &idx)
    {

    };

    static double ClusteringError(const ArrayXi &indices, const ArrayXi &trueLabels)
    {

    };

    
}; // class def

} // namespace MAPA

#endif
