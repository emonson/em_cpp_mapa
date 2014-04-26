#ifndef COMPUTINGBASES_H
#define COMPUTINGBASES_H

/* ComputingBases.h

Based on the Matlab code of
Guangliang Chen & Mauro Maggioni

Eric E Monson – 2014
Duke University

*/

#include "Options.h"
#include <vector>
#include "Eigen/Dense"

using namespace Eigen;


namespace MAPA {

class ComputingBases {

public:

    ComputingBases(const ArrayXXd &Xin, unsigned int Kin)
    {

    };

private:

    std::vector<ArrayXd> centers;
    std::vector<ArrayXXd> bases;

}; // class def

} // namespace MAPA

#endif
