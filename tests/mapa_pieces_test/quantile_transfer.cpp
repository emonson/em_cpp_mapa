#include <Eigen/Core>

using namespace Eigen;

#include <iostream>

#include <igl/slice.h>

#include <stdlib.h>		/* NULL */
#include <time.h>			/* time */
#include <stdio.h>		/* srand */

#include "UtilityCalcs.h"


int main(int argc, char * argv[])
{
    // Reseed random number generator since Eigen Random.h doesn't do this itself
    srand( (unsigned int)time(NULL) );
    
    int N = 10;
    double q_cutoff = 0.2;
    
    // Doubles -----------------
    ArrayXd randvec = ArrayXd::Random(N);
    std::cout << std::endl << "random double vector" << std::endl;
    std::cout << randvec.transpose() << std::endl;

    ArrayXi Yi = MAPA::UtilityCalcs::IdxsAboveQuantile( randvec, q_cutoff );
    
    // Use slice to grab the desired indices of the original array that passed the test
    ArrayXd outdvec;
    igl::slice(randvec, Yi, outdvec);
    std::cout << outdvec.transpose() << std::endl;
    
    // Integers ---------
    ArrayXi randints = (10.0 * ArrayXd::Random(N) + 10).cast<int>();
    std::cout << std::endl << "random int vector" << std::endl;
    std::cout << randints.transpose() << std::endl;

    Yi = MAPA::UtilityCalcs::IdxsAboveQuantile( randints, q_cutoff );
    
    // Use slice to grab the desired indices of the original array that passed the test
    ArrayXi outivec;
    igl::slice(randints, Yi, outivec);
    std::cout << outivec.transpose() << std::endl;
    
    return 0;
}
