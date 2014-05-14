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

    // Doubles ---------
    ArrayXd randvec = ArrayXd::Random(10);
    std::cout << std::endl << "random double vector" << std::endl;
    std::cout << randvec.transpose() << std::endl;
    
    // ---------
    ArrayXi IX = MAPA::UtilityCalcs::IdxsFromComparison(randvec, "gt", 0.2);
    
    // Use slice to grab the desired indices of the original array that passed the test
    ArrayXd out_d;
    igl::slice(randvec, IX, out_d);
    std::cout << std::endl << "case: > 0.2" << std::endl;
    std::cout << IX.transpose() << std::endl;
    std::cout << out_d.transpose() << std::endl;
    
    // ---------
    IX = MAPA::UtilityCalcs::IdxsFromComparison(randvec, "lt", -0.2);
    
    igl::slice(randvec, IX, out_d);
    std::cout << std::endl << "case: < -0.2" << std::endl;
    std::cout << IX.transpose() << std::endl;
    std::cout << out_d.transpose() << std::endl;
    
    // Integers ---------
    ArrayXi randints = (4.0 * ArrayXd::Random(20)).cast<int>();
    std::cout << std::endl << "random int vector" << std::endl;
    std::cout << randints.transpose() << std::endl;

    // ---------
    IX = MAPA::UtilityCalcs::IdxsFromComparison(randints, "eq", 0);
    
    ArrayXi out_i;
    igl::slice(randints, IX, out_i);
    std::cout << std::endl << "case: == 0" << std::endl;
    std::cout << IX.transpose() << std::endl;
    std::cout << out_i.transpose() << std::endl;
    
    // ---------
    IX = MAPA::UtilityCalcs::IdxsFromComparison(randints, "gte", 1);
    
    igl::slice(randints, IX, out_i);
    std::cout << std::endl << "case: >= 1" << std::endl;
    std::cout << IX.transpose() << std::endl;
    std::cout << out_i.transpose() << std::endl;
    
    // ---------
    IX = MAPA::UtilityCalcs::IdxsFromComparison(randints, "lte", -1);
    
    igl::slice(randints, IX, out_i);
    std::cout << std::endl << "case: <= -1" << std::endl;
    std::cout << IX.transpose() << std::endl;
    std::cout << out_i.transpose() << std::endl;
    
    return 0;
}
