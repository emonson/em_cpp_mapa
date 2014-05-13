#include <Eigen/Core>
using namespace Eigen;

#include "UtilityCalcs.h"

#include <iostream>
#include <vector>
#include <set>

int main(int argc, char * argv[])
{
    std::cout << "random integer neighborhoods of different sizes" << std::endl;
    std::vector<ArrayXi> neighborhoods;
    for (int ii = 0; ii < 5; ii++)
    {
        ArrayXi randints = (10.0 * ArrayXd::Random(ii+7)).cast<int>();
        std::cout << randints.transpose() << std::endl;
        neighborhoods.push_back(randints);
    }

    ArrayXi unique_array_out = MAPA::UtilityCalcs::UniqueMembers( neighborhoods );
    
    std::cout << std::endl << "unique members of all neighborhoods" << std::endl;
    std::cout << unique_array_out.transpose() << std::endl;

    return 0;
}
