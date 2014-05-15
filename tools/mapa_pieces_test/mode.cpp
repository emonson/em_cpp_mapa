#include <Eigen/Core>
using namespace Eigen;

#include <iostream>
#include <cmath>
#include <map>

#include <igl/sort.h>

#include <stdlib.h>		/* NULL */
#include <time.h>			/* time */
#include <stdio.h>		/* srand */

int main(int argc, char * argv[])
{
    // Reseed random number generator since Eigen Random.h doesn't do this itself
    srand( (unsigned int)time(NULL) );

    // Integers ---------
    ArrayXi randints = (4.0 * ArrayXd::Random(20)).cast<int>();
    std::cout << std::endl << "random int vector" << std::endl;
    std::cout << randints.transpose() << std::endl;
    
    // overall results
    int max_count = 0;
    int low_key = INFINITY;
    
    std::map<int, int> counts;
    std::map<int, int>::iterator it;
    
    for (int ii = 0; ii < randints.size(); ii++)
    {
        int vv = randints(ii);
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
    
    ArrayXi Y;
    ArrayXi IX;
    ArrayXi randints_sorted;
    igl::sort(randints, 1, true, Y, IX);
    std::cout << Y.transpose() << std::endl;
    std::cout << "mode = " << low_key << " with count of " << max_count << std::endl;
}
