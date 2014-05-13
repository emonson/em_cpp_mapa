#include <Eigen/Core>

using namespace Eigen;

#include <algorithm>
#include <iostream>
#include <boost/math/distributions/uniform.hpp>

#include <igl/slice.h>

#include <stdlib.h>		/* NULL */
#include <time.h>			/* time */
#include <stdio.h>		/* srand */

bool gtezero(int val) { return val >= 0; }

int main(int argc, char * argv[])
{
// // stable_partition example
// #include <iostream>     // std::cout
// #include <algorithm>    // std::stable_partition
// #include <vector>       // std::vector
// 
// bool IsOdd (int i) { return (i%2)==1; }
// 
// int main () {
//   std::vector<int> myvector;
// 
//   // set some values:
//   for (int i=1; i<10; ++i) myvector.push_back(i); // 1 2 3 4 5 6 7 8 9
// 
//   std::vector<int>::iterator bound;
//   bound = std::stable_partition (myvector.begin(), myvector.end(), IsOdd);
// 
//   // print out content:
//   std::cout << "odd elements:";
//   for (std::vector<int>::iterator it=myvector.begin(); it!=bound; ++it)
//     std::cout << ' ' << *it;
//   std::cout << '\n';
// 
//   std::cout << "even elements:";
//   for (std::vector<int>::iterator it=bound; it!=myvector.end(); ++it)
//     std::cout << ' ' << *it;
//   std::cout << '\n';
// 
//   return 0;
// }

// VectorXi IP = I;
// IP.conservativeResize(stable_partition(
//   IP.data(), 
//   IP.data()+IP.size(), 
//   [&P](int i){return P(i)==0;})-IP.data());

    // Reseed random number generator since Eigen Random.h doesn't do this itself
    srand( (unsigned int)time(NULL) );

    ArrayXd randvec = ArrayXd::Random(10);
    
    // std::cout.precision(3);
    std::cout << randvec.transpose() << std::endl;
    
    // Create an array of indices
    ArrayXi idxs = ArrayXi::LinSpaced(Eigen::Sequential, randvec.size(), 0, randvec.size()-1);
    // Replace all indices with -1 that don't pass the test
    ArrayXi found = (randvec.array() > 0).select(idxs, -1);
    
    // Use stable_partition and the gtezero ( >= 0 ) to place all good indices
    // still in their original order, before bound
    int *bound;
    bound = std::stable_partition( found.data(), found.data()+found.size(), gtezero);
    std::cout << found.transpose() << std::endl;
    
    // Resize indices array to exclude all of the -1s
    found.conservativeResize(bound-found.data());
    std::cout << found.transpose() << std::endl;
    
    // Use slice to grab the desired indices of the original array that passed the test
    ArrayXd gt0;
    igl::slice(randvec, found, gt0);
    std::cout << gt0.transpose() << std::endl;
    
    return 0;
}
