#include <Eigen/Core>

using namespace Eigen;

#include <algorithm>
#include <iostream>

#include <igl/slice.h>
#include <igl/sort.h>

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
    
    int N = 10;
    double q_cutoff = 0.2;

    ArrayXd randvec = ArrayXd::Random(N);
    
    // std::cout.precision(3);
    std::cout << randvec.transpose() << std::endl;
    
    ArrayXd Yd;
    ArrayXi IX;
    bool ascending = true;
    igl::sort(randvec, 1, ascending, Yd, IX);
    
    // Create an array cumulative probabilities
    ArrayXd quants = ArrayXd::LinSpaced(Eigen::Sequential, N, 0.5, N-0.5) / (double)N;
    std::cout << quants << std::endl;

    // Replace all indices with -1 that don't pass the test
    ArrayXi IX_found = (quants.array() > q_cutoff).select(IX, -1);
    
    // Use stable_partition and the gtezero ( >= 0 ) to place all good indices
    // still in their original order, before bound
    int *bound;
    bound = std::stable_partition( IX_found.data(), IX_found.data()+IX_found.size(), gtezero);
    std::cout << IX_found.transpose() << std::endl;
    
    // Resize indices array to exclude all of the -1s
    IX_found.conservativeResize(bound-IX_found.data());
    std::cout << IX_found.transpose() << std::endl;
    
    // Resort indices back to original order
    ArrayXi Yi; 
    igl::sort(IX_found, 1, ascending, Yi, IX);
    std::cout << Yi.transpose() << std::endl;
    
    // Use slice to grab the desired indices of the original array that passed the test
    ArrayXd outvec;
    igl::slice(randvec, Yi, outvec);
    std::cout << outvec.transpose() << std::endl;
    
    return 0;
}
