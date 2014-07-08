#include "mersenneTwister2002.h"

#include <iostream>
#include <assert.h>
// #include <stdlib.h>		/* NULL */
// #include <time.h>			/* time */
// #include <stdio.h>		/* srand */

#include <stdexcept>
#undef eigen_assert
#define eigen_assert(x) \
  if (!x) { throw (std::runtime_error("TEST FAILED: Some wrong values were generated")); }

#include <cmath>
#define close_enough(a,b) (std::abs(a-b) < 0.0001)
	
int main(int argc, char * argv[])
{
  // unsigned int seed = (unsigned int)time(NULL);
  unsigned int seed = 0;

  KMeans::MersenneTwister twist;
  std::cout << "MersenneTwister2002 test with seed = 0" << std::endl;
  twist.InitGenrand(seed);
  double d1 = twist.GenrandDouble();
  double d2 = twist.GenrandDouble();
  
  std::cout.unsetf ( std::ios::floatfield );                // floatfield not set
  std::cout.precision(10);
  std::cout << "Two random double values" << std::endl;
  std::cout << d1 << ", " << d2 << std::endl;
	
	eigen_assert(close_enough(d1, 0.5488135024));
	eigen_assert(close_enough(d2, 0.5928446165))
	
	std::cout << "TEST PASSED" << std::endl;
	
  return 0;
}
