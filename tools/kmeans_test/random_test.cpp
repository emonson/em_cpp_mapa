#include "mersenneTwister2002.h"

#include <iostream>
#include <assert.h>
// #include <stdlib.h>		/* NULL */
// #include <time.h>			/* time */
// #include <stdio.h>		/* srand */

int main(int argc, char * argv[])
{
  // unsigned int seed = (unsigned int)time(NULL);
  unsigned int seed = 0;

  KMeans::MersenneTwister twist(seed);
  
  double d1 = twist.GenrandDouble();
  double d2 = twist.GenrandDouble();
  
  std::cout << d1 << ", " << d2 << std::endl;

	  
  return 0;
}
